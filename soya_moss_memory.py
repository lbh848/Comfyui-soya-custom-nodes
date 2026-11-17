"""Optional MOSS phase boundaries that retain the host weight caches."""

from contextlib import contextmanager
import logging
import time


def _offload_models(models, device, phase):
    import torch
    import comfy.model_management as model_management

    models = [model for model in models if model.loaded_size() > 0]
    if not models:
        return
    started = time.monotonic()
    logging.info("[Soya:MossMemory] %s: returning %d model(s) to RAM", phase, len(models))
    with torch.cuda.device(device):
        # Complete previous work, including prefetch streams, before releasing
        # their VRAM mappings. This is a boundary, never an in-forward hook.
        torch.cuda.synchronize(device)
        for model in models:
            before = model.loaded_size()
            host_before = model.loaded_ram_size() if model.is_dynamic() else None
            # detach()/unload_all_models() also discard DynamicVRAM host
            # buffers. The patcher's VRAM-only API keeps those buffers and
            # LoRA patch caches registered for the next managed load.
            model.partially_unload(model.offload_device, 1e30)
            logging.info(
                "[Soya:MossMemory] %s: %s VRAM %.1f -> %.1f MiB; host cache retained",
                phase, type(model.model).__name__, before / (1024 ** 2),
                model.loaded_size() / (1024 ** 2),
            )
            if host_before is not None:
                logging.info("[Soya:MossMemory] %s: host weights %.1f -> %.1f MiB",
                             phase, host_before / (1024 ** 2), model.loaded_ram_size() / (1024 ** 2))
        model_management.soft_empty_cache()
    logging.info("[Soya:MossMemory] %s: handoff completed in %.3fs", phase, time.monotonic() - started)


@contextmanager
def moss_model_phase(model, enabled, phase, *, release_after=False):
    """Keep the active model on its device and return inactive weights to RAM.

    Normal Comfy loaders still perform all uploads and patch registration.
    No model objects, conditioning results, or CPU weight caches are deleted.
    """
    if not enabled:
        yield
        return

    import comfy.memory_management
    import comfy.model_management as model_management

    device = model.load_device
    if not comfy.memory_management.aimdo_enabled or device.type != "cuda":
        yield
        return

    active = [model, *model.get_nested_additional_models(), *model.model_patches_models()]
    active_ids = {patcher.clone_base_uuid for patcher in active}
    loaded = [patcher for patcher in model_management.loaded_models()
              if patcher is not None and patcher.load_device == device]
    _offload_models([patcher for patcher in loaded if patcher.clone_base_uuid not in active_ids],
                    device, phase)
    yield
    if release_after:
        # Only release after successful sampling: do not mask an inference
        # failure with another CUDA operation in a finally block.
        loaded = [patcher for patcher in model_management.loaded_models()
                  if patcher is not None and patcher.load_device == device
                  and patcher.clone_base_uuid in active_ids]
        _offload_models(loaded, device, phase + " complete")
