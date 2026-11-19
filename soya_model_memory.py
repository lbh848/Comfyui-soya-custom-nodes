"""Optional model phase boundaries that retain the host weight caches."""

from contextlib import contextmanager
from functools import wraps
import logging
import time


class PhaseIO:
    """Apply a phase at each actual encoder/decoder call inside a composite node."""

    def __init__(self, component, phase, methods):
        self.component = component
        self.phase = phase
        self.methods = methods

    def __getattr__(self, name):
        method = getattr(self.component, name)
        if name not in self.methods:
            return method

        @wraps(method)
        def run(*args, **kwargs):
            with model_phase(self.component.patcher, True, self.phase):
                return method(*args, **kwargs)
        return run


def _sampling_phase(executor, *args, **kwargs):
    with model_phase(executor.class_obj.model_patcher, True, "DiT", release_after=True):
        return executor(*args, **kwargs)


def sequential_sampling_model(model):
    """Start the DiT phase after any internal conditioning and SnapRef encoding."""
    from comfy import memory_management
    from comfy.patcher_extension import WrappersMP

    if not memory_management.aimdo_enabled or model.load_device.type != "cuda":
        return model
    key = "soya_sequential_offload"
    if model.get_wrappers(WrappersMP.OUTER_SAMPLE, key):
        return model
    patched = model.clone()
    patched.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, key, _sampling_phase)
    return patched


def _offload_models(models, device, phase):
    import torch
    import comfy.model_management as model_management

    models = [model for model in models if model.loaded_size() > 0]
    if not models:
        return
    started = time.monotonic()
    logging.info("[Soya:ModelMemory] %s: returning %d model(s) to RAM", phase, len(models))
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
                "[Soya:ModelMemory] %s: %s VRAM %.1f -> %.1f MiB; host cache retained",
                phase, type(model.model).__name__, before / (1024 ** 2),
                model.loaded_size() / (1024 ** 2),
            )
            if host_before is not None:
                logging.info("[Soya:ModelMemory] %s: host weights %.1f -> %.1f MiB",
                             phase, host_before / (1024 ** 2), model.loaded_ram_size() / (1024 ** 2))
        model_management.soft_empty_cache()
    logging.info("[Soya:ModelMemory] %s: handoff completed in %.3fs", phase, time.monotonic() - started)


@contextmanager
def model_phase(model, enabled, phase, *, release_after=False):
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

    import torch
    import comfy_aimdo.control as aimdo_control

    active = [model, *model.get_nested_additional_models(), *model.model_patches_models()]
    active_ids = {patcher.clone_base_uuid for patcher in active}
    loaded = [patcher for patcher in model_management.loaded_models()
              if patcher is not None and patcher.load_device == device]
    previous_headroom = aimdo_control.get_simple_vram_headroom()
    capacity = torch.cuda.get_device_properties(device).total_memory
    # On the observed 8 GB device, the encoder alone stalled at ~6.5 GB
    # framebuffer usage. Leave 3 GiB outside AIMDO's tracked allocation budget
    # for activation bursts and driver/WDDM overhead. AIMDO counts both Torch
    # allocations and weight pages, evicting weight pages as activations grow.
    # This is a conservative opt-in budget, not a measured working-set size.
    headroom = max(previous_headroom, min(3 * 1024 ** 3, capacity // 2))
    aimdo_control.set_simple_vram_headroom(headroom)
    logging.info("[Soya:ModelMemory] %s: tracked VRAM budget %.1f MiB; headroom %.1f MiB",
                 phase, max(0, capacity - headroom) / (1024 ** 2), headroom / (1024 ** 2))
    try:
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
    finally:
        # The public setter does not launch or synchronize CUDA work. Restore
        # the user's process setting even when inference raises or is cancelled.
        aimdo_control.set_simple_vram_headroom(previous_headroom)
