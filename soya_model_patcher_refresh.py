import threading
import traceback


_REFRESH_LOCK = threading.Lock()
_REFRESH_GENERATION = 0


class SoyaModelPatcherRefresh_mdsoya:
    """Return a fresh ModelPatcher wrapper without unloading or changing patches.

    This intentionally mirrors the two state-refreshing lines used by the old
    image-diagnostic ModelProbe, but does not install any probe wrapper or
    instrumentation.  LoRA patches, model options, callbacks, wrappers, and
    the shared resident model are preserved by ComfyUI's ``ModelPatcher.clone``.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "refresh"
    CATEGORY = "Soya/Model"
    DESCRIPTION = (
        "Creates a fresh ModelPatcher wrapper while preserving the resident "
        "model and all configured patches. It does not unload models or clear VRAM."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # The purpose of this node is to replace the wrapper for every prompt,
        # including prompts whose upstream MODEL output was served from cache.
        return float("nan")

    def refresh(self, model):
        global _REFRESH_GENERATION
        try:
            refreshed = model.clone()
            refreshed.model_options = {**refreshed.model_options}
            with _REFRESH_LOCK:
                _REFRESH_GENERATION += 1
                generation = _REFRESH_GENERATION
            print(
                "[Soya:ModelPatcherRefresh] "
                f"generation={generation}, input_id={id(model)}, "
                f"output_id={id(refreshed)}, "
                f"clone_base_uuid={getattr(refreshed, 'clone_base_uuid', None)}, "
                f"patches_uuid={getattr(refreshed, 'patches_uuid', None)}, "
                f"patch_count={len(getattr(refreshed, 'patches', {}) or {})}, "
                f"model_option_keys={sorted((refreshed.model_options or {}).keys())}"
            )
            return (refreshed,)
        except Exception as exc:
            print(
                "[Soya:ModelPatcherRefresh] Failed to clone ModelPatcher: "
                f"input_type={type(model).__name__}, input_id={id(model)}, "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise
