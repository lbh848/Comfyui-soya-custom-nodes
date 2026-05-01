class SoyaIPAdapterPatchCleaner_mdsoya:
    """Clears stale attn2 patches from model to prevent IPAdapter patch accumulation.

    Place this node before any IPAdapter node (FaceID, Advanced, etc.) to ensure
    no leftover attention patches from previous executions persist on a cached model.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "execute"
    CATEGORY = "ipadapter/faceid"

    def execute(self, model):
        transformer_options = model.model_options.setdefault("transformer_options", {})
        patches_replace = transformer_options.setdefault("patches_replace", {})
        if "attn2" in patches_replace:
            del patches_replace["attn2"]
            print("[Soya:PatchCleaner] Cleared stale attn2 patches")
        return (model,)
