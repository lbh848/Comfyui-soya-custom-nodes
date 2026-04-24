class SoyaFaceIDModelSwitch_mdsoya:
    """Switch between IPAdapter FaceID model and normal model based on update_model flag."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "update_model": ("STRING", {"default": "true", "multiline": False}),
                "model_faceid": ("MODEL",),
                "model_normal": ("MODEL",),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "execute"
    CATEGORY = "ipadapter/faceid"

    def execute(self, update_model, model_faceid, model_normal):
        if update_model == "true":
            return (model_faceid,)
        return (model_normal,)
