class SoyaIPAPatchConfig_mdsoya:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device": (["CPU", "CUDA"], {"default": "CPU"}),
                "max_face_count": ("INT", {"default": 5, "min": 1, "max": 20}),
                "yolo_confidence": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.01}),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IPA_PATCH_CONFIG",)
    RETURN_NAMES = ("config",)
    FUNCTION = "build_config"
    CATEGORY = "Soya/IPA"

    def build_config(self, device, max_face_count, yolo_confidence, debug):
        return ({
            "device": device,
            "max_face_count": max_face_count,
            "yolo_confidence": yolo_confidence,
            "debug": debug,
        },)
