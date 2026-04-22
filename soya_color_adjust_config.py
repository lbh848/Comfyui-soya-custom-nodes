"""
SoyaColorAdjustConfig – packages color adjustment parameters into a single output
for cleaner wiring into SoyaBatchDetailer.
"""


class SoyaColorAdjustConfig_mdsoya:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": False}),
                "temperature": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "tint": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "saturation": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "vibrance": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "brightness": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "contrast": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 2.2, "step": 0.1}),
                "mask_sigma": ("FLOAT", {"default": 0.4, "min": 0.01, "max": 2.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("COLOR_ADJUST",)
    RETURN_NAMES = ("color_adjustment",)
    FUNCTION = "build"
    CATEGORY = "Soya/Detailer"

    def build(self, enabled, temperature, tint, saturation, vibrance,
              brightness, contrast, gamma, mask_sigma):
        return ({
            "enabled": enabled,
            "temperature": temperature,
            "tint": tint,
            "saturation": saturation,
            "vibrance": vibrance,
            "brightness": brightness,
            "contrast": contrast,
            "gamma": gamma,
            "mask_sigma": mask_sigma,
        },)
