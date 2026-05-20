import torch
import comfy.utils


class SoyaScaleBy_mdsoya:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "scale_by": ("FLOAT", {"default": 1.25, "min": 0.01, "max": 100.0, "step": 0.01}),
                "upscale_method": (
                    "STRING",
                    {
                        "default": "lanczos",
                        "choices": ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"],
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "doit"
    CATEGORY = "Soya"

    def doit(self, image, scale_by, upscale_method):
        if scale_by == 1.0:
            return (image,)

        _, h, w, _ = image.shape
        new_w = max(1, round(w * scale_by))
        new_h = max(1, round(h * scale_by))

        samples = image.movedim(-1, 1)  # (B, C, H, W)
        resized = comfy.utils.common_upscale(samples, new_w, new_h, upscale_method, "disabled")
        resized = resized.movedim(1, -1)  # (B, H, W, C)

        return (resized,)
