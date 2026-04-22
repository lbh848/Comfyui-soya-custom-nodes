"""
SoyaMaskBrightness – Brightness & contrast adjustment in masked areas only.
Uses the same -100~100 unit scale as SoyaColorAdjust.
Mask value acts as per-pixel intensity: higher mask = stronger effect.
"""

import torch


class SoyaMaskBrightness_mdsoya:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "brightness": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "contrast": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "adjust"
    CATEGORY = "Soya/Image"

    def adjust(self, image, mask, brightness, contrast):
        result = image.clone().float()

        # Expand mask: (B, H, W) -> (B, H, W, 1)
        m = mask.unsqueeze(-1)
        if m.shape[0] == 1 and result.shape[0] > 1:
            m = m.expand(result.shape[0], -1, -1, -1)

        # Scale -100~100 -> -1~1 (same convention as ColorAdjust)
        b = brightness * 0.01
        c = contrast * 0.01

        # Brightness: additive shift weighted by mask
        if b != 0.0:
            result = result + m * b

        # Contrast: push/pull toward 0.5, weighted by mask
        if c != 0.0:
            factor = 1.0 + c * 2.0
            contrasted = torch.tensor(0.5, device=result.device).lerp(result, factor)
            result = result + m * (contrasted - result)

        return (result.clamp(0.0, 1.0),)
