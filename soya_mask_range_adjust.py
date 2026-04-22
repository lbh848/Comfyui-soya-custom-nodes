"""
SoyaMaskRangeAdjust – Remap mask value distribution to a target range.
Linearly remaps 0~1 mask values to [min_value, max_value],
with an optional gamma curve for non-linear distribution control.
"""

import torch


class SoyaMaskRangeAdjust_mdsoya:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "min_value": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_value": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "adjust"
    CATEGORY = "Soya/Mask"

    def adjust(self, mask, min_value, max_value, gamma):
        # mask shape: (B, H, W) or (H, W)
        result = mask.clone().float()

        # Apply gamma curve for non-linear distribution shaping
        if gamma != 1.0:
            result = result.pow(gamma)

        # Linear remap: 0→min_value, 1→max_value
        result = result * (max_value - min_value) + min_value

        return (result.clamp(0.0, 1.0),)
