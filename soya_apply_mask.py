"""
SoyaApplyMask – Multiply image by mask, keeping only the masked region.
Pixels outside the mask become 0 (black), preserving the mask shape.
"""

import torch


class SoyaApplyMask_mdsoya:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply"
    CATEGORY = "Soya/Image"

    def apply(self, image, mask):
        # mask: (B, H, W) -> (B, H, W, 1) for broadcast with image (B, H, W, C)
        m = mask.unsqueeze(-1)
        if m.shape[0] == 1 and image.shape[0] > 1:
            m = m.expand(image.shape[0], -1, -1, -1)

        return (image * m,)