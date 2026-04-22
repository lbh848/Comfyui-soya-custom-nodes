"""
SoyaColorAdjust – Pure PyTorch color adjustment node.
Replicates the GLSL shader logic without disk I/O.
Output is always 3-channel RGB.
"""

import torch


class SoyaColorAdjust_mdsoya:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "temperature": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "tint": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "saturation": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "vibrance": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 2.2, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "adjust"
    CATEGORY = "Soya/Image"

    # Constants matching the GLSL shader
    INPUT_SCALE = 0.01
    TEMP_TINT_PRIMARY = 0.3
    TEMP_TINT_SECONDARY = 0.15
    VIBRANCE_BOOST = 2.0
    SATURATION_BOOST = 2.0
    SKIN_PROTECTION = 0.5
    EPSILON = 1e-4
    LUMA_WEIGHTS = torch.tensor([0.299, 0.587, 0.114])

    def adjust(self, image, temperature, tint, saturation, vibrance, gamma):
        # Ensure RGB – strip alpha if present
        if image.shape[-1] == 4:
            image = image[:, :, :, :3]

        color = image.clone()  # (B, H, W, 3)

        # Scale inputs: -100/100 → -1/1
        t = temperature * self.INPUT_SCALE
        ti = tint * self.INPUT_SCALE
        v = vibrance * self.INPUT_SCALE
        s = saturation * self.INPUT_SCALE

        # Temperature (warm/cool)
        color[..., 0] += t * self.TEMP_TINT_PRIMARY  # R
        color[..., 2] -= t * self.TEMP_TINT_PRIMARY  # B

        # Tint (green/magenta)
        color[..., 1] += ti * self.TEMP_TINT_PRIMARY  # G
        color[..., 0] -= ti * self.TEMP_TINT_SECONDARY  # R
        color[..., 2] -= ti * self.TEMP_TINT_SECONDARY  # B

        color = color.clamp(0.0, 1.0)

        # Vibrance with skin protection
        if v != 0.0:
            r, g, b = color[..., 0], color[..., 1], color[..., 2]
            maxC = torch.maximum(r, torch.maximum(g, b))
            minC = torch.minimum(r, torch.minimum(g, b))
            sat = maxC - minC
            luma_w = self.LUMA_WEIGHTS.to(color.device)
            gray = (color * luma_w).sum(dim=-1)

            if v < 0.0:
                # Desaturate: -100 → gray
                gray3 = gray.unsqueeze(-1).expand_as(color)
                color = gray3.lerp(color, 1.0 + v)
            else:
                # Boost less saturated colors more
                vibranceAmt = v * (1.0 - sat)
                # Branchless skin tone protection
                isWarmTone = ((b <= g) & (g <= r)).float()
                warmth = (r - b) / maxC.clamp(min=self.EPSILON)
                skinTone = isWarmTone * warmth * sat * (1.0 - sat)
                vibranceAmt = vibranceAmt * (1.0 - skinTone * self.SKIN_PROTECTION)
                gray3 = gray.unsqueeze(-1).expand_as(color)
                color = gray3.lerp(color, 1.0 + vibranceAmt.unsqueeze(-1) * self.VIBRANCE_BOOST)

        # Saturation
        if s != 0.0:
            luma_w = self.LUMA_WEIGHTS.to(color.device)
            gray = (color * luma_w).sum(dim=-1, keepdim=True)
            satMix = 1.0 + (s if s < 0.0 else s * self.SATURATION_BOOST)
            color = gray.lerp(color, satMix)

        # Gamma (ColorCorrect style: power function)
        if gamma != 1.0:
            color = torch.pow(color.clamp(0.0, 1.0), gamma)

        return (color.clamp(0.0, 1.0),)
