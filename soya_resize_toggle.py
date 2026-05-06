"""
SoyaResizeToggle – Lanczos resize in a single node.

enable=true: resize image to target_width x target_height using Lanczos.
enable=false: pass image through unchanged.
"""

import torch


class SoyaResizeToggle_mdsoya:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "doit"
    CATEGORY = "Soya"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enable": ("STRING", {"default": "true"}),
                "image": ("IMAGE",),
                "target_width": ("INT", {"default": 0, "min": 0, "max": 16384}),
                "target_height": ("INT", {"default": 0, "min": 0, "max": 16384}),
            },
        }

    def doit(self, *, enable, image, target_width, target_height):
        use = enable.strip().lower() in ("true", "1", "yes")

        if not use:
            print("[SoyaResizeToggle] DISABLED — bypassing")
            return (image,)

        _, img_h, img_w, _ = image.shape
        out_w = target_width if target_width > 0 else img_w
        out_h = target_height if target_height > 0 else img_h

        if image.shape[1] == out_h and image.shape[2] == out_w:
            print("[SoyaResizeToggle] ENABLED — already at target size, skipping")
            return (image,)

        print(f"[SoyaResizeToggle] ENABLED — resizing {img_w}x{img_h} -> {out_w}x{out_h} (Lanczos)")
        resized = self._lanczos_resize(image, out_h, out_w)
        return (resized,)

    @staticmethod
    def _lanczos_resize(tensor, out_h, out_w):
        """Lanczos resize using PIL (high quality, negligible overhead)."""
        import numpy as np
        from PIL import Image

        resized = []
        for i in range(tensor.shape[0]):
            img = tensor[i].cpu().numpy()
            pil_img = Image.fromarray((img * 255).astype(np.uint8))
            pil_img = pil_img.resize((out_w, out_h), Image.LANCZOS)
            resized.append(torch.from_numpy(np.array(pil_img)).float() / 255.0)
        return torch.stack(resized)
