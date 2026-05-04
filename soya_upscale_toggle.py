"""
SoyaUpscaleToggle – Upscale with enable toggle.

enable=true  → upscale image with model → output
enable=false → pass image through as-is
"""

import torch


class SoyaUpscaleToggle_mdsoya:
    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "original_width", "original_height")
    FUNCTION = "doit"
    CATEGORY = "Soya"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enable": ("STRING", {"default": "true"}),
                "image": ("IMAGE",),
                "upscale_model": ("UPSCALE_MODEL",),
            },
        }

    def doit(self, *, enable, image, upscale_model):
        use = enable.strip().lower() in ("true", "1", "yes")
        _, orig_h, orig_w, _ = image.shape

        if not use:
            print("[SoyaUpscaleToggle] DISABLED — bypassing")
            return (image, orig_w, orig_h)

        print("[SoyaUpscaleToggle] ENABLED — upscaling")
        return (self._upscale_with_model(upscale_model, image), orig_w, orig_h)

    @staticmethod
    def _upscale_with_model(upscale_model, image):
        import comfy.utils
        import comfy.model_management

        device = comfy.model_management.get_torch_device()
        upscale_model.to(device)
        in_img = image.movedim(-1, -3).to(device)

        tile = 512
        overlap = 32
        oom = True
        try:
            while oom:
                try:
                    steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(
                        in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap
                    )
                    pbar = comfy.utils.ProgressBar(steps)
                    s = comfy.utils.tiled_scale(
                        in_img, lambda a: upscale_model(a),
                        tile_x=tile, tile_y=tile, overlap=overlap,
                        upscale_amount=upscale_model.scale, pbar=pbar,
                    )
                    oom = False
                except comfy.model_management.OOM_EXCEPTION:
                    tile //= 2
                    if tile < 128:
                        raise
        finally:
            upscale_model.to("cpu")

        return torch.clamp(s.movedim(-3, -1), min=0, max=1.0)
