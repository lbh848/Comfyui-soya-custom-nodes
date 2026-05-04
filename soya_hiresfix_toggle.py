"""
SoyaHiresfixToggle – Hires.fix in a single node.

Pipeline (enable=true):
  VAE encode → KSampler → VAE decode → resize to original
  tiled_vae: VRAM-saving option for 12GB GPUs (uses tiled encode/decode)

Upscaling is handled by SoyaUpscaleToggle – feed its output into this node.
enable=false passes the original image through.
"""

import torch
import torch.nn.functional as F


class SoyaHiresfixToggle_mdsoya:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "doit"
    CATEGORY = "Soya"

    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers
        return {
            "required": {
                "enable": ("STRING", {"default": "true"}),
                "image": ("IMAGE",),
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "tiled_vae": ("STRING", {"default": "false"}),
                "tile_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 32}),
            },
            "optional": {
                "target_width": ("INT", {"default": 0, "min": 0, "max": 16384}),
                "target_height": ("INT", {"default": 0, "min": 0, "max": 16384}),
            },
        }

    def doit(self, *, enable, image, model, positive, negative, vae,
             seed, steps, cfg, sampler_name, scheduler, denoise,
             tiled_vae, tile_size, target_width=0, target_height=0):

        use = enable.strip().lower() in ("true", "1", "yes")

        if not use:
            print("[SoyaHiresfixToggle] DISABLED — bypassing")
            return (image,)

        use_tiled = tiled_vae.strip().lower() in ("true", "1", "yes")
        print(f"[SoyaHiresfixToggle] ENABLED — running hires.fix pipeline (tiled_vae={use_tiled})")

        # Use target dimensions if provided, otherwise use input image size
        _, img_h, img_w, _ = image.shape
        out_h = target_height if target_height > 0 else img_h
        out_w = target_width if target_width > 0 else img_w

        # 1. VAE encode
        if use_tiled:
            latent = vae.encode_tiled(
                image[:, :, :, :3],
                tile_x=tile_size, tile_y=tile_size, overlap=tile_size // 8,
            )
        else:
            latent = vae.encode(image[:, :, :, :3])
        latent_dict = {"samples": latent}

        # 2. KSampler
        from nodes import common_ksampler
        result = common_ksampler(
            model, seed, steps, cfg, sampler_name, scheduler,
            positive, negative, latent_dict,
            denoise=denoise,
        )
        sampled = result[0]["samples"]

        # 3. VAE decode
        if use_tiled:
            compression = vae.spacial_compression_decode()
            decoded = vae.decode_tiled(
                sampled,
                tile_x=tile_size // compression,
                tile_y=tile_size // compression,
                overlap=tile_size // compression // 8,
            )
        else:
            decoded = vae.decode(sampled)

        # 4. Resize back to original dimensions (lanczos)
        if decoded.shape[1] != out_h or decoded.shape[2] != out_w:
            decoded = self._lanczos_resize(decoded, out_h, out_w)

        return (torch.clamp(decoded, 0.0, 1.0),)

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
