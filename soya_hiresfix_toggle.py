"""
SoyaHiresfixToggle – Hires.fix in a single node.

Pipeline (enable=true):
  upscale_model → VAE encode → KSampler → VAE decode → resize to original
  tiled_vae: VRAM-saving option for 12GB GPUs (uses tiled encode/decode)

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
                "upscale_model": ("UPSCALE_MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "tiled_vae": ("STRING", {"default": "false"}),
                "tile_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 32}),
            },
        }

    def doit(self, *, enable, image, model, positive, negative, vae,
             upscale_model, seed, steps, cfg, sampler_name, scheduler, denoise,
             tiled_vae, tile_size):

        use = enable.strip().lower() in ("true", "1", "yes")

        if not use:
            print("[SoyaHiresfixToggle] DISABLED — bypassing")
            return (image,)

        use_tiled = tiled_vae.strip().lower() in ("true", "1", "yes")
        print(f"[SoyaHiresfixToggle] ENABLED — running hires.fix pipeline (tiled_vae={use_tiled})")
        _, orig_h, orig_w, _ = image.shape

        # 1. Upscale
        upscaled = self._upscale_with_model(upscale_model, image)

        # 2. VAE encode
        if use_tiled:
            latent = vae.encode_tiled(
                upscaled[:, :, :, :3],
                tile_x=tile_size, tile_y=tile_size, overlap=tile_size // 8,
            )
        else:
            latent = vae.encode(upscaled[:, :, :, :3])
        latent_dict = {"samples": latent}

        # 3. KSampler
        from nodes import common_ksampler
        result = common_ksampler(
            model, seed, steps, cfg, sampler_name, scheduler,
            positive, negative, latent_dict,
            denoise=denoise,
        )
        sampled = result[0]["samples"]

        # 4. VAE decode
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

        # 5. Resize back to original dimensions
        if decoded.shape[1] != orig_h or decoded.shape[2] != orig_w:
            decoded = F.interpolate(
                decoded.movedim(-1, 1),
                size=(orig_h, orig_w),
                mode="bicubic",
                align_corners=False,
            ).movedim(1, -1)

        return (torch.clamp(decoded, 0.0, 1.0),)

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
