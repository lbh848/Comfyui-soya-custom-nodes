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

        # 4. Resize back to original dimensions (lanczos on GPU)
        if decoded.shape[1] != out_h or decoded.shape[2] != out_w:
            decoded = self._lanczos_resize(decoded, out_h, out_w)

        return (torch.clamp(decoded, 0.0, 1.0),)

    @staticmethod
    def _lanczos_resize(tensor, out_h, out_w):
        """Lanczos3 resize via separable convolution on GPU."""
        import math
        _, in_h, in_w, ch = tensor.shape
        img = tensor.movedim(-1, 1)  # (B, C, H, W)

        a = 3  # Lanczos3

        def _make_kernel(size, scale):
            """1D Lanczos kernel."""
            x = torch.arange(size, dtype=torch.float32) - size // 2
            x = x * (1.0 / scale)
            kernel = torch.where(
                x == 0, torch.ones_like(x),
                torch.where(
                    torch.abs(x) < a,
                    (a * torch.sin(math.pi * x) * torch.sin(math.pi * x / a))
                    / (math.pi * math.pi * x * x),
                    torch.zeros_like(x),
                ),
            )
            return kernel / kernel.sum()

        # Horizontal pass
        kw = max(int(in_w / out_w * a) * 2 + 1, 3)
        kernel_w = _make_kernel(kw, in_w / out_w).to(img.device)
        pad = kw // 2
        padded = F.pad(img, [pad, pad, 0, 0], mode='reflect')
        # Apply per-channel
        B, C, H, Wp = padded.shape
        kh = kernel_w.view(1, 1, 1, -1).expand(C, 1, 1, -1)
        out = F.conv2d(padded.view(B * C, 1, H, Wp), kh, groups=1)
        out = out.view(B, C, H, -1)[:, :, :, :out_w]

        # Vertical pass
        kh_size = max(int(in_h / out_h * a) * 2 + 1, 3)
        kernel_h = _make_kernel(kh_size, in_h / out_h).to(img.device)
        pad = kh_size // 2
        padded = F.pad(out, [0, 0, pad, pad], mode='reflect')
        B, C, Hp, W = padded.shape
        kv = kernel_h.view(1, 1, -1, 1).expand(C, 1, -1, 1)
        out = F.conv2d(padded.view(B * C, 1, Hp, W), kv, groups=1)
        out = out.view(B, C, -1, W)[:, :, :out_h, :]

        return out.movedim(1, -1)  # (B, out_h, out_w, C)
