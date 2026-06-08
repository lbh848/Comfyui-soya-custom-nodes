"""
SoyaHiresfixDDToggle – Hires.fix + Differential Diffusion in a single node.

VAE encode → DD mask application → KSampler → VAE decode, all in one node.

When dd_mask is connected (e.g. combined_mask from IPA Apply Face Patches):
  - Background (mask=0) → full denoise (upscale quality)
  - Face regions (mask=1) → controlled by face_denoise param (preserve + IP-Adapter)
When dd_mask is not connected → behaves like regular Hiresfix Toggle.
"""

import torch


class SoyaHiresfixDDToggle_mdsoya:
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
                "face_denoise": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "dd_mask": ("MASK",),
            },
        }

    @staticmethod
    def _dd_forward(sigma, denoise_mask, extra_options):
        model = extra_options["model"]
        step_sigmas = extra_options["sigmas"]

        sigma_to = model.inner_model.model_sampling.sigma_min
        if step_sigmas[-1] > sigma_to:
            sigma_to = step_sigmas[-1]
        sigma_from = step_sigmas[0]

        ts_from = model.inner_model.model_sampling.timestep(sigma_from)
        ts_to = model.inner_model.model_sampling.timestep(sigma_to)
        current_ts = model.inner_model.model_sampling.timestep(sigma[0])

        threshold = (current_ts - ts_to) / (ts_from - ts_to)
        return (denoise_mask >= threshold).to(denoise_mask.dtype)

    def doit(self, *, enable, image, model, positive, negative, vae,
             seed, steps, cfg, sampler_name, scheduler, denoise,
             tiled_vae, tile_size, face_denoise, dd_mask=None):

        use = enable.strip().lower() in ("true", "1", "yes")
        if not use:
            print("[SoyaHiresfixDDToggle] DISABLED — bypassing")
            return (image,)

        use_tiled = tiled_vae.strip().lower() in ("true", "1", "yes")
        has_dd = dd_mask is not None and dd_mask.numel() > 0

        print(f"[SoyaHiresfixDDToggle] ENABLED — hires.fix (tiled_vae={use_tiled}, dd={has_dd})")

        # 1. VAE encode
        if use_tiled:
            latent = vae.encode_tiled(
                image[:, :, :, :3],
                tile_x=tile_size, tile_y=tile_size, overlap=tile_size // 8,
            )
        else:
            latent = vae.encode(image[:, :, :, :3])
        latent_dict = {"samples": latent}

        # 2. Apply Differential Diffusion
        if has_dd:
            # Input mask: face regions = ~1.0, background = ~0.0
            # DD semantics: higher value → pixel stays active longer → more denoise
            # Want: bg=1.0 (full denoise), face=face_denoise (controlled)
            scaled_mask = 1.0 - dd_mask * (1.0 - face_denoise)
            latent_dict["noise_mask"] = scaled_mask.reshape(
                (-1, 1, scaled_mask.shape[-2], scaled_mask.shape[-1])
            )
            model = model.clone()
            model.set_model_denoise_mask_function(self._dd_forward)
            print(f"[SoyaHiresfixDDToggle] DD — face_denoise={face_denoise:.2f}")

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

        result = torch.clamp(decoded, 0.0, 1.0)
        if result.ndim == 5 and result.shape[1] == 1:
            result = result.squeeze(1)
        return (result,)
