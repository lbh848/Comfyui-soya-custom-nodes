"""Optional hires.fix pass using the shared Spectrum Mod Guidance sampler."""

import traceback

import torch

import comfy.samplers

from .soya_spectrum_mod_guidance import (
    _SPECTRUM_MOD_OPTIONS_TYPE,
    sample_spectrum_mod_guidance,
)


class SoyaHiresfixSpectrumToggle_mdsoya:
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
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "clip": ("CLIP",),
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
                "spectrum_options": (
                    _SPECTRUM_MOD_OPTIONS_TYPE,
                    {
                        "tooltip": (
                            "Spectrum Mod Guidance Options 노드 출력입니다. "
                            "미연결 시 1st sampler와 동일한 기본값을 사용합니다."
                        ),
                    },
                ),
            },
        }

    def doit(self, *, enable, image, model, positive, negative, vae, clip,
             seed, steps, cfg, sampler_name, scheduler, denoise,
             tiled_vae, tile_size, spectrum_options=None):

        use = str(enable or "").strip().casefold() in ("true", "1", "yes", "on")

        if not use:
            print("[SoyaHiresfixSpectrumToggle] DISABLED — bypassing")
            return (image,)

        use_tiled = str(tiled_vae or "").strip().casefold() in (
            "true",
            "1",
            "yes",
            "on",
        )
        print(
            "[SoyaHiresfixSpectrumToggle] ENABLED — running hires.fix pipeline "
            f"(tiled_vae={use_tiled}, spectrum_options_connected={spectrum_options is not None})"
        )

        try:
            # 1. VAE encode
            if use_tiled:
                latent = vae.encode_tiled(
                    image[:, :, :, :3],
                    tile_x=tile_size,
                    tile_y=tile_size,
                    overlap=tile_size // 8,
                )
            else:
                latent = vae.encode(image[:, :, :, :3])
            latent_dict = {"samples": latent}

            # 2. Run the exact same Spectrum Mod Guidance core as 1st sampler.
            result = sample_spectrum_mod_guidance(
                model,
                clip,
                seed,
                steps,
                cfg,
                sampler_name,
                scheduler,
                positive,
                negative,
                latent_dict,
                denoise,
                spectrum_options,
                log_prefix="[SoyaHiresfixSpectrumToggle]",
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

            out = torch.clamp(decoded, 0.0, 1.0)
            if out.ndim == 5 and out.shape[1] == 1:
                out = out.squeeze(1)
            return (out,)
        except Exception as exc:
            print(
                "[SoyaHiresfixSpectrumToggle] hires.fix 실패: "
                f"tiled_vae={use_tiled}, tile_size={tile_size}, "
                f"steps={steps}, cfg={cfg}, denoise={denoise}, error={exc}"
            )
            traceback.print_exc()
            raise
