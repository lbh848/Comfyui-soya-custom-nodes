"""
SoyaHiresfixSpectrumToggle – Hires.fix using KSampler (Spectrum + Mod Guidance).

Pipeline (enable=true):
  VAE encode → Spectrum + Mod Guidance KSampler → VAE decode
  tiled_vae: VRAM-saving option for 12GB GPUs (uses tiled encode/decode)

enable=false passes the original image through.
"""

import importlib.util
import os
import sys
import torch

import comfy.samplers

# The sibling custom node directory uses hyphens (e.g. "comfyui-spectrum-ksampler")
# which Python can't import directly. We locate it case-insensitively, then
# register it as a proper package via importlib so that internal relative
# imports (from .forecaster, etc.) work.
_parent = os.path.normpath(os.path.dirname(__file__) + "/..")
_pkg_name = "comfyui_spectrum_ksampler"

if _pkg_name not in sys.modules:
    _pkg_dir = None
    _target = "comfyui-spectrum-ksampler".lower()
    for entry in os.listdir(_parent):
        if entry.lower() == _target:
            _pkg_dir = os.path.join(_parent, entry)
            break
    if _pkg_dir is None:
        raise ImportError("Cannot find comfyui-spectrum-ksampler in custom_nodes")

    _spec = importlib.util.spec_from_file_location(
        _pkg_name,
        os.path.join(_pkg_dir, "__init__.py"),
        submodule_search_locations=[_pkg_dir],
    )
    _pkg = importlib.util.module_from_spec(_spec)
    sys.modules[_pkg_name] = _pkg
    _spec.loader.exec_module(_pkg)

from comfyui_spectrum_ksampler.mod_guidance import AUTO_ADAPTER_SENTINEL, setup_mod_guidance
from comfyui_spectrum_ksampler.spectrum import spectrum_sample

# Constants from comfyui_spectrum_ksampler/nodes.py — inlined here to avoid
# importing that module (its bare name "nodes" clashes with ComfyUI's nodes.py
# when accessed through the package).
MOD_W_PROFILE_OFF = "off"
MOD_W_PROFILES = {
    "step_i8_skip27": dict(w=3.0, start_layer=8,  end_layer=27, taper=0, taper_scale=0.25, final_w=0.0),
    "step_i14":       dict(w=3.0, start_layer=14, end_layer=-1, taper=0, taper_scale=0.25, final_w=0.0),
    "uniform_w3":     dict(w=3.0, start_layer=0,  end_layer=-1, taper=0, taper_scale=0.25, final_w=0.0),
}
DEFAULT_MOD_W_PROFILE = "step_i8_skip27"

_SPECTRUM_DEFAULTS = dict(
    window_size=2.0,
    flex_window=0.25,
    warmup_steps=7,
    blend_w=0.3,
    cheby_degree=3,
    ridge_lambda=0.1,
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
                "quality_tags": (
                    "STRING",
                    {
                        "default": "absurdres, highres, masterpiece, best quality, score_9, score_8, newest, year 2025, year 2024",
                        "multiline": True,
                        "dynamicPrompts": True,
                    },
                ),
                "mod_w_profile": (
                    [MOD_W_PROFILE_OFF] + list(MOD_W_PROFILES.keys()),
                    {"default": DEFAULT_MOD_W_PROFILE},
                ),
                "dcw": (["on", "off"], {"default": "on"}),
                "tiled_vae": ("STRING", {"default": "false"}),
                "tile_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 32}),
            },
        }

    def doit(self, *, enable, image, model, positive, negative, vae, clip,
             seed, steps, cfg, sampler_name, scheduler, denoise,
             quality_tags, mod_w_profile, dcw,
             tiled_vae, tile_size):

        use = enable.strip().lower() in ("true", "1", "yes")

        if not use:
            print("[SoyaHiresfixSpectrumToggle] DISABLED — bypassing")
            return (image,)

        use_tiled = tiled_vae.strip().lower() in ("true", "1", "yes")
        print(f"[SoyaHiresfixSpectrumToggle] ENABLED — running hires.fix pipeline "
              f"(tiled_vae={use_tiled}, mod_w_profile={mod_w_profile}, dcw={dcw})")

        # 1. VAE encode
        if use_tiled:
            latent = vae.encode_tiled(
                image[:, :, :, :3],
                tile_x=tile_size, tile_y=tile_size, overlap=tile_size // 8,
            )
        else:
            latent = vae.encode(image[:, :, :, :3])
        latent_dict = {"samples": latent}

        # 2. Setup mod guidance on model
        if mod_w_profile == MOD_W_PROFILE_OFF:
            m = model
        else:
            profile = MOD_W_PROFILES.get(mod_w_profile) or MOD_W_PROFILES[DEFAULT_MOD_W_PROFILE]
            m = model.clone()
            setup_mod_guidance(
                m, clip, positive, negative, None, quality_tags,
                profile["w"],
                start_layer=profile["start_layer"],
                end_layer=profile["end_layer"],
                taper=profile["taper"],
                taper_scale=profile["taper_scale"],
                final_w=profile["final_w"],
            )

        # 3. Spectrum sample
        dcw_mode = "auto" if dcw == "on" else "off"
        result = spectrum_sample(
            m, seed, steps, cfg, sampler_name, scheduler,
            positive, negative, latent_dict, denoise,
            **_SPECTRUM_DEFAULTS,
            dcw_mode=dcw_mode,
            dcw_lambda=0.01,
            dcw_band_mask="LL",
            dcw_calibrator=AUTO_ADAPTER_SENTINEL,
            clip=clip,
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

        out = torch.clamp(decoded, 0.0, 1.0)
        if out.ndim == 5 and out.shape[1] == 1:
            out = out.squeeze(1)
        return (out,)
