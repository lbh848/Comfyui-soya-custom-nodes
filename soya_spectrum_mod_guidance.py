"""Shared Spectrum Mod Guidance options and sampling implementation."""

from __future__ import annotations

import math
import traceback


_SPECTRUM_MOD_OPTIONS_TYPE = "SOYA_SPECTRUM_MOD_GUIDANCE_OPTIONS"

_MOD_GUIDANCE_QUALITY_TAGS = (
    "masterpiece, best quality, highres, absurdres, very aesthetic"
)
_MOD_GUIDANCE_QUALITY_NEG = (
    "score_1, score_2, score_3, worst quality, lowres, old, bad hands, bad anatomy"
)
_MOD_GUIDANCE_PROFILE = "step_i8_skip27"
_MOD_GUIDANCE_ADAPTIVE_SMC_ALPHA = 0.10
_MOD_GUIDANCE_REFRESH_RATIO = -1.0
_MOD_GUIDANCE_PROFILE_OFF = "off"
_MOD_GUIDANCE_PROFILES = {
    "step_i8_skip27": {
        "w": 3.0,
        "start_layer": 8,
        "end_layer": 27,
        "taper": 0,
        "taper_scale": 0.25,
        "final_w": 0.0,
    },
    "step_i14": {
        "w": 3.0,
        "start_layer": 14,
        "end_layer": -1,
        "taper": 0,
        "taper_scale": 0.25,
        "final_w": 0.0,
    },
    "uniform_w3": {
        "w": 3.0,
        "start_layer": 0,
        "end_layer": -1,
        "taper": 0,
        "taper_scale": 0.25,
        "final_w": 0.0,
    },
}
_MOD_GUIDANCE_PROFILE_CHOICES = [
    _MOD_GUIDANCE_PROFILE_OFF,
    *_MOD_GUIDANCE_PROFILES.keys(),
]
_SPECTRUM_DEFAULTS = {
    "window_size": 2.0,
    "flex_window": 0.25,
    "warmup_steps": 6,
    "blend_w": 0.3,
    "cheby_degree": 3,
    "ridge_lambda": 0.1,
}


def _default_spectrum_mod_options():
    return {
        "positive": _MOD_GUIDANCE_QUALITY_TAGS,
        "negative": _MOD_GUIDANCE_QUALITY_NEG,
        "mod_w_profile": _MOD_GUIDANCE_PROFILE,
        "refresh_ratio": _MOD_GUIDANCE_REFRESH_RATIO,
        "adaptive_smc_alpha": _MOD_GUIDANCE_ADAPTIVE_SMC_ALPHA,
    }


def _resolve_spectrum_mod_options(options, *, log_prefix="[Spectrum Mod Guidance]"):
    if options is None:
        print(f"{log_prefix} 옵션 미연결 · 기존 기본값을 사용합니다")
        return _default_spectrum_mod_options()
    if not isinstance(options, dict):
        raise ValueError(
            "spectrum_options는 Spectrum Mod Guidance Options 노드의 출력이어야 "
            f"합니다: type={type(options)!r}"
        )

    missing = [
        key for key in _default_spectrum_mod_options()
        if key not in options
    ]
    if missing:
        raise ValueError(f"spectrum_options 필수 값이 없습니다: {missing}")

    positive = options["positive"]
    negative = options["negative"]
    if not isinstance(positive, str):
        raise ValueError(f"spectrum_options.positive는 문자열이어야 합니다: {positive!r}")
    if not isinstance(negative, str):
        raise ValueError(f"spectrum_options.negative는 문자열이어야 합니다: {negative!r}")

    profile_name = str(options["mod_w_profile"] or "").strip()
    if profile_name not in _MOD_GUIDANCE_PROFILE_CHOICES:
        raise ValueError(
            f"지원하지 않는 mod_w_profile입니다: {profile_name!r} "
            f"(지원값: {', '.join(_MOD_GUIDANCE_PROFILE_CHOICES)})"
        )

    try:
        refresh_ratio = float(options["refresh_ratio"])
        adaptive_smc_alpha = float(options["adaptive_smc_alpha"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "spectrum_options의 refresh_ratio와 adaptive_smc_alpha는 숫자여야 "
            f"합니다: refresh_ratio={options['refresh_ratio']!r}, "
            f"adaptive_smc_alpha={options['adaptive_smc_alpha']!r}"
        ) from exc
    if not math.isfinite(refresh_ratio) or not -1.0 <= refresh_ratio <= 1.0:
        raise ValueError(
            "spectrum_options.refresh_ratio는 -1.0~1.0의 유한수여야 합니다: "
            f"{refresh_ratio!r}"
        )
    if not math.isfinite(adaptive_smc_alpha) or not 0.0 <= adaptive_smc_alpha <= 1.0:
        raise ValueError(
            "spectrum_options.adaptive_smc_alpha는 0.0~1.0의 유한수여야 합니다: "
            f"{adaptive_smc_alpha!r}"
        )
    return {
        "positive": positive.strip(),
        "negative": negative.strip(),
        "mod_w_profile": profile_name,
        "refresh_ratio": refresh_ratio,
        "adaptive_smc_alpha": adaptive_smc_alpha,
    }


class SoyaSpectrumModGuidanceOptions_mdsoya:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": (
                    "STRING",
                    {
                        "default": _MOD_GUIDANCE_QUALITY_TAGS,
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "Mod Guidance가 향하도록 만들 품질 프롬프트입니다.",
                    },
                ),
                "negative": (
                    "STRING",
                    {
                        "default": _MOD_GUIDANCE_QUALITY_NEG,
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "Mod Guidance 품질 축의 반대편 프롬프트입니다.",
                    },
                ),
                "mod_w_profile": (
                    _MOD_GUIDANCE_PROFILE_CHOICES,
                    {"default": _MOD_GUIDANCE_PROFILE},
                ),
                "refresh_ratio": (
                    "FLOAT",
                    {
                        "default": _MOD_GUIDANCE_REFRESH_RATIO,
                        "min": -1.0,
                        "max": 1.0,
                        "step": 0.01,
                        "round": 0.001,
                        "tooltip": (
                            "-1 = 기존 window 스케줄, 0 = SEA 자동, "
                            "0보다 크면 SEA 명시 비율"
                        ),
                    },
                ),
                "adaptive_smc_alpha": (
                    "FLOAT",
                    {
                        "default": _MOD_GUIDANCE_ADAPTIVE_SMC_ALPHA,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "round": 0.001,
                    },
                ),
            }
        }

    RETURN_TYPES = (_SPECTRUM_MOD_OPTIONS_TYPE,)
    RETURN_NAMES = ("spectrum_options",)
    FUNCTION = "build"
    CATEGORY = "sampling"
    DESCRIPTION = (
        "1st sampler와 Hiresfix Spectrum Toggle에서 사용할 품질 프롬프트, "
        "Mod Guidance 프로필, SEA 비율과 adaptive SMC alpha를 묶어 전달합니다."
    )

    def build(
        self,
        positive,
        negative,
        mod_w_profile,
        refresh_ratio,
        adaptive_smc_alpha,
    ):
        try:
            options = _resolve_spectrum_mod_options({
                "positive": positive,
                "negative": negative,
                "mod_w_profile": mod_w_profile,
                "refresh_ratio": refresh_ratio,
                "adaptive_smc_alpha": adaptive_smc_alpha,
            })
            print(
                "[Spectrum Mod Guidance Options] 옵션 생성: "
                f"profile={options['mod_w_profile']}, "
                f"refresh_ratio={options['refresh_ratio']:.3f}, "
                f"adaptive_smc_alpha={options['adaptive_smc_alpha']:.3f}"
            )
            return (options,)
        except Exception as exc:
            print(f"[Spectrum Mod Guidance Options] 옵션 생성 실패: error={exc}")
            traceback.print_exc()
            raise


def _spectrum_mod_guidance_runtime(*, log_prefix="[Spectrum Mod Guidance]"):
    """Return Spectrum's loaded low-level functions without re-importing its package."""
    import nodes as comfy_nodes

    sampler_class = comfy_nodes.NODE_CLASS_MAPPINGS.get("SpectrumKSampler")
    if sampler_class is None:
        message = (
            "Spectrum 저수준 런타임을 찾지 못했습니다. "
            "comfyui-spectrum-ksampler가 설치·로드되었는지 확인하세요."
        )
        print(f"{log_prefix} {message}")
        raise RuntimeError(message)

    namespace = getattr(getattr(sampler_class, "sample", None), "__globals__", {})
    setup_mod_guidance = namespace.get("setup_mod_guidance")
    spectrum_sample = namespace.get("spectrum_sample")
    if not callable(setup_mod_guidance) or not callable(spectrum_sample):
        message = (
            "Spectrum 저수준 함수가 없습니다: "
            f"setup_mod_guidance={callable(setup_mod_guidance)}, "
            f"spectrum_sample={callable(spectrum_sample)}"
        )
        print(f"{log_prefix} {message}")
        raise RuntimeError(message)
    return setup_mod_guidance, spectrum_sample


def sample_spectrum_mod_guidance(
    model,
    clip,
    seed,
    steps,
    cfg,
    sampler_name,
    scheduler,
    positive,
    negative,
    latent_image,
    denoise,
    spectrum_options=None,
    *,
    log_prefix="[Spectrum Mod Guidance]",
):
    """Run the single shared Spectrum Mod Guidance sampling path."""
    options = _resolve_spectrum_mod_options(
        spectrum_options,
        log_prefix=log_prefix,
    )
    profile_name = options["mod_w_profile"]
    if clip is None and profile_name != _MOD_GUIDANCE_PROFILE_OFF:
        message = "SpectrumKSamplerModGuidance 실행에 CLIP 입력이 필요합니다"
        print(f"{log_prefix} {message}")
        raise ValueError(message)

    setup_mod_guidance, spectrum_sample = _spectrum_mod_guidance_runtime(
        log_prefix=log_prefix,
    )
    if profile_name == _MOD_GUIDANCE_PROFILE_OFF:
        mod_model = model
        print(f"{log_prefix} mod_w_profile=off · Mod Guidance 적용 생략")
    else:
        profile = _MOD_GUIDANCE_PROFILES[profile_name]
        mod_model = model.clone()
        setup_mod_guidance(
            mod_model,
            clip,
            positive,
            negative,
            None,
            options["positive"],
            profile["w"],
            quality_neg=options["negative"],
            start_layer=profile["start_layer"],
            end_layer=profile["end_layer"],
            taper=profile["taper"],
            taper_scale=profile["taper_scale"],
            final_w=profile["final_w"],
        )

    refresh_ratio = options["refresh_ratio"]
    schedule = "window" if refresh_ratio < 0.0 else "sea"
    if schedule == "window":
        refresh_ratio = -1.0
    print(
        f"{log_prefix} SpectrumKSamplerModGuidance 옵션 적용: "
        f"profile={profile_name}, "
        f"adaptive_smc_alpha={options['adaptive_smc_alpha']:.3f}, "
        f"schedule={schedule}, refresh_ratio={refresh_ratio:.3f}"
    )
    return spectrum_sample(
        mod_model,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise,
        **_SPECTRUM_DEFAULTS,
        dcw_mode="off",
        smc_cfg_alpha=options["adaptive_smc_alpha"],
        smc_cfg_lambda=5.0,
        schedule=schedule,
        refresh_ratio=refresh_ratio,
    )
