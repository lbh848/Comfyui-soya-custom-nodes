"""Selectable stock/Spectrum samplers with optional Anima regional conditioning."""

from __future__ import annotations

import json
import math
import os
import traceback
from collections import OrderedDict

import comfy.hooks
import comfy.samplers
import comfy.sd
import comfy.utils
import folder_paths
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .anima_regional_conditioning import (
    AnimaConditioningRegionChain,
    ApplyAnimaRegionalConditioningPatch,
)

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
_MAX_LORA_CACHE_ENTRIES = 8
_SAMPLER_MODE_KSAMPLER = "KSampler"
_SAMPLER_MODE_FAST = "FAST"
_SAMPLER_MODE_SPECTRUM_MOD_GUIDANCE = "SpectrumKSamplerModGuidance"
_SAMPLER_MODES = [
    _SAMPLER_MODE_KSAMPLER,
    _SAMPLER_MODE_FAST,
    _SAMPLER_MODE_SPECTRUM_MOD_GUIDANCE,
]

_MOD_GUIDANCE_QUALITY_TAGS = (
    "masterpiece, best quality, highres, absurdres, very aesthetic"
)
_MOD_GUIDANCE_QUALITY_NEG = (
    "score_1, score_2, score_3, worst quality, lowres, old, bad hands, bad anatomy"
)
_MOD_GUIDANCE_PROFILE = "step_i8_skip27"
_MOD_GUIDANCE_WEIGHT = 3.0
_MOD_GUIDANCE_START_LAYER = 8
_MOD_GUIDANCE_END_LAYER = 27
_MOD_GUIDANCE_ADAPTIVE_SMC_ALPHA = 0.10
_MOD_GUIDANCE_REFRESH_RATIO = -1.0
_SPECTRUM_DEFAULTS = {
    "window_size": 2.0,
    "flex_window": 0.25,
    "warmup_steps": 6,
    "blend_w": 0.3,
    "cheby_degree": 3,
    "ridge_lambda": 0.1,
}


def _spectrum_mod_guidance_runtime():
    """Return Spectrum's loaded low-level functions without invoking its node class."""
    import nodes as comfy_nodes

    sampler_class = comfy_nodes.NODE_CLASS_MAPPINGS.get("SpectrumKSampler")
    if sampler_class is None:
        message = (
            "Spectrum 저수준 런타임을 찾지 못했습니다. "
            "comfyui-spectrum-ksampler가 설치·로드되었는지 확인하세요."
        )
        print(f"[1st sampler] {message}")
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
        print(f"[1st sampler] {message}")
        raise RuntimeError(message)
    return setup_mod_guidance, spectrum_sample


def _parse_enabled(value, field_name):
    if isinstance(value, bool):
        return value
    normalized = str(value or "").strip().casefold()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"", "false", "0", "no", "off"}:
        return False
    raise ValueError(f"{field_name} 값이 boolean 문자열이 아닙니다: {value!r}")


def _resolve_lora_path(lora_path):
    raw = str(lora_path or "").strip()
    if not raw:
        raise ValueError("LoRA 경로가 비어 있습니다")
    if os.path.isfile(raw):
        return os.path.realpath(raw)
    try:
        resolved = folder_paths.get_full_path("loras", raw)
        if resolved and os.path.isfile(resolved):
            return os.path.realpath(resolved)
    except Exception as exc:
        print(f"[1st sampler] folder_paths LoRA 경로 조회 실패: path={raw!r}, error={exc}")
        traceback.print_exc()
    for base in folder_paths.get_folder_paths("loras"):
        candidate = os.path.join(base, raw)
        if os.path.isfile(candidate):
            return os.path.realpath(candidate)
    raise FileNotFoundError(f"LoRA 파일을 찾지 못했습니다: {raw}")


def _parse_lora_entries(lora_activate, lora_data, character_names=None):
    if not _parse_enabled(lora_activate, "LORA_ACT"):
        print("[1st sampler] LORA_ACT=false · 캐릭터 LoRA 적용 생략")
        return []
    if isinstance(lora_data, dict):
        payload = lora_data
    else:
        try:
            payload = json.loads(str(lora_data or ""))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"LORA_DATA JSON 파싱 실패: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("list"), list):
        raise ValueError("LORA_DATA는 list 배열을 가진 JSON object여야 합니다")

    expected = None
    expected_lookup = None
    if character_names is not None:
        expected = [str(name or "").strip() for name in character_names]
        expected_lookup = {name.casefold(): name for name in expected}

    entries = []
    for index, raw_entry in enumerate(payload["list"]):
        if not isinstance(raw_entry, dict):
            raise ValueError(f"LORA_DATA.list[{index}]가 object가 아닙니다")
        base = str(raw_entry.get("BASE") or "").strip()
        if base.casefold() != "anima":
            continue
        path = str(raw_entry.get("lora_path") or "").strip()
        if not path:
            raise ValueError(f"LORA_DATA.list[{index}].lora_path가 비어 있습니다")
        try:
            strength = float(raw_entry.get("str", 1.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"LORA_DATA.list[{index}].str이 숫자가 아닙니다: {raw_entry.get('str')!r}"
            ) from exc
        if not math.isfinite(strength):
            raise ValueError(f"LORA_DATA.list[{index}].str이 유한수가 아닙니다: {strength!r}")
        if strength == 0.0:
            print(f"[1st sampler] strength=0 LoRA 생략: index={index}, path={path}")
            continue
        entries.append({
            "source_index": index,
            "lora_path": path,
            "strength": strength,
            "character": str(raw_entry.get("CHAR") or "").strip(),
        })

    if not entries:
        print("[1st sampler] 활성화된 Anima 캐릭터 LoRA가 없어 LoRA 적용 생략")
        return []
    if expected is None:
        return entries

    has_character = [bool(entry["character"]) for entry in entries]
    if not any(has_character):
        if len(entries) != len(expected):
            raise ValueError(
                "구버전 LORA_DATA에 CHAR가 없고 Anima LoRA 수가 캐릭터 수와 다릅니다: "
                f"loras={len(entries)}, characters={len(expected)}"
            )
        print(
            "[1st sampler] 구버전 LORA_DATA 호환: CHAR가 없어 1인당 1개 LoRA를 "
            f"캐릭터 순서대로 매핑합니다: order={expected}"
        )
        for entry, name in zip(entries, expected):
            entry["character"] = name
        return entries
    if not all(has_character):
        raise ValueError("LORA_DATA의 Anima LoRA 일부에만 CHAR가 있습니다")

    for entry in entries:
        key = entry["character"].casefold()
        if key not in expected_lookup:
            raise ValueError(
                f"LORA_DATA에 MULTI_CHAR에 없는 CHAR가 있습니다: {entry['character']!r}"
            )
        entry["character"] = expected_lookup[key]
    return entries


def _as_prompt_parts(value, field_name):
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        parts = []
        for index, item in enumerate(value):
            if not isinstance(item, str):
                raise ValueError(f"{field_name}[{index}]가 문자열이 아닙니다")
            if item.strip():
                parts.append(item.strip())
        return parts
    raise ValueError(f"{field_name}는 문자열 또는 문자열 배열이어야 합니다")


def _parse_multi_char(value):
    if value is None or (isinstance(value, str) and not value.strip()):
        return {"enable": False}
    if isinstance(value, dict):
        payload = value
    else:
        try:
            payload = json.loads(str(value))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"MULTI_CHAR JSON 파싱 실패: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("MULTI_CHAR 루트가 object가 아닙니다")
    enable = payload.get("enable", False)
    if not isinstance(enable, bool):
        raise ValueError(f"MULTI_CHAR.enable은 boolean이어야 합니다: {enable!r}")
    if not enable:
        return {"enable": False}

    try:
        char_num = int(payload.get("char_num"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"MULTI_CHAR.char_num이 정수가 아닙니다: {payload.get('char_num')!r}") from exc
    if not 2 <= char_num <= 3:
        raise ValueError(f"MULTI_CHAR는 2~3명만 지원합니다: {char_num}")

    names = payload.get("char_name_list")
    char_inform = payload.get("char_inform")
    trigger_groups = payload.get("char_trigger_list")
    if not isinstance(names, list) or len(names) != char_num:
        raise ValueError("MULTI_CHAR.char_name_list 길이가 char_num과 다릅니다")
    if not isinstance(char_inform, list) or len(char_inform) != char_num:
        raise ValueError("MULTI_CHAR.char_inform 길이가 char_num과 다릅니다")
    if not isinstance(trigger_groups, list) or len(trigger_groups) != char_num:
        raise ValueError("MULTI_CHAR.char_trigger_list 길이가 char_num과 다릅니다")

    clean_names = []
    clean_inform = []
    clean_triggers = []
    seen = set()
    for index in range(char_num):
        name = str(names[index] or "").strip()
        info = str(char_inform[index] or "").strip()
        if not name or name.casefold() in seen:
            raise ValueError(f"MULTI_CHAR 캐릭터 이름이 비었거나 중복됩니다: {name!r}")
        if not info:
            raise ValueError(f"MULTI_CHAR.char_inform[{index}]가 비어 있습니다")
        seen.add(name.casefold())
        clean_names.append(name)
        clean_inform.append(info)
        clean_triggers.append(_as_prompt_parts(
            trigger_groups[index],
            f"MULTI_CHAR.char_trigger_list[{index}]",
        ))

    shared = payload.get("shared_tag")
    if not isinstance(shared, dict):
        raise ValueError("MULTI_CHAR.shared_tag가 object가 아닙니다")
    background_prompt = payload.get("background_prompt", "")
    if background_prompt is None:
        background_prompt = ""
    if not isinstance(background_prompt, str):
        raise ValueError("MULTI_CHAR.background_prompt는 문자열이어야 합니다")
    composition_prompt = payload.get("composition_prompt", "")
    if not isinstance(composition_prompt, str):
        raise ValueError("MULTI_CHAR.composition_prompt는 문자열이어야 합니다")
    composition_prompt = composition_prompt.strip()
    if not composition_prompt:
        raise ValueError("MULTI_CHAR.composition_prompt가 비어 있습니다")
    mask_fingerprint = str(payload.get("mask_fingerprint") or "").strip().casefold()
    if len(mask_fingerprint) != 64 or any(
        character not in "0123456789abcdef" for character in mask_fingerprint
    ):
        raise ValueError("MULTI_CHAR.mask_fingerprint는 SHA-256 hex 문자열이어야 합니다")
    return {
        "enable": True,
        "char_num": char_num,
        "char_name_list": clean_names,
        "char_inform": clean_inform,
        "char_trigger_list": clean_triggers,
        "background_trigger_list": _as_prompt_parts(
            payload.get("background_trigger_list", []),
            "MULTI_CHAR.background_trigger_list",
        ),
        "shared_before": _as_prompt_parts(shared.get("before_char", []), "shared_tag.before_char"),
        "shared_after": _as_prompt_parts(shared.get("after_char", []), "shared_tag.after_char"),
        "background_prompt": background_prompt.strip(),
        "composition_prompt": composition_prompt,
        "mask_fingerprint": mask_fingerprint,
    }


def _safe_mask_directory(mask_location):
    input_root = os.path.realpath(os.path.abspath(folder_paths.get_input_directory()))
    relative = str(mask_location or "region_mask").strip() or "region_mask"
    target = os.path.realpath(os.path.abspath(os.path.join(input_root, relative)))
    try:
        inside_root = os.path.commonpath([input_root, target]) == input_root
    except ValueError as exc:
        raise ValueError(f"mask_location이 ComfyUI input과 다른 드라이브입니다: {relative!r}") from exc
    if not inside_root or target == input_root:
        raise ValueError(f"mask_location이 ComfyUI input 폴더 밖을 가리킵니다: {relative!r}")
    if not os.path.isdir(target):
        raise FileNotFoundError(f"마스크 폴더가 없습니다: {target}")
    return target


def _load_channel_masks(mask_location, char_num):
    directory = _safe_mask_directory(mask_location)
    images = sorted(
        entry.path
        for entry in os.scandir(directory)
        if entry.is_file(follow_symlinks=False)
        and os.path.splitext(entry.name)[1].lower() in _IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"마스크 이미지가 없습니다: {directory}")
    if len(images) > 1:
        print(f"[1st sampler] 마스크 이미지가 {len(images)}개여서 첫 파일을 사용합니다: {images[0]}")
    with Image.open(images[0]) as source:
        rgb = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0
    masks = []
    for channel_index in range(char_num):
        channel = np.ascontiguousarray(rgb[:, :, channel_index]).copy()
        if float(channel.max()) <= 0.0:
            raise ValueError(f"마스크 채널 {'RGB'[channel_index]}가 비어 있습니다: {images[0]}")
        masks.append(torch.from_numpy(channel).unsqueeze(0))
    print(
        f"[1st sampler] Regional RGB 마스크 로드: file={images[0]}, "
        f"size={rgb.shape[1]}x{rgb.shape[0]}, channels={char_num}"
    )
    return masks


def _spectrum_sampler():
    import nodes as comfy_nodes

    sampler_class = comfy_nodes.NODE_CLASS_MAPPINGS.get("SpectrumSPDKSampler")
    if sampler_class is None:
        raise RuntimeError(
            "SpectrumSPDKSampler를 찾지 못했습니다. comfyui-spectrum-ksampler가 설치·로드되었는지 확인하세요."
        )
    return sampler_class()


class SoyaFirstSampler_mdsoya:
    def __init__(self):
        self._loaded_loras = OrderedDict()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "clip": ("CLIP",),
                "multi_char": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "forceInput": True,
                    },
                ),
                "latent_image": ("LATENT",),
                "LORA_ACT": ("STRING", {"default": "false", "forceInput": True}),
                "LORA_DATA": (
                    "STRING",
                    {"default": '{"list": []}', "multiline": True, "forceInput": True},
                ),
                "mask_location": ("STRING", {"default": "region_mask"}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 10000}),
                "cfg": (
                    "FLOAT",
                    {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01},
                ),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "split_mode": (["single"], {"default": "single"}),
                "spd_scale": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.25, "max": 1.0, "step": 0.05, "round": 0.01},
                ),
                "spd_sigma": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001},
                ),
                "adaptive_smc_alpha": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05, "round": 0.001},
                ),
            },
            "optional": {
                "sampler_mode": (
                    _SAMPLER_MODES,
                    {
                        "default": _SAMPLER_MODE_FAST,
                        "tooltip": (
                            "KSampler = 원본 ComfyUI KSampler, "
                            "FAST = Spectrum SPD/SPEED 가속 sampler, "
                            "SpectrumKSamplerModGuidance = 개인용 고정 Mod Guidance + "
                            "legacy window Spectrum sampler"
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("LATENT", "MODEL")
    RETURN_NAMES = ("output", "model")
    FUNCTION = "sample"
    CATEGORY = "sampling"
    DESCRIPTION = (
        "ANIMA_MODEL_WO_CHAR와 LORA_DATA를 받아 캐릭터 LoRA를 내부 적용합니다. "
        "MULTI_CHAR=true이면 캐릭터 LoRA를 한 번 병합하고 RGB 마스크별 프롬프트를 "
        "Anima Regional Attention 단일 패스로 처리합니다. sampler_mode에서 원본 "
        "ComfyUI KSampler, Spectrum SPD/SPEED FAST sampler 또는 고정 프로필 "
        "Spectrum Mod Guidance sampler를 선택할 수 있습니다."
    )

    def _load_lora(self, entry):
        resolved = _resolve_lora_path(entry["lora_path"])
        stat = os.stat(resolved)
        signature = (stat.st_mtime_ns, stat.st_size)
        cached = self._loaded_loras.get(resolved)
        if cached is not None and cached[0] == signature:
            self._loaded_loras.move_to_end(resolved)
            return cached[1], resolved
        if cached is not None:
            print(f"[1st sampler] 변경된 LoRA를 다시 로드합니다: path={resolved}")
        try:
            lora = comfy.utils.load_torch_file(resolved, safe_load=True)
        except Exception as exc:
            print(f"[1st sampler] LoRA 로드 실패: path={resolved}, error={exc}")
            traceback.print_exc()
            raise
        self._loaded_loras[resolved] = (signature, lora)
        self._loaded_loras.move_to_end(resolved)
        while len(self._loaded_loras) > _MAX_LORA_CACHE_ENTRIES:
            evicted_path, _evicted = self._loaded_loras.popitem(last=False)
            print(f"[1st sampler] LoRA CPU 캐시 제거: path={evicted_path}")
        print(f"[1st sampler] LoRA 로드 완료: path={resolved}")
        return lora, resolved

    def _apply_global_loras(self, model, entries):
        current_model = model
        for entry in entries:
            lora, resolved = self._load_lora(entry)
            try:
                current_model, _ = comfy.sd.load_lora_for_models(
                    current_model,
                    None,
                    lora,
                    entry["strength"],
                    0.0,
                )
            except Exception as exc:
                print(
                    f"[1st sampler] 전역 LoRA 적용 실패: path={resolved}, "
                    f"strength={entry['strength']}, error={exc}"
                )
                traceback.print_exc()
                raise
            print(
                f"[1st sampler] 전역 캐릭터 LoRA 적용: path={resolved}, "
                f"strength={entry['strength']}"
            )
        return current_model

    def _build_character_hooks(self, model, entries, character_names):
        registered_model = model
        hooks_by_character = {name: None for name in character_names}
        counts = {name: 0 for name in character_names}
        for entry in entries:
            name = entry["character"]
            lora, resolved = self._load_lora(entry)
            try:
                registered_model, _, new_hooks = comfy.hooks.load_hook_lora_for_models(
                    registered_model,
                    None,
                    lora,
                    entry["strength"],
                    0.0,
                )
            except Exception as exc:
                print(
                    f"[1st sampler] Hook LoRA 등록 실패: char={name!r}, path={resolved}, "
                    f"strength={entry['strength']}, error={exc}"
                )
                traceback.print_exc()
                raise
            current_hooks = hooks_by_character[name]
            hooks_by_character[name] = (
                new_hooks
                if current_hooks is None
                else current_hooks.clone_and_combine(new_hooks)
            )
            counts[name] += 1
            print(
                f"[1st sampler] 캐릭터 Hook LoRA 등록: char={name!r}, "
                f"path={resolved}, strength={entry['strength']}"
            )
        for name in character_names:
            if counts[name] == 0:
                print(f"[1st sampler] 캐릭터에 적용할 Anima LoRA가 없습니다: char={name!r}")
        return registered_model, [hooks_by_character[name] for name in character_names], counts

    @staticmethod
    def _sample_stock_padded(
        model,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise,
    ):
        import nodes as comfy_nodes

        if not isinstance(latent_image, dict) or "samples" not in latent_image:
            raise ValueError("latent_image에 samples tensor가 없습니다")
        samples = latent_image["samples"]
        if not torch.is_tensor(samples) or samples.ndim < 4:
            raise ValueError(
                f"latent_image.samples 형식이 올바르지 않습니다: {type(samples)!r}, "
                f"shape={getattr(samples, 'shape', None)}"
            )
        original_h, original_w = samples.shape[-2:]
        pad_h = (-original_h) % 2
        pad_w = (-original_w) % 2
        padded_latent = latent_image.copy()
        if pad_h or pad_w:
            spatial_pad = (
                (0, pad_w, 0, pad_h, 0, 0)
                if samples.ndim == 5
                else (0, pad_w, 0, pad_h)
            )
            padded_latent["samples"] = F.pad(samples, spatial_pad, mode="replicate")
            noise_mask = padded_latent.get("noise_mask")
            if noise_mask is not None:
                if not torch.is_tensor(noise_mask) or noise_mask.ndim < 2:
                    raise ValueError(
                        "latent_image.noise_mask 형식이 올바르지 않습니다: "
                        f"{type(noise_mask)!r}, shape={getattr(noise_mask, 'shape', None)}"
                    )
                padded_latent["noise_mask"] = F.pad(
                    noise_mask,
                    (0, pad_w, 0, pad_h),
                    mode="constant",
                    value=1.0,
                )
            print(
                f"[1st sampler] stock KSampler용 latent padding: "
                f"{original_w}x{original_h} -> {original_w + pad_w}x{original_h + pad_h}"
            )
        result = comfy_nodes.common_ksampler(
            model,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            padded_latent,
            denoise=denoise,
        )
        if pad_h or pad_w:
            output = result[0].copy()
            output["samples"] = output["samples"][..., :original_h, :original_w]
            return (output,)
        return result

    @staticmethod
    def _sample_spectrum(
        model,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise,
        split_mode,
        spd_scale,
        spd_sigma,
        adaptive_smc_alpha,
    ):
        return _spectrum_sampler().sample(
            model=model,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image=latent_image,
            denoise=denoise,
            split_mode=split_mode,
            spd_scale=spd_scale,
            spd_sigma=spd_sigma,
            adaptive_smc_alpha=adaptive_smc_alpha,
        )

    @staticmethod
    def _sample_spectrum_mod_guidance(
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
    ):
        if clip is None:
            message = "SpectrumKSamplerModGuidance 실행에 CLIP 입력이 필요합니다"
            print(f"[1st sampler] {message}")
            raise ValueError(message)

        setup_mod_guidance, spectrum_sample = _spectrum_mod_guidance_runtime()
        mod_model = model.clone()
        setup_mod_guidance(
            mod_model,
            clip,
            positive,
            negative,
            None,
            _MOD_GUIDANCE_QUALITY_TAGS,
            _MOD_GUIDANCE_WEIGHT,
            quality_neg=_MOD_GUIDANCE_QUALITY_NEG,
            start_layer=_MOD_GUIDANCE_START_LAYER,
            end_layer=_MOD_GUIDANCE_END_LAYER,
            taper=0,
            taper_scale=0.25,
            final_w=0.0,
        )
        print(
            "[1st sampler] SpectrumKSamplerModGuidance 고정 프로필 적용: "
            f"profile={_MOD_GUIDANCE_PROFILE}, "
            f"adaptive_smc_alpha={_MOD_GUIDANCE_ADAPTIVE_SMC_ALPHA:.2f}, "
            f"refresh_ratio={_MOD_GUIDANCE_REFRESH_RATIO:.1f}"
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
            smc_cfg_alpha=_MOD_GUIDANCE_ADAPTIVE_SMC_ALPHA,
            smc_cfg_lambda=5.0,
            schedule="window",
            refresh_ratio=_MOD_GUIDANCE_REFRESH_RATIO,
        )

    def _sample_selected(
        self,
        sampler_mode,
        model,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        clip,
        latent_image,
        denoise,
        split_mode,
        spd_scale,
        spd_sigma,
        adaptive_smc_alpha,
    ):
        if sampler_mode == _SAMPLER_MODE_KSAMPLER:
            return self._sample_stock_padded(
                model,
                seed,
                steps,
                cfg,
                sampler_name,
                scheduler,
                positive,
                negative,
                latent_image,
                denoise,
            )
        if sampler_mode == _SAMPLER_MODE_FAST:
            return self._sample_spectrum(
                model,
                seed,
                steps,
                cfg,
                sampler_name,
                scheduler,
                positive,
                negative,
                latent_image,
                denoise,
                split_mode,
                spd_scale,
                spd_sigma,
                adaptive_smc_alpha,
            )
        if sampler_mode == _SAMPLER_MODE_SPECTRUM_MOD_GUIDANCE:
            return self._sample_spectrum_mod_guidance(
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
            )
        raise ValueError(
            f"지원하지 않는 sampler_mode입니다: {sampler_mode!r} "
            f"(지원값: {', '.join(_SAMPLER_MODES)})"
        )

    def sample(
        self,
        model,
        positive,
        negative,
        clip,
        multi_char,
        seed,
        latent_image,
        LORA_ACT,
        LORA_DATA,
        mask_location,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        split_mode,
        spd_scale,
        spd_sigma,
        adaptive_smc_alpha,
        sampler_mode=_SAMPLER_MODE_FAST,
    ):
        try:
            payload = _parse_multi_char(multi_char)
            if not payload["enable"]:
                lora_entries = _parse_lora_entries(LORA_ACT, LORA_DATA)
                global_model = self._apply_global_loras(model, lora_entries)
                print(
                    "[1st sampler] MULTI_CHAR=false · 캐릭터 LoRA 전역 적용 후 "
                    f"{sampler_mode} 실행: loras={len(lora_entries)}"
                )
                result = self._sample_selected(
                    sampler_mode,
                    global_model,
                    seed,
                    steps,
                    cfg,
                    sampler_name,
                    scheduler,
                    positive,
                    negative,
                    clip,
                    latent_image,
                    denoise,
                    split_mode,
                    spd_scale,
                    spd_sigma,
                    adaptive_smc_alpha,
                )
                return result[0], global_model

            lora_entries = _parse_lora_entries(
                LORA_ACT,
                LORA_DATA,
                payload["char_name_list"],
            )
            masks = _load_channel_masks(mask_location, payload["char_num"])

            regional_model = self._apply_global_loras(model, lora_entries)
            lora_counts = {
                name: sum(
                    1 for entry in lora_entries
                    if entry["character"].casefold() == name.casefold()
                )
                for name in payload["char_name_list"]
            }
            regions = None
            region_prompts = []
            for index in range(payload["char_num"]):
                prompt_parts = (
                    payload["char_trigger_list"][index]
                    + payload["shared_before"]
                    + [payload["char_inform"][index]]
                    + [payload["composition_prompt"]]
                    + payload["shared_after"]
                )
                prompt = ", ".join(part for part in prompt_parts if part)
                tokens = clip.tokenize(prompt)
                conditioning = clip.encode_from_tokens_scheduled(tokens)
                regions = AnimaConditioningRegionChain(
                    previous=regions,
                    mask=masks[index],
                    conditioning=conditioning,
                    weight=1.0,
                )
                region_prompts.append(prompt)

            background_parts = (
                payload["background_trigger_list"]
                + payload["shared_before"]
                + payload["shared_after"]
            )
            if payload["composition_prompt"] not in background_parts:
                background_parts.append(payload["composition_prompt"])
            if (
                payload["background_prompt"]
                and payload["background_prompt"] not in background_parts
            ):
                background_parts.append(payload["background_prompt"])
            background_prompt = ", ".join(part for part in background_parts if part)
            if not background_prompt:
                raise ValueError(
                    "MULTI_CHAR.shared_tag에 배경용 공통 프롬프트가 없습니다"
                )
            background_conditioning = clip.encode_from_tokens_scheduled(
                clip.tokenize(background_prompt)
            )
            regional_model = ApplyAnimaRegionalConditioningPatch().apply(
                regional_model,
                regions,
                base_mode="uncovered_only",
                base_strength=1.0,
                end_percent=1.0,
                cross_mask_strength=1.0,
                self_mask_strength=0.0,
                base_ratio=0.0,
                cross_inject_every_n_blocks=1,
                self_inject_every_n_blocks=1,
                start_percent=0.0,
                background_conditioning=background_conditioning,
            )[0]
            print(
                f"[1st sampler] MULTI_CHAR Regional Attention 패치 적용: "
                f"base_mode=uncovered_only, cross_mask_strength=1.0, "
                f"self_mask_strength=0.0, base_ratio=0.0"
            )
            print(
                f"[1st sampler] MULTI_CHAR=true · 전역 캐릭터 LoRA + RGB Regional Attention "
                f"+ {sampler_mode} 단일 패스 실행: "
                f"order={payload['char_name_list']}, loras={lora_counts}, "
                f"prompt_lengths={[len(p) for p in region_prompts]}, "
                f"background_length={len(background_prompt)}, "
                f"steps={steps}, denoise={denoise}, "
                f"mask_fingerprint={payload['mask_fingerprint'][:12]}"
            )
            result = self._sample_selected(
                sampler_mode,
                regional_model,
                seed,
                steps,
                cfg,
                sampler_name,
                scheduler,
                background_conditioning,
                negative,
                clip,
                latent_image,
                denoise,
                split_mode,
                spd_scale,
                spd_sigma,
                adaptive_smc_alpha,
            )
            # 후단 HRF/디테일러에는 전역 all-character LoRA 모델이 아니라
            # 원래 clean model을 넘겨 첫 패스의 영역 분리를 재오염하지 않는다.
            return result[0], model
        except Exception as exc:
            print(
                f"[1st sampler] 샘플링 실패: mask_location={mask_location!r}, "
                f"sampler_mode={sampler_mode!r}, "
                f"multi_char_len={len(str(multi_char or ''))}, "
                f"lora_data_len={len(str(LORA_DATA or ''))}, error={exc}"
            )
            traceback.print_exc()
            raise
