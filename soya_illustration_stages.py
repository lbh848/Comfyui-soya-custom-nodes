"""ANIMA reference and regional LoRA preparation for illustration HF/FD/ED."""
import math

import torch
import torch.nn.functional as F
import comfy.utils

from .soya_first_sampler_v2_core import _parse_multi_char, _parse_lora_entries, _resolve_lora_path
from .soya_snapref import SnapRef
from .soya_spatial_lora import apply_spatial_loras
from .soya_character_lora import load_character_lora_file


def _resolve_snapref_path(char_model, style_model, has_style):
    if has_style:
        candidates = [style_model or "snapref_v2s_pre1.safetensors"]
        if candidates[0] == "snapref_v2s_pre1.safetensors":
            candidates.append("snapref_v2s_pre1_fp8.safetensors")
        for name in candidates:
            try:
                return _resolve_lora_path(name)
            except FileNotFoundError:
                continue
        raise FileNotFoundError(
            "STYLE_REF 모델(snapref_v2s_pre1.safetensors)을 찾을 수 없습니다. "
            "설정의 업데이트 → 개별적 업데이트 → 5. 누락 모델 다운로드로 설치하거나 models/loras 폴더에 추가하세요."
        )
    candidates = [char_model or "snapref_v2a_pre2.safetensors"]
    if candidates[0] == "snapref_v2a_pre2.safetensors":
        candidates.extend(("snapref_v2s_pre1.safetensors", "snapref_v2s_pre1_fp8.safetensors"))
    for name in candidates:
        try:
            return _resolve_lora_path(name)
        except FileNotFoundError:
            continue
    return _resolve_lora_path(candidates[0])


def _snapref(model, clip, vae, images, settings, style_images=None, style_settings=None):
    settings = settings or {}
    style_settings = style_settings or {}
    strength = float(settings.get("strength", 1.0 if settings.get("strength_scale") == "linear" else 0.0))
    if settings.get("strength_scale") != "linear":
        strength = math.exp(strength)
    char_active = bool(images) and bool(settings.get("enabled")) and strength > 0
    style_strength = float(style_settings.get("strength", 1.0))
    style_active = bool(style_images) and bool(style_settings.get("enabled", True)) and style_strength > 0
    if not char_active and not style_active:
        return None
    path = _resolve_snapref_path(
        settings.get("model", "snapref_v2a_pre2.safetensors"),
        style_settings.get("model", "snapref_v2s_pre1.safetensors"),
        style_active,
    )
    if not style_active:
        return SnapRef(model, clip, vae, images, path,
                       settings.get("mode", "char"), float(settings.get("megapixels", 1.0)),
                       strength, float(settings.get("fidelity", 1.0)))
    return SnapRef(
        model, clip, vae, images if char_active else {}, path,
        settings.get("mode", "char"), float(settings.get("megapixels", 1.0)),
        strength if char_active else 0.0, float(settings.get("fidelity", 1.0)),
        style_images=style_images,
        style_megapixels=float(style_settings.get("megapixels", 1.0)),
        style_strength=style_strength,
        style_fidelity=float(style_settings.get("fidelity", 1.0)),
    )


def prepare_hires_model(model, clip, vae, positive, image, *, multi, masks,
                        lora_activate, lora_data, images, settings,
                        style_images=None, style_settings=None):
    multi = _parse_multi_char(multi)
    if not multi["enable"] and not images and not style_images:
        return model
    if clip is None:
        raise ValueError("ANIMA HF regional LoRA/reference processing requires CLIP")
    if not multi["enable"]:
        # The first sampler already applied the solo character's global LoRA.
        char_images = {1: next(iter(images.values()))} if images else {}
        reference = _snapref(model, clip, vae, char_images, settings, style_images, style_settings)
        if reference is None:
            return model
        return apply_spatial_loras(model, [], [torch.ones(1, *image.shape[1:3])],
                                   [positive], positive, snapref=reference)

    names = multi["char_name_list"]
    loaded = []
    for entry in _parse_lora_entries(lora_activate, lora_data, names):
        lora = load_character_lora_file(_resolve_lora_path(entry["lora_path"]))
        loaded.append((names.index(entry["character"]), lora, entry["strength"]))
    conditionings = []
    for index in range(multi["char_num"]):
        parts = (multi["char_trigger_list"][index] + multi["shared_before"]
                 + [multi["char_inform"][index], multi["char_composition_list"][index]] + multi["shared_after"])
        conditionings.append(clip.encode_from_tokens_scheduled(clip.tokenize(", ".join(part for part in parts if part))))
    parts = multi["background_trigger_list"] + multi["shared_before"] + multi["shared_after"]
    if multi["char_num"] > 1 and multi["composition_prompt"] not in parts:
        parts.append(multi["composition_prompt"])
    if multi["background_prompt"] and multi["background_prompt"] not in parts:
        parts.append(multi["background_prompt"])
    background = clip.encode_from_tokens_scheduled(clip.tokenize(", ".join(part for part in parts if part)))
    references = {index + 1: images[name] for index, name in enumerate(names) if name in images}
    reference = _snapref(model, clip, vae, references, settings, style_images, style_settings)
    # Reuse the supplied region ownership at the hires resolution.
    return apply_spatial_loras(model, loaded, masks, conditionings, background, snapref=reference)


def prepare_detailer_model(model, clip, vae, positive, mask, name, images, settings,
                           style_images=None, style_settings=None):
    char_images = {1: images[name]} if images and name in images else {}
    reference = _snapref(model, clip, vae, char_images, settings, style_images, style_settings)
    if reference is None:
        return model
    return apply_spatial_loras(model, [], [mask], [positive], positive, snapref=reference)


def prepare_hand_model(model, vae, positive, mask, crop, image_size, *, clip, masks, names, images, settings,
                       style_images=None, style_settings=None):
    x1, y1, x2, y2 = crop
    regions, references = [], {}
    for index, name in enumerate(names):
        if masks:
            full = F.interpolate(masks[index].unsqueeze(1), size=image_size, mode="bilinear", align_corners=False)
            cropped = full[:, :, y1:y2, x1:x2]
            region = F.interpolate(cropped, size=mask.shape[-2:], mode="bilinear", align_corners=False)[:, 0] * mask
        else:
            region = mask
        regions.append(region)
        if name in images and torch.any(region > 0):
            references[index + 1] = images[name]
    if not regions and style_images:
        regions = [mask]
    reference = _snapref(model, clip, vae, references, settings, style_images, style_settings)
    if reference is None:
        return model
    return apply_spatial_loras(model, [], regions, [positive] * len(regions), positive, snapref=reference)
