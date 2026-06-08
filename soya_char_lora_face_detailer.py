"""
SoyaCharLoraFaceDetailer – Per-face detailer with character-specific LoRA patches.

For each face in face_context:
  1. Identify character → find matching LoRA (filtered by base_model)
  2. Apply ONLY that character's LoRA to a fresh copy of the original model
  3. Assemble prompt: quality_tags + artist_tags + FACE_TAGS + EYE_TAGS
  4. Crop face region from image (8-aligned coords, crop_expand_factor padding)
  5. Upscale crop by upscale_factor for higher-resolution processing
  6. VAE encode → KSampler → VAE decode → downscale back → paste with feathered mask
"""

import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import comfy.sd
import comfy.utils
import folder_paths


# ── Shared helpers ──────────────────────────────────────────────────

def _resolve_lora_path(lora_path):
    if os.path.isfile(lora_path):
        return lora_path
    try:
        resolved = folder_paths.get_full_path("loras", lora_path)
        if resolved and os.path.isfile(resolved):
            return resolved
    except Exception:
        pass
    try:
        for base in folder_paths.get_folder_paths("loras"):
            candidate = os.path.join(base, lora_path)
            if os.path.isfile(candidate):
                return candidate
    except Exception:
        pass
    return None


def _extract_char_from_path(path, char_names):
    normalized = path.replace("\\", "/")
    segments = normalized.split("/")
    for char_name in char_names:
        for seg in segments:
            if seg.startswith(f"{char_name}-") or seg == char_name:
                return char_name
    return None


def _build_lora_map(filtered_loras, char_names):
    lora_map = {}
    for entry in filtered_loras:
        path = entry.get("lora_path", "")
        char = _extract_char_from_path(path, char_names)
        if char and char not in lora_map:
            lora_map[char] = entry
    return lora_map


def _parse_inputs(char_tags, lora_list, base_model):
    char_data = json.loads(char_tags).get("list", [])
    lora_data = json.loads(lora_list).get("list", [])

    char_map = {}
    char_names = []
    for entry in char_data:
        name = entry.get("CHAR", "").strip()
        if name:
            char_map[name] = {
                "FACE_TAGS": entry.get("FACE_TAGS", ""),
                "EYE_TAGS": entry.get("EYE_TAGS", ""),
            }
            char_names.append(name)

    filtered_loras = [l for l in lora_data if l.get("BASE", "").strip() == base_model.strip()]
    lora_map = _build_lora_map(filtered_loras, char_names)

    return char_map, char_names, lora_map, len(lora_data), len(filtered_loras)


def _compute_info(face_context, char_tags, quality_tags, artist_tags,
                  lora_list, base_model):
    char_map, char_names, lora_map, total_loras, filtered_loras = \
        _parse_inputs(char_tags, lora_list, base_model)

    lines = [
        f"Base model: {base_model}",
        f"LoRAs: {filtered_loras} matched / {total_loras} total",
        f"Characters: {', '.join(char_names) or '(none)'}",
        "─" * 50,
    ]

    if not face_context or not face_context.get("matches"):
        lines.append("No faces in face_context")
        return "\n".join(lines)

    matches = face_context["matches"]
    lines.append(f"Detected faces: {len(matches)}")
    lines.append("─" * 50)

    for i, match in enumerate(matches):
        name = match["name"]
        crop = match["crop"]
        score = match["score"]

        if name == "unknown":
            lines.append(f"Face {i + 1}: unknown (score: {score:.4f}) — SKIPPED")
            continue

        tags = char_map.get(name, {})
        face_tags = tags.get("FACE_TAGS", "")
        eye_tags = tags.get("EYE_TAGS", "")
        parts = [p for p in [quality_tags, artist_tags, face_tags, eye_tags] if p.strip()]
        prompt = ", ".join(parts)

        lora_entry = lora_map.get(name)
        lines.append(f"Face {i + 1}: {name}")
        lines.append(f"  Score: {score:.4f}")
        lines.append(f"  Crop: {crop}")
        if lora_entry:
            lines.append(f"  LoRA: {os.path.basename(lora_entry['lora_path'])} @ {lora_entry.get('str', 1.0)}")
        else:
            lines.append("  LoRA: NOT FOUND")
        lines.append(f"  Prompt: {prompt}")

    return "\n".join(lines)


# ── GPU helpers ─────────────────────────────────────────────────────

def _gaussian_blur_gpu(mask, sigma):
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1

    x = torch.arange(ksize, dtype=torch.float32, device=mask.device) - ksize // 2
    kernel = torch.exp(-x ** 2 / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()

    mask_4d = mask.unsqueeze(0).unsqueeze(0)
    pad = ksize // 2

    padded = F.pad(mask_4d, [pad, pad, 0, 0], mode='reflect')
    blurred = F.conv2d(padded, kernel.view(1, 1, 1, -1))

    padded = F.pad(blurred, [0, 0, pad, pad], mode='reflect')
    blurred = F.conv2d(padded, kernel.view(1, 1, -1, 1))

    return blurred.squeeze(0).squeeze(0)


def _encode_conditioning(clip, text):
    from nodes import CLIPTextEncode
    return CLIPTextEncode().encode(clip, text)[0]


def _run_ksampler(model, seed, steps, cfg, sampler_name, scheduler,
                  positive, negative, latent_dict, denoise):
    from nodes import common_ksampler
    result = common_ksampler(
        model, seed, steps, cfg, sampler_name, scheduler,
        positive, negative, latent_dict, denoise=denoise,
    )
    return result[0]["samples"]


def _compute_voronoi_zones(matches, H, W):
    """Compute per-face Voronoi ownership masks (each pixel assigned to nearest face center)."""
    if not matches:
        return []

    centers = []
    for m in matches:
        crop = m["crop"]
        centers.append(((crop[0] + crop[2]) / 2.0, (crop[1] + crop[3]) / 2.0))

    if len(centers) == 1:
        mask = torch.ones((H, W), dtype=torch.float32)
        return [mask]

    # Bounding box of all face regions for efficiency
    all_x1 = [int(m["crop"][0]) for m in matches]
    all_y1 = [int(m["crop"][1]) for m in matches]
    all_x2 = [int(m["crop"][2]) for m in matches]
    all_y2 = [int(m["crop"][3]) for m in matches]
    bx1, by1 = max(0, min(all_x1)), max(0, min(all_y1))
    bx2, by2 = min(W, max(all_x2)), min(H, max(all_y2))

    yy, xx = torch.meshgrid(
        torch.arange(by1, by2, dtype=torch.float32),
        torch.arange(bx1, bx2, dtype=torch.float32),
        indexing='ij'
    )

    nearest = torch.zeros_like(xx, dtype=torch.long)
    min_dist = torch.full_like(xx, float('inf'))
    for i, (cx, cy) in enumerate(centers):
        dist = (xx - cx) ** 2 + (yy - cy) ** 2
        closer = dist < min_dist
        nearest[closer] = i
        min_dist[closer] = dist[closer]

    masks = []
    for i in range(len(matches)):
        mask = torch.zeros((H, W), dtype=torch.float32)
        mask[by1:by2, bx1:bx2] = (nearest == i).float()
        masks.append(mask)
    return masks


# ── Main Face Detailer Node ────────────────────────────────────────

class SoyaCharLoraFaceDetailer_mdsoya:
    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers
        return {
            "required": {
                "enable": ("STRING", {"default": "true"}),
                "image": ("IMAGE",),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "face_context": ("IPA_FACE_CONTEXT",),
                "char_tags": ("STRING", {"multiline": True, "default": '{"list":[{"CHAR":"name","FACE_TAGS":"face tags","EYE_TAGS":"eye tags"}]}'}),
                "quality_tags": ("STRING", {"default": ""}),
                "artist_tags": ("STRING", {"default": ""}),
                "negative": ("STRING", {"multiline": True, "default": ""}),
                "lora_list": ("STRING", {"multiline": True, "default": '{"list":[{"CHAR":"name","lora_path":"filename.safetensors","str":1.0,"BASE":"anima"}]}'}),
                "base_model": ("STRING", {"default": "anima"}),
                "crop_expand_factor": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 4.0, "step": 0.1}),
                "upscale_factor": ("FLOAT", {"default": 1.2, "min": 1.0, "max": 4.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "feather": ("INT", {"default": 5, "min": 0, "max": 100}),
                "corner_roundness": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "noise_mask": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("image", "mask", "crop_preview", "info")
    FUNCTION = "execute"
    CATEGORY = "Soya/FaceDetailer"

    def execute(self, *, enable, image, model, clip, vae, face_context,
                char_tags, quality_tags, artist_tags, negative,
                lora_list, base_model,
                crop_expand_factor, upscale_factor, seed, steps, cfg, sampler_name, scheduler,
                denoise, feather, corner_roundness, noise_mask):

        B, H, W, C = image.shape
        use = enable.strip().lower() in ("true", "1", "yes")

        if not use:
            return (image, torch.zeros((B, H, W), dtype=torch.float32),
                    torch.zeros_like(image),
                    "DISABLED")

        char_map, char_names, lora_map, _, _ = \
            _parse_inputs(char_tags, lora_list, base_model)

        if not face_context or not face_context.get("matches"):
            return (image, torch.zeros((B, H, W), dtype=torch.float32),
                    torch.zeros_like(image),
                    "No faces in face_context")

        matches = face_context["matches"]
        result_image = image.clone()
        combined_mask = torch.zeros((B, H, W), dtype=torch.float32)
        crop_region = torch.zeros((B, H, W), dtype=torch.float32)
        log_lines = []

        # Encode negative once (shared across all faces)
        negative_cond = _encode_conditioning(clip, negative)

        # Compute Voronoi zones so overlapping faces get unique regions
        voronoi_masks = _compute_voronoi_zones(matches, H, W)

        for batch_idx in range(B):
            for match_idx, match in enumerate(matches):
                name = match["name"]
                crop = match["crop"]

                if name == "unknown":
                    log_lines.append(f"  unknown — SKIPPED")
                    continue

                if name not in char_map:
                    log_lines.append(f"  {name} — NO TAGS, SKIPPED")
                    continue

                # ── LoRA: always start from original model ──────
                lora_entry = lora_map.get(name)
                patched_model = model
                lora_info = "none"

                if lora_entry:
                    resolved = _resolve_lora_path(lora_entry["lora_path"])
                    if resolved:
                        lora_loaded = comfy.utils.load_torch_file(resolved, safe_load=True)
                        strength = float(lora_entry.get("str", 1.0))
                        patched_model, _ = comfy.sd.load_lora_for_models(
                            model, clip, lora_loaded, strength, 0.0
                        )
                        lora_info = f"{os.path.basename(resolved)} @ {strength}"
                    else:
                        lora_info = f"FILE NOT FOUND: {lora_entry['lora_path']}"

                # ── Tags & prompt ───────────────────────────────
                tags = char_map[name]
                face_tags = tags["FACE_TAGS"]
                eye_tags = tags["EYE_TAGS"]
                parts = [p for p in [quality_tags, artist_tags, face_tags, eye_tags] if p.strip()]
                prompt = ", ".join(parts)

                positive_cond = _encode_conditioning(clip, prompt)

                # ── Crop region from face_context (expanded with CROP_TOP/BOTTOM) ──
                cr_x1, cr_y1, cr_x2, cr_y2 = int(crop[0]), int(crop[1]), int(crop[2]), int(crop[3])
                cr_w, cr_h = cr_x2 - cr_x1, cr_y2 - cr_y1
                cr_cx, cr_cy = (cr_x1 + cr_x2) / 2.0, (cr_y1 + cr_y2) / 2.0

                # Apply crop_expand_factor for additional context padding
                ew = cr_w * crop_expand_factor
                eh = cr_h * crop_expand_factor

                # Round UP crop dimensions to multiple of 8
                crop_w = ((int(ew) + 7) // 8) * 8
                crop_h = ((int(eh) + 7) // 8) * 8

                if crop_w < 8 or crop_h < 8:
                    log_lines.append(f"  {name} — crop too small ({crop_w}x{crop_h})")
                    continue

                # Center the crop, align start to multiple of 8
                cx1 = (int(cr_cx - crop_w / 2) // 8) * 8
                cy1 = (int(cr_cy - crop_h / 2) // 8) * 8
                cx2 = cx1 + crop_w
                cy2 = cy1 + crop_h

                # Shift left/up if extends past image edge (re-align to 8)
                if cx2 > W:
                    cx1 = max(0, (W - crop_w) // 8 * 8)
                    cx2 = cx1 + crop_w
                if cy2 > H:
                    cy1 = max(0, (H - crop_h) // 8 * 8)
                    cy2 = cy1 + crop_h

                # Final clamp (crop may be larger than image)
                cx1 = max(0, cx1)
                cy1 = max(0, cy1)
                cx2 = min(W, cx2)
                cy2 = min(H, cy2)

                # Ensure crop fully contains the original face region (8-aligned)
                # 8-alignment can shift the crop so original area pokes out at edges
                cx1 = min(cx1, (cr_x1 // 8) * 8)
                cy1 = min(cy1, (cr_y1 // 8) * 8)
                cx2 = max(cx2, ((cr_x2 + 7) // 8) * 8)
                cy2 = max(cy2, ((cr_y2 + 7) // 8) * 8)

                # Re-clamp after containment expansion
                cx1 = max(0, cx1)
                cy1 = max(0, cy1)
                cx2 = min(W, cx2)
                cy2 = min(H, cy2)
                actual_w, actual_h = cx2 - cx1, cy2 - cy1

                # ── Crop from original image ──
                crop_tensor = image[batch_idx, cy1:cy2, cx1:cx2].clone()

                # ── Mask from crop region (superellipse = rounded rectangle) ──
                # corner_roundness: 0=rectangle, 1=ellipse
                # Internally maps to superellipse exponent: n = 2/max(t, 0.01)
                #   t=0   → hard rectangle
                #   t=0.25 → n=8 (subtle rounding)
                #   t=0.5 → n=4 (squircle)
                #   t=1.0 → n=2 (ellipse)
                sx = actual_w / max(1, crop_w)
                sy = actual_h / max(1, crop_h)
                local_cx = (cr_cx - cx1) * sx
                local_cy = (cr_cy - cy1) * sy
                local_hw = max(1.0, (cr_x2 - cr_x1) / 2.0 * sx)
                local_hh = max(1.0, (cr_y2 - cr_y1) / 2.0 * sy)

                if corner_roundness <= 0.0:
                    local_x1 = max(0, int((cr_x1 - cx1) * sx))
                    local_y1 = max(0, int((cr_y1 - cy1) * sy))
                    local_x2 = min(actual_w, int((cr_x2 - cx1) * sx))
                    local_y2 = min(actual_h, int((cr_y2 - cy1) * sy))
                    mask = torch.zeros((actual_h, actual_w), dtype=torch.float32)
                    if local_x2 > local_x1 and local_y2 > local_y1:
                        mask[local_y1:local_y2, local_x1:local_x2] = 1.0
                else:
                    n = 2.0 / max(corner_roundness, 0.01)
                    yy, xx = torch.meshgrid(
                        torch.arange(actual_h, dtype=torch.float32),
                        torch.arange(actual_w, dtype=torch.float32),
                        indexing='ij'
                    )
                    nx = (xx - local_cx) / local_hw
                    ny = (yy - local_cy) / local_hh
                    superellipse = (nx.abs() ** n + ny.abs() ** n)
                    mask = (superellipse <= 1.0).float()

                # Intersect with Voronoi zone (only pixels owned by this face)
                if voronoi_masks:
                    voronoi_crop = voronoi_masks[match_idx][cy1:cy2, cx1:cx2]
                    mask = mask * voronoi_crop

                if feather > 0:
                    mask = _gaussian_blur_gpu(mask, feather)

                # ── Upscale for processing ──
                if upscale_factor > 1.0:
                    process_w = ((int(actual_w * upscale_factor) + 7) // 8) * 8
                    process_h = ((int(actual_h * upscale_factor) + 7) // 8) * 8

                    crop_pil = Image.fromarray((crop_tensor.cpu().numpy() * 255).astype(np.uint8))
                    crop_pil = crop_pil.resize((process_w, process_h), Image.Resampling.LANCZOS)
                    crop_tensor = torch.from_numpy(np.array(crop_pil).astype(np.float32) / 255.0)
                    crop_tensor = crop_tensor.unsqueeze(0)  # (1, process_h, process_w, C)

                    mask_pil = Image.fromarray((mask.cpu().numpy() * 255).astype(np.uint8))
                    mask_pil = mask_pil.resize((process_w, process_h), Image.Resampling.LANCZOS)
                    process_mask = torch.from_numpy(np.array(mask_pil).astype(np.float32) / 255.0)
                    mask_batch = process_mask.unsqueeze(0)
                else:
                    process_w, process_h = actual_w, actual_h
                    crop_tensor = crop_tensor.unsqueeze(0)  # (1, actual_h, actual_w, C)
                    mask_batch = mask.unsqueeze(0)

                # ── VAE encode → KSampler → VAE decode ──
                latent = vae.encode(crop_tensor[:, :, :, :3])
                latent_dict = {"samples": latent}
                if noise_mask:
                    latent_dict["noise_mask"] = mask_batch

                refined_latent = _run_ksampler(
                    patched_model, seed, steps, cfg,
                    sampler_name, scheduler,
                    positive_cond, negative_cond, latent_dict, denoise,
                )
                enhanced = vae.decode(refined_latent)

                # ── Downscale back to target size ──
                # vae.decode returns (B, H, W, C); squeeze batch dim
                enhanced_crop = enhanced[0].clamp(0, 1)
                while enhanced_crop.dim() > 3:
                    enhanced_crop = enhanced_crop[0]

                if upscale_factor > 1.0 and (enhanced_crop.shape[1] != actual_w or enhanced_crop.shape[0] != actual_h):
                    enhanced_np = (enhanced_crop.cpu().numpy() * 255).astype(np.uint8)
                    enhanced_pil = Image.fromarray(enhanced_np)
                    enhanced_pil = enhanced_pil.resize((actual_w, actual_h), Image.Resampling.LANCZOS)
                    enhanced_crop = torch.from_numpy(np.array(enhanced_pil).astype(np.float32) / 255.0)

                # ── Paste back with feathered mask ──
                alpha = mask.unsqueeze(-1)
                result_image[batch_idx, cy1:cy2, cx1:cx2] = (
                    alpha * enhanced_crop
                    + (1 - alpha) * result_image[batch_idx, cy1:cy2, cx1:cx2]
                )
                combined_mask[batch_idx, cy1:cy2, cx1:cx2] = torch.maximum(
                    combined_mask[batch_idx, cy1:cy2, cx1:cx2], mask
                )
                crop_region[batch_idx, cy1:cy2, cx1:cx2] = 1.0

                log_lines.append(
                    f"  {name} | LoRA: {lora_info} | "
                    f"crop: {actual_w}x{actual_h} | "
                    f"process: {process_w}x{process_h} | "
                    f"Prompt: {prompt}"
                )
                seed += 1

        info = _compute_info(
            face_context, char_tags, quality_tags, artist_tags,
            lora_list, base_model,
        )
        info += "\n" + "═" * 50 + "\nProcessing log:\n" + "\n".join(log_lines)
        # crop_preview: expanded crop area (bright) + superellipse mask (highlight)
        crop_preview = image * (crop_region * 0.3 + combined_mask * 0.7).unsqueeze(-1)
        print(f"[CharLoraFaceDetailer] Processed {len(matches)} faces")
        return (result_image, combined_mask, crop_preview, info)
