"""
SoyaCharLoraEyeDetailer – Per-face eye detailer with character-specific LoRA patches.

For each face in face_context:
  1. Identify character → find matching LoRA (filtered by base_model)
  2. Apply ONLY that character's LoRA to a fresh copy of the original model
  3. Assemble prompt: quality_tags + artist_tags + EYE_TAGS
  4. Crop face region from image (8-aligned coords, crop_expand_factor padding)
  5. Build eye inpainting mask from eye_context (eye - eyebrow)
  6. Upscale crop by upscale_factor for higher-resolution processing
  7. VAE encode → KSampler → VAE decode
  8. Eyebrow HSV restoration (H,S from original, V from enhanced)
  9. Downscale back → paste with feathered eye mask
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


# ── Shared helpers (from soya_char_lora_face_detailer) ─────────────

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
        char = entry.get("CHAR", "").strip()
        if not char:
            char = _extract_char_from_path(entry.get("lora_path", ""), char_names)
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
                "EYE_TAGS": entry.get("EYE_TAGS", ""),
                "TRIGGER_ANIMA": entry.get("TRIGGER_ANIMA", ""),
                "TRIGGER_SDXL": entry.get("TRIGGER_SDXL", ""),
            }
            char_names.append(name)

    filtered_loras = [l for l in lora_data if l.get("BASE", "").strip() == base_model.strip()]
    lora_map = _build_lora_map(filtered_loras, char_names)

    return char_map, char_names, lora_map, len(lora_data), len(filtered_loras)


# ── GPU / HSV helpers ──────────────────────────────────────────────

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


def _rgb_to_hsv(rgb):
    """(H, W, 3) float32 RGB [0,1] → HSV [0,1]."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc

    h = np.zeros_like(maxc)
    mask = delta > 0
    m = mask & (maxc == r)
    h[m] = ((g[m] - b[m]) / delta[m]) % 6.0
    m = mask & (maxc == g)
    h[m] = (b[m] - r[m]) / delta[m] + 2.0
    m = mask & (maxc == b)
    h[m] = (r[m] - g[m]) / delta[m] + 4.0
    h = h / 6.0
    h[h < 0] += 1.0

    s = np.where(maxc > 0, delta / maxc, 0.0)
    v = maxc

    return np.stack([h, s, v], axis=-1)


def _hsv_to_rgb(hsv):
    """(H, W, 3) float32 HSV [0,1] → RGB [0,1]."""
    h = hsv[..., 0] * 6.0
    s = hsv[..., 1]
    v = hsv[..., 2]

    i = np.floor(h).astype(np.int32) % 6
    f = h - np.floor(h)
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))

    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])

    return np.stack([r, g, b], axis=-1).clip(0, 1)


def _compute_voronoi_zones(matches, H, W):
    if not matches:
        return []

    centers = []
    for m in matches:
        crop = m["crop"]
        centers.append(((crop[0] + crop[2]) / 2.0, (crop[1] + crop[3]) / 2.0))

    if len(centers) == 1:
        mask = torch.ones((H, W), dtype=torch.float32)
        return [mask]

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


# ── Main Eye Detailer Node ─────────────────────────────────────────

class SoyaCharLoraEyeDetailer_mdsoya:
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
                "eye_context": ("EYE_CONTEXT",),
                "char_tags": ("STRING", {"multiline": True, "default": '{"list":[{"CHAR":"name","FACE_TAGS":"face tags","EYE_TAGS":"eye tags","POSITIVE":"positive tags"}]}'}),
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
                "noise_mask": ("BOOLEAN", {"default": True}),
                "eyebrow_restore": ("BOOLEAN", {"default": True}),
                "eyebrow_opacity": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "eyebrow_th": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("image", "mask", "crop_preview", "info")
    FUNCTION = "execute"
    CATEGORY = "Soya/FaceDetailer"

    def execute(self, *, enable, image, model, clip, vae, face_context,
                eye_context, char_tags, quality_tags, artist_tags, negative,
                lora_list, base_model,
                crop_expand_factor, upscale_factor, seed, steps, cfg,
                sampler_name, scheduler,
                denoise, feather, noise_mask,
                eyebrow_restore, eyebrow_opacity, eyebrow_th):

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
        eye_faces = eye_context.get("faces", [])

        if len(eye_faces) != len(matches):
            return (image, torch.zeros((B, H, W), dtype=torch.float32),
                    torch.zeros_like(image),
                    f"face_context has {len(matches)} faces but eye_context has {len(eye_faces)}")

        result_image = image.clone()
        combined_mask = torch.zeros((B, H, W), dtype=torch.float32)
        crop_region = torch.zeros((B, H, W), dtype=torch.float32)
        log_lines = []

        negative_cond = _encode_conditioning(clip, negative)
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

                eye_face = eye_faces[match_idx]
                eye_mask_local = eye_face.get("eye_mask")
                eb_mask_local = eye_face.get("eyebrow_mask")
                bbox_clamped = eye_face.get("bbox_clamped")

                if eye_mask_local is None or bbox_clamped is None:
                    log_lines.append(f"  {name} — NO EYE MASK, SKIPPED")
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

                # ── Tags & prompt (EYE_TAGS only) ───────────────
                tags = char_map[name]
                eye_tags = tags["EYE_TAGS"]
                trigger_key = "TRIGGER_ANIMA" if base_model.strip() == "anima" else "TRIGGER_SDXL"
                trigger = tags.get(trigger_key, "").strip()
                parts = [p for p in [trigger, quality_tags, artist_tags, eye_tags] if p.strip()]
                prompt = ", ".join(parts)

                positive_cond = _encode_conditioning(clip, prompt)

                # ── Crop region from face_context ──
                cr_x1, cr_y1, cr_x2, cr_y2 = int(crop[0]), int(crop[1]), int(crop[2]), int(crop[3])
                cr_w, cr_h = cr_x2 - cr_x1, cr_y2 - cr_y1
                cr_cx, cr_cy = (cr_x1 + cr_x2) / 2.0, (cr_y1 + cr_y2) / 2.0

                ew = cr_w * crop_expand_factor
                eh = cr_h * crop_expand_factor

                crop_w = ((int(ew) + 7) // 8) * 8
                crop_h = ((int(eh) + 7) // 8) * 8

                if crop_w < 8 or crop_h < 8:
                    log_lines.append(f"  {name} — crop too small ({crop_w}x{crop_h})")
                    continue

                cx1 = (int(cr_cx - crop_w / 2) // 8) * 8
                cy1 = (int(cr_cy - crop_h / 2) // 8) * 8
                cx2 = cx1 + crop_w
                cy2 = cy1 + crop_h

                if cx2 > W:
                    cx1 = max(0, (W - crop_w) // 8 * 8)
                    cx2 = cx1 + crop_w
                if cy2 > H:
                    cy1 = max(0, (H - crop_h) // 8 * 8)
                    cy2 = cy1 + crop_h

                cx1 = max(0, cx1)
                cy1 = max(0, cy1)
                cx2 = min(W, cx2)
                cy2 = min(H, cy2)

                cx1 = min(cx1, (cr_x1 // 8) * 8)
                cy1 = min(cy1, (cr_y1 // 8) * 8)
                cx2 = max(cx2, ((cr_x2 + 7) // 8) * 8)
                cy2 = max(cy2, ((cr_y2 + 7) // 8) * 8)

                cx1 = max(0, cx1)
                cy1 = max(0, cy1)
                cx2 = min(W, cx2)
                cy2 = min(H, cy2)
                actual_w, actual_h = cx2 - cx1, cy2 - cy1

                # ── Crop from original image ──
                crop_tensor = image[batch_idx, cy1:cy2, cx1:cx2].clone()

                # ── Build eye mask in crop space ──
                bx1, by1, bx2, by2 = bbox_clamped
                offset_x = bx1 - cx1
                offset_y = by1 - cy1

                # Place eye mask into crop coordinate space
                eye_full = np.zeros((actual_h, actual_w), dtype=np.uint8)
                eb_full = np.zeros((actual_h, actual_w), dtype=np.float32)

                em_h, em_w = eye_mask_local.shape[:2]
                ex1_local = max(0, offset_x)
                ey1_local = max(0, offset_y)
                ex2_local = min(actual_w, offset_x + em_w)
                ey2_local = min(actual_h, offset_y + em_h)

                # Source region in mask space (handles negative offsets)
                sx1 = max(0, -offset_x)
                sy1 = max(0, -offset_y)
                sx2 = sx1 + (ex2_local - ex1_local)
                sy2 = sy1 + (ey2_local - ey1_local)

                if ex2_local > ex1_local and ey2_local > ey1_local:
                    eye_full[ey1_local:ey2_local, ex1_local:ex2_local] = \
                        eye_mask_local[sy1:sy2, sx1:sx2]
                    if eb_mask_local is not None:
                        eb_full[ey1_local:ey2_local, ex1_local:ex2_local] = \
                            eb_mask_local[sy1:sy2, sx1:sx2]

                # Pure eye mask: eye minus eyebrow
                eb_binary = (eb_full > eyebrow_th).astype(np.uint8)
                pure_eye = eye_full * (1 - eb_binary)

                mask = torch.from_numpy(pure_eye.astype(np.float32))

                # For paste-back and eyebrow opacity: full eye area (before subtract)
                full_eye_mask = torch.from_numpy(eye_full.astype(np.float32))
                eb_mask_t = torch.from_numpy(eb_binary.astype(np.float32))

                if mask.sum() == 0:
                    log_lines.append(f"  {name} — empty eye mask after eyebrow subtract")
                    continue

                # Intersect with Voronoi zone
                if voronoi_masks:
                    voronoi_crop = voronoi_masks[match_idx][cy1:cy2, cx1:cx2]
                    mask = mask * voronoi_crop
                    full_eye_mask = full_eye_mask * voronoi_crop
                    eb_mask_t = eb_mask_t * voronoi_crop

                if feather > 0:
                    mask = _gaussian_blur_gpu(mask, feather)
                    full_eye_mask = _gaussian_blur_gpu(full_eye_mask, feather)

                # ── Store original crop for eyebrow restoration ──
                original_crop_np = crop_tensor.cpu().numpy()

                # ── Upscale for processing ──
                if upscale_factor > 1.0:
                    process_w = ((int(actual_w * upscale_factor) + 7) // 8) * 8
                    process_h = ((int(actual_h * upscale_factor) + 7) // 8) * 8

                    crop_pil = Image.fromarray((crop_tensor.cpu().numpy() * 255).astype(np.uint8))
                    crop_pil = crop_pil.resize((process_w, process_h), Image.Resampling.LANCZOS)
                    crop_tensor = torch.from_numpy(np.array(crop_pil).astype(np.float32) / 255.0)
                    crop_tensor = crop_tensor.unsqueeze(0)

                    mask_pil = Image.fromarray((mask.cpu().numpy() * 255).astype(np.uint8))
                    mask_pil = mask_pil.resize((process_w, process_h), Image.Resampling.LANCZOS)
                    process_mask = torch.from_numpy(np.array(mask_pil).astype(np.float32) / 255.0)
                    mask_batch = process_mask.unsqueeze(0)

                    # Also upscale full_eye_mask and eb_mask for eyebrow restoration
                    fem_pil = Image.fromarray((full_eye_mask.cpu().numpy() * 255).astype(np.uint8))
                    fem_pil = fem_pil.resize((process_w, process_h), Image.Resampling.LANCZOS)
                    process_fem = torch.from_numpy(np.array(fem_pil).astype(np.float32) / 255.0)

                    ebm_pil = Image.fromarray((eb_mask_t.cpu().numpy() * 255).astype(np.uint8))
                    ebm_pil = ebm_pil.resize((process_w, process_h), Image.Resampling.LANCZOS)
                    process_ebm = torch.from_numpy(np.array(ebm_pil).astype(np.float32) / 255.0)
                else:
                    process_w, process_h = actual_w, actual_h
                    crop_tensor = crop_tensor.unsqueeze(0)
                    mask_batch = mask.unsqueeze(0)
                    process_fem = full_eye_mask
                    process_ebm = eb_mask_t

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

                enhanced_crop = enhanced[0].clamp(0, 1)
                while enhanced_crop.dim() > 3:
                    enhanced_crop = enhanced_crop[0]

                # ── Eyebrow HSV restoration ──
                if eyebrow_restore and process_ebm.sum() > 0:
                    enhanced_np = enhanced_crop.cpu().numpy()

                    # Resize original crop to match process size if upscaled
                    if process_h != actual_h or process_w != actual_w:
                        orig_pil = Image.fromarray((original_crop_np * 255).astype(np.uint8))
                        orig_pil = orig_pil.resize((process_w, process_h), Image.Resampling.LANCZOS)
                        original_resized = np.array(orig_pil).astype(np.float32) / 255.0
                    else:
                        original_resized = original_crop_np.copy()

                    # hs_preserve: H,S from original, V from enhanced
                    orig_hsv = _rgb_to_hsv(original_resized)
                    enh_hsv = _rgb_to_hsv(enhanced_np)

                    merged_hsv = enh_hsv.copy()
                    merged_hsv[..., 0] = orig_hsv[..., 0]
                    merged_hsv[..., 1] = orig_hsv[..., 1]
                    merged_rgb = _hsv_to_rgb(merged_hsv)

                    eb_binary_process = (process_ebm.cpu().numpy() > 0.5).astype(np.float32)
                    mask_3ch = eb_binary_process[..., np.newaxis]
                    enhanced_np = mask_3ch * merged_rgb + (1 - mask_3ch) * enhanced_np

                    # Eyebrow opacity: blend eyebrow area toward original
                    if eyebrow_opacity > 0:
                        opacity_3ch = eb_binary_process[..., np.newaxis] * eyebrow_opacity
                        enhanced_np = opacity_3ch * original_resized + (1 - opacity_3ch) * enhanced_np

                    enhanced_crop = torch.from_numpy(enhanced_np.astype(np.float32))

                # ── Downscale back to target size ──
                if upscale_factor > 1.0 and (enhanced_crop.shape[1] != actual_h or enhanced_crop.shape[0] != actual_w):
                    enhanced_np = (enhanced_crop.cpu().numpy() * 255).astype(np.uint8)
                    enhanced_pil = Image.fromarray(enhanced_np)
                    enhanced_pil = enhanced_pil.resize((actual_w, actual_h), Image.Resampling.LANCZOS)
                    enhanced_crop = torch.from_numpy(np.array(enhanced_pil).astype(np.float32) / 255.0)

                # ── Paste back with feathered eye mask ──
                alpha = full_eye_mask.unsqueeze(-1)
                result_image[batch_idx, cy1:cy2, cx1:cx2] = (
                    alpha * enhanced_crop
                    + (1 - alpha) * result_image[batch_idx, cy1:cy2, cx1:cx2]
                )
                combined_mask[batch_idx, cy1:cy2, cx1:cx2] = torch.maximum(
                    combined_mask[batch_idx, cy1:cy2, cx1:cx2], full_eye_mask
                )
                crop_region[batch_idx, cy1:cy2, cx1:cx2] = 1.0

                log_lines.append(
                    f"  {name} | LoRA: {lora_info} | "
                    f"crop:({cx1},{cy1},{cx2},{cy2}) {actual_w}x{actual_h} | "
                    f"process: {process_w}x{process_h} | "
                    f"Prompt: {prompt}"
                )
                seed += 1

        # Build info
        info_lines = [
            f"Faces: {len(matches)} | eye_context: {len(eye_faces)}",
            "─" * 50,
        ]
        for match_idx, match in enumerate(matches):
            name = match["name"]
            if name == "unknown":
                info_lines.append(f"Face {match_idx + 1}: unknown — SKIPPED")
                continue
            if name not in char_map:
                info_lines.append(f"Face {match_idx + 1}: {name} — NO TAGS")
                continue
            tags = char_map[name]
            eye_tags = tags["EYE_TAGS"]
            trigger_key = "TRIGGER_ANIMA" if base_model.strip() == "anima" else "TRIGGER_SDXL"
            trigger = tags.get(trigger_key, "").strip()
            parts = [p for p in [trigger, quality_tags, artist_tags, eye_tags] if p.strip()]
            prompt = ", ".join(parts)
            lora_entry = lora_map.get(name)
            info_lines.append(f"Face {match_idx + 1}: {name}")
            info_lines.append(f"  Crop: {match['crop']}")
            if lora_entry:
                info_lines.append(f"  LoRA: {os.path.basename(lora_entry['lora_path'])} @ {lora_entry.get('str', 1.0)}")
            else:
                info_lines.append("  LoRA: NOT FOUND")
            info_lines.append(f"  Prompt: {prompt}")

        info = "\n".join(info_lines)
        info += "\n" + "═" * 50 + "\nProcessing log:\n" + "\n".join(log_lines)

        crop_preview = image * (crop_region * 0.3 + combined_mask * 0.7).unsqueeze(-1)
        print(f"[CharLoraEyeDetailer] Processed {len(matches)} faces")
        return (result_image, combined_mask, crop_preview, info)
