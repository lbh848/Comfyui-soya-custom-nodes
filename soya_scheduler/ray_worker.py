"""
Ray remote worker for parallel face analysis pipeline.

Steps:
  A. Prompt filtering  – find which characters appear in the prompt
  B. Face extraction   – YOLO bbox detection + crop
  C+D. Batch embedding – encode reference + face crops in single CLIP pass, then Hungarian matching
  E-F. Prompt build    – [LAB] format output
"""

# ── Fix module name for Ray pickling (MUST be before any function defs) ──
# ComfyUI loads custom nodes using spec_from_file_location() with the
# full directory path as __name__ (e.g. 'E:\...\comfyui-soya-custom').
# cloudpickle serializes ALL referenced functions by __module__, so every
# function in this file must have __module__ = 'soya_scheduler.ray_worker'.
import sys as _sys
_clean_name = "soya_scheduler.ray_worker"
_this_mod = _sys.modules[__name__]
_sys.modules[_clean_name] = _this_mod
# Changing globals()['__name__'] ensures all subsequent function definitions
# get __module__ = _clean_name automatically.
globals()['__name__'] = _clean_name

import io
import time
import base64
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment

# ── model caches (per worker process) ──────────────────────────
_yolo_cache = {}       # key: (model_path, device) -> YOLO model
_clip_cache = {}       # key: (model_path, device) -> clip_vision model


def _cleanup_old_device(cache, model_path, new_device):
    """Remove cached models for the same model_path but different device."""
    to_remove = [k for k in cache if k[0] == model_path and k[1] != new_device]
    if not to_remove:
        return
    for k in to_remove:
        old_model = cache.pop(k, None)
        if old_model is not None:
            try:
                if hasattr(old_model, 'patcher') and hasattr(old_model.patcher, 'model'):
                    old_model.patcher.model = old_model.patcher.model.to("cpu")
                elif hasattr(old_model, 'to'):
                    old_model.to("cpu")
            except Exception:
                pass
    import gc
    gc.collect()
    # empty_cache() uses the current device – only call when actually using CUDA
    if new_device.startswith("cuda:"):
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _get_yolo(model_path, device):
    key = (model_path, device)
    if key not in _yolo_cache:
        _cleanup_old_device(_yolo_cache, model_path, device)
        from ultralytics import YOLO
        _yolo_cache[key] = YOLO(model_path).to(device)
    return _yolo_cache[key]


def _get_clip_vision(model_path, device):
    key = (model_path, device)
    if key not in _clip_cache:
        _cleanup_old_device(_clip_cache, model_path, device)
        import comfy.model_management
        import comfy.clip_vision
        # Patch ComfyUI's device detection – Ray workers may have different
        # GPU visibility, causing text_encoder_device() → should_use_fp16()
        # to fail with "Invalid device id".
        _orig = comfy.model_management.text_encoder_device
        comfy.model_management.text_encoder_device = lambda: torch.device(device)
        try:
            clip_v = comfy.clip_vision.load(model_path)
        finally:
            comfy.model_management.text_encoder_device = _orig
        # Bypass ComfyUI's model management: load directly to target device
        dev = torch.device(device)
        clip_v.load_device = dev
        clip_v.offload_device = dev
        clip_v.patcher.model = clip_v.patcher.model.to(dev)
        _clip_cache[key] = clip_v
    return _clip_cache[key]


# ── helper functions ────────────────────────────────────────────

def _filter_characters_by_prompt(characters, prompt):
    """A. Return list of characters whose name appears in the prompt."""
    prompt_norm = prompt.lower().replace('_', ' ')
    matched = []
    for ch in characters:
        name = ch["name"].lower().replace('_', ' ')
        if name in prompt_norm:
            matched.append(ch)
    return matched


def _detect_faces_yolo(image_numpy, yolo_model, threshold=0.5, crop_factor=3.0, max_faces=0):
    """B. Detect faces with YOLO and return cropped face images + bboxes."""
    # image_numpy: (H, W, 3) uint8
    img = Image.fromarray(image_numpy.astype(np.uint8))
    W, H = img.size

    results = yolo_model(image_numpy, verbose=False)
    faces = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            conf = float(box.conf[0])
            if conf < threshold:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            # expand bbox by crop_factor
            bw = x2 - x1
            bh = y2 - y1
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            new_w = bw * crop_factor
            new_h = bh * crop_factor
            nx1 = max(0, int(cx - new_w / 2))
            ny1 = max(0, int(cy - new_h / 2))
            nx2 = min(W, int(cx + new_w / 2))
            ny2 = min(H, int(cy + new_h / 2))

            crop = img.crop((nx1, ny1, nx2, ny2))
            crop_np = np.array(crop)
            faces.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "crop_region": [nx1, ny1, nx2, ny2],
                "crop": crop_np,
                "confidence": conf,
            })

    # sort by bbox size descending, keep top max_faces
    for f in faces:
        f["area"] = (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1])
    faces.sort(key=lambda f: f["area"], reverse=True)
    if max_faces > 0:
        faces = faces[:max_faces]
    # re-sort left to right by bbox x1 for consistent ordering
    faces.sort(key=lambda f: f["bbox"][0])
    return faces


def _encode_images(clip_vision, images_numpy_list, device):
    """Encode a list of (H,W,3) numpy arrays using CLIP Vision.
    Returns tensor of shape (N, embed_dim)."""
    if not images_numpy_list:
        return None

    # Resize all images to the same size, then stack into (N, H, W, 3) tensor
    tensors = []
    target_size = (224, 224)
    for img_np in images_numpy_list:
        img = Image.fromarray(img_np.astype(np.uint8))
        img = img.resize(target_size, Image.BILINEAR)
        t = torch.from_numpy(np.array(img).astype(np.float32) / 255.0)
        tensors.append(t)
    batch = torch.stack(tensors).to(device)

    # Bypass ComfyUI's VRAM management: model is already on target device
    import comfy.model_management
    _orig = comfy.model_management.load_model_gpu
    comfy.model_management.load_model_gpu = lambda *a, **kw: None
    try:
        with torch.no_grad():
            output = clip_vision.encode_image(batch, crop="center")
    finally:
        comfy.model_management.load_model_gpu = _orig

    embeds = output.image_embeds if hasattr(output, 'image_embeds') else output.get("image_embeds")
    if embeds is None:
        raise ValueError("Failed to get image embeddings from CLIP Vision")
    if embeds.dim() > 2:
        N = embeds.shape[0]
        embeds = embeds.view(N, -1)
    return embeds.detach().cpu()


def _match_characters_hungarian(query_embeds, ref_embeds, char_names):
    """Hungarian algorithm matching between detected faces (M) and references (N).
    Each reference is assigned at most once. Unassigned faces get 'unknown'."""
    M = query_embeds.shape[0]
    N = ref_embeds.shape[0]

    sim_matrix = F.cosine_similarity(query_embeds.unsqueeze(1), ref_embeds.unsqueeze(0), dim=2)

    cost_matrix = -sim_matrix.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    final_names = ["unknown"] * M
    final_scores = [0.0] * M
    for r, c in zip(row_ind, col_ind):
        final_names[r] = char_names[c]
        final_scores[r] = float(sim_matrix[r, c].item())

    return final_names, final_scores


def _build_lab_prompt(assignments, scores, characters_dict, settings):
    """E-F. Build [LAB] format prompt string."""
    male_enhance = settings.get("male_enhance_prompt", "")
    female_enhance = settings.get("female_enhance_prompt", "")
    common = settings.get("common_prompt", "")
    unknown_gender = settings.get("unknown_gender", "boy")
    unknown_eye_prompt = settings.get("unknown_eye_prompt", "unknown, black eyes")

    entries = []
    for i, (name, score) in enumerate(zip(assignments, scores)):
        if name == "unknown" or name not in characters_dict:
            # unknown character
            gender_display = "boy" if unknown_gender == "boy" else "girl"
            parts = [unknown_eye_prompt, gender_display]
            if common:
                parts.append(common)
            entries.append(", ".join(parts))
            continue

        ch = characters_dict[name]
        gender = ch.get("gender", "girl")
        eye_prompt = ch.get("eye_prompt", "")

        gender_display = "boy" if gender == "boy" else "girl"
        parts = [gender_display]
        if eye_prompt:
            parts.insert(0, eye_prompt)
        elif gender == "boy":
            if male_enhance:
                parts.insert(0, male_enhance)
        else:
            if female_enhance:
                parts.insert(0, female_enhance)
        if common:
            parts.append(common)
        entries.append(", ".join(parts))

    result = "[LAB]\n" + "\n".join([f"[{i+1}] {e}" for i, e in enumerate(entries)])
    return result


def _crop_to_base64(crop_np):
    """Convert numpy crop to base64 string."""
    img = Image.fromarray(crop_np.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── main remote function ────────────────────────────────────────

def analyze_faces_sync(task_id, face_data, ref_data, config):
    """
    CLIP matching pipeline (face detection done in preprocess by Divider).

    Args:
        task_id: unique task identifier
        face_data: list of dicts with bbox, crop_region, crop, confidence, area
        ref_data: dict of {character_name: numpy_crop} (pre-extracted by Divider)
        config: full config dict

    Returns dict with prompts, face_crops, similarities, assignments, timing.
    """
    timing = {}
    t0 = time.time()

    settings = config.get("settings", {})
    characters = config.get("characters", [])
    clip_device = settings.get("clip_device", settings.get("device", "cuda:1"))

    # CPU 모드에서 PyTorch 연산에 멀티스레드 사용
    if clip_device == "cpu":
        num_cpus = settings.get("num_cpus", 1)
        torch.set_num_threads(num_cpus)

    # Collect device info
    num_cpus = settings.get("num_cpus", 1)
    device_info = f"clip: {clip_device}"
    if clip_device.startswith("cuda:"):
        try:
            idx = int(clip_device.split(":")[1])
            alloc = torch.cuda.memory_allocated(idx) / (1024**2)
            total = torch.cuda.get_device_properties(idx).total_mem / (1024**2)
            device_info += f" | GPU {idx} VRAM: {alloc:.0f}/{total:.0f} MB"
        except Exception:
            pass
    else:
        device_info += f" | CPU threads: {num_cpus}"

    # Use pre-detected faces from Divider (no face detection here)
    faces = face_data

    if not faces:
        timing["total"] = time.time() - t0
        return {
            "prompts": "[LAB]\n[1] no face detected",
            "face_crops": [],
            "similarities": [],
            "assignments": [],
            "timing": timing,
        }

    # ── CLIP encoding + matching ─────────────────────────────────
    t3 = time.time()

    # Load CLIP Vision
    clip_model_name = settings.get("clip_vision_model", "clip_vision_vit_h.safetensors")
    clip_path = settings.get("_resolved_clip_path", clip_model_name)

    clip_vision = _get_clip_vision(clip_path, clip_device)

    # Reference images (pre-extracted by Divider)
    ref_images = []
    ref_names = []
    for ch in characters:
        name = ch["name"]
        if name in ref_data:
            ref_images.append(ref_data[name])
            ref_names.append(name)
        else:
            print(f"[RayWorker] WARNING: no reference crop for '{name}'")

    # Face crops from pre-detected faces
    face_crops = [f["crop"] for f in faces]

    # Batch encode ALL images (reference + detected faces) in a single CLIP forward pass
    all_images = ref_images + face_crops
    all_embeds = _encode_images(clip_vision, all_images, clip_device) if all_images else None

    # Split results: first N are reference embeddings, rest are face embeddings
    n_ref = len(ref_images)
    if all_embeds is not None:
        ref_embeds = all_embeds[:n_ref] if n_ref > 0 else None
        query_embeds = all_embeds[n_ref:]
    else:
        ref_embeds = None
        query_embeds = None

    del all_embeds

    # Hungarian matching
    assignments = []
    similarities = []

    if query_embeds is not None and ref_embeds is not None and len(ref_names) > 0:
        assignments, similarities = _match_characters_hungarian(query_embeds, ref_embeds, ref_names)

        for i in range(len(assignments)):
            if assignments[i] is None:
                assignments[i] = "unknown"
                similarities[i] = 0.0
    else:
        if not ref_names:
            print(f"[RayWorker] WARNING: no reference names available for matching "
                  f"(characters={len(characters)}, prompt may not contain character names)")
        assignments = ["unknown"] * len(faces)
        similarities = [0.0] * len(faces)

    timing["clip_embedding_matching"] = time.time() - t3

    # ── Build prompt ─────────────────────────────────────────────
    t5 = time.time()

    # Build characters_dict for quick lookup
    characters_dict = {ch["name"]: ch for ch in characters}

    # ── Separate tracked faces into kept / remain ────────────────
    keep_face_count = settings.get("keep_face_count", settings.get("tracking_face_count", 3))
    keep_matched_only = settings.get("keep_matched_only", False)

    # Build full tracked_faces list with assignment info
    tracked_faces = []
    for i, face in enumerate(faces):
        tracked_faces.append({
            "bbox": face["bbox"],
            "crop_region": face["crop_region"],
            "assignment": assignments[i],
            "similarity": similarities[i],
            "area": face["area"],
        })

    # Sort: matched (non-unknown) first, then unknown; each group by bbox area desc
    matched = [f for f in tracked_faces if f["assignment"] != "unknown"]
    unknown = [f for f in tracked_faces if f["assignment"] == "unknown"]
    matched.sort(key=lambda f: f["area"], reverse=True)
    unknown.sort(key=lambda f: f["area"], reverse=True)
    sorted_tracked = matched + unknown

    if keep_matched_only:
        kept_faces = matched
        remain_faces = unknown
    else:
        # Always keep all matched faces; fill remaining slots with unknown
        kept_count = max(keep_face_count, len(matched))
        kept_faces = sorted_tracked[:kept_count]
        remain_faces = sorted_tracked[kept_count:]

    # Mark kept/remain on sorted_tracked entries for UI display
    kept_bboxes = {tuple(f["bbox"]) for f in kept_faces}
    for f in sorted_tracked:
        f["kept"] = tuple(f["bbox"]) in kept_bboxes

    # Build prompt only for kept faces
    kept_assignments = [f["assignment"] for f in kept_faces]
    kept_similarities = [f["similarity"] for f in kept_faces]
    prompts = _build_lab_prompt(kept_assignments, kept_similarities, characters_dict, settings)

    # All detected faces raw info (from preprocess, before matching)
    all_detected_faces = [
        {"bbox": f["bbox"], "confidence": f["confidence"], "area": f["area"]}
        for f in faces
    ]

    timing["prompt_build"] = time.time() - t5
    timing["total"] = time.time() - t0

    # Encode face crops to base64 for API
    face_crops_b64 = [_crop_to_base64(f["crop"]) for f in faces]

    # Encode reference crops to base64, keyed by character name
    ref_crops_map = {}
    for i, name in enumerate(ref_names):
        if name not in ref_crops_map:
            ref_crops_map[name] = _crop_to_base64(ref_images[i])

    result = {
        "prompts": prompts,
        "face_crops": face_crops_b64,
        "ref_crops_map": ref_crops_map,
        "similarities": similarities,
        "assignments": assignments,
        "timing": timing,
        "device_info": device_info,
        "all_detected_faces": all_detected_faces,
        "tracked_faces": sorted_tracked,
        "kept_faces": kept_faces,
        "keep_face_count": keep_face_count,
        "remain_faces": [{"bbox": f["bbox"]} for f in remain_faces],
    }

    # Free intermediate data – no accumulation across runs
    del faces, ref_images, ref_embeds, query_embeds, face_crops, face_crops_b64, ref_crops_map
    import gc
    gc.collect()

    return result


# ── Ray Actor (single persistent worker) ───────────────────────

import ray

@ray.remote
class FaceAnalyzer:
    """Singleton actor – guarantees one worker, one copy of cached models."""

    def analyze(self, task_id, face_data, ref_data, config):
        return analyze_faces_sync(task_id, face_data, ref_data, config)
