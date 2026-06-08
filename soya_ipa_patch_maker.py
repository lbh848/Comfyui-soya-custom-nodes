import os
import json
import gc
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
import folder_paths
import comfy.model_management
import comfy.clip_vision


def _parse_json_stream(text):
    """Parse one or more concatenated JSON objects from a string."""
    text = text.strip()
    if not text:
        return []
    decoder = json.JSONDecoder()
    objects = []
    pos = 0
    while pos < len(text):
        if text[pos] in ' \t\n\r':
            pos += 1
            continue
        obj, end = decoder.raw_decode(text, idx=pos)
        objects.append(obj)
        pos = end
    return objects


# ── Main-process CLIP vision cache ─────────────────────────────
_main_clip_cache = {}


def _load_clip_vision(model_name, device):
    """Load and cache CLIP vision model for the main process."""
    key = (model_name, device)
    if key in _main_clip_cache:
        return _main_clip_cache[key]

    # Evict old device for same model
    for k in list(_main_clip_cache):
        if k[0] == model_name and k[1] != device:
            _main_clip_cache.pop(k, None)
    gc.collect()

    clip_path = folder_paths.get_full_path_or_raise("clip_vision", model_name)
    clip_v = comfy.clip_vision.load(clip_path)

    if device == "CPU":
        dev = torch.device("cpu")
        clip_v.load_device = dev
        clip_v.offload_device = dev
        clip_v.patcher.model = clip_v.patcher.model.to(dev)

    _main_clip_cache[key] = clip_v
    return clip_v



COMBINE_METHODS = [
    "average", "norm average", "concat", "add", "subtract", "max", "min",
]


def _combine_embeds(embeds, method):
    if method == "concat":
        return embeds
    elif method == "add":
        return torch.sum(embeds, dim=0).unsqueeze(0)
    elif method == "subtract":
        return (embeds[0] - torch.mean(embeds[1:], dim=0)).unsqueeze(0)
    elif method == "average":
        return torch.mean(embeds, dim=0).unsqueeze(0)
    elif method == "norm average":
        return torch.mean(
            embeds / torch.norm(embeds, dim=0, keepdim=True), dim=0,
        ).unsqueeze(0)
    elif method == "max":
        return torch.max(embeds, dim=0).values.unsqueeze(0)
    elif method == "min":
        return torch.min(embeds, dim=0).values.unsqueeze(0)
    return embeds


class SoyaIPAPatchMaker_mdsoya:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "character_names": ("STRING", {"default": "", "multiline": False}),
                "ipa_cache_data": ("STRING", {"default": "", "multiline": True}),
                "face_crop_top": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "face_crop_bottom": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "embed_cache_data": ("STRING", {"default": "", "multiline": True}),
                "bbox_detector": ("BBOX_DETECTOR",),
                "image": ("IMAGE",),
                "config": ("IPA_PATCH_CONFIG",),
                "combine_method": (COMBINE_METHODS,),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING", "IPA_FACE_CONTEXT")
    RETURN_NAMES = ("detected_faces", "named_faces", "names", "info", "face_context")
    OUTPUT_IS_LIST = (True, True, True, False, False)
    FUNCTION = "process"
    CATEGORY = "Soya/FaceMatch"

    def process(self, character_names, ipa_cache_data, face_crop_top, face_crop_bottom,
                embed_cache_data, bbox_detector, image, config, combine_method):

        num_cpus = config.get("num_cpus", 1)
        clip_model_name = config.get("clip_vision_model", "")
        device = config.get("device", "CPU")
        max_faces = config["max_face_count"]
        yolo_conf = config["yolo_confidence"]
        debug = config["debug"]

        # Parse inputs
        char_names = [n.strip() for n in character_names.split(",") if n.strip()]
        if not char_names:
            return ([], [], [], "No character names provided.")

        # Parse ipa_cache_data → per-character IPA embeds
        ipa_parsed = _parse_json_stream(ipa_cache_data)
        ipa_caches = []
        for obj in ipa_parsed:
            if isinstance(obj, dict) and "list" in obj:
                ipa_caches.extend(obj["list"])
            elif isinstance(obj, list):
                ipa_caches.extend(obj)

        input_dir = folder_paths.get_input_directory()
        ipa_embeds_by_char = {}
        for entry in ipa_caches:
            char_name = entry["CHAR"]
            ipa_path = entry["ipa_path"]
            strength = entry.get("str", 0.7)

            if not os.path.isabs(ipa_path):
                ipa_path = os.path.join(input_dir, ipa_path)
            if not os.path.isfile(ipa_path):
                print(f"[IPAPatchMaker] WARNING: IPA cache not found for {char_name}: {ipa_path}")
                continue

            raw = torch.load(ipa_path, map_location="cpu", weights_only=True)
            ipa_embeds_by_char[char_name] = {
                "embeds": _combine_embeds(raw, combine_method),
                "strength": strength,
            }

        # Parse embed_cache_data — may be single JSON or concatenated
        embed_parsed = _parse_json_stream(embed_cache_data)
        embed_caches = []
        for obj in embed_parsed:
            if isinstance(obj, dict) and "list" in obj:
                embed_caches.extend(obj["list"])
            elif isinstance(obj, list):
                embed_caches.extend(obj)

        if not embed_caches:
            return ([], [], [], "No embed cache entries provided.", {})

        # ── STEP 1: YOLO face detection + crop ──
        yolo = bbox_detector.bbox_model
        side_factor = (face_crop_top + face_crop_bottom) / 2

        detected_faces = []
        face_bbox_areas = []
        all_bboxes = []
        img_H, img_W = 0, 0

        for img_tensor in image:
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            W, H = pil_img.size
            img_H, img_W = H, W

            detections = yolo(img_np, verbose=False)
            for result in detections:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    conf = float(box.conf[0])
                    if conf < yolo_conf:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                    bw = x2 - x1
                    bh = y2 - y1
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2

                    nx1 = max(0, int(cx - bw * side_factor / 2))
                    ny1 = max(0, int(cy - bh * face_crop_top / 2))
                    nx2 = min(W, int(cx + bw * side_factor / 2))
                    ny2 = min(H, int(cy + bh * face_crop_bottom / 2))

                    if nx2 <= nx1 or ny2 <= ny1:
                        continue

                    cropped = pil_img.crop((nx1, ny1, nx2, ny2))
                    cropped_np = np.array(cropped).astype(np.float32) / 255.0
                    face_tensor = torch.from_numpy(cropped_np).unsqueeze(0)

                    detected_faces.append(face_tensor)
                    all_bboxes.append((float(x1), float(y1), float(x2), float(y2)))
                    face_bbox_areas.append((bw * bh, len(detected_faces) - 1))

        total_detected = len(detected_faces)
        if total_detected > max_faces:
            face_bbox_areas.sort(key=lambda x: x[0], reverse=True)
            keep_indices = sorted([idx for _, idx in face_bbox_areas[:max_faces]])
            detected_faces = [detected_faces[i] for i in keep_indices]
            all_bboxes = [all_bboxes[i] for i in keep_indices]

        if not detected_faces:
            info = f"No faces detected (YOLO conf: {yolo_conf}, total before filter: {total_detected})"
            return ([], [], [], info, {})

        # ── STEP 2: CLIP Vision encode detected faces ──
        n_workers = min(num_cpus, len(detected_faces))
        use_ray = n_workers > 1
        ray_fallback_error = None
        if use_ray:
            try:
                face_embeds = self._encode_parallel(detected_faces, clip_model_name, n_workers)
            except Exception as e:
                ray_fallback_error = str(e)
                print(f"[IPAPatchMaker] Ray parallel encoding failed ({e}), falling back to sequential")
                clip_vision = _load_clip_vision(clip_model_name, device)
                face_embeds = self._encode_sequential(detected_faces, clip_vision)
        else:
            clip_vision = _load_clip_vision(clip_model_name, device)
            face_embeds = self._encode_sequential(detected_faces, clip_vision)

        face_embeds = torch.cat(face_embeds, dim=0)

        # ── STEP 3: Load cached embeddings + Hungarian matching ──
        char_embeds = {}
        for entry in embed_caches:
            char_name = entry["CHAR"]
            emb_path = os.path.join(input_dir, entry["emb_path"])
            cache = torch.load(emb_path, map_location="cpu", weights_only=True)
            embeds = cache["embeds"]
            if embeds.dim() > 2:
                embeds = embeds.view(embeds.size(0), -1)
            char_embeds[char_name] = embeds

        M = face_embeds.shape[0]
        N = len(char_names)
        sim_matrix = torch.zeros(M, N)

        for j, char_name in enumerate(char_names):
            if char_name not in char_embeds:
                continue
            char_embs = char_embeds[char_name]
            for i in range(M):
                sims = F.cosine_similarity(face_embeds[i].unsqueeze(0), char_embs, dim=1)
                sim_matrix[i, j] = sims.max().item()

        row_ind, col_ind = linear_sum_assignment(-sim_matrix.cpu().numpy())

        final_names = ["unknown"] * M
        final_scores = [0.0] * M
        for r, c in zip(row_ind, col_ind):
            if c < N:
                final_names[r] = char_names[c]
                final_scores[r] = float(sim_matrix[r, c].item())

        matched_count = sum(1 for n in final_names if n != "unknown")
        if ray_fallback_error:
            encode_mode = f"Ray x{n_workers} FAILED -> sequential ({ray_fallback_error})"
        elif use_ray:
            encode_mode = f"Ray x{n_workers}"
        else:
            encode_mode = "sequential"
        info_lines = [
            f"Detected {total_detected} face(s), kept {len(detected_faces)}, "
            f"matched {matched_count} [{encode_mode}]",
        ]
        for i, (name, score) in enumerate(zip(final_names, final_scores)):
            info_lines.append(f"  Face {i + 1}: {name} ({score:.4f})")
        info = "\n".join(info_lines)
        print(f"[IPAPatchMaker] {info}")

        # ── Build face context for downstream nodes ──
        face_context = {
            "matches": [],
            "ipa_embeds": ipa_embeds_by_char,
            "img_H": img_H,
            "img_W": img_W,
        }
        for i, (name, score) in enumerate(zip(final_names, final_scores)):
            bbox = all_bboxes[i] if i < len(all_bboxes) else (0, 0, 0, 0)
            face_context["matches"].append({
                "name": name,
                "bbox": bbox,
                "score": score,
            })

        if not debug:
            return ([], [], [], info, face_context)

        named_faces = []
        named_names = []
        for i, name in enumerate(final_names):
            if name != "unknown":
                named_faces.append(detected_faces[i])
                named_names.append(name)

        return (detected_faces, named_faces, named_names, info, face_context)

    def _encode_sequential(self, detected_faces, clip_vision):
        comfy.model_management.load_model_gpu(clip_vision.patcher)
        embeds = []
        for face in detected_faces:
            encoded = clip_vision.encode_image(face, crop=True)
            embed = encoded.image_embeds
            if embed.dim() > 2:
                embed = embed.view(1, -1)
            embeds.append(embed)
        return embeds

    def _encode_parallel(self, detected_faces, clip_model_name, n_workers):
        import ray
        from .soya_scheduler import get_encoder_pool

        clip_path = folder_paths.get_full_path("clip_vision", clip_model_name)
        if not clip_path:
            raise ValueError(f"CLIP vision model not found: {clip_model_name}")

        print(f"[IPAPatchMaker] Ray parallel encoding: {len(detected_faces)} faces, {n_workers} workers")
        pool = get_encoder_pool(n_workers, clip_path)

        futures = []
        for i, face in enumerate(detected_faces):
            face_np = face.cpu().numpy()
            actor = pool[i % n_workers]
            futures.append(actor.encode.remote(face_np))

        results = ray.get(futures)
        return [torch.from_numpy(r) for r in results]
