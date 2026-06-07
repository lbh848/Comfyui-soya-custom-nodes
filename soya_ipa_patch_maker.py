import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
import folder_paths
import comfy.model_management


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
                "clip_vision": ("CLIP_VISION",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("detected_faces", "named_faces", "names", "info")
    OUTPUT_IS_LIST = (True, True, True, False)
    FUNCTION = "process"
    CATEGORY = "Soya/IPA"

    def process(self, character_names, ipa_cache_data, face_crop_top, face_crop_bottom,
                embed_cache_data, bbox_detector, image, config, clip_vision):

        device_str = config["device"]
        max_faces = config["max_face_count"]
        yolo_conf = config["yolo_confidence"]
        debug = config["debug"]

        # Parse inputs
        char_names = [n.strip() for n in character_names.split(",") if n.strip()]
        if not char_names:
            return ([], [], [], "No character names provided.")

        embed_caches = json.loads(embed_cache_data)["list"]
        json.loads(ipa_cache_data)  # validate for future use

        # ── STEP 1: YOLO face detection + crop ──
        yolo = bbox_detector.bbox_model
        side_factor = (face_crop_top + face_crop_bottom) / 2

        detected_faces = []   # list of (1, H, W, 3) tensors
        face_bbox_areas = []  # (area, index) for size-based filtering

        for img_tensor in image:
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            W, H = pil_img.size

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
                    face_bbox_areas.append((bw * bh, len(detected_faces) - 1))

        # Filter by max face count — keep largest faces
        total_detected = len(detected_faces)
        if total_detected > max_faces:
            face_bbox_areas.sort(key=lambda x: x[0], reverse=True)
            keep_indices = sorted([idx for _, idx in face_bbox_areas[:max_faces]])
            detected_faces = [detected_faces[i] for i in keep_indices]

        if not detected_faces:
            info = f"No faces detected (YOLO conf: {yolo_conf}, total before filter: {total_detected})"
            return ([], [], [], info)

        # ── STEP 2: CLIP Vision encode detected faces ──
        comfy.model_management.load_model_gpu(clip_vision.patcher)

        face_embeds = []
        for face in detected_faces:
            encoded = clip_vision.encode_image(face, crop=True)
            embed = encoded.image_embeds
            if embed.dim() > 2:
                embed = embed.view(1, -1)
            face_embeds.append(embed)

        face_embeds = torch.cat(face_embeds, dim=0)  # (M, embed_dim)

        # ── STEP 3: Load cached embeddings + Hungarian matching ──
        input_dir = folder_paths.get_input_directory()

        char_embeds = {}
        for entry in embed_caches:
            char_name = entry["CHAR"]
            emb_path = os.path.join(input_dir, entry["emb_path"])
            cache = torch.load(emb_path, map_location="cpu")
            embeds = cache["embeds"]  # (K, embed_dim)
            if embeds.dim() > 2:
                embeds = embeds.view(embeds.size(0), -1)
            char_embeds[char_name] = embeds

        # Build similarity matrix: max cosine similarity per (face, character)
        M = face_embeds.shape[0]
        N = len(char_names)
        sim_matrix = torch.zeros(M, N)

        for j, char_name in enumerate(char_names):
            if char_name not in char_embeds:
                continue
            char_embs = char_embeds[char_name]  # (K, embed_dim)
            for i in range(M):
                sims = F.cosine_similarity(face_embeds[i].unsqueeze(0), char_embs, dim=1)
                sim_matrix[i, j] = sims.max().item()

        # Hungarian algorithm for 1:1 optimal assignment
        row_ind, col_ind = linear_sum_assignment(-sim_matrix.cpu().numpy())

        final_names = ["unknown"] * M
        final_scores = [0.0] * M
        for r, c in zip(row_ind, col_ind):
            if c < N:
                final_names[r] = char_names[c]
                final_scores[r] = float(sim_matrix[r, c].item())

        # ── Build outputs ──
        matched_count = sum(1 for n in final_names if n != "unknown")
        info_lines = [
            f"Detected {total_detected} face(s), kept {len(detected_faces)}, matched {matched_count}",
        ]
        for i, (name, score) in enumerate(zip(final_names, final_scores)):
            info_lines.append(f"  Face {i + 1}: {name} ({score:.4f})")
        info = "\n".join(info_lines)
        print(f"[IPAPatchMaker] {info}")

        if not debug:
            return ([], [], [], info)

        # Debug outputs: named faces only
        named_faces = []
        named_names = []
        for i, name in enumerate(final_names):
            if name != "unknown":
                named_faces.append(detected_faces[i])
                named_names.append(name)

        return (detected_faces, named_faces, named_names, info)
