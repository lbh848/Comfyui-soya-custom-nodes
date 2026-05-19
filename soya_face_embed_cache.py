"""
SoyaFaceEmbedCache – Detect faces in images, crop, encode via CLIP Vision, cache result.

Flow:
  1. Check if cache file exists at <path>/cache.ipadpt
     - If exists: load raw embeds from cache, combine with current method, return
  2. Otherwise: load images → face detection (insightface + YOLO fallback) → crop faces
     → CLIP Vision encode → save raw embeds to cache → combine → return
"""

import os
import sys

import numpy as np
import torch
import folder_paths
from PIL import Image, ImageOps
import node_helpers

from .soya_ipadapter_embed_cache import (
    SUPPORTED_EXTENSIONS, CACHE_FILENAME, COMBINE_METHODS,
    _detect_is_plus, _combine_embeds, _import_encode_image_masked,
)


# ---------------------------------------------------------------------------
# Face detection helpers (moved from soya_faceid_yolo_fallback.py)
# ---------------------------------------------------------------------------

def _crop_by_bbox(source, bbox, crop_factor, target_size):
    """Crop region around bbox expanded by crop_factor, resize to target_size.

    Args:
        source: (H, W, 3) uint8 numpy array (BGR)
        bbox: [x1, y1, x2, y2] bounding box
        crop_factor: expansion factor (1.0 = tight, 3.0 = wide)
        target_size: output size (224 or 256)

    Returns:
        (1, target_size, target_size, 3) float32 [0,1] tensor (RGB)
    """
    H, W = source.shape[:2]
    x1, y1, x2, y2 = bbox
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

    crop = source[ny1:ny2, nx1:nx2].copy()
    from PIL import Image as PILImage
    crop_rgb = crop[:, :, ::-1]  # BGR → RGB
    pil_img = PILImage.fromarray(crop_rgb)
    pil_img = pil_img.resize((target_size, target_size), PILImage.BILINEAR)
    tensor = torch.from_numpy(np.array(pil_img).astype(np.float32) / 255.0).unsqueeze(0)
    return tensor


def _detect_face_with_yolo_fallback(image_numpy, insightface_model, yolo_model,
                                     yolo_threshold, yolo_crop_factor, is_sdxl,
                                     need_insightface_embed=True):
    """Detect face: insightface first, YOLO crop as fallback.

    Uses face bbox expanded by yolo_crop_factor for the crop,
    so hair, accessories, and head shape are captured.

    Args:
        need_insightface_embed: If False, InsightFace embed extraction is skipped
            when YOLO finds the face. Only the crop image is needed (for non-FaceID
            models that use CLIP Vision instead of InsightFace embeddings).

    Returns:
        (face_embed, face_crop_tensor)
    """
    norm_size = 256 if is_sdxl else 224
    H, W = image_numpy.shape[:2]

    # 1차: insightface progressive loop (640→256)
    for size in range(640, 256, -64):
        insightface_model.det_model.input_size = (size, size)
        face = insightface_model.get(image_numpy)
        if face:
            face_embed = torch.from_numpy(face[0].normed_embedding).unsqueeze(0)
            face_crop = _crop_by_bbox(
                image_numpy, face[0].bbox, yolo_crop_factor, norm_size,
            )
            return face_embed, face_crop

    # 2차: YOLO fallback → crop
    if yolo_model is not None:
        print(f"[Soya:FaceEmbedCache] InsightFace failed on {W}x{H}, trying YOLO fallback...")
        results = yolo_model(image_numpy, verbose=False)
        best_bbox = None
        best_conf = 0.0
        best_area = 0

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                conf = float(box.conf[0])
                if conf < yolo_threshold:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                area = (x2 - x1) * (y2 - y1)
                if area > best_area:
                    best_area = area
                    best_bbox = (x1, y1, x2, y2)
                    best_conf = conf

        if best_bbox is not None:
            yolo_crop = _crop_by_bbox(
                image_numpy, best_bbox, yolo_crop_factor, norm_size,
            )

            if not need_insightface_embed:
                dummy_embed = torch.zeros(1, 512)
                print(f"[Soya:FaceEmbedCache] YOLO fallback: using crop directly")
                return dummy_embed, yolo_crop

            # FaceID: must extract InsightFace embedding from crop
            yolo_crop_np = (yolo_crop[0].numpy() * 255).astype(np.uint8)[:, :, ::-1].copy()

            for det_size in range(640, 192, -64):
                insightface_model.det_model.input_size = (det_size, det_size)
                face = insightface_model.get(yolo_crop_np)
                if face:
                    face_embed = torch.from_numpy(face[0].normed_embedding).unsqueeze(0)
                    print(f"[Soya:FaceEmbedCache] YOLO fallback succeeded, insightface det_size={det_size}")
                    return face_embed, yolo_crop

            print(f"[Soya:FaceEmbedCache] YOLO found face but insightface failed on crop")

    raise Exception(
        f"InsightFace: No face detected in {W}x{H} image. "
        f"Insightface progressive loop and YOLO fallback both failed."
    )


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class SoyaFaceEmbedCache_mdsoya:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": "", "multiline": False}),
                "ipadapter": ("IPADAPTER",),
                "clip_vision": ("CLIP_VISION",),
                "combine_method": (COMBINE_METHODS,),
                "bbox_detector": ("BBOX_DETECTOR",),
                "yolo_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "yolo_crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.1}),
            },
            "optional": {
                "insightface": ("INSIGHTFACE",),
            },
        }

    RETURN_TYPES = ("EMBEDS",)
    RETURN_NAMES = ("embeds",)
    FUNCTION = "execute"
    CATEGORY = "ipadapter/faceid"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def execute(self, path, ipadapter, clip_vision, combine_method,
                bbox_detector, yolo_threshold, yolo_crop_factor,
                insightface=None):
        import comfy.model_management

        # ── Resolve path ──
        path = path.strip()
        if not os.path.isabs(path):
            input_dir = folder_paths.get_input_directory()
            path = os.path.join(input_dir, path)
        if not os.path.isdir(path):
            raise ValueError(f"Directory not found: {path}")

        cache_path = os.path.join(path, CACHE_FILENAME)

        # ── Cache hit ──
        if os.path.isfile(cache_path):
            print(f"[Soya:FaceEmbedCache] Loading cached embeds from {cache_path}")
            raw_embeds = torch.load(cache_path).cpu()
            combined = _combine_embeds(raw_embeds, combine_method)
            print(f"[Soya:FaceEmbedCache] Combined ({combine_method}): {raw_embeds.shape[0]} faces → {combined.shape}")
            return (combined,)

        # ── Cache miss ──
        encode_image_masked = _import_encode_image_masked()
        is_plus = _detect_is_plus(ipadapter)

        # Detect SDXL from cross_attention_dim
        output_cross_attention_dim = ipadapter["ip_adapter"]["1.to_k_ip.weight"].shape[1]
        is_sdxl = output_cross_attention_dim == 2048

        # Load insightface if not provided
        if insightface is None:
            from .soya_scheduler.model_manager import get_insightface_model
            insightface = get_insightface_model()

        yolo_model = bbox_detector.bbox_model

        # Load images
        files = sorted([
            f for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f))
            and os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
        ])
        if not files:
            raise ValueError(f"No supported image files found in: {path}")

        face_crops = []
        for filename in files:
            filepath = os.path.join(path, filename)
            try:
                img = node_helpers.pillow(Image.open, filepath)
                img = ImageOps.exif_transpose(img)
                if img.mode == 'I':
                    img = img.point(lambda i: i * (1 / 255))
                img = img.convert("RGB")
            except Exception:
                continue

            img_rgb = np.array(img).astype(np.uint8)
            img_bgr = img_rgb[:, :, ::-1].copy()

            # Detect face: insightface first, YOLO fallback
            # need_insightface_embed=False since we only need the crop for CLIP Vision
            _, face_crop = _detect_face_with_yolo_fallback(
                img_bgr, insightface, yolo_model,
                yolo_threshold, yolo_crop_factor, is_sdxl,
                need_insightface_embed=False,
            )
            face_crops.append(face_crop)

        if not face_crops:
            raise ValueError(f"No faces detected in images from: {path}")

        batch = torch.cat(face_crops, dim=0)

        # CLIP Vision encode
        comfy.model_management.load_model_gpu(clip_vision.patcher)
        encoded = encode_image_masked(clip_vision, batch, batch_size=0)

        if is_plus:
            raw_embeds = encoded.penultimate_hidden_states
        else:
            raw_embeds = encoded.image_embeds

        # Save raw embeds to cache
        torch.save(raw_embeds, cache_path)
        print(f"[Soya:FaceEmbedCache] Saved raw embeds to {cache_path}")
        print(f"[Soya:FaceEmbedCache] Faces: {len(face_crops)}, Plus: {is_plus}, SDXL: {is_sdxl}, Raw shape: {raw_embeds.shape}")

        # Combine and return
        combined = _combine_embeds(raw_embeds, combine_method)
        print(f"[Soya:FaceEmbedCache] Combined ({combine_method}): {combined.shape}")

        return (combined,)
