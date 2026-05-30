"""
SoyaFaceEmbedCacheV2 – Same as Face Embed Cache but with separate top/bottom crop factors.
"""

import os
import numpy as np
import torch
import folder_paths
from PIL import Image, ImageOps
import node_helpers

from .soya_ipadapter_embed_cache import (
    SUPPORTED_EXTENSIONS, CACHE_FILENAME, COMBINE_METHODS,
    _detect_is_plus, _combine_embeds, _import_encode_image_masked,
)


def _crop_by_bbox_v2(source, bbox, crop_factor_top, crop_factor_bottom, target_size):
    H, W = source.shape[:2]
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    side_factor = (crop_factor_top + crop_factor_bottom) / 2

    nx1 = max(0, int(cx - bw * side_factor / 2))
    ny1 = max(0, int(cy - bh * crop_factor_top / 2))
    nx2 = min(W, int(cx + bw * side_factor / 2))
    ny2 = min(H, int(cy + bh * crop_factor_bottom / 2))

    crop = source[ny1:ny2, nx1:nx2].copy()
    from PIL import Image as PILImage
    crop_rgb = crop[:, :, ::-1]  # BGR → RGB
    pil_img = PILImage.fromarray(crop_rgb)
    pil_img = pil_img.resize((target_size, target_size), PILImage.BILINEAR)
    tensor = torch.from_numpy(np.array(pil_img).astype(np.float32) / 255.0).unsqueeze(0)
    return tensor


def _detect_face_v2(image_numpy, insightface_model, yolo_model,
                    yolo_threshold, crop_factor_top, crop_factor_bottom, is_sdxl):
    norm_size = 256 if is_sdxl else 224
    H, W = image_numpy.shape[:2]

    # insightface progressive loop
    for size in range(640, 256, -64):
        insightface_model.det_model.input_size = (size, size)
        face = insightface_model.get(image_numpy)
        if face:
            face_crop = _crop_by_bbox_v2(
                image_numpy, face[0].bbox, crop_factor_top, crop_factor_bottom, norm_size,
            )
            return face_crop

    # YOLO fallback
    if yolo_model is not None:
        print(f"[Soya:FaceEmbedCacheV2] InsightFace failed on {W}x{H}, trying YOLO fallback...")
        results = yolo_model(image_numpy, verbose=False)
        best_bbox = None
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

        if best_bbox is not None:
            return _crop_by_bbox_v2(
                image_numpy, best_bbox, crop_factor_top, crop_factor_bottom, norm_size,
            )

    raise Exception(
        f"InsightFace: No face detected in {W}x{H} image. "
        f"Insightface progressive loop and YOLO fallback both failed."
    )


class SoyaFaceEmbedCacheV2_mdsoya:
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
                "face_crop_top": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "face_crop_bottom": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.1}),
            },
            "optional": {
                "insightface": ("INSIGHTFACE",),
            },
        }

    RETURN_TYPES = ("EMBEDS", "STRING")
    RETURN_NAMES = ("embeds", "info")
    FUNCTION = "execute"
    CATEGORY = "ipadapter/faceid"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def execute(self, path, ipadapter, clip_vision, combine_method,
                bbox_detector, yolo_threshold, face_crop_top, face_crop_bottom,
                insightface=None):
        import comfy.model_management

        path = path.strip()
        if not os.path.isabs(path):
            input_dir = folder_paths.get_input_directory()
            path = os.path.join(input_dir, path)
        if not os.path.isdir(path):
            raise ValueError(f"Directory not found: {path}")

        cache_path = os.path.join(path, CACHE_FILENAME)
        from_cache = False

        if os.path.isfile(cache_path):
            print(f"[Soya:FaceEmbedCacheV2] Loading cached embeds from {cache_path}")
            raw_embeds = torch.load(cache_path).cpu()
            combined = _combine_embeds(raw_embeds, combine_method)
            print(f"[Soya:FaceEmbedCacheV2] Combined ({combine_method}): {raw_embeds.shape[0]} faces → {combined.shape}")
            from_cache = True
            info = f"[CACHE HIT] path: {path}, faces: {raw_embeds.shape[0]}, combine: {combine_method}, shape: {combined.shape}"
            return (combined, info)

        encode_image_masked = _import_encode_image_masked()
        is_plus = _detect_is_plus(ipadapter)

        output_cross_attention_dim = ipadapter["ip_adapter"]["1.to_k_ip.weight"].shape[1]
        is_sdxl = output_cross_attention_dim == 2048

        if insightface is None:
            from .soya_scheduler.model_manager import get_insightface_model
            insightface = get_insightface_model()

        yolo_model = bbox_detector.bbox_model

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

            face_crop = _detect_face_v2(
                img_bgr, insightface, yolo_model,
                yolo_threshold, face_crop_top, face_crop_bottom, is_sdxl,
            )
            face_crops.append(face_crop)

        if not face_crops:
            raise ValueError(f"No faces detected in images from: {path}")

        batch = torch.cat(face_crops, dim=0)

        comfy.model_management.load_model_gpu(clip_vision.patcher)
        encoded = encode_image_masked(clip_vision, batch, batch_size=0)

        if is_plus:
            raw_embeds = encoded.penultimate_hidden_states
        else:
            raw_embeds = encoded.image_embeds

        torch.save(raw_embeds, cache_path)
        print(f"[Soya:FaceEmbedCacheV2] Saved raw embeds to {cache_path}")
        print(f"[Soya:FaceEmbedCacheV2] Faces: {len(face_crops)}, Plus: {is_plus}, SDXL: {is_sdxl}, Raw shape: {raw_embeds.shape}")

        combined = _combine_embeds(raw_embeds, combine_method)
        print(f"[Soya:FaceEmbedCacheV2] Combined ({combine_method}): {combined.shape}")

        info = (
            f"[NEW] path: {path}, "
            f"faces: {len(face_crops)}, plus: {is_plus}, sdxl: {is_sdxl}, "
            f"crop_top: {face_crop_top}, crop_bottom: {face_crop_bottom}, "
            f"combine: {combine_method}, shape: {combined.shape}"
        )
        return (combined, info)
