"""
SoyaProcessDivider – preprocesses face detection, dispatches CLIP matching to Ray worker,
and returns task_ref + circular face mask.
"""

import time
import uuid
import numpy as np
import torch

from .soya_scheduler import ensure_ray_initialized
from .soya_scheduler.config_manager import load_config, load_characters, load_reference_image
from .soya_scheduler.task_store import put


class SoyaProcessDivider_mdsoya:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "MASK")
    RETURN_NAMES = ("image", "task_ref", "mask")
    FUNCTION = "divide"
    CATEGORY = "Soya/Scheduler"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def divide(self, image, prompt):
        ensure_ray_initialized()
        config = load_config()

        # Load characters from base_path
        settings = config.get("settings", {})
        base_path = settings.get("base_path", "")

        from .soya_scheduler.ray_worker import _filter_characters_by_prompt, _detect_faces_yolo
        all_chars = load_characters(base_path)
        matched_chars = _filter_characters_by_prompt(all_chars, prompt)
        config["characters"] = matched_chars

        # Resolve model paths before dispatching to Ray worker
        try:
            import folder_paths
            bbox_model = settings.get("bbox_model", "face_yolov8m.pt")
            resolved_bbox = folder_paths.get_full_path("ultralytics_bbox", bbox_model)
            if resolved_bbox:
                settings["_resolved_bbox_path"] = resolved_bbox

            clip_model = settings.get("clip_vision_model", "clip_vision_vit_h.safetensors")
            resolved_clip = folder_paths.get_full_path("clip_vision", clip_model)
            if resolved_clip:
                settings["_resolved_clip_path"] = resolved_clip

            upscale_model = settings.get("upscale_model", "")
            if upscale_model:
                resolved_upscale = folder_paths.get_full_path("upscale_models", upscale_model)
                if resolved_upscale:
                    settings["_resolved_upscale_path"] = resolved_upscale
        except Exception:
            pass
        config["settings"] = settings

        # ── Preprocess: Upscale ────────────────────────────────────
        preprocess_upscale_time = 0.0
        preprocess_vram_peak = 0
        preprocess_upscale = settings.get("preprocess_upscale", False)
        if preprocess_upscale and settings.get("upscale_model"):
            from .soya_scheduler.model_manager import get_upscale_model, upscale_image
            upscale_dev = settings.get("upscale_device", "cuda:1")
            if upscale_dev.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(upscale_dev)
                vram_before = torch.cuda.memory_allocated(upscale_dev)
            _t0 = time.time()
            model = get_upscale_model(settings["upscale_model"], upscale_dev)
            image = upscale_image(model, image, upscale_dev)
            preprocess_upscale_time = time.time() - _t0
            if upscale_dev.startswith("cuda"):
                torch.cuda.synchronize(upscale_dev)
                preprocess_vram_peak = torch.cuda.max_memory_allocated(upscale_dev) - vram_before

        # ── Preprocess: Face Detection ─────────────────────────────
        preprocess_face_detect_time = 0.0

        # Convert image for YOLO
        image_numpy = image.cpu().numpy()
        if image_numpy.ndim == 4:
            img_single = image_numpy[0]
        else:
            img_single = image_numpy
        if img_single.dtype != np.uint8:
            img_single = (img_single * 255).clip(0, 255).astype(np.uint8)

        H, W = img_single.shape[:2]

        from .soya_scheduler.model_manager import get_yolo_model
        bbox_device = settings.get("bbox_device", settings.get("device", "cuda:1"))
        bbox_model_path = settings.get("_resolved_bbox_path", settings.get("bbox_model", "face_yolov8m.pt"))
        threshold = settings.get("bbox_threshold", 0.5)
        crop_factor = settings.get("face_crop_factor", 3.0)
        max_faces = settings.get("tracking_face_count", settings.get("max_faces", 3))

        _t0 = time.time()
        yolo = get_yolo_model(bbox_model_path, bbox_device)
        detected_faces = _detect_faces_yolo(img_single, yolo, threshold, crop_factor, max_faces)
        preprocess_face_detect_time = time.time() - _t0

        # ── Preprocess: Reference image face extraction ────────────
        ref_crop_factor = settings.get("ref_face_crop_factor", crop_factor)
        ref_data = {}
        for ch in matched_chars:
            image_file = ch.get("image_file", "")
            if image_file:
                ref_img = load_reference_image(base_path, image_file)
                if ref_img is not None:
                    ref_faces = _detect_faces_yolo(ref_img, yolo, threshold, ref_crop_factor, max_faces=1)
                    if ref_faces:
                        ref_data[ch["name"]] = ref_faces[0]["crop"]
                    else:
                        ref_data[ch["name"]] = ref_img
                else:
                    print(f"[Soya:ProcessDivider] WARNING: reference image failed to load for '{ch['name']}': {image_file}")
            else:
                print(f"[Soya:ProcessDivider] WARNING: no reference image configured for '{ch['name']}'")

        # ── Create circular mask from face bboxes ──────────────────
        mask_sigma_factor = settings.get("mask_sigma_factor", 0.0)
        mask = torch.zeros(1, H, W, dtype=torch.float32)
        for face in detected_faces:
            x1, y1, x2, y2 = face["bbox"]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            radius = max(x2 - x1, y2 - y1) / 2

            yy, xx = np.ogrid[:H, :W]
            dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2

            if mask_sigma_factor > 0:
                sigma = max(radius * mask_sigma_factor, 1e-6)
                circle = np.exp(-0.5 * dist_sq / (sigma ** 2)).astype(np.float32)
            else:
                circle = (dist_sq <= radius ** 2).astype(np.float32)

            mask[0] = torch.maximum(mask[0], torch.from_numpy(circle))

        # ── Prepare face data for Ray worker ───────────────────────
        face_data = []
        for f in detected_faces:
            face_data.append({
                "bbox": f["bbox"],
                "crop_region": f["crop_region"],
                "crop": f["crop"],
                "confidence": f["confidence"],
                "area": (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]),
            })

        task_id = str(uuid.uuid4())[:8]

        from .soya_scheduler import get_analyzer
        analyzer = get_analyzer()
        future = analyzer.analyze.remote(task_id, face_data, ref_data, config)

        preprocess_time = preprocess_upscale_time + preprocess_face_detect_time

        put(task_id, {
            "future": future,
            "start_time": time.time(),
            "divide_end_time": time.time(),
            "preprocess_time": preprocess_time,
            "preprocess_upscale_time": preprocess_upscale_time,
            "preprocess_face_detect_time": preprocess_face_detect_time,
            "preprocess_vram_peak": preprocess_vram_peak,
        })
        print(f"[Soya:ProcessDivider] Dispatched task {task_id}")
        return (image, task_id, mask)
