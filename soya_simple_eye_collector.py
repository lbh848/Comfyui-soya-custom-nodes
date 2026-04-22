"""
SoyaSimpleEyeCollector – Standalone eye detailer collector without Ray dependency.

Uses YOLO (Ultralytics) for face detection and ISNet for eye/eyebrow segmentation.
Outputs are compatible with Soya Batch Detailer.

Pipeline:
  1. YOLO detects face bbox -> expand by crop_factor (default 2x)
  2. ISNet eye segmentation on expanded crop
  3. ISNet eyebrow segmentation on eye region
  4. Subtract eyebrow from eye -> pure eye mask
  5. Build SEGS + context for Soya Batch Detailer

When enabled="false", returns empty segs so Batch Detailer passes through the image.
"""

import time
import numpy as np
import torch
from collections import namedtuple

# Impact Pack SEG namedtuple compatibility
SEG = namedtuple("SEG", [
    'cropped_image', 'cropped_mask', 'confidence',
    'crop_region', 'bbox', 'label', 'control_net_wrapper',
], defaults=[None])


class SoyaSimpleEyeCollector_mdsoya:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "bbox_detector": ("BBOX_DETECTOR",),
                "positive_prompt": ("STRING", {"multiline": True}),
                "enabled": ("STRING", {"default": "true", "multiline": False}),
                "yolo_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "crop_factor": ("FLOAT", {"default": 1.3, "min": 1.0, "max": 5.0, "step": 0.1}),
                "eye_seg_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "eyebrow_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "eye_model": ("SOYA_SEG_MODEL", {"forceInput": True}),
                "eyebrow_model": ("SOYA_SEG_MODEL", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE", "STRING", "SEGS", "CONTEXT")
    RETURN_NAMES = ("prompts", "main_image", "info", "segs", "context")
    FUNCTION = "execute"
    CATEGORY = "Soya/FaceDetailer"

    def execute(self, image, bbox_detector, positive_prompt, enabled,
                yolo_threshold, crop_factor, eye_seg_threshold, eyebrow_threshold,
                eye_model=None, eyebrow_model=None):
        t0 = time.time()
        info_lines = ["[Soya Simple Eye Collector]"]

        H, W = image.shape[1], image.shape[2]
        image_np = (image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

        # Format prompt for Batch Detailer: [LAB]\n[1] <prompt>
        prompts = f"[LAB]\n[1] {positive_prompt}"

        # ── Helper: empty result ──
        def _empty(msg=""):
            if msg:
                info_lines.append(msg)
            empty_segs = ((H, W), [])
            empty_ctx = {
                "kept_faces": [], "remain_faces": [],
                "segs": empty_segs, "batch_groups": [],
                "crop_mode": "preserve",
                "eyebrow_restore": False, "eyebrow_restore_mode": "hs_preserve",
                "eyebrow_blur": 0, "eyebrow_hs_percentile": 0.0,
                "eyebrow_v_range": 1.0, "eyebrow_opacity": 0.0,
            }
            return (prompts, image, "\n".join(info_lines), empty_segs, empty_ctx)

        # ── Toggle: disabled -> empty segs ──
        if enabled.lower().strip() != "true":
            return _empty("Disabled (enabled='false'). Returning empty segs.")

        # ── Extract models from provider inputs ──
        seg_dev = "cuda:0"
        eye_net = None
        eyebrow_net = None

        if eye_model is not None:
            eye_net = eye_model.get("model")
            seg_dev = eye_model.get("device", seg_dev)
        if eyebrow_model is not None:
            eyebrow_net = eyebrow_model.get("model")

        if eye_net is None:
            info_lines.append("Warning: No eye model provided, skipping eye segmentation.")

        # ── Step 1: YOLO face detection ──
        yolo = bbox_detector.bbox_model
        results = yolo(image_np, verbose=False)

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
                    best_bbox = (int(x1), int(y1), int(x2), int(y2))

        if best_bbox is None:
            return _empty("No face detected by YOLO.")

        info_lines.append(f"Face bbox: {best_bbox}")

        # ── Expand bbox by crop_factor ──
        fx1, fy1, fx2, fy2 = best_bbox
        bw, bh = fx2 - fx1, fy2 - fy1
        cx, cy = (fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0
        ew, eh = bw * crop_factor, bh * crop_factor

        cx1 = max(0, int(cx - ew / 2))
        cy1 = max(0, int(cy - eh / 2))
        cx2 = min(W, int(cx + ew / 2))
        cy2 = min(H, int(cy + eh / 2))

        crop_region = (cx1, cy1, cx2, cy2)
        info_lines.append(f"Crop region (x{crop_factor}): {crop_region}")

        # Crop image
        crop_np = image_np[cy1:cy2, cx1:cx2].copy()

        # ── Step 1b: Upscale crop 2x (face becomes 2x larger) ──
        from PIL import Image as PILImage
        ch, cw = crop_np.shape[:2]
        crop_pil = PILImage.fromarray(crop_np)
        crop_pil = crop_pil.resize((cw * 2, ch * 2), PILImage.LANCZOS)
        crop_np = np.array(crop_pil)
        info_lines.append(f"Crop {cw}x{ch} -> upscaled 2x -> {crop_np.shape[1]}x{crop_np.shape[0]}")

        # ── Step 2: ISNet eye segmentation ──
        mask = np.zeros(crop_np.shape[:2], dtype=np.uint8)

        if eye_net is not None:
            from .soya_scheduler.model_manager import eye_seg_segment
            t_seg = time.time()
            eye_mask = eye_seg_segment(eye_net, crop_np, seg_dev)
            mask = (eye_mask > eye_seg_threshold).astype(np.uint8)
            info_lines.append(f"Eye seg: {time.time() - t_seg:.2f}s, pixels: {int(mask.sum())}")

        # ── Steps 3 & 4: Eyebrow segmentation + subtraction ──
        eyebrow_mask_float = None
        if eyebrow_net is not None and mask.max() > 0:
            t_eb = time.time()
            eyebrow_mask_float = self._run_eyebrow_segmentation(
                crop_np, mask, eyebrow_net, seg_dev
            )
            info_lines.append(f"Eyebrow seg: {time.time() - t_eb:.2f}s")

            eb_binary = (eyebrow_mask_float > eyebrow_threshold).astype(np.uint8)
            mask = mask * (1 - eb_binary)
            info_lines.append(f"After eyebrow subtract: pixels: {int(mask.sum())}")

        # ── Build SEGS + context ──
        crop_tensor = torch.from_numpy(crop_np.astype(np.float32) / 255.0).unsqueeze(0)

        seg = SEG(
            cropped_image=crop_tensor,
            cropped_mask=mask,
            confidence=1.0,
            crop_region=crop_region,
            bbox=best_bbox,
            label="1",
        )

        segs = ((H, W), [seg])

        # upscale_passes=1 → uf=2 → paste-back downscales enhanced 2x back to original.
        # No zero-padding: crop is clamped to image bounds, so all pads are 0.
        context = {
            "kept_faces": [{
                "image": crop_tensor,
                "upscale_passes": 1,
                "is_large": False,
                "original_bbox": best_bbox,
                "label": "1",
                "eyebrow_mask": eyebrow_mask_float,
                "eyebrow_threshold": eyebrow_threshold,
                "crop_pad_left": 0,
                "crop_pad_top": 0,
                "crop_pad_right": 0,
                "crop_pad_bottom": 0,
                "crop_x1_raw": crop_region[0],
                "crop_y1_raw": crop_region[1],
            }],
            "remain_faces": [],
            "segs": segs,
            "batch_groups": [[0]],
            "crop_mode": "preserve",
            "eyebrow_restore": False,
            "eyebrow_restore_mode": "hs_preserve",
            "eyebrow_blur": 0,
            "eyebrow_hs_percentile": 0.0,
            "eyebrow_v_range": 1.0,
            "eyebrow_opacity": 0.0,
        }

        elapsed = time.time() - t0
        info_lines.append(f"Total: {elapsed:.2f}s")
        return (prompts, image, "\n".join(info_lines), segs, context)

    @staticmethod
    def _run_eyebrow_segmentation(image_np, seg_mask, eyebrow_model, device):
        """Run eyebrow ISNet on the eye mask region, map back to full size."""
        from .soya_scheduler.model_manager import eyebrow_segment

        ys, xs = np.where(seg_mask > 0)
        if len(xs) == 0:
            return np.zeros(image_np.shape[:2], dtype=np.float32)

        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1

        margin = max(10, int(max(x2 - x1, y2 - y1) * 0.1))
        x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
        x2 = min(image_np.shape[1], x2 + margin)
        y2 = min(image_np.shape[0], y2 + margin)

        eye_crop = image_np[y1:y2, x1:x2]
        eb_crop = eyebrow_segment(eyebrow_model, eye_crop, device, img_size=384)

        full_mask = np.zeros(image_np.shape[:2], dtype=np.float32)
        full_mask[y1:y2, x1:x2] = eb_crop
        return full_mask
