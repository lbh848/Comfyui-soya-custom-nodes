"""
SoyaEyeStateDetector – Detects eye open/closed state per character.

Takes face_context from Face Match and the original image, crops each face
using bbox from context, runs ISNet eye/eyebrow segmentation, and computes
overlap ratio (eyebrow coverage over eye).

Outputs:
  - overlap_ratios: per-face overlap ratio (for Eye Tag Override)
  - info: human-readable debug text
  - eye_context: per-face eye/eyebrow masks + metadata (for Eye Detailer)
"""

import numpy as np
import torch
import time


class SoyaEyeStateDetector_mdsoya:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "face_context": ("IPA_FACE_CONTEXT",),
                "image": ("IMAGE",),
                "eye_model": ("SOYA_SEG_MODEL",),
                "eyebrow_model": ("SOYA_SEG_MODEL",),
                "eye_seg_th": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                }),
                "eyebrow_th": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                }),
            },
        }

    RETURN_TYPES = ("FLOAT", "STRING", "EYE_CONTEXT")
    RETURN_NAMES = ("overlap_ratios", "info", "eye_context")
    OUTPUT_IS_LIST = (True, False, False)
    FUNCTION = "detect"
    CATEGORY = "Soya/FaceMatch"

    def detect(self, face_context, image, eye_model, eyebrow_model,
               eye_seg_th, eyebrow_th):
        from .soya_scheduler.model_manager import eye_seg_segment

        matches = face_context.get("matches", [])
        if not matches:
            return ([], "No faces in face_context.", {"faces": []})

        # image: [B, H, W, 3] float32 [0,1] → use first frame
        img_np = (image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        img_H, img_W = img_np.shape[:2]

        eye_net = eye_model.get("model") if eye_model else None
        eyebrow_net = eyebrow_model.get("model") if eyebrow_model else None
        seg_dev = (eye_model or eyebrow_model).get("device", "cpu")

        if eye_net is None:
            ratios = [0.0] * len(matches)
            info_lines = [
                f"Eye model is None – returning 0.0 for {len(matches)} face(s)."
            ]
            faces = []
            for m in matches:
                info_lines.append(f"  {m['name']}: ratio=0.0000 (no eye model)")
                faces.append(self._empty_face(m["name"]))
            return (ratios, "\n".join(info_lines), {"faces": faces})

        ratios = []
        info_lines = []
        faces = []
        t0 = time.time()

        for match in matches:
            name = match["name"]
            bx1, by1, bx2, by2 = match["bbox"]

            # Clamp bbox to image bounds
            x1 = max(0, int(bx1))
            y1 = max(0, int(by1))
            x2 = min(img_W, int(bx2))
            y2 = min(img_H, int(by2))

            face_data = {
                "name": name,
                "bbox_clamped": (x1, y1, x2, y2),
                "eye_mask": None,
                "eyebrow_mask": None,
                "overlap_ratio": 0.0,
            }

            if x2 <= x1 or y2 <= y1:
                ratios.append(0.0)
                info_lines.append(f"  {name}: ratio=0.0000 (invalid bbox)")
                print(f"[EyeState] {name}: invalid bbox ({x1},{y1},{x2},{y2})")
                faces.append(face_data)
                continue

            face_np = img_np[y1:y2, x1:x2].copy()

            # ── Eye segmentation ──
            eye_mask_float = eye_seg_segment(eye_net, face_np, seg_dev)
            eye_binary = (eye_mask_float > eye_seg_th).astype(np.uint8)

            eye_area = float(eye_binary.sum())
            if eye_area == 0:
                ratios.append(0.0)
                info_lines.append(f"  {name}: ratio=0.0000 (no eye detected)")
                print(f"[EyeState] {name}: no eye pixels detected")
                face_data["eye_mask"] = eye_binary
                faces.append(face_data)
                continue

            # ── Eyebrow segmentation ──
            eb_mask_float = np.zeros(face_np.shape[:2], dtype=np.float32)
            if eyebrow_net is not None:
                eb_mask_float = self._run_eyebrow_segmentation(
                    face_np, eye_binary, eyebrow_net, seg_dev
                )

            eb_binary = (eb_mask_float > eyebrow_th).astype(np.uint8)

            # ── Overlap ratio: how much eyebrow covers eye ──
            overlap = (eye_binary & eb_binary).astype(np.float32)
            ratio = float(overlap.sum()) / eye_area
            ratios.append(ratio)

            face_data["eye_mask"] = eye_binary
            face_data["eyebrow_mask"] = eb_mask_float
            face_data["overlap_ratio"] = ratio

            info_lines.append(
                f"  {name}: ratio={ratio:.4f} "
                f"(eye={int(eye_area)}, overlap={int(overlap.sum())})"
            )
            faces.append(face_data)

        elapsed = time.time() - t0
        header = f"Analyzed {len(ratios)} face(s) in {elapsed:.2f}s"
        info_lines.insert(0, header)
        info = "\n".join(info_lines)
        print(f"[EyeState] {header}")

        eye_context = {"faces": faces}
        return (ratios, info, eye_context)

    @staticmethod
    def _empty_face(name):
        return {
            "name": name,
            "bbox_clamped": None,
            "eye_mask": None,
            "eyebrow_mask": None,
            "overlap_ratio": 0.0,
        }

    @staticmethod
    def _run_eyebrow_segmentation(image_np, eye_mask, eyebrow_model, device):
        """Run eyebrow ISNet on the eye mask region, map back to full size."""
        from .soya_scheduler.model_manager import eyebrow_segment

        ys, xs = np.where(eye_mask > 0)
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
