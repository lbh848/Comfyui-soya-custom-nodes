"""
SoyaFacePasteback – Paste enhanced faces back into the original image.

Uses FACE_CONTEXT from SoyaFaceDetailer to paste 2x-downscaled enhanced faces
with Voronoi overlap resolution.
"""

import torch
import numpy as np
from PIL import Image as PILImage

from .soya_batch_detailer import SoyaBatchDetailer_mdsoya


class SoyaFacePasteback_mdsoya:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "face_context": ("FACE_CONTEXT", {"forceInput": True}),
                "context": ("CONTEXT", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "paste_back"
    CATEGORY = "Soya/FaceDetailer"

    def paste_back(self, image, face_context, context):
        import time
        t0 = time.time()
        info_lines = ["[Soya Face Paste-back]"]

        faces = face_context.get("faces", [])
        remain_faces = face_context.get("remain_faces", [])

        if not faces:
            info_lines.append("No enhanced faces to paste back.")
            return (image, "\n".join(info_lines))

        img_h, img_w = image.shape[1], image.shape[2]
        result = image.clone()

        # Collect all bboxes
        kept_bboxes = [f["original_bbox"] for f in faces]
        remain_bboxes = [rf.get("original_bbox", []) for rf in remain_faces if rf.get("original_bbox")]
        all_bboxes = kept_bboxes + remain_bboxes

        if not all_bboxes:
            return (image, "\n".join(info_lines))

        centers = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in all_bboxes]

        # Step 1: Create individual ellipse masks per face
        individual_masks = []
        for bbox in all_bboxes:
            mask = np.zeros((img_h, img_w), dtype=np.float32)
            SoyaBatchDetailer_mdsoya._draw_ellipse_on_mask(mask, bbox, value=1.0)
            individual_masks.append(mask)

        # Step 2: Resolve overlaps with Voronoi
        overlap_count = np.zeros((img_h, img_w), dtype=np.int32)
        for m in individual_masks:
            overlap_count += (m > 0).astype(np.int32)

        overlap_pixels = overlap_count > 1
        if np.any(overlap_pixels):
            ys, xs = np.where(overlap_pixels)
            points = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
            centers_arr = np.array(centers, dtype=np.float64)

            dists = ((points[:, None, :] - centers_arr[None, :, :]) ** 2).sum(axis=2)
            nearest = np.argmin(dists, axis=1)

            for m in individual_masks:
                m[ys, xs] = 0

            for idx in range(len(all_bboxes)):
                sel = nearest == idx
                if np.any(sel):
                    individual_masks[idx][ys[sel], xs[sel]] = 1.0

        # Step 3: Punch holes
        hole_mask = np.maximum.reduce(individual_masks)
        hole_t = torch.from_numpy(hole_mask).unsqueeze(2)  # (H, W, 1)
        result[0] = result[0] * (1 - hole_t)

        # Step 4: Fill enhanced faces
        # Collector2 uses _compute_crop_region which SHIFTS crops (not pads them).
        # The crop_region is the actual clamped position — paste directly there.
        for fi, face in enumerate(faces):
            enhanced = face["enhanced_image"]  # (1, H_up, W_up, 3)
            crop_region = face["crop_region"]
            uf = face["upscale_factor"]
            cr_x1, cr_y1, cr_x2, cr_y2 = crop_region
            crop_w = cr_x2 - cr_x1
            crop_h = cr_y2 - cr_y1

            if crop_w <= 0 or crop_h <= 0:
                continue

            # Resize by exact upscale factor to avoid staircase artifacts.
            # uf is always 2^upscale_passes (1, 2, or 4), so enh_w // uf is exact.
            enh_h, enh_w = enhanced.shape[1], enhanced.shape[2]
            target_w = enh_w // uf
            target_h = enh_h // uf

            _enh_np = (enhanced[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            _enh_pil = PILImage.fromarray(_enh_np).resize((target_w, target_h), PILImage.LANCZOS)
            resized = torch.from_numpy(np.array(_enh_pil).astype(np.float32) / 255.0)

            # Clamp paste region to image bounds
            paste_w = min(target_w, img_w - cr_x1)
            paste_h = min(target_h, img_h - cr_y1)
            if paste_w <= 0 or paste_h <= 0:
                continue

            # Blend at crop_region position using Voronoi-resolved ellipse mask
            face_mask = individual_masks[fi][cr_y1:cr_y1 + paste_h, cr_x1:cr_x1 + paste_w]
            fill_t = torch.from_numpy(face_mask.astype(np.float32)).unsqueeze(2)
            resized_clamped = resized[:paste_h, :paste_w]
            region = result[0, cr_y1:cr_y1 + paste_h, cr_x1:cr_x1 + paste_w, :]
            result[0, cr_y1:cr_y1 + paste_h, cr_x1:cr_x1 + paste_w, :] = (
                fill_t * resized_clamped + (1 - fill_t) * region
            )

            info_lines.append(
                f"  Face {fi} ({face.get('label', '')}): "
                f"crop=({cr_x1},{cr_y1},{cr_x2},{cr_y2}) "
                f"enhanced=({enh_w},{enh_h})→({target_w},{target_h}) "
                f"paste=({cr_x1},{cr_y1},{cr_x1 + paste_w},{cr_y1 + paste_h})"
            )

        # Step 5: Fill remain faces with original pixels
        for j, bbox in enumerate(remain_bboxes):
            face_idx = len(faces) + j
            face_mask = individual_masks[face_idx]

            active = face_mask > 0
            if not np.any(active):
                continue
            rows, cols = np.where(active)
            ry1, ry2 = rows.min(), rows.max() + 1
            rx1, rx2 = cols.min(), cols.max() + 1

            local_mask = face_mask[ry1:ry2, rx1:rx2]
            mask_t = torch.from_numpy(local_mask).unsqueeze(2)
            result[0, ry1:ry2, rx1:rx2] = (
                mask_t * image[0, ry1:ry2, rx1:rx2]
                + (1 - mask_t) * result[0, ry1:ry2, rx1:rx2]
            )

        elapsed = time.time() - t0
        info_lines.append(f"Paste-back completed in {elapsed:.2f}초")

        return (result, "\n".join(info_lines))
