"""
SoyaProcessCollector2 – face-only extraction pipeline (no eye segmentation).

Same as SoyaProcessCollector but skips eye/eyebrow segmentation.
Outputs face crops as SEGS for SoyaFaceDetailer.
"""

import math
import time
import numpy as np
import torch

from .soya_process_collector import SoyaProcessCollector_mdsoya, SEG
from .soya_scheduler.task_store import pop
from .soya_scheduler.config_manager import load_config, save_config


class SoyaProcessCollector2_mdsoya:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task_ref": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "main_image": ("IMAGE", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE", "STRING", "SEGS", "CONTEXT")
    RETURN_NAMES = ("prompts", "main_image", "info", "segs", "context")
    FUNCTION = "collect"
    CATEGORY = "Soya/Scheduler"

    def collect(self, task_ref, main_image=None):
        collect_start = time.time()

        task = pop(task_ref)
        if task is None:
            info = f"[Soya Process Collector2]\ntask_ref '{task_ref}' not found."
            empty_segs = ((64, 64), [])
            empty_ctx = {}
            return ("", SoyaProcessCollector_mdsoya._empty_image(), info, empty_segs, empty_ctx)

        future = task["future"]
        start_time = task["start_time"]
        divide_end_time = task.get("divide_end_time", start_time)
        preprocess_time = task.get("preprocess_time", 0.0)

        main_process_time = max(0, collect_start - divide_end_time)

        import ray
        result = ray.get(future)
        elapsed = time.time() - start_time

        prompts = result.get("prompts", "")
        timing = result.get("timing", {})
        assignments = result.get("assignments", [])

        # ── Info string ────────────────────────────────────
        lines = ["[Soya Process Collector2]"]
        lines.append(f"Device: {result.get('device_info', 'unknown')}")
        lines.append(f"Preprocess: {preprocess_time:.2f}초")

        sub_total = timing.get("total", 0)
        parallel_time = max(sub_total, main_process_time)
        lines.append(f"Parallel: {parallel_time:.2f}초")

        lines.append(f"감지된 얼굴: {len(assignments)}개")
        lines.append(f"매칭: {', '.join(assignments) if assignments else 'none'}")

        # ── Image ────────────────────────────────────
        img = main_image if main_image is not None else SoyaProcessCollector_mdsoya._empty_image()
        if img.shape[-1] == 4:
            img = img[:, :, :, :3].clone()

        settings = load_config().get("settings", {})

        kept_faces_ray = result.get("kept_faces", [])
        remain_faces_ray = result.get("remain_faces", [])

        segs_items = []
        context_kept = []
        context_remain = []

        postprocess_time = 0.0
        total_upscale_time = 0.0

        if kept_faces_ray and img is not None:
            try:
                segs_items, context_kept, context_remain, post_info, \
                    postprocess_time, total_upscale_time = self._postprocess_faces(
                    img, kept_faces_ray, remain_faces_ray, prompts, settings
                )
                lines.extend(post_info)
            except Exception as e:
                lines.append(f"⚠ Postprocess 오류: {e}")
                import traceback
                traceback.print_exc()

        total_time = preprocess_time + parallel_time + postprocess_time

        H, W = img.shape[1], img.shape[2]
        segs = ((H, W), segs_items)

        batch_groups = SoyaProcessCollector_mdsoya._build_batch_groups(segs_items, context_kept) if segs_items else []

        lines.append(f"Total: {total_time:.2f}초")
        info = "\n".join(lines)

        context = {
            "kept_faces": context_kept,
            "remain_faces": context_remain,
            "segs": segs,
            "batch_groups": batch_groups,
            "crop_mode": settings.get("crop_mode", "preserve"),
            "eyebrow_restore": settings.get("eyebrow_restore", False),
            "eyebrow_restore_mode": settings.get("eyebrow_restore_mode", "hs_preserve"),
            "eyebrow_blur": settings.get("eyebrow_blur", 0),
            "eyebrow_hs_percentile": settings.get("eyebrow_hs_percentile", 0.0),
            "eyebrow_v_range": settings.get("eyebrow_v_range", 1.0),
            "eyebrow_opacity": settings.get("eyebrow_opacity", 0.0),
        }

        # Save to config for web UI
        config = load_config()
        config["last_process_result"] = {
            "timing": {"total": total_time},
            "assignments": assignments,
            "elapsed": elapsed,
            "info_text": info,
        }
        config["last_final_prompts"] = prompts
        save_config(config)

        print(f"[Soya:ProcessCollector2] Task {task_ref} completed in {elapsed:.2f}s")
        return (prompts, img, info, segs, context)

    def _postprocess_faces(self, image_tensor, kept_faces_ray, remain_faces_ray, prompts, settings):
        """Face extraction: crop + blackbox + upscale. No segmentation."""
        info_lines = []
        post_t0 = time.time()

        image_np = image_tensor[0].cpu().numpy()
        image_uint8 = (image_np * 255).clip(0, 255).astype(np.uint8)
        H, W = image_uint8.shape[:2]

        target_size = settings.get("target_size", 512)
        detailer_crop_factor = settings.get("detailer_face_crop_factor", 3.0)
        segs_min_distance = settings.get("segs_min_distance", 0)

        # Load upscale model
        upscale_model = None
        upscale_dev = settings.get("upscale_device", "cuda:1")
        upscale_model_name = settings.get("upscale_model", "")
        if upscale_model_name:
            from .soya_scheduler.model_manager import get_upscale_model
            upscale_model = get_upscale_model(upscale_model_name, upscale_dev)

        # Compute face info
        matched_faces = []
        for f in kept_faces_ray:
            upscale_passes = SoyaProcessCollector_mdsoya._calc_upscale_passes(f["bbox"], target_size)
            bw, bh = f["bbox"][2] - f["bbox"][0], f["bbox"][3] - f["bbox"][1]
            is_large = max(bw, bh) >= target_size
            matched_faces.append({
                "bbox": f["bbox"],
                "assignment": f.get("assignment", "unknown"),
                "similarity": f.get("similarity", 0.0),
                "upscale_passes": upscale_passes,
                "is_large": is_large,
            })

        # Compute common crop dimensions (same as Collector)
        common_final_w = 0
        common_final_h = 0
        for face in matched_faces:
            if not face.get("is_large", False):
                bw = face["bbox"][2] - face["bbox"][0]
                bh = face["bbox"][3] - face["bbox"][1]
                longest = max(bw, bh)
                min_w = int(longest * detailer_crop_factor)
                min_h = int(longest * detailer_crop_factor)
                if segs_min_distance > 0:
                    min_w = max(min_w, bw + 2 * segs_min_distance)
                    min_h = max(min_h, bh + 2 * segs_min_distance)
                common_final_w = max(common_final_w, min_w * (2 ** face["upscale_passes"]))
                common_final_h = max(common_final_h, min_h * (2 ** face["upscale_passes"]))

        if common_final_w > 0 or common_final_h > 0:
            max_pow2 = max(
                (2 ** f["upscale_passes"] for f in matched_faces if not f.get("is_large", False)),
                default=1,
            )
            if common_final_w > 0:
                common_final_w = -(-common_final_w // max_pow2) * max_pow2
            if common_final_h > 0:
                common_final_h = -(-common_final_h // max_pow2) * max_pow2
            max_w_upscaled = (W // max_pow2) * max_pow2
            max_h_upscaled = (H // max_pow2) * max_pow2
            common_final_w = min(common_final_w, max_w_upscaled)
            common_final_h = min(common_final_h, max_h_upscaled)

        for face in matched_faces:
            bw = face["bbox"][2] - face["bbox"][0]
            bh = face["bbox"][3] - face["bbox"][1]
            if face.get("is_large", False):
                longest = max(bw, bh)
                crop_w = int(longest * detailer_crop_factor)
                crop_h = int(longest * detailer_crop_factor)
                if segs_min_distance > 0:
                    crop_w = max(crop_w, bw + 2 * segs_min_distance)
                    crop_h = max(crop_h, bh + 2 * segs_min_distance)
                face["crop_w"] = min(crop_w, W)
                face["crop_h"] = min(crop_h, H)
            else:
                face["crop_w"] = common_final_w // (2 ** face["upscale_passes"])
                face["crop_h"] = common_final_h // (2 ** face["upscale_passes"])

        all_bboxes = [f["bbox"] for f in matched_faces]
        all_bboxes += [f["bbox"] for f in remain_faces_ray]

        labels = SoyaProcessCollector_mdsoya._parse_prompt_labels(prompts)

        segs_items = []
        context_kept = []
        context_remain = []
        total_upscale_time = 0.0
        _collector = SoyaProcessCollector_mdsoya()

        for i, face in enumerate(matched_faces):
            label = labels[i] if i < len(labels) else str(i + 1)
            face["label"] = label
            bbox = face["bbox"]
            upscale_passes = face["upscale_passes"]

            # Crop with blackbox
            cropped_np, crop_region = SoyaProcessCollector_mdsoya._crop_with_blackbox(
                image_uint8, bbox, all_bboxes, face["crop_w"], face["crop_h"]
            )

            # Upscale
            crop_tensor, up_time, _ = _collector._upscale_crop(
                cropped_np, upscale_model, upscale_passes, upscale_dev
            )
            total_upscale_time += up_time

            # Compute padding offsets
            cx1_ideal = int((bbox[0] + bbox[2]) / 2 - face["crop_w"] / 2)
            cy1_ideal = int((bbox[1] + bbox[3]) / 2 - face["crop_h"] / 2)
            pad_left = max(0, -cx1_ideal)
            pad_top = max(0, -cy1_ideal)
            pad_right = max(0, (cx1_ideal + face["crop_w"]) - W)
            pad_bottom = max(0, (cy1_ideal + face["crop_h"]) - H)

            # Face mask: full ones (no eye segmentation)
            enh_h, enh_w = crop_tensor.shape[1], crop_tensor.shape[2]
            face_mask = np.ones((enh_h, enh_w), dtype=np.uint8)

            seg = SEG(
                cropped_image=crop_tensor,
                cropped_mask=face_mask,
                confidence=1.0,
                crop_region=tuple(crop_region),
                bbox=tuple(bbox),
                label=label,
            )
            segs_items.append(seg)

            context_kept.append({
                "image": crop_tensor,
                "upscale_passes": upscale_passes,
                "is_large": face.get("is_large", False),
                "original_bbox": bbox,
                "label": label,
                "assignment": face.get("assignment", "unknown"),
                "crop_pad_left": pad_left,
                "crop_pad_top": pad_top,
                "crop_pad_right": pad_right,
                "crop_pad_bottom": pad_bottom,
                "crop_x1_raw": cx1_ideal,
                "crop_y1_raw": cy1_ideal,
                "crop_w": face["crop_w"],
                "crop_h": face["crop_h"],
            })

            info_lines.append(
                f"  Face {i} ({label}): bbox={bbox} crop={crop_region} "
                f"upscale={upscale_passes}x size={crop_tensor.shape[2]}x{crop_tensor.shape[1]}"
            )

        # Remain faces context
        for rf in remain_faces_ray:
            context_remain.append({
                "image": None,
                "original_bbox": rf["bbox"],
            })

        post_elapsed = time.time() - post_t0
        info_lines.append(f"Face extraction: {post_elapsed:.2f}초 (upscale: {total_upscale_time:.2f}초)")

        return segs_items, context_kept, context_remain, info_lines, post_elapsed, total_upscale_time
