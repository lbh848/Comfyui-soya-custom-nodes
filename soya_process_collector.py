"""
SoyaProcessCollector – awaits Ray worker result, runs postprocess pipeline,
and returns prompts + image + info + SEGS + context.
"""

import time
import math
import os
import datetime
import numpy as np
import torch
from PIL import Image
from collections import namedtuple

from .soya_scheduler.task_store import pop
from .soya_scheduler.config_manager import load_config, save_config

# Impact Pack SEG namedtuple compatibility
SEG = namedtuple("SEG", [
    'cropped_image', 'cropped_mask', 'confidence',
    'crop_region', 'bbox', 'label', 'control_net_wrapper',
], defaults=[None])


class SoyaProcessCollector_mdsoya:
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
            info = f"[Soya Scheduler Info]\ntask_ref '{task_ref}' not found."
            empty_segs = ((64, 64), [])
            empty_ctx = {}
            return ("", self._empty_image(), info, empty_segs, empty_ctx)

        future = task["future"]
        start_time = task["start_time"]
        divide_end_time = task.get("divide_end_time", start_time)
        preprocess_time = task.get("preprocess_time", 0.0)
        preprocess_upscale_time = task.get("preprocess_upscale_time", 0.0)
        preprocess_face_detect_time = task.get("preprocess_face_detect_time", 0.0)
        preprocess_vram_peak = task.get("preprocess_vram_peak", 0)

        # Main process time = Divider return → Collector start
        main_process_time = max(0, collect_start - divide_end_time)

        import ray
        result = ray.get(future)
        elapsed = time.time() - start_time

        prompts = result.get("prompts", "")
        timing = result.get("timing", {})
        assignments = result.get("assignments", [])

        # ── Timing breakdown ────────────────────────────────────
        sub_total = timing.get("total", 0)
        parallel_time = max(sub_total, main_process_time)

        # ── Basic info string ────────────────────────────────────
        lines = ["[Soya Scheduler Info]"]
        lines.append(f"Device: {result.get('device_info', 'unknown')}")

        # Preprocess
        lines.append(f"Preprocess: {preprocess_time:.2f}\ucd08")
        lines.append(f"  Upscale: {preprocess_upscale_time:.2f}\ucd08")
        lines.append(f"  Face Detection: {preprocess_face_detect_time:.2f}\ucd08")

        # Parallel block
        lines.append(f"Parallel: {parallel_time:.2f}\ucd08")
        lines.append(f"  Subprocess: {sub_total:.2f}\ucd08")
        if timing:
            for step, dur in timing.items():
                if step != "total":
                    lines.append(f"    {step}: {dur:.2f}\ucd08")
        lines.append(f"  Main process: {main_process_time:.2f}\ucd08")

        lines.append(f"\uac10\uc9c0\ub41c \uc5bc\uad74: {len(assignments)}\uac1c")
        lines.append(f"\ub9e4\uce6d: {', '.join(assignments) if assignments else 'none'}")

        # ── Postprocess pipeline ─────────────────────────────────
        img = main_image if main_image is not None else self._empty_image()
        # Ensure RGB (3 channels) – strip alpha if present (e.g. GLSL Shader outputs RGBA)
        if img.shape[-1] == 4:
            img = img[:, :, :, :3].clone()
        settings = load_config().get("settings", {})

        kept_faces_ray = result.get("kept_faces", [])
        remain_faces_ray = result.get("remain_faces", [])
        keep_face_count = result.get("keep_face_count", 0)

        segs_items = []
        context_kept = []
        context_remain = []
        before_post = []
        after_post = []

        postprocess_time = 0.0
        post_upscale_time = 0.0
        post_segs_time = 0.0
        post_upscale_vram = 0
        post_segs_vram = 0
        post_eyebrow_time = 0.0
        post_eyebrow_vram = 0

        if kept_faces_ray and img is not None:
            try:
                segs_items, context_kept, context_remain, post_info, before_post, after_post, \
                    postprocess_time, post_upscale_time, post_segs_time, post_upscale_vram, post_segs_vram, \
                    post_eyebrow_time, post_eyebrow_vram = self._postprocess(
                    img, kept_faces_ray, remain_faces_ray, prompts, settings
                )
                lines.extend(post_info)
                # Save face data if toggle is enabled
                self._save_face_data(before_post, settings)
            except Exception as e:
                lines.append(f"\u26a0 Postprocess \uc624\ub958: {e}")
                import traceback
                traceback.print_exc()

        total_time = preprocess_time + parallel_time + postprocess_time

        # Build SEGS output
        H, W = img.shape[1], img.shape[2]
        segs = ((H, W), segs_items)

        # Build batch groups
        batch_groups = self._build_batch_groups(segs_items, context_kept) if segs_items else []

        # Add batch grouping info to info lines
        if batch_groups:
            lines.append("\ubc30\uce58 \uadf8\ub8f9\ud551:")
            for gi, group in enumerate(batch_groups):
                mode = "\ubc30\uce58" if len(group) > 1 else "\uac1c\ubcc4"
                segs_info_parts = []
                for idx in group:
                    s = segs_items[idx]
                    up = context_kept[idx].get("upscale_passes", "?")
                    h, w = s.cropped_image.shape[1], s.cropped_image.shape[2]
                    segs_info_parts.append(f'SEGS #{idx} (label="{s.label}", Upscale {up}x, {w}\u00d7{h})')
                lines.append(f"  Group {gi + 1}: {mode} - {' + '.join(segs_info_parts)}")

        lines.append(f"Total: {total_time:.2f}\ucd08")
        info = "\n".join(lines)

        # Build CONTEXT output
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

        # Save to config (full debug info)
        config = load_config()
        # Build structured subprocess steps (exclude 'total')
        subprocess_steps = {k: v for k, v in timing.items() if k != "total"}

        config["last_process_result"] = {
            "timing": {
                "preprocess": preprocess_time,
                "preprocess_upscale": preprocess_upscale_time,
                "preprocess_face_detect": preprocess_face_detect_time,
                "preprocess_vram": preprocess_vram_peak / (1024**3),
                "parallel": parallel_time,
                "subprocess": sub_total,
                "subprocess_steps": subprocess_steps,
                "main_process": main_process_time,
                "postprocess": postprocess_time,
                "postprocess_upscale": post_upscale_time,
                "postprocess_segs": post_segs_time,
                "postprocess_upscale_vram": post_upscale_vram / (1024**3),
                "postprocess_segs_vram": post_segs_vram / (1024**3),
                "postprocess_eyebrow": post_eyebrow_time,
                "postprocess_eyebrow_vram": post_eyebrow_vram / (1024**3),
                "total": total_time,
            },
            "assignments": assignments,
            "similarities": result.get("similarities", []),
            "face_crops": result.get("face_crops", []),
            "ref_crops_map": result.get("ref_crops_map", {}),
            "elapsed": elapsed,
            # New debug fields
            "all_detected_faces": result.get("all_detected_faces", []),
            "tracked_faces": result.get("tracked_faces", []),
            "kept_faces": result.get("kept_faces", []),
            "remain_faces": result.get("remain_faces", []),
            "keep_face_count": result.get("keep_face_count", 0),
            "info_text": info,
            # Postprocess results (serializable subset)
            "postprocess": {
                "segs_count": len(segs_items),
                "segs_labels": [s.label for s in segs_items],
                "segs_upscale_passes": [f.get("upscale_passes") for f in context_kept],
                "segs_bboxes": [list(s.bbox) for s in segs_items],
                "segs_crop_regions": [list(s.crop_region) for s in segs_items],
                "remain_bboxes": [f["original_bbox"] for f in context_remain],
                "before": before_post,
                "after": after_post,
                "batch_groups": [
                    {
                        "indices": group,
                        "type": "\ubc30\uce58" if len(group) > 1 else "\uac1c\ubcc4",
                        "segs": [
                            {
                                "index": idx,
                                "label": segs_items[idx].label,
                                "upscale_passes": context_kept[idx].get("upscale_passes"),
                                "crop_h": int(segs_items[idx].cropped_image.shape[1]),
                                "crop_w": int(segs_items[idx].cropped_image.shape[2]),
                            }
                            for idx in group
                        ],
                    }
                    for group in batch_groups
                ],
            },
        }
        config["last_final_prompts"] = prompts
        save_config(config)

        print(f"[Soya:ProcessCollector] Task {task_ref} completed in {elapsed:.2f}s")
        return (prompts, img, info, segs, context)

    def _postprocess(self, image_tensor, kept_faces_ray, remain_faces_ray, prompts, settings):
        """Run the full postprocess pipeline on the main image.

        Uses bboxes directly from Ray worker result – no YOLO redetection needed.

        Returns (segs_items, context_kept, context_remain, info_lines,
                 before_postprocess, after_postprocess)
        """
        info_lines = []
        post_t0 = time.time()

        # Convert image tensor to numpy
        image_np = image_tensor[0].cpu().numpy()  # (H, W, 3) float32 [0,1]
        image_uint8 = (image_np * 255).clip(0, 255).astype(np.uint8)
        H, W = image_uint8.shape[:2]

        # Use Ray bboxes directly – no redetection
        target_size = settings.get("target_size", 512)
        detailer_crop_factor = settings.get("detailer_face_crop_factor", 3.0)
        segs_min_distance = settings.get("segs_min_distance", 0)
        crop_mode = settings.get("crop_mode", "preserve")  # "preserve" or "maximize_segment_ratio"
        is_maximize = crop_mode == "maximize_segment_ratio"

        # Load upscale model (cached)
        upscale_model = None
        upscale_dev = settings.get("upscale_device", "cuda:1")
        upscale_model_name = settings.get("upscale_model", "")
        if upscale_model_name:
            from .soya_scheduler.model_manager import get_upscale_model, upscale_image
            upscale_model = get_upscale_model(upscale_model_name, upscale_dev)

        # Load SAM2 + GroundingDINO (cached)
        sam_model = None
        dino_model = None
        eye_model = None
        segment_method = settings.get("segment_method", "sam2")
        sam2_model_name = settings.get("sam2_model", "")
        dino_model_name = settings.get("grounding_dino_model", "")
        segment_dev = settings.get("segment_device", "cuda:1")
        segment_prompt = settings.get("segment_prompt", "eyes")
        segment_threshold = settings.get("segment_threshold", 0.3)
        eye_seg_model_name = settings.get("eye_seg_model", "")
        eye_seg_threshold = settings.get("eye_seg_threshold", 0.5)
        if segment_method == "sam2" and sam2_model_name and dino_model_name:
            from .soya_scheduler.model_manager import get_sam2_model, get_grounding_dino_model
            sam_model = get_sam2_model(sam2_model_name, segment_dev)
            dino_model = get_grounding_dino_model(dino_model_name, segment_dev)
        elif segment_method == "custom" and eye_seg_model_name:
            from .soya_scheduler.model_manager import get_eye_seg_model
            eye_model = get_eye_seg_model(eye_seg_model_name, segment_dev)

        # Load eyebrow model (cached)
        eyebrow_model = None
        eyebrow_mode = settings.get("eyebrow_mode", "off")
        # Backward compat: old eyebrow_subtract=True → "subtract"
        if eyebrow_mode == "off" and settings.get("eyebrow_subtract", False):
            eyebrow_mode = "subtract"
        eyebrow_threshold = settings.get("eyebrow_threshold", 0.5)
        closed_eye_threshold = settings.get("closed_eye_threshold", 0.8)
        eyebrow_model_name = settings.get("eyebrow_model", "")
        if eyebrow_mode != "off" and eyebrow_model_name:
            from .soya_scheduler.model_manager import get_eyebrow_model
            eyebrow_model = get_eyebrow_model(eyebrow_model_name, segment_dev)

        matched_faces = []
        for f in kept_faces_ray:
            upscale_passes = self._calc_upscale_passes(f["bbox"], target_size)
            # is_large: face bbox already >= target_size (originally Level 0)
            # These faces get x2 upscale but stay individual (not batched)
            bw, bh = f["bbox"][2] - f["bbox"][0], f["bbox"][3] - f["bbox"][1]
            is_large = max(bw, bh) >= target_size
            matched_faces.append({
                "bbox": f["bbox"],
                "assignment": f.get("assignment", "unknown"),
                "similarity": f.get("similarity", 0.0),
                "area": f.get("area", 0),
                "upscale_passes": upscale_passes,
                "is_large": is_large,
            })

        info_lines.append(f"Target Size: {target_size}px")
        info_lines.append(f"Crop Mode: {crop_mode}")
        info_lines.append(f"Segment Method: {segment_method}")

        # Compute crop dimensions per face, independently for width and height.
        # Modes:
        #   "preserve": minimize crop area, generous padding – current behavior.
        #     When segs_min_distance > 0: rectangular crops (bbox_w/h + 2*margin).
        #     When segs_min_distance = 0: square crops (backward compatible, detailer_crop_factor).
        #   "maximize_segment_ratio": tight rectangular crops around bbox to maximize
        #     segment area / crop area ratio. Uses minimal padding regardless of segs_min_distance.
        common_final_w = 0
        common_final_h = 0
        for face in matched_faces:
            if not face.get("is_large", False):
                bw = face["bbox"][2] - face["bbox"][0]
                bh = face["bbox"][3] - face["bbox"][1]
                longest = max(bw, bh)
                # detailer_crop_factor as base, segs_min_distance as floor
                min_w = int(longest * detailer_crop_factor)
                min_h = int(longest * detailer_crop_factor)
                if segs_min_distance > 0:
                    min_w = max(min_w, bw + 2 * segs_min_distance)
                    min_h = max(min_h, bh + 2 * segs_min_distance)
                common_final_w = max(common_final_w, min_w * (2 ** face["upscale_passes"]))
                common_final_h = max(common_final_h, min_h * (2 ** face["upscale_passes"]))

        # Round up to ensure clean integer division for all upscale_passes values
        if common_final_w > 0 or common_final_h > 0:
            max_pow2 = max(
                (2 ** f["upscale_passes"] for f in matched_faces if not f.get("is_large", False)),
                default=1,
            )
            if common_final_w > 0:
                common_final_w = -(-common_final_w // max_pow2) * max_pow2
            if common_final_h > 0:
                common_final_h = -(-common_final_h // max_pow2) * max_pow2
            # Clamp to image dimensions to prevent crops extending beyond boundaries
            max_w_upscaled = (W // max_pow2) * max_pow2
            max_h_upscaled = (H // max_pow2) * max_pow2
            common_final_w = min(common_final_w, max_w_upscaled)
            common_final_h = min(common_final_h, max_h_upscaled)

        if is_maximize:
            info_lines.append(f"  Maximize segment ratio (two-pass)")
        elif segs_min_distance > 0:
            info_lines.append(f"  Trim mode: segs_min_distance={segs_min_distance}px")

        for face in matched_faces:
            bw = face["bbox"][2] - face["bbox"][0]
            bh = face["bbox"][3] - face["bbox"][1]
            if face.get("is_large", False):
                # Large face (Level 0): individual crop size
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
            info_lines.append(
                f"  Upscale {face['upscale_passes']}x - bbox {face['bbox']} "
                f"crop={face['crop_w']}x{face['crop_h']}"
            )

        # Collect all bboxes for blackboxing
        all_bboxes = [f["bbox"] for f in matched_faces]
        all_bboxes += [f["bbox"] for f in remain_faces_ray]

        # Parse labels from prompt ([1], [2], ...)
        labels = self._parse_prompt_labels(prompts)

        segs_items = []
        context_kept = []
        context_remain = []
        before_postprocess = []
        after_postprocess = []
        pass1_data = []  # for maximize_segment_ratio two-pass

        total_upscale_time = 0.0
        total_segs_time = 0.0
        total_eyebrow_time = 0.0
        upscale_vram_peak = 0  # bytes, peak across all faces
        segs_vram_peak = 0
        eyebrow_vram_peak = 0

        for i, face in enumerate(matched_faces):
            label = labels[i] if i < len(labels) else str(i + 1)
            face["label"] = label
            bbox = face["bbox"]
            upscale_passes = face["upscale_passes"]

            # Before: raw crop (no blackbox) – same dimensions as _crop_with_blackbox
            raw_region = self._compute_crop_region(bbox, face["crop_w"], face["crop_h"], H, W)
            raw_crop = image_uint8[raw_region[1]:raw_region[3], raw_region[0]:raw_region[2]].copy()

            before_postprocess.append({
                "label": label,
                "assignment": face.get("assignment", "unknown"),
                "bbox": bbox,
                "upscale_passes": upscale_passes,
                "crop_region": raw_region,
                "image_b64": self._np_to_base64(raw_crop),
                "image_w": raw_crop.shape[1],
                "image_h": raw_crop.shape[0],
            })

            if is_maximize:
                # ── Maximize: raw crop → upscale → blackbox (segment only) ──
                crop_region = list(raw_region)
                crop_tensor, up_time, up_vram = self._upscale_crop(
                    raw_crop, upscale_model, upscale_passes, upscale_dev
                )
                total_upscale_time += up_time
                upscale_vram_peak = max(upscale_vram_peak, up_vram)
                raw_final_np = (crop_tensor[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

                # Apply blackbox to upscaled image for segmentation only
                uf = 2 ** upscale_passes
                blackbox_np = self._apply_blackbox_to_crop(
                    raw_final_np, crop_region, bbox, all_bboxes, uf
                )

                # Segment on blackbox version
                mask, has_segment_mask, seg_time, seg_vram = self._run_segmentation(
                    blackbox_np, sam_model, dino_model,
                    segment_prompt, segment_threshold, segment_dev,
                    eye_model=eye_model, eye_seg_threshold=eye_seg_threshold
                )
                total_segs_time += seg_time
                segs_vram_peak = max(segs_vram_peak, seg_vram)

                # Eyebrow detection
                eyebrow_mask_float = None
                closed_eye_ratio = 0.0
                if eyebrow_model is not None:
                    eyebrow_mask_float, eb_time, eb_vram = self._run_eyebrow_segmentation(
                        blackbox_np, mask, eyebrow_model, segment_dev
                    )
                    total_eyebrow_time += eb_time
                    eyebrow_vram_peak = max(eyebrow_vram_peak, eb_vram)
                    # Closed-eye detection: if eyebrow overlaps heavily with segment, eyes are closed
                    seg_area = float(np.sum(mask > 0))
                    if seg_area > 0 and closed_eye_threshold > 0:
                        eb_binary = (eyebrow_mask_float > eyebrow_threshold).astype(np.uint8)
                        overlap = float(np.sum((mask > 0) & (eb_binary > 0)))
                        closed_eye_ratio = overlap / seg_area
                        if closed_eye_ratio >= closed_eye_threshold:
                            info_lines.append(f"    Face {label}: closed eye detected (overlap {closed_eye_ratio:.1%} >= {closed_eye_threshold:.0%})")
                            mask = np.zeros_like(mask)
                    if eyebrow_mode == "subtract":
                        eyebrow_binary = (eyebrow_mask_float > eyebrow_threshold).astype(np.uint8)
                        mask = mask * (1 - eyebrow_binary)

                # Store raw upscaled (no blackbox) for extraction
                pass1_data.append({
                    "upscaled_np": raw_final_np,
                    "mask": mask,
                    "crop_region": crop_region,
                    "upscale_passes": upscale_passes,
                    "is_large": face.get("is_large", False),
                    "has_segment_mask": has_segment_mask,
                    "label": label,
                    "bbox": bbox,
                    "assignment": face.get("assignment", "unknown"),
                    "eyebrow_mask_float": eyebrow_mask_float,
                    "closed_eye_ratio": closed_eye_ratio,
                })
            else:
                # ── Preserve: blackbox crop → upscale → segment → build outputs ──
                cropped_np, crop_region = self._crop_with_blackbox(
                    image_uint8, bbox, all_bboxes, face["crop_w"], face["crop_h"]
                )
                crop_tensor, up_time, up_vram = self._upscale_crop(
                    cropped_np, upscale_model, upscale_passes, upscale_dev
                )
                total_upscale_time += up_time
                upscale_vram_peak = max(upscale_vram_peak, up_vram)
                final_np = (crop_tensor[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

                mask, has_segment_mask, seg_time, seg_vram = self._run_segmentation(
                    final_np, sam_model, dino_model,
                    segment_prompt, segment_threshold, segment_dev,
                    eye_model=eye_model, eye_seg_threshold=eye_seg_threshold
                )
                total_segs_time += seg_time
                segs_vram_peak = max(segs_vram_peak, seg_vram)

                # Eyebrow detection
                eyebrow_mask_float = None
                closed_eye_ratio = 0.0
                if eyebrow_model is not None:
                    eyebrow_mask_float, eb_time, eb_vram = self._run_eyebrow_segmentation(
                        final_np, mask, eyebrow_model, segment_dev
                    )
                    total_eyebrow_time += eb_time
                    eyebrow_vram_peak = max(eyebrow_vram_peak, eb_vram)
                    # Closed-eye detection: if eyebrow overlaps heavily with segment, eyes are closed
                    seg_area = float(np.sum(mask > 0))
                    if seg_area > 0 and closed_eye_threshold > 0:
                        eb_binary = (eyebrow_mask_float > eyebrow_threshold).astype(np.uint8)
                        overlap = float(np.sum((mask > 0) & (eb_binary > 0)))
                        closed_eye_ratio = overlap / seg_area
                        if closed_eye_ratio >= closed_eye_threshold:
                            info_lines.append(f"    Face {label}: closed eye detected (overlap {closed_eye_ratio:.1%} >= {closed_eye_threshold:.0%})")
                            mask = np.zeros_like(mask)
                    if eyebrow_mode == "subtract":
                        eyebrow_binary = (eyebrow_mask_float > eyebrow_threshold).astype(np.uint8)
                        mask = mask * (1 - eyebrow_binary)

                # Build outputs
                after_postprocess.append({
                    "label": label,
                    "assignment": face.get("assignment", "unknown"),
                    "bbox": bbox,
                    "upscale_passes": upscale_passes,
                    "crop_region": crop_region,
                    "image_b64": self._np_to_base64(final_np),
                    "image_w": final_np.shape[1],
                    "image_h": final_np.shape[0],
                    "closed_eye_ratio": closed_eye_ratio,
                })

                if has_segment_mask:
                    overlay_b64 = self._build_mask_overlay_b64(final_np, mask)
                    if overlay_b64:
                        after_postprocess[-1]["mask_overlay_b64"] = overlay_b64

                # Eyebrow mask overlay (red/magenta tint)
                if eyebrow_mask_float is not None:
                    eb_overlay = self._build_eyebrow_overlay_b64(final_np, eyebrow_mask_float, eyebrow_threshold)
                    if eb_overlay:
                        after_postprocess[-1]["eyebrow_mask_overlay_b64"] = eb_overlay

                seg = SEG(
                    cropped_image=crop_tensor,
                    cropped_mask=mask,
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
                    "eyebrow_mask": eyebrow_mask_float if eyebrow_mode == "restore_color" else None,
                    "eyebrow_threshold": eyebrow_threshold if eyebrow_mode == "restore_color" else None,
                })

        # ── Maximize segment ratio: reverse calc + extraction ──────────
        if is_maximize and pass1_data:
            seg_centers = []
            tight_dims = []
            uf_list = []

            for p1 in pass1_data:
                msk = p1["mask"]
                uf = 2 ** p1["upscale_passes"]
                uf_list.append(uf)

                if p1["has_segment_mask"]:
                    ys, xs = np.where(msk > 0)
                    if len(xs) > 0:
                        seg_x1, seg_y1 = int(xs.min()), int(ys.min())
                        seg_x2, seg_y2 = int(xs.max()), int(ys.max())
                        seg_cx = (seg_x1 + seg_x2) / 2
                        seg_cy = (seg_y1 + seg_y2) / 2
                        margin = max(segs_min_distance, max(5, int(min(seg_x2 - seg_x1, seg_y2 - seg_y1) * 0.05)))
                        tight_w = (seg_x2 - seg_x1) + 2 * margin
                        tight_h = (seg_y2 - seg_y1) + 2 * margin
                    else:
                        # Empty mask – fall back to bbox center in upscaled coords
                        bx = p1["bbox"]
                        cr = p1["crop_region"]
                        bx1, by1 = (bx[0] - cr[0]) * uf, (bx[1] - cr[1]) * uf
                        bx2, by2 = (bx[2] - cr[0]) * uf, (bx[3] - cr[1]) * uf
                        seg_cx = (bx1 + bx2) / 2
                        seg_cy = (by1 + by2) / 2
                        margin = max(segs_min_distance, max(5, int(min(bx2 - bx1, by2 - by1) * 0.05)))
                        tight_w = (bx2 - bx1) + 2 * margin
                        tight_h = (by2 - by1) + 2 * margin
                else:
                    # No segment mask – fall back to bbox center in upscaled coords
                    bx = p1["bbox"]
                    cr = p1["crop_region"]
                    bx1, by1 = (bx[0] - cr[0]) * uf, (bx[1] - cr[1]) * uf
                    bx2, by2 = (bx[2] - cr[0]) * uf, (bx[3] - cr[1]) * uf
                    seg_cx = (bx1 + bx2) / 2
                    seg_cy = (by1 + by2) / 2
                    margin = max(segs_min_distance, max(5, int(min(bx2 - bx1, by2 - by1) * 0.05)))
                    tight_w = (bx2 - bx1) + 2 * margin
                    tight_h = (by2 - by1) + 2 * margin

                seg_centers.append((seg_cx, seg_cy))
                tight_dims.append((tight_w, tight_h))

            # Expand tight dims so the segment is fully contained when
            # extraction is centered on bbox center (not segment center).
            # Without this, any offset between segment and bbox centers
            # causes the tight crop to clip the segment on one side.
            for i, p1 in enumerate(pass1_data):
                bx = p1["bbox"]
                cr = p1["crop_region"]
                uf = 2 ** p1["upscale_passes"]
                bbox_cx_ups = (bx[0] - cr[0]) * uf + (bx[2] - bx[0]) * uf / 2
                bbox_cy_ups = (bx[1] - cr[1]) * uf + (bx[3] - bx[1]) * uf / 2
                seg_cx, seg_cy = seg_centers[i]
                offset_cx = abs(seg_cx - bbox_cx_ups)
                offset_cy = abs(seg_cy - bbox_cy_ups)
                if offset_cx > 0 or offset_cy > 0:
                    tw, th = tight_dims[i]
                    tight_dims[i] = (tw + 2 * offset_cx, th + 2 * offset_cy)

            # Common tight size for Level1/2 faces (exclude large faces)
            common_tight_w = 0
            common_tight_h = 0
            for i, p1 in enumerate(pass1_data):
                if not p1.get("is_large", False):
                    common_tight_w = max(common_tight_w, tight_dims[i][0])
                    common_tight_h = max(common_tight_h, tight_dims[i][1])

            # Round common tight size for clean dimensions
            # Use lcm(uf, 8) so that VAE encode/decode (8x alignment) doesn't change size
            if common_tight_w > 0 or common_tight_h > 0:
                max_pow2 = max(
                    (2 ** p1["upscale_passes"] for p1 in pass1_data if not p1.get("is_large", False)),
                    default=1,
                )
                snap = math.lcm(max_pow2, 8)
                if common_tight_w > 0:
                    common_tight_w = -(-int(common_tight_w) // snap) * snap
                if common_tight_h > 0:
                    common_tight_h = -(-int(common_tight_h) // snap) * snap

            info_lines.append(f"  Pass 2 extraction:")
            for i, p1 in enumerate(pass1_data):
                if p1.get("is_large", False):
                    tw, th = int(tight_dims[i][0]), int(tight_dims[i][1])
                    tw = -(-tw // 8) * 8
                    th = -(-th // 8) * 8
                else:
                    tw, th = common_tight_w, common_tight_h
                info_lines.append(
                    f"    Face {p1['label']}: tight {int(tw)}\u00d7{int(th)} "
                    f"(segment center ({int(seg_centers[i][0])},{int(seg_centers[i][1])}))"
                )
                # Debug: verify tight dimensions are multiples of lcm(uf, 8)
                uf_i = 2 ** p1["upscale_passes"]
                snap_i = math.lcm(uf_i, 8) if uf_i > 0 else 8
                info_lines.append(
                    f"      [DEBUG] uf={uf_i} snap={snap_i} tw%snap={tw % snap_i} th%snap={th % snap_i} "
                    f"divisible={'YES' if tw % snap_i == 0 and th % snap_i == 0 else 'NO'}"
                )

            # Extraction loop
            for i, p1 in enumerate(pass1_data):
                upscaled_np = p1["upscaled_np"]
                msk = p1["mask"]
                uf = uf_list[i]
                crop_region = p1["crop_region"]

                if p1.get("is_large", False):
                    tw, th = int(tight_dims[i][0]), int(tight_dims[i][1])
                    # Snap large face tight dims to multiples of 8 for VAE alignment
                    tw = -(-tw // 8) * 8
                    th = -(-th // 8) * 8
                else:
                    tw, th = int(common_tight_w), int(common_tight_h)

                cx, cy = seg_centers[i]

                # Compute bbox center in upscaled coords for comparison
                bx = p1["bbox"]
                bbox_cx_ups = (bx[0] - crop_region[0]) * uf + (bx[2] - bx[0]) * uf / 2
                bbox_cy_ups = (bx[1] - crop_region[1]) * uf + (bx[3] - bx[1]) * uf / 2
                offset_cx = cx - bbox_cx_ups
                offset_cy = cy - bbox_cy_ups
                info_lines.append(
                    f"    Face {p1['label']} [CENTER OFFSET]: "
                    f"seg_center=({cx:.0f},{cy:.0f}) bbox_center=({bbox_cx_ups:.0f},{bbox_cy_ups:.0f}) "
                    f"offset=({offset_cx:.1f},{offset_cy:.1f})px (in upscaled) "
                    f"→ orig offset=({offset_cx/uf:.1f},{offset_cy/uf:.1f})px"
                )

                # Extract tight region centered on bbox center (not segment center)
                # so paste-back aligns with the face bbox
                snap_val = math.lcm(uf, 8) if uf > 1 else 8
                extracted_img, extract_x1, extract_y1 = self._extract_region_centered(
                    upscaled_np, bbox_cx_ups, bbox_cy_ups, tw, th, snap=snap_val
                )
                extracted_mask = self._extract_region_centered(
                    msk, bbox_cx_ups, bbox_cy_ups, tw, th, snap=snap_val
                )[0].astype(np.uint8)

                # Compute crop_region in original coords (for paste-back)
                # extract_x1 is a multiple of uf, so division is exact (no rounding)
                cr_w = tw // uf
                cr_h = th // uf
                cr_x1_raw = crop_region[0] + extract_x1 // uf
                cr_y1_raw = crop_region[1] + extract_y1 // uf
                # Compute zero-padding from _extract_region_centered behavior.
                # Padding occurs when extraction extends beyond the upscaled CROP bounds
                # (not just the image bounds). extract_x1/y1 are in upscaled-crop coords.
                ups_h, ups_w = upscaled_np.shape[:2]
                pad_left = max(0, -extract_x1) // uf
                pad_top = max(0, -extract_y1) // uf
                pad_right = max(0, (extract_x1 + tw) - ups_w) // uf
                pad_bottom = max(0, (extract_y1 + th) - ups_h) // uf
                # Clamp so that (x1 + cr_w) <= W and (y1 + cr_h) <= H
                cr_x1 = max(0, min(cr_x1_raw, W - cr_w))
                cr_y1 = max(0, min(cr_y1_raw, H - cr_h))
                new_crop_region = [cr_x1, cr_y1, cr_x1 + cr_w, cr_y1 + cr_h]
                new_cr_w = new_crop_region[2] - new_crop_region[0]
                new_cr_h = new_crop_region[3] - new_crop_region[1]
                info_lines.append(
                    f"      [DEBUG] crop_region={new_crop_region} size=({new_cr_w},{new_cr_h}) "
                    f"extract=({extract_x1},{extract_y1}) extract/uf=({extract_x1//uf},{extract_y1//uf}) "
                    f"upscaled_size=({ups_w},{ups_h}) "
                    f"padding=({pad_left},{pad_top},{pad_right},{pad_bottom}) "
                    f"bbox={p1['bbox']} "
                    f"match={'YES' if new_cr_w == cr_w and new_cr_h == cr_h else 'MISMATCH'}"
                )

                # Convert to tensor
                crop_tensor = torch.from_numpy(extracted_img.astype(np.float32) / 255.0).unsqueeze(0)

                # Build after_postprocess entry
                after_entry = {
                    "label": p1["label"],
                    "assignment": p1["assignment"],
                    "bbox": p1["bbox"],
                    "upscale_passes": p1["upscale_passes"],
                    "crop_region": new_crop_region,
                    "image_b64": self._np_to_base64(extracted_img),
                    "image_w": extracted_img.shape[1],
                    "image_h": extracted_img.shape[0],
                    "closed_eye_ratio": p1.get("closed_eye_ratio", 0.0),
                }

                # Mask overlay for web UI
                if p1["has_segment_mask"]:
                    overlay_b64 = self._build_mask_overlay_b64(extracted_img, extracted_mask)
                    if overlay_b64:
                        after_entry["mask_overlay_b64"] = overlay_b64

                # Eyebrow mask overlay for web UI
                eb_extracted = None
                eb_original_pixels = None
                if p1.get("eyebrow_mask_float") is not None:
                    eb_extracted = self._extract_region_centered(
                        p1["eyebrow_mask_float"], bbox_cx_ups, bbox_cy_ups, tw, th, snap=uf
                    )[0]
                    eb_overlay = self._build_eyebrow_overlay_b64(extracted_img, eb_extracted, eyebrow_threshold)
                    if eb_overlay:
                        after_entry["eyebrow_mask_overlay_b64"] = eb_overlay
                    # Extract original eyebrow pixels (mask applied to original crop)
                    if eyebrow_mode == "restore_color" and eb_extracted is not None:
                        eb_binary = (eb_extracted > eyebrow_threshold).astype(np.float32)
                        eb_original_pixels = (extracted_img.astype(np.float32) * eb_binary[:, :, np.newaxis]).astype(np.uint8)

                after_postprocess.append(after_entry)

                seg = SEG(
                    cropped_image=crop_tensor,
                    cropped_mask=extracted_mask,
                    confidence=1.0,
                    crop_region=tuple(new_crop_region),
                    bbox=tuple(p1["bbox"]),
                    label=p1["label"],
                )
                segs_items.append(seg)

                context_kept.append({
                    "image": crop_tensor,
                    "upscale_passes": p1["upscale_passes"],
                    "is_large": p1.get("is_large", False),
                    "original_bbox": p1["bbox"],
                    "label": p1["label"],
                    "eyebrow_mask": eb_extracted if eyebrow_mode == "restore_color" else None,
                    "eyebrow_threshold": eyebrow_threshold if eyebrow_mode == "restore_color" else None,
                    "eyebrow_original_pixels": eb_original_pixels,
                    "crop_pad_left": pad_left,
                    "crop_pad_top": pad_top,
                    "crop_pad_right": pad_right,
                    "crop_pad_bottom": pad_bottom,
                    "crop_x1_raw": cr_x1_raw,
                    "crop_y1_raw": cr_y1_raw,
                })

        # Remain faces context
        for rf in remain_faces_ray:
            context_remain.append({
                "image": None,
                "original_bbox": rf["bbox"],
            })

        post_elapsed = time.time() - post_t0
        info_lines.append(f"Postprocess: {post_elapsed:.2f}\ucd08")
        info_lines.append(f"  Upscale: {total_upscale_time:.2f}\ucd08 / VRAM peak {upscale_vram_peak / (1024**3):.2f}GB")
        info_lines.append(f"  Segs: {total_segs_time:.2f}\ucd08 / VRAM peak {segs_vram_peak / (1024**3):.2f}GB")
        if eyebrow_model is not None:
            info_lines.append(f"  Eyebrow: {total_eyebrow_time:.2f}\ucd08 / VRAM peak {eyebrow_vram_peak / (1024**3):.2f}GB")

        return segs_items, context_kept, context_remain, info_lines, before_postprocess, after_postprocess, post_elapsed, total_upscale_time, total_segs_time, upscale_vram_peak, segs_vram_peak, total_eyebrow_time, eyebrow_vram_peak

    # ── Helper methods ───────────────────────────────────────────

    @staticmethod
    def _build_batch_groups(segs_items, context_kept, max_batch=2):
        """Group SEGS into batch-processing groups.

        is_large (Level 0) → always individual.
        Others (Level 1/2) → batched together (detailer resizes to common size).

        Returns list of groups, each group is a list of SEGS indices.
        """
        if not segs_items:
            return []

        groups = []
        batchable_pool = []

        for i, ctx in enumerate(context_kept):
            if ctx.get("is_large", False):
                for j in range(0, len(batchable_pool), max_batch):
                    groups.append(batchable_pool[j:j + max_batch])
                batchable_pool = []
                groups.append([i])
            else:
                batchable_pool.append(i)

        for j in range(0, len(batchable_pool), max_batch):
            groups.append(batchable_pool[j:j + max_batch])

        return groups

    @staticmethod
    def _flush_batchable(pool, segs_items, max_batch=2):
        """Split batchable pool into groups by matching tensor dimensions."""
        if not pool:
            return []

        # Group by actual cropped_image tensor size (H, W)
        size_groups = {}
        for idx in pool:
            img = segs_items[idx].cropped_image
            key = (img.shape[1], img.shape[2])
            if key not in size_groups:
                size_groups[key] = []
            size_groups[key].append(idx)

        result = []
        for indices in size_groups.values():
            for j in range(0, len(indices), max_batch):
                result.append(indices[j:j + max_batch])
        return result

    @staticmethod
    def _calc_upscale_passes(bbox, target_size):
        """Calculate upscale passes (1/2) to reach target_size from bbox longest edge.
        Minimum 1 pass (x2) so all faces go through upscaler before detailer."""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        longest = max(w, h)
        scale = target_size / longest
        passes = max(1, math.ceil(math.log2(scale)))
        return min(passes, 2)

    def _upscale_crop(self, crop_np, upscale_model, upscale_passes, upscale_dev):
        """Upscale a numpy crop image. Returns (tensor[1,H,W,3], elapsed, vram_peak)."""
        crop_tensor = torch.from_numpy(crop_np.astype(np.float32) / 255.0).unsqueeze(0)
        elapsed = 0.0
        vram_peak = 0
        if upscale_model is not None and upscale_passes > 0:
            from .soya_scheduler.model_manager import upscale_image
            t0 = time.time()
            if upscale_dev.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(upscale_dev)
                vram_before = torch.cuda.memory_allocated(upscale_dev)
            for _ in range(upscale_passes):
                crop_tensor = upscale_image(upscale_model, crop_tensor, upscale_dev)
            if upscale_dev.startswith("cuda"):
                torch.cuda.synchronize(upscale_dev)
                vram_peak = torch.cuda.max_memory_allocated(upscale_dev) - vram_before
            elapsed = time.time() - t0
        return crop_tensor, elapsed, vram_peak

    def _run_segmentation(self, image_np, sam_model, dino_model,
                          segment_prompt, segment_threshold, segment_dev,
                          eye_model=None, eye_seg_threshold=0.5):
        """Run segmentation. Returns (mask, has_mask, elapsed, vram_peak).

        Supports two modes:
        - SAM2 + GroundingDINO (sam_model and dino_model provided)
        - Custom ISNet eye segmentation (eye_model provided)
        """
        elapsed = 0.0
        vram_peak = 0
        mask = None
        has_mask = False

        if eye_model is not None:
            # Custom eye segmentation (ISNet)
            from .soya_scheduler.model_manager import eye_seg_segment
            t0 = time.time()
            if segment_dev.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(segment_dev)
                vram_before = torch.cuda.memory_allocated(segment_dev)
            mask_float = eye_seg_segment(eye_model, image_np, segment_dev)
            mask_binary = (mask_float > eye_seg_threshold).astype(np.uint8)
            if mask_binary.max() > 0:
                mask = mask_binary
                has_mask = True
            if segment_dev.startswith("cuda"):
                torch.cuda.synchronize(segment_dev)
                vram_peak = torch.cuda.max_memory_allocated(segment_dev) - vram_before
            elapsed = time.time() - t0
        elif sam_model is not None and dino_model is not None:
            # SAM2 + GroundingDINO
            from .soya_scheduler.model_manager import grounding_dino_predict, sam2_segment
            from PIL import Image as PILImage
            t0 = time.time()
            if segment_dev.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(segment_dev)
                vram_before = torch.cuda.memory_allocated(segment_dev)
            crop_pil = PILImage.fromarray(image_np)
            boxes = grounding_dino_predict(dino_model, crop_pil, segment_prompt, segment_threshold)
            if boxes:
                masks = sam2_segment(sam_model, image_np, boxes)
                combined = np.zeros(image_np.shape[:2], dtype=np.uint8)
                for m in masks:
                    combined = np.maximum(combined, m)
                mask = combined
                has_mask = True
            if segment_dev.startswith("cuda"):
                torch.cuda.synchronize(segment_dev)
                vram_peak = torch.cuda.max_memory_allocated(segment_dev) - vram_before
            elapsed = time.time() - t0

        if mask is None:
            mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
        return mask, has_mask, elapsed, vram_peak

    def _run_eyebrow_segmentation(self, image_np, seg_mask, eyebrow_model, device):
        """Run eyebrow ISNet model on the segment mask region only.

        Crops the area covered by seg_mask, runs eyebrow detection,
        then maps the result back to full image size.

        Args:
            image_np: (H, W, 3) uint8 full crop image
            seg_mask: (H, W) uint8 segment mask (eyes)
            eyebrow_model: ISNetDIS model
            device: torch device string

        Returns:
            (full_size_eyebrow_mask_float, elapsed, vram_peak)
            eyebrow mask is (H, W) float32, same size as input.
        """
        from .soya_scheduler.model_manager import eyebrow_segment

        # Find bounding box of the segment mask
        ys, xs = np.where(seg_mask > 0)
        if len(xs) == 0:
            empty = np.zeros(image_np.shape[:2], dtype=np.float32)
            return empty, 0.0, 0

        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1

        # Add margin so the eyebrow model has enough context
        margin = max(10, int(max(x2 - x1, y2 - y1) * 0.1))
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(image_np.shape[1], x2 + margin)
        y2 = min(image_np.shape[0], y2 + margin)

        # Crop the eye region
        eye_crop = image_np[y1:y2, x1:x2]

        elapsed = 0.0
        vram_peak = 0

        t0 = time.time()
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
            vram_before = torch.cuda.memory_allocated(device)

        eyebrow_crop = eyebrow_segment(eyebrow_model, eye_crop, device, img_size=384)

        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
            vram_peak = torch.cuda.max_memory_allocated(device) - vram_before
        elapsed = time.time() - t0

        # Map back to full image size
        full_mask = np.zeros(image_np.shape[:2], dtype=np.float32)
        full_mask[y1:y2, x1:x2] = eyebrow_crop

        return full_mask, elapsed, vram_peak

    def _build_mask_overlay_b64(self, image_np, mask):
        """Green-tinted mask overlay as base64. Returns str or None."""
        if mask is None or mask.max() == 0:
            return None
        mask_float = mask.astype(np.float32) / max(mask.max(), 1.0)
        mask_3ch = np.stack([mask_float] * 3, axis=-1)
        overlay = image_np.astype(np.float32) / 255.0
        overlay = overlay * (1 - mask_3ch * 0.5) + np.array([0.0, 0.8, 0.4]) * mask_3ch * 0.5
        overlay = (overlay.clip(0, 1) * 255).astype(np.uint8)
        return self._np_to_base64(overlay)

    @staticmethod
    def _build_eyebrow_overlay_b64(image_np, eyebrow_mask_float, threshold):
        """Red/magenta eyebrow mask overlay as base64. Returns str or None."""
        if eyebrow_mask_float is None or eyebrow_mask_float.max() <= threshold:
            return None
        mask_binary = (eyebrow_mask_float > threshold).astype(np.float32)
        mask_3ch = np.stack([mask_binary] * 3, axis=-1)
        overlay = image_np.astype(np.float32) / 255.0
        overlay = overlay * (1 - mask_3ch * 0.5) + np.array([0.8, 0.0, 0.4]) * mask_3ch * 0.5
        overlay = (overlay.clip(0, 1) * 255).astype(np.uint8)
        return SoyaProcessCollector_mdsoya._np_to_base64(overlay)

    @staticmethod
    def _match_bboxes(kept_faces_ray, redetected, img_w, img_h):
        """Match Ray worker tracked faces to redetected faces by bbox proximity.

        Returns list of dicts with 'bbox', 'assignment', 'similarity', etc.
        """
        if not redetected:
            # Fallback: use Ray bboxes directly
            return [
                {
                    "bbox": f["bbox"],
                    "assignment": f.get("assignment", "unknown"),
                    "similarity": f.get("similarity", 0.0),
                    "confidence": 1.0,
                }
                for f in kept_faces_ray
            ]

        matched = []
        used_redet = set()

        for ray_face in kept_faces_ray:
            rb = ray_face["bbox"]
            rcx = (rb[0] + rb[2]) / 2
            rcy = (rb[1] + rb[3]) / 2

            best_idx = -1
            best_dist = float('inf')
            for j, rd in enumerate(redetected):
                if j in used_redet:
                    continue
                db = rd["bbox"]
                dcx = (db[0] + db[2]) / 2
                dcy = (db[1] + db[3]) / 2
                dist = ((rcx - dcx) ** 2 + (rcy - dcy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_idx = j

            if best_idx >= 0:
                used_redet.add(best_idx)
                rd = redetected[best_idx]
                matched.append({
                    "bbox": rd["bbox"],
                    "assignment": ray_face.get("assignment", "unknown"),
                    "similarity": ray_face.get("similarity", 0.0),
                    "confidence": rd.get("confidence", 1.0),
                })
            else:
                matched.append({
                    "bbox": rb,
                    "assignment": ray_face.get("assignment", "unknown"),
                    "similarity": ray_face.get("similarity", 0.0),
                    "confidence": 1.0,
                })

        return matched

    @staticmethod
    def _crop_with_blackbox(image_np, target_bbox, all_bboxes, crop_w, crop_h):
        """Crop rectangular region around target face, blacking out other faces.

        When other faces' ellipses overlap with the target face or each other,
        the overlapping area is split via Voronoi (perpendicular bisector of centers).
        Each face only blacks out pixels closer to its own center, preventing
        nearby non-target ellipses from eating into the target face's region.

        Args:
            crop_w, crop_h: dimensions of the crop region

        Returns (cropped_np, crop_region [x1, y1, x2, y2])
        """
        H, W = image_np.shape[:2]
        crop_region = SoyaProcessCollector_mdsoya._compute_crop_region(
            target_bbox, crop_w, crop_h, H, W
        )

        masked = image_np.copy()

        # Separate target and other bboxes
        other_bboxes = [b for b in all_bboxes if b != target_bbox]

        if not other_bboxes:
            cropped = masked[crop_region[1]:crop_region[3], crop_region[0]:crop_region[2]]
            return cropped, crop_region

        # Target center (used as Voronoi reference – pixels closer to target stay visible)
        target_cx = (target_bbox[0] + target_bbox[2]) / 2
        target_cy = (target_bbox[1] + target_bbox[3]) / 2

        # Build list of valid other faces with ellipse params
        others = []
        for ob in other_bboxes:
            ox1, oy1, ox2, oy2 = ob
            ecx, ecy = (ox1 + ox2) / 2, (oy1 + oy2) / 2
            a, b = (ox2 - ox1) / 2, (oy2 - oy1) / 2
            if a > 0 and b > 0:
                others.append({'bbox': ob, 'cx': ecx, 'cy': ecy, 'a': a, 'b': b})

        if not others:
            cropped = masked[crop_region[1]:crop_region[3], crop_region[0]:crop_region[2]]
            return cropped, crop_region

        # All centers: index 0 = target, 1..n = others
        all_centers = [(target_cx, target_cy)] + [(o['cx'], o['cy']) for o in others]

        for i, o in enumerate(others):
            ox1, oy1, ox2, oy2 = o['bbox']
            ecx, ecy, a, b = o['cx'], o['cy'], o['a'], o['b']

            ry1, ry2 = max(0, oy1), min(H, oy2)
            rx1, rx2 = max(0, ox1), min(W, ox2)
            if ry1 >= ry2 or rx1 >= rx2:
                continue

            yy, xx = np.ogrid[ry1:ry2, rx1:rx2]
            ellipse = ((xx - ecx) / a) ** 2 + ((yy - ecy) / b) ** 2 <= 1

            if not np.any(ellipse):
                continue

            # Voronoi split: keep only pixels closer to this face than any other
            # (including target – protects target face from being blacked out)
            voronoi = np.ones(ellipse.shape, dtype=bool)
            for j, (cx, cy) in enumerate(all_centers):
                if j == i + 1:  # skip self (index 0 = target, 1..n = others)
                    continue
                dx, dy = ecx - cx, ecy - cy
                if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                    continue  # coincident centers
                mid_x, mid_y = (ecx + cx) / 2, (ecy + cy) / 2
                # Perpendicular bisector: True on this face's side
                bisector = (xx - mid_x) * dx + (yy - mid_y) * dy >= 0
                voronoi &= bisector

            masked[ry1:ry2, rx1:rx2][ellipse & voronoi] = 0

        # Extract crop from masked image
        cropped = masked[crop_region[1]:crop_region[3], crop_region[0]:crop_region[2]]

        return cropped, crop_region

    @staticmethod
    def _apply_blackbox_to_crop(cropped_np, crop_region, target_bbox, all_bboxes, upscale_factor=1):
        """Apply Voronoi blackbox to an already-cropped image.

        Translates all bboxes from original image coords to the cropped image's
        own coordinate space (× upscale_factor), then applies the same Voronoi
        ellipse logic as _crop_with_blackbox.

        The resulting blackbox is mathematically identical to what
        _crop_with_blackbox would produce on the original image, then crop+upscale.
        This is because the ellipse equation and Voronoi bisector are invariant
        under the linear transform (translate + scale).

        Args:
            cropped_np: (H, W, 3) already-cropped image
            crop_region: [x1, y1, x2, y2] crop position in original image coords
            target_bbox: target face bbox in original image coords
            all_bboxes: all face bboxes in original image coords
            upscale_factor: multiplier to translate original→cropped coords

        Returns:
            blackbox_np: cropped image with other faces blacked out (new copy)
        """
        masked = cropped_np.copy()
        uf = upscale_factor
        H, W = masked.shape[:2]

        # Translate target center to crop-relative coords
        target_cx = ((target_bbox[0] + target_bbox[2]) / 2 - crop_region[0]) * uf
        target_cy = ((target_bbox[1] + target_bbox[3]) / 2 - crop_region[1]) * uf

        # Build list of other faces with ellipse params in crop-relative coords
        others = []
        for ob in all_bboxes:
            if ob == target_bbox:
                continue
            ox1, oy1, ox2, oy2 = ob
            ecx = ((ox1 + ox2) / 2 - crop_region[0]) * uf
            ecy = ((oy1 + oy2) / 2 - crop_region[1]) * uf
            a = (ox2 - ox1) / 2 * uf
            b = (oy2 - oy1) / 2 * uf
            if a > 0 and b > 0:
                others.append({'cx': ecx, 'cy': ecy, 'a': a, 'b': b})

        if not others:
            return masked

        # All centers for Voronoi (index 0 = target, 1..n = others)
        all_centers = [(target_cx, target_cy)] + [(o['cx'], o['cy']) for o in others]

        for i, o in enumerate(others):
            ecx, ecy, a, b = o['cx'], o['cy'], o['a'], o['b']

            ry1, ry2 = max(0, int(ecy - b)), min(H, int(ecy + b))
            rx1, rx2 = max(0, int(ecx - a)), min(W, int(ecx + a))
            if ry1 >= ry2 or rx1 >= rx2:
                continue

            yy, xx = np.ogrid[ry1:ry2, rx1:rx2]
            ellipse = ((xx - ecx) / a) ** 2 + ((yy - ecy) / b) ** 2 <= 1

            if not np.any(ellipse):
                continue

            # Voronoi split (same logic as _crop_with_blackbox)
            voronoi = np.ones(ellipse.shape, dtype=bool)
            for j, (cx, cy) in enumerate(all_centers):
                if j == i + 1:  # skip self
                    continue
                dx, dy = ecx - cx, ecy - cy
                if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                    continue
                mid_x, mid_y = (ecx + cx) / 2, (ecy + cy) / 2
                bisector = (xx - mid_x) * dx + (yy - mid_y) * dy >= 0
                voronoi &= bisector

            masked[ry1:ry2, rx1:rx2][ellipse & voronoi] = 0

        return masked

    @staticmethod
    def _extract_region_centered(img_np, center_x, center_y, target_w, target_h, snap=1):
        """Extract a region of target_w x target_h centered on (center_x, center_y).

        If the region extends beyond image bounds, the out-of-bounds area is zero-padded.

        Args:
            img_np: (H, W, ...) or (H, W) numpy array
            center_x, center_y: center of extraction in pixel coords
            target_w, target_h: size of extraction
            snap: snap extraction position to multiples of this value (e.g. upscale factor)

        Returns:
            (extracted_region, x1_offset, y1_offset) where x1/y1 are the
            top-left corner position in source image coordinates.
        """
        H, W = img_np.shape[:2]

        # Compute ideal bounds
        x1 = int(round(center_x - target_w / 2))
        y1 = int(round(center_y - target_h / 2))

        # Snap to grid so x1/uf is always an integer
        if snap > 1:
            x1 = (x1 // snap) * snap
            y1 = (y1 // snap) * snap
        x2 = x1 + target_w
        y2 = y1 + target_h

        # Clamp to image bounds
        src_x1 = max(0, x1)
        src_y1 = max(0, y1)
        src_x2 = min(W, x2)
        src_y2 = min(H, y2)

        # Create output array (zero-padded)
        out_shape = (target_h, target_w) + img_np.shape[2:]
        result = np.zeros(out_shape, dtype=img_np.dtype)

        # Compute where to place the valid region in the output
        dst_x1 = src_x1 - x1
        dst_y1 = src_y1 - y1
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)

        # Copy valid region
        if src_x2 > src_x1 and src_y2 > src_y1:
            result[dst_y1:dst_y2, dst_x1:dst_x2] = img_np[src_y1:src_y2, src_x1:src_x2]

        return result, x1, y1

    @staticmethod
    def _compute_crop_region(bbox, crop_w, crop_h, H, W):
        """Compute a rectangular crop region centered on bbox center.
        Each dimension shifts independently when hitting image boundaries."""
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        half_w = crop_w / 2
        half_h = crop_h / 2
        x1 = int(cx - half_w)
        y1 = int(cy - half_h)
        x2 = int(cx + half_w)
        y2 = int(cy + half_h)

        # Shift each dimension independently to compensate for boundary hits
        if x1 < 0:
            x2 -= x1
            x1 = 0
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if x2 > W:
            x1 -= (x2 - W)
            x2 = W
        if y2 > H:
            y1 -= (y2 - H)
            y2 = H

        # Final clamp
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(W, x2)
        y2 = min(H, y2)

        return [x1, y1, x2, y2]

    @staticmethod
    def _parse_prompt_labels(prompts):
        """Parse [1], [2], [3] labels from [LAB] prompt."""
        import re
        labels = re.findall(r'^\[(\d+)\]', prompts, re.MULTILINE)
        return labels

    @staticmethod
    def _np_to_base64(np_image):
        """Convert (H, W, 3) uint8 numpy to base64 PNG string."""
        import io
        import base64
        from PIL import Image as PILImage
        img = PILImage.fromarray(np_image.astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _save_face_data(self, before_post, settings):
        """Save before-postprocess face crops to disk if toggle is enabled."""
        if not settings.get("save_face_data", False):
            return
        if not before_post:
            return

        # Determine output directory
        try:
            import folder_paths
            output_dir = folder_paths.get_output_directory()
        except Exception:
            output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")

        face_data_subdir = settings.get("face_data_path", "face_data")
        save_dir = os.path.join(output_dir, face_data_subdir)
        os.makedirs(save_dir, exist_ok=True)

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        saved = 0
        for i, entry in enumerate(before_post):
            label = entry.get("label", "unknown")
            assignment = entry.get("assignment", "unknown")
            # Decode base64 image
            import base64
            import io
            b64 = entry.get("image_b64", "")
            if not b64:
                continue
            img_data = base64.b64decode(b64)
            img = Image.open(io.BytesIO(img_data))
            filename = f"{ts}_{i}_{label}_{assignment}.png"
            img.save(os.path.join(save_dir, filename))
            saved += 1

        if saved:
            print(f"[Soya:ProcessCollector] Face data saved to {save_dir} ({saved} faces)")

    @staticmethod
    def _empty_image():
        return torch.zeros((1, 64, 64, 3), dtype=torch.float32)
