"""
SoyaEyeExtractor – Extract eyes from face-enhanced image using segmentation.

Takes the paste-back result (face-enhanced image) and context from Collector2,
delegates to the original Collector's _postprocess for crop/upscale/segment,
and builds SEGS+context compatible with SoyaBatchDetailer (Eye Detailer).
"""

import time

from .soya_process_collector import SoyaProcessCollector_mdsoya
from .soya_scheduler.config_manager import load_config


class SoyaEyeExtractor_mdsoya:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompts": ("STRING", {"multiline": True, "forceInput": True}),
                "context": ("CONTEXT", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "SEGS", "STRING", "CONTEXT")
    RETURN_NAMES = ("image", "segs", "prompts", "context")
    FUNCTION = "extract"
    CATEGORY = "Soya/FaceDetailer"

    def extract(self, image, prompts, context):
        t0 = time.time()
        info_lines = ["[Soya Eye Extractor]"]

        kept_faces = context.get("kept_faces", [])
        if not kept_faces:
            info_lines.append("No kept faces in context.")
            H, W = image.shape[1], image.shape[2]
            return (image, ((H, W), []), prompts, context)

        settings = load_config().get("settings", {})

        # Reconstruct kept_faces_ray from Collector2 context.
        # Only bbox is required; assignment/similarity are optional.
        kept_faces_ray = []
        for kf in kept_faces:
            entry = {"bbox": kf["original_bbox"]}
            if "assignment" in kf:
                entry["assignment"] = kf["assignment"]
            if "similarity" in kf:
                entry["similarity"] = kf["similarity"]
            kept_faces_ray.append(entry)

        remain_faces_ray = context.get("remain_faces", [])

        # Delegate to the original Collector's _postprocess — handles
        # crop, upscale, segmentation, eyebrow, common dimensions, etc.
        collector = SoyaProcessCollector_mdsoya()
        result = collector._postprocess(
            image, kept_faces_ray, remain_faces_ray, prompts, settings
        )
        (segs_items, context_kept, context_remain, post_info,
         before_post, after_post, post_elapsed, total_upscale_time,
         total_segs_time, upscale_vram, segs_vram,
         total_eyebrow_time, eyebrow_vram) = result

        # Build SEGS
        H, W = image.shape[1], image.shape[2]
        new_segs = ((H, W), segs_items)

        # Build batch groups
        batch_groups = SoyaProcessCollector_mdsoya._build_batch_groups(
            segs_items, context_kept
        )

        # Build context (compatible with BatchDetailer)
        new_context = {
            "kept_faces": context_kept,
            "remain_faces": context_remain,
            "segs": new_segs,
            "batch_groups": batch_groups,
            "crop_mode": settings.get("crop_mode", "preserve"),
            "eyebrow_restore": settings.get("eyebrow_restore", False),
            "eyebrow_restore_mode": settings.get("eyebrow_restore_mode", "hs_preserve"),
            "eyebrow_blur": settings.get("eyebrow_blur", 0),
            "eyebrow_hs_percentile": settings.get("eyebrow_hs_percentile", 0.0),
            "eyebrow_v_range": settings.get("eyebrow_v_range", 1.0),
            "eyebrow_opacity": settings.get("eyebrow_opacity", 0.0),
        }

        elapsed = time.time() - t0
        info_lines.extend(post_info)
        info_lines.append(
            f"Extracted {len(segs_items)} eye regions in {elapsed:.2f}초 "
            f"(upscale: {total_upscale_time:.2f}초, segs: {total_segs_time:.2f}초, "
            f"eyebrow: {total_eyebrow_time:.2f}초)"
        )

        return (image, new_segs, prompts, new_context)
