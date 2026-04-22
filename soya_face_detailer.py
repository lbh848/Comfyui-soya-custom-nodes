"""
SoyaFaceDetailer – KSampler on face crops from Collector2.

Takes SEGS from SoyaProcessCollector2 (already cropped+upscaled faces),
runs KSampler on each face crop, outputs FACE_CONTEXT for paste-back.
"""

import time
import torch
import numpy as np
from scipy.ndimage import gaussian_filter

from .soya_batch_detailer import SoyaBatchDetailer_mdsoya
from .soya_scheduler.config_manager import load_config


class SoyaFaceDetailer_mdsoya:
    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers
        return {
            "required": {
                "image": ("IMAGE",),
                "segs": ("SEGS",),
                "prompts": ("STRING", {"multiline": True, "forceInput": True}),
                "context": ("CONTEXT", {"forceInput": True}),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "edge_blur": ("INT", {"default": 5, "min": 0, "max": 100}),
            },
        }

    RETURN_TYPES = ("IMAGE", "FACE_CONTEXT", "STRING")
    RETURN_NAMES = ("image", "face_context", "info")
    FUNCTION = "detail"
    CATEGORY = "Soya/FaceDetailer"

    def detail(self, image, segs, prompts, context, model, clip, vae,
               negative_prompt, seed, steps, cfg, sampler_name, scheduler,
               denoise, edge_blur):
        t0 = time.time()
        info_lines = ["[Soya Face Detailer]"]

        config = load_config().get("settings", {})
        enabled = config.get("face_detailer_enabled", False)
        if not enabled:
            info_lines.append("Face Detailer is disabled in settings.")
            face_ctx = {"faces": [], "remain_faces": context.get("remain_faces", []),
                        "image_size": (image.shape[1], image.shape[2])}
            return (image, face_ctx, "\n".join(info_lines))

        _, segs_list = segs
        kept_faces = context.get("kept_faces", [])

        if not segs_list:
            info_lines.append("No face SEGS to process.")
            face_ctx = {"faces": [], "remain_faces": context.get("remain_faces", []),
                        "image_size": (image.shape[1], image.shape[2])}
            return (image, face_ctx, "\n".join(info_lines))

        # Parse prompts and encode negative
        prompt_map = SoyaBatchDetailer_mdsoya._parse_prompts(prompts)
        negative_cond = SoyaBatchDetailer_mdsoya._encode_conditioning(clip, negative_prompt)

        face_results = []

        for i, seg in enumerate(segs_list):
            crop = seg.cropped_image  # (1, H_up, W_up, 3) already cropped+upscaled by Collector2
            bbox = seg.bbox
            crop_region = seg.crop_region
            label = seg.label
            kf = kept_faces[i] if i < len(kept_faces) else {}
            upscale_passes = kf.get("upscale_passes", 0)

            enh_h, enh_w = crop.shape[1], crop.shape[2]

            # Create face ellipse mask in upscaled crop coordinates
            uf = 2 ** upscale_passes
            cr_x1, cr_y1 = crop_region[0], crop_region[1]
            bbox_in_crop = [
                (bbox[0] - cr_x1) * uf,
                (bbox[1] - cr_y1) * uf,
                (bbox[2] - cr_x1) * uf,
                (bbox[3] - cr_y1) * uf,
            ]
            mask = np.zeros((enh_h, enh_w), dtype=np.float32)
            SoyaBatchDetailer_mdsoya._draw_ellipse_on_mask(mask, bbox_in_crop, value=1.0)

            if edge_blur > 0:
                mask = gaussian_filter(mask, sigma=edge_blur)

            mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

            # KSampler: VAE encode → sample → VAE decode
            prompt_text = prompt_map.get(label, "")
            positive_cond = SoyaBatchDetailer_mdsoya._encode_conditioning(clip, prompt_text)

            latent = vae.encode(crop[:, :, :, :3])
            refined_latent = SoyaBatchDetailer_mdsoya._run_ksampler(
                model, seed, steps, cfg, sampler_name, scheduler,
                positive_cond, negative_cond, latent, denoise, mask_tensor,
            )
            enhanced_image = vae.decode(refined_latent)

            info_lines.append(
                f"  Face {i} ({label}): crop={crop_region} size={enh_w}x{enh_h} "
                f"uf={uf} prompt='{prompt_text[:50]}...'"
            )

            face_results.append({
                "original_bbox": bbox,
                "crop_region": crop_region,
                "crop_region_raw": [
                    kf.get("crop_x1_raw", crop_region[0]),
                    kf.get("crop_y1_raw", crop_region[1]),
                    kf.get("crop_x1_raw", crop_region[0]) + (crop_region[2] - crop_region[0]),
                    kf.get("crop_y1_raw", crop_region[1]) + (crop_region[3] - crop_region[1]),
                ],
                "crop_pad_left": kf.get("crop_pad_left", 0),
                "crop_pad_top": kf.get("crop_pad_top", 0),
                "crop_pad_right": kf.get("crop_pad_right", 0),
                "crop_pad_bottom": kf.get("crop_pad_bottom", 0),
                "enhanced_image": enhanced_image,
                "upscale_factor": uf,
                "label": label,
            })

        remain_faces = context.get("remain_faces", [])
        elapsed = time.time() - t0
        info_lines.append(f"Processed {len(face_results)} faces in {elapsed:.2f}초")

        face_context = {
            "faces": face_results,
            "remain_faces": remain_faces,
            "image_size": (image.shape[1], image.shape[2]),
        }

        return (image, face_context, "\n".join(info_lines))
