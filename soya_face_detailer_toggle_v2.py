"""
SoyaFaceDetailerToggleV2 – Face detailer with mask_expand + crop_expand + upscale mechanism.

For each detected face:
  1. Detect face bbox via YOLO bbox_detector
  2. Expand mask area by mask_expand (1.0 = raw bbox, 1.5 = 1.5x → what gets redrawn)
  3. Expand crop area by crop_expand (1.0 = mask only, 2.0 = 2x → context visible to model)
  4. Upscale crop by upscale_factor for higher-resolution processing
  5. VAE encode → KSampler → VAE decode → downscale back → paste with feathered mask
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image


class SoyaFaceDetailerToggleV2_mdsoya:
    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers
        return {
            "required": {
                "enable": ("STRING", {"default": "true"}),
                "image": ("IMAGE",),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "positive": ("STRING", {"multiline": True, "default": ""}),
                "negative": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "bbox_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mask_expand": ("FLOAT", {"default": 1.3, "min": 1.0, "max": 4.0, "step": 0.05}),
                "crop_expand": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 6.0, "step": 0.1}),
                "upscale_factor": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 4.0, "step": 0.05}),
                "feather": ("INT", {"default": 5, "min": 0, "max": 100}),
                "corner_roundness": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "noise_mask": ("BOOLEAN", {"default": True}),
                "drop_size": ("INT", {"default": 10, "min": 1, "max": 1024}),
                "bbox_detector": ("BBOX_DETECTOR",),
                "cycle": ("INT", {"default": 1, "min": 1, "max": 10}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("image", "mask", "crop_preview")
    FUNCTION = "doit"
    CATEGORY = "Soya"

    def doit(self, *, enable, image, model, clip, vae, positive, negative,
             seed, steps, cfg, sampler_name, scheduler, denoise,
             bbox_threshold, mask_expand, crop_expand, upscale_factor,
             feather, corner_roundness, noise_mask, drop_size,
             bbox_detector, cycle):

        use = enable.strip().lower() in ("true", "1", "yes")

        B, H, W, C = image.shape
        if not use:
            print("[FaceDetailerToggleV2] DISABLED — bypassing")
            return (image, torch.zeros((B, H, W), dtype=torch.float32),
                    torch.zeros_like(image))

        print("[FaceDetailerToggleV2] ENABLED — running face detailer")

        result_image = image.clone()
        combined_mask = torch.zeros((B, H, W), dtype=torch.float32)
        crop_region = torch.zeros((B, H, W), dtype=torch.float32)

        positive_cond = self._encode_conditioning(clip, positive)
        negative_cond = self._encode_conditioning(clip, negative)

        for batch_idx in range(B):
            img_single = result_image[batch_idx:batch_idx + 1]

            for cy in range(cycle):
                img_single, face_mask, crop_coords = self._detail_single(
                    img_single, model, vae, positive_cond, negative_cond,
                    seed, steps, cfg, sampler_name, scheduler, denoise,
                    bbox_threshold, mask_expand, crop_expand, upscale_factor,
                    feather, corner_roundness, noise_mask, drop_size,
                    bbox_detector,
                )
                seed += 1

            result_image[batch_idx] = img_single[0]
            combined_mask[batch_idx] = face_mask[0]
            for cx1, cy1, cx2, cy2 in crop_coords:
                crop_region[batch_idx, cy1:cy2, cx1:cx2] = 1.0

        crop_preview = image * (crop_region * 0.3 + combined_mask * 0.7).unsqueeze(-1)
        return (result_image, combined_mask, crop_preview)

    def _detail_single(self, image, model, vae, positive_cond, negative_cond,
                       seed, steps, cfg, sampler_name, scheduler, denoise,
                       bbox_threshold, mask_expand, crop_expand, upscale_factor,
                       feather, corner_roundness, noise_mask, drop_size,
                       bbox_detector):

        _, H, W, C = image.shape

        bboxes = self._detect_faces(image, bbox_detector, bbox_threshold)
        if not bboxes:
            print("[FaceDetailerToggleV2] No faces detected — returning original")
            return image, torch.zeros((1, H, W), dtype=torch.float32), []

        full_mask = torch.zeros((1, H, W), dtype=torch.float32)
        result = image.clone()
        crop_regions = []

        for bbox in bboxes:
            rx1, ry1, rx2, ry2 = bbox
            rbw, rbh = rx2 - rx1, ry2 - ry1

            if rbw < drop_size or rbh < drop_size:
                continue

            # ── Step 1: Expand raw bbox → mask region ──
            rcx, rcy = (rx1 + rx2) / 2.0, (ry1 + ry2) / 2.0
            mw = rbw * mask_expand
            mh = rbh * mask_expand
            mx1 = max(0, int(rcx - mw / 2))
            my1 = max(0, int(rcy - mh / 2))
            mx2 = min(W, int(rcx + mw / 2))
            my2 = min(H, int(rcy + mh / 2))
            mw, mh = mx2 - mx1, my2 - my1

            # ── Step 2: Expand mask region → crop region ──
            mcx, mcy = (mx1 + mx2) / 2.0, (my1 + my2) / 2.0
            cw = mw * crop_expand
            ch = mh * crop_expand

            # Round UP to multiple of 8
            crop_w = ((int(cw) + 7) // 8) * 8
            crop_h = ((int(ch) + 7) // 8) * 8

            if crop_w < 8 or crop_h < 8:
                continue

            cx1 = (int(mcx - crop_w / 2) // 8) * 8
            cy1 = (int(mcy - crop_h / 2) // 8) * 8
            cx2 = cx1 + crop_w
            cy2 = cy1 + crop_h

            if cx2 > W:
                cx1 = max(0, (W - crop_w) // 8 * 8)
                cx2 = cx1 + crop_w
            if cy2 > H:
                cy1 = max(0, (H - crop_h) // 8 * 8)
                cy2 = cy1 + crop_h

            cx1 = max(0, cx1)
            cy1 = max(0, cy1)
            cx2 = min(W, cx2)
            cy2 = min(H, cy2)

            actual_w, actual_h = cx2 - cx1, cy2 - cy1
            crop_regions.append((cx1, cy1, cx2, cy2))

            # ── Crop from image ──
            crop_tensor = image[0, cy1:cy2, cx1:cx2].clone()

            # ── Mask (superellipse based on mask region) ──
            local_mcx = mcx - cx1
            local_mcy = mcy - cy1
            local_mhw = max(1.0, mw / 2.0)
            local_mhh = max(1.0, mh / 2.0)

            if corner_roundness <= 0.0:
                local_x1 = max(0, int(mx1 - cx1))
                local_y1 = max(0, int(my1 - cy1))
                local_x2 = min(actual_w, int(mx2 - cx1))
                local_y2 = min(actual_h, int(my2 - cy1))
                mask = torch.zeros((actual_h, actual_w), dtype=torch.float32)
                if local_x2 > local_x1 and local_y2 > local_y1:
                    mask[local_y1:local_y2, local_x1:local_x2] = 1.0
            else:
                n = 2.0 / max(corner_roundness, 0.01)
                yy, xx = torch.meshgrid(
                    torch.arange(actual_h, dtype=torch.float32),
                    torch.arange(actual_w, dtype=torch.float32),
                    indexing='ij'
                )
                nx = (xx - local_mcx) / local_mhw
                ny = (yy - local_mcy) / local_mhh
                superellipse = (nx.abs() ** n + ny.abs() ** n)
                mask = (superellipse <= 1.0).float()

            if feather > 0:
                mask = self._gaussian_blur_gpu(mask, feather)

            # ── Upscale for processing ──
            if upscale_factor > 1.0:
                process_w = ((int(actual_w * upscale_factor) + 7) // 8) * 8
                process_h = ((int(actual_h * upscale_factor) + 7) // 8) * 8

                crop_pil = Image.fromarray((crop_tensor.cpu().numpy() * 255).astype(np.uint8))
                crop_pil = crop_pil.resize((process_w, process_h), Image.Resampling.LANCZOS)
                crop_tensor = torch.from_numpy(np.array(crop_pil).astype(np.float32) / 255.0)
                crop_tensor = crop_tensor.unsqueeze(0)

                mask_pil = Image.fromarray((mask.cpu().numpy() * 255).astype(np.uint8))
                mask_pil = mask_pil.resize((process_w, process_h), Image.Resampling.LANCZOS)
                process_mask = torch.from_numpy(np.array(mask_pil).astype(np.float32) / 255.0)
                mask_batch = process_mask.unsqueeze(0)
            else:
                process_w, process_h = actual_w, actual_h
                crop_tensor = crop_tensor.unsqueeze(0)
                mask_batch = mask.unsqueeze(0)

            # ── VAE encode → KSampler → VAE decode ──
            latent = vae.encode(crop_tensor[:, :, :, :3])
            latent_dict = {"samples": latent}
            if noise_mask:
                latent_dict["noise_mask"] = mask_batch

            refined_latent = self._run_ksampler(
                model, seed, steps, cfg, sampler_name, scheduler,
                positive_cond, negative_cond, latent_dict, denoise,
            )
            enhanced = vae.decode(refined_latent)

            # ── Downscale back ──
            enhanced_crop = enhanced[0].clamp(0, 1)
            while enhanced_crop.dim() > 3:
                enhanced_crop = enhanced_crop[0]

            if upscale_factor > 1.0 and (enhanced_crop.shape[1] != actual_w or enhanced_crop.shape[0] != actual_h):
                enhanced_np = (enhanced_crop.cpu().numpy() * 255).astype(np.uint8)
                enhanced_pil = Image.fromarray(enhanced_np)
                enhanced_pil = enhanced_pil.resize((actual_w, actual_h), Image.Resampling.LANCZOS)
                enhanced_crop = torch.from_numpy(np.array(enhanced_pil).astype(np.float32) / 255.0)

            # ── Paste back with feathered mask ──
            alpha = mask.unsqueeze(-1)
            result[0, cy1:cy2, cx1:cx2] = (
                alpha * enhanced_crop
                + (1 - alpha) * result[0, cy1:cy2, cx1:cx2]
            )
            full_mask[0, cy1:cy2, cx1:cx2] = torch.maximum(
                full_mask[0, cy1:cy2, cx1:cx2], mask
            )

        return result, full_mask, crop_regions

    # ── Helpers ──

    @staticmethod
    def _gaussian_blur_gpu(mask, sigma):
        ksize = int(6 * sigma + 1)
        if ksize % 2 == 0:
            ksize += 1

        x = torch.arange(ksize, dtype=torch.float32, device=mask.device) - ksize // 2
        kernel = torch.exp(-x ** 2 / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()

        mask_4d = mask.unsqueeze(0).unsqueeze(0)
        pad = ksize // 2

        padded = F.pad(mask_4d, [pad, pad, 0, 0], mode='reflect')
        blurred = F.conv2d(padded, kernel.view(1, 1, 1, -1))

        padded = F.pad(blurred, [0, 0, pad, pad], mode='reflect')
        blurred = F.conv2d(padded, kernel.view(1, 1, -1, 1))

        return blurred.squeeze(0).squeeze(0)

    @staticmethod
    def _detect_faces(image, bbox_detector, threshold):
        yolo = bbox_detector.bbox_model
        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        results = yolo(img_np, verbose=False)

        faces = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < threshold:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                faces.append((int(x1), int(y1), int(x2), int(y2)))

        faces.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        return faces

    @staticmethod
    def _encode_conditioning(clip, text):
        from nodes import CLIPTextEncode
        return CLIPTextEncode().encode(clip, text)[0]

    @staticmethod
    def _run_ksampler(model, seed, steps, cfg, sampler_name, scheduler,
                      positive, negative, latent_dict, denoise):
        from nodes import common_ksampler
        result = common_ksampler(
            model, seed, steps, cfg, sampler_name, scheduler,
            positive, negative, latent_dict, denoise=denoise,
        )
        return result[0]["samples"]
