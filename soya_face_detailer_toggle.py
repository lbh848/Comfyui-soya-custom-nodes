"""
SoyaFaceDetailerToggle – Face detailer without Impact Pack dependency.

Uses ComfyUI's built-in KSampler, VAE encode/decode, and YOLO-based
bbox detection directly.  enable=False passes the original image through.

Optimized: All image resize/blur done with torch ops (F.interpolate, F.conv2d)
to eliminate GPU↔CPU transfers — critical for Docker environments.
"""

import torch
import torch.nn.functional as F
import numpy as np


class SoyaFaceDetailerToggle_mdsoya:
    """
    Face detailer — enable=False면 원본 이미지를 그대로 반환,
    enable=True면 YOLO 감지 + KSampler로 얼굴 디테일링.
    """

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
                "denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "guide_size": ("FLOAT", {"default": 512, "min": 0.0, "max": 2048.0, "step": 1.0}),
                "bbox_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "bbox_dilation": ("INT", {"default": 10, "min": -512, "max": 512}),
                "crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "feather": ("INT", {"default": 5, "min": 0, "max": 100}),
                "noise_mask": ("BOOLEAN", {"default": True}),
                "drop_size": ("INT", {"default": 10, "min": 1, "max": 1024}),
                "bbox_detector": ("BBOX_DETECTOR",),
                "cycle": ("INT", {"default": 1, "min": 1, "max": 10}),
            },
            "optional": {},
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "doit"
    CATEGORY = "Soya"

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def doit(self, *, enable, image, model, clip, vae, positive, negative,
             seed, steps, cfg, sampler_name, scheduler, denoise,
             guide_size, bbox_threshold, bbox_dilation, crop_factor,
             feather, noise_mask, drop_size, bbox_detector, cycle):

        use = enable.strip().lower() in ("true", "1", "yes")

        if not use:
            print("[SoyaFaceDetailerToggle] DISABLED — bypassing")
            B, H, W, C = image.shape
            return (image, torch.zeros((B, H, W), dtype=torch.float32))

        # Defensive type checks – the workflow may have wrong wiring
        if not hasattr(model, 'get_model_object'):
            raise TypeError(
                f"[SoyaFaceDetailerToggle] 'model' input received a "
                f"{type(model).__name__} object instead of a MODEL. "
                f"Check your workflow wiring — make sure a MODEL (not CLIP/VAE) "
                f"is connected to the 'model' input."
            )

        print("[SoyaFaceDetailerToggle] ENABLED — running face detailer")

        B, H, W, C = image.shape
        result_image = image.clone()
        combined_mask = torch.zeros((B, H, W), dtype=torch.float32)

        # Encode conditioning once
        positive_cond = self._encode_conditioning(clip, positive)
        negative_cond = self._encode_conditioning(clip, negative)

        for batch_idx in range(B):
            img_single = result_image[batch_idx:batch_idx + 1]  # (1, H, W, C)

            for cy in range(cycle):
                img_single, face_mask = self._detail_single(
                    img_single, model, vae, positive_cond, negative_cond,
                    seed, steps, cfg, sampler_name, scheduler, denoise,
                    guide_size, bbox_threshold, bbox_dilation, crop_factor,
                    feather, noise_mask, drop_size, bbox_detector,
                )
                seed += 1  # advance seed per cycle

            result_image[batch_idx] = img_single[0]
            combined_mask[batch_idx] = face_mask[0]

        return (result_image, combined_mask)

    # ------------------------------------------------------------------
    # Detail a single image (GPU-optimized, no PIL/scipy transfers)
    # ------------------------------------------------------------------
    def _detail_single(self, image, model, vae, positive_cond, negative_cond,
                       seed, steps, cfg, sampler_name, scheduler, denoise,
                       guide_size, bbox_threshold, bbox_dilation, crop_factor,
                       feather, noise_mask, drop_size, bbox_detector):

        _, H, W, C = image.shape

        # 1. Detect faces with YOLO (only CPU transfer needed here)
        bboxes = self._detect_faces(image, bbox_detector, bbox_threshold)

        if not bboxes:
            print("[SoyaFaceDetailerToggle] No faces detected — returning original")
            return image, torch.zeros((1, H, W), dtype=torch.float32)

        full_mask = torch.zeros((1, H, W), dtype=torch.float32)
        result = image.clone()

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            bw, bh = x2 - x1, y2 - y1

            if bw < drop_size or bh < drop_size:
                continue

            # 2. Compute crop region with crop_factor
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            ew, eh = bw * crop_factor, bh * crop_factor

            if guide_size > 0:
                scale = guide_size / max(bw, bh)
                ew = bw * max(crop_factor, scale)
                eh = bh * max(crop_factor, scale)

            cx1 = max(0, int(cx - ew / 2))
            cy1 = max(0, int(cy - eh / 2))
            cx2 = min(W, int(cx + ew / 2))
            cy2 = min(H, int(cy + eh / 2))

            crop_w, crop_h = cx2 - cx1, cy2 - cy1
            if crop_w < 1 or crop_h < 1:
                continue

            # 3. Crop and resize on GPU (no PIL)
            crop_tensor = image[0, cy1:cy2, cx1:cx2].clone()  # (crop_h, crop_w, C)

            if guide_size > 0:
                target_h = int(crop_h * guide_size / max(crop_h, crop_w))
                target_w = int(crop_w * guide_size / max(crop_h, crop_w))
                target_h = max(8, target_h // 8 * 8)
                target_w = max(8, target_w // 8 * 8)
            else:
                target_h = crop_h
                target_w = crop_w

            # GPU resize via F.interpolate (NCHW)
            crop_4d = crop_tensor.permute(2, 0, 1).unsqueeze(0)
            crop_4d = F.interpolate(crop_4d, size=(target_h, target_w), mode='bicubic', align_corners=False)
            crop_tensor = crop_4d.squeeze(0).permute(1, 2, 0).unsqueeze(0)  # (1, target_h, target_w, C)

            # 4. Build mask on GPU
            mask = torch.ones((target_h, target_w), dtype=torch.float32)

            local_x1 = max(0, x1 - cx1 + bbox_dilation)
            local_y1 = max(0, y1 - cy1 + bbox_dilation)
            local_x2 = min(crop_w, x2 - cx1 - bbox_dilation)
            local_y2 = min(crop_h, y2 - cy1 - bbox_dilation)

            sx = target_w / crop_w
            sy = target_h / crop_h
            local_x1 = int(local_x1 * sx)
            local_y1 = int(local_y1 * sy)
            local_x2 = int(local_x2 * sx)
            local_y2 = int(local_y2 * sy)

            if local_x2 > local_x1 and local_y2 > local_y1:
                inner = torch.zeros((target_h, target_w), dtype=torch.float32)
                inner[local_y1:local_y2, local_x1:local_x2] = 1.0
                mask = inner

            # Feather with GPU Gaussian blur (replaces scipy)
            if feather > 0:
                mask = self._gaussian_blur_gpu(mask, feather)

            mask_batch = mask.unsqueeze(0)  # (1, target_h, target_w)

            # 5. VAE encode → KSampler → VAE decode
            latent = vae.encode(crop_tensor[:, :, :, :3])

            latent_dict = {"samples": latent}
            if noise_mask:
                latent_dict["noise_mask"] = mask_batch

            refined_latent = self._run_ksampler(
                model, seed, steps, cfg, sampler_name, scheduler,
                positive_cond, negative_cond, latent_dict, denoise,
            )

            enhanced = vae.decode(refined_latent)  # (1, target_h, target_w, 3)

            # 6. Resize enhanced image back on GPU (no PIL)
            enhanced_4d = enhanced[0].permute(2, 0, 1).unsqueeze(0)
            enhanced_4d = F.interpolate(enhanced_4d, size=(crop_h, crop_w), mode='bicubic', align_corners=False)
            enhanced_crop = enhanced_4d.squeeze(0).permute(1, 2, 0).clamp(0, 1)

            # Resize mask back on GPU (no PIL)
            mask_4d = mask.unsqueeze(0).unsqueeze(0)
            mask_4d = F.interpolate(mask_4d, size=(crop_h, crop_w), mode='bilinear', align_corners=False)
            mask_resized = mask_4d.squeeze(0).squeeze(0)

            # 7. Paste back with feather mask (all on same device)
            alpha = mask_resized.unsqueeze(-1)  # (crop_h, crop_w, 1)
            result[0, cy1:cy2, cx1:cx2] = (
                alpha * enhanced_crop + (1 - alpha) * result[0, cy1:cy2, cx1:cx2]
            )
            full_mask[0, cy1:cy2, cx1:cx2] = torch.maximum(
                full_mask[0, cy1:cy2, cx1:cx2], mask_resized
            )

        return result, full_mask

    # ------------------------------------------------------------------
    # GPU Gaussian blur (replaces scipy.ndimage.gaussian_filter)
    # ------------------------------------------------------------------
    @staticmethod
    def _gaussian_blur_gpu(mask, sigma):
        """Gaussian blur using separable convolution — no CPU transfer."""
        ksize = int(6 * sigma + 1)
        if ksize % 2 == 0:
            ksize += 1

        x = torch.arange(ksize, dtype=torch.float32, device=mask.device) - ksize // 2
        kernel = torch.exp(-x ** 2 / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()

        mask_4d = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        pad = ksize // 2

        # Horizontal pass
        padded = F.pad(mask_4d, [pad, pad, 0, 0], mode='reflect')
        blurred = F.conv2d(padded, kernel.view(1, 1, 1, -1))

        # Vertical pass
        padded = F.pad(blurred, [0, 0, pad, pad], mode='reflect')
        blurred = F.conv2d(padded, kernel.view(1, 1, -1, 1))

        return blurred.squeeze(0).squeeze(0)

    # ------------------------------------------------------------------
    # YOLO face detection
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_faces(image, bbox_detector, threshold):
        """Detect faces using bbox_detector's YOLO model. Returns list of (x1,y1,x2,y2)."""
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

        # Sort by area (largest first)
        faces.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        return faces

    # ------------------------------------------------------------------
    # Helpers (same pattern as SoyaBatchDetailer)
    # ------------------------------------------------------------------
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
            positive, negative, latent_dict,
            denoise=denoise,
        )
        return result[0]["samples"]
