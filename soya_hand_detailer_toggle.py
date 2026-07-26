"""
Hand-specific detailer node for ComfyUI.

The old workflows reused the face detailer for hand detection.  This node keeps
the V2 workflow socket layout, but owns the hand processing path and fixes
overlapping detections, multi-cycle mask accumulation, and failure logging.
"""

import traceback

import numpy as np
import torch
import torch.nn.functional as F


class SoyaHandDetailerToggle_mdsoya:
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
                "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "bbox_threshold": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mask_expand": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 4.0, "step": 0.05}),
                "crop_expand": ("FLOAT", {"default": 1.2, "min": 1.0, "max": 6.0, "step": 0.1}),
                "upscale_factor": ("FLOAT", {"default": 1.2, "min": 1.0, "max": 4.0, "step": 0.05}),
                "feather": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1}),
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
    CATEGORY = "Soya/Detailer"

    def doit(
        self,
        *,
        enable,
        image,
        model,
        clip,
        vae,
        positive,
        negative,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        bbox_threshold,
        mask_expand,
        crop_expand,
        upscale_factor,
        feather,
        corner_roundness,
        noise_mask,
        drop_size,
        bbox_detector,
        cycle,
    ):
        shape = tuple(image.shape) if isinstance(image, torch.Tensor) else None
        try:
            if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[-1] < 3:
                raise TypeError(
                    f"image must be a BHWC tensor with at least 3 channels; "
                    f"received type={type(image).__name__}, shape={shape}"
                )

            batch_size, height, width, _ = image.shape
            if not self._is_enabled(enable):
                print(
                    f"[HandDetailer] DISABLED - bypassing: "
                    f"enable={enable!r}, shape={shape}"
                )
                empty_mask = image.new_zeros((batch_size, height, width), dtype=torch.float32)
                return image, empty_mask, torch.zeros_like(image)

            if not hasattr(model, "get_model_object"):
                raise TypeError(
                    f"model input must be a ComfyUI MODEL; received {type(model).__name__}"
                )
            if not callable(getattr(bbox_detector, "bbox_model", None)):
                raise TypeError(
                    "bbox_detector must expose a callable bbox_model; "
                    f"received {type(bbox_detector).__name__}"
                )

            print(
                f"[HandDetailer] ENABLED: shape={shape}, cycles={cycle}, "
                f"threshold={bbox_threshold}, drop_size={drop_size}, "
                f"mask_expand={mask_expand}, crop_expand={crop_expand}, "
                f"upscale_factor={upscale_factor}"
            )

            positive_cond = self._encode_conditioning(clip, positive)
            negative_cond = self._encode_conditioning(clip, negative)

            result_image = image.clone()
            combined_mask = image.new_zeros(
                (batch_size, height, width), dtype=torch.float32
            )
            combined_crop_region = image.new_zeros(
                (batch_size, height, width), dtype=torch.float32
            )
            seed_cursor = int(seed)

            for batch_idx in range(batch_size):
                image_single = result_image[batch_idx:batch_idx + 1]
                batch_mask = image.new_zeros((1, height, width), dtype=torch.float32)

                for cycle_idx in range(int(cycle)):
                    image_single, cycle_mask, crop_regions, processed_count = self._detail_single(
                        image_single,
                        model,
                        vae,
                        positive_cond,
                        negative_cond,
                        seed_cursor,
                        steps,
                        cfg,
                        sampler_name,
                        scheduler,
                        denoise,
                        bbox_threshold,
                        mask_expand,
                        crop_expand,
                        upscale_factor,
                        feather,
                        corner_roundness,
                        bool(noise_mask),
                        drop_size,
                        bbox_detector,
                    )
                    batch_mask = torch.maximum(batch_mask, cycle_mask)
                    for crop_x1, crop_y1, crop_x2, crop_y2 in crop_regions:
                        combined_crop_region[
                            batch_idx, crop_y1:crop_y2, crop_x1:crop_x2
                        ] = 1.0
                    print(
                        f"[HandDetailer] batch={batch_idx}, cycle={cycle_idx + 1}/"
                        f"{cycle}, processed_hands={processed_count}"
                    )
                    seed_cursor += max(processed_count, 1)

                result_image[batch_idx] = image_single[0]
                combined_mask[batch_idx] = batch_mask[0]

            preview_weight = (
                combined_crop_region * 0.3 + combined_mask * 0.7
            ).clamp(0.0, 1.0)
            crop_preview = result_image * preview_weight.unsqueeze(-1)
            print(
                f"[HandDetailer] COMPLETED: batches={batch_size}, "
                f"mask_pixels={(combined_mask > 0).sum().item()}"
            )
            return result_image, combined_mask, crop_preview
        except Exception as exc:
            print(
                f"[HandDetailer] FAILED: error={exc}, enable={enable!r}, "
                f"shape={shape}, detector={type(bbox_detector).__name__}, "
                f"steps={steps}, cfg={cfg}, denoise={denoise}, seed={seed}"
            )
            traceback.print_exc()
            raise

    def _detail_single(
        self,
        image,
        model,
        vae,
        positive_cond,
        negative_cond,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        bbox_threshold,
        mask_expand,
        crop_expand,
        upscale_factor,
        feather,
        corner_roundness,
        noise_mask,
        drop_size,
        bbox_detector,
    ):
        _, height, width, channels = image.shape
        bboxes = self._detect_hands(image, bbox_detector, bbox_threshold)
        if not bboxes:
            print(
                f"[HandDetailer] NO_HANDS_DETECTED: "
                f"threshold={bbox_threshold}, shape={tuple(image.shape)}"
            )
            return (
                image,
                image.new_zeros((1, height, width), dtype=torch.float32),
                [],
                0,
            )

        full_mask = image.new_zeros((1, height, width), dtype=torch.float32)
        result = image.clone()
        crop_regions = []
        processed_count = 0

        for detection_idx, (raw_x1, raw_y1, raw_x2, raw_y2) in enumerate(bboxes):
            bbox_width = raw_x2 - raw_x1
            bbox_height = raw_y2 - raw_y1
            if bbox_width < drop_size or bbox_height < drop_size:
                print(
                    f"[HandDetailer] SKIP_SMALL_HAND: bbox="
                    f"{(raw_x1, raw_y1, raw_x2, raw_y2)}, drop_size={drop_size}"
                )
                continue

            center_x = (raw_x1 + raw_x2) / 2.0
            center_y = (raw_y1 + raw_y2) / 2.0
            mask_width = bbox_width * float(mask_expand)
            mask_height = bbox_height * float(mask_expand)
            mask_x1 = max(0, int(center_x - mask_width / 2.0))
            mask_y1 = max(0, int(center_y - mask_height / 2.0))
            mask_x2 = min(width, int(center_x + mask_width / 2.0))
            mask_y2 = min(height, int(center_y + mask_height / 2.0))
            mask_width = mask_x2 - mask_x1
            mask_height = mask_y2 - mask_y1
            if mask_width < 1 or mask_height < 1:
                print(
                    f"[HandDetailer] SKIP_INVALID_MASK: bbox="
                    f"{(raw_x1, raw_y1, raw_x2, raw_y2)}"
                )
                continue

            mask_center_x = (mask_x1 + mask_x2) / 2.0
            mask_center_y = (mask_y1 + mask_y2) / 2.0
            requested_crop_width = mask_width * float(crop_expand)
            requested_crop_height = mask_height * float(crop_expand)
            crop_width = min(width, max(8, self._round_up_to_8(requested_crop_width)))
            crop_height = min(height, max(8, self._round_up_to_8(requested_crop_height)))

            crop_x1 = int(round(mask_center_x - crop_width / 2.0))
            crop_y1 = int(round(mask_center_y - crop_height / 2.0))
            crop_x1 = min(max(0, crop_x1), max(0, width - crop_width))
            crop_y1 = min(max(0, crop_y1), max(0, height - crop_height))
            crop_x2 = crop_x1 + crop_width
            crop_y2 = crop_y1 + crop_height
            actual_width = crop_x2 - crop_x1
            actual_height = crop_y2 - crop_y1
            if actual_width < 8 or actual_height < 8:
                print(
                    f"[HandDetailer] SKIP_INVALID_CROP: crop="
                    f"{(crop_x1, crop_y1, crop_x2, crop_y2)}, shape={(height, width)}"
                )
                continue

            crop_regions.append((crop_x1, crop_y1, crop_x2, crop_y2))

            # Always crop from the progressively updated result.  The old face
            # node cropped from the original image, so overlapping hands could
            # erase refinements made by an earlier detection.
            crop_tensor = result[0, crop_y1:crop_y2, crop_x1:crop_x2].clone()
            mask = self._build_mask(
                actual_height,
                actual_width,
                mask_center_x - crop_x1,
                mask_center_y - crop_y1,
                mask_width,
                mask_height,
                corner_roundness,
                crop_tensor.device,
            )
            if feather > 0:
                mask = self._gaussian_blur(mask, float(feather))

            process_width = max(
                8, self._round_up_to_8(actual_width * float(upscale_factor))
            )
            process_height = max(
                8, self._round_up_to_8(actual_height * float(upscale_factor))
            )
            crop_batch = crop_tensor.unsqueeze(0)
            if process_width != actual_width or process_height != actual_height:
                crop_batch = self._resize_bhwc(
                    crop_batch, process_height, process_width, mode="bicubic"
                )
                process_mask = self._resize_mask(mask, process_height, process_width)
            else:
                process_mask = mask

            latent = vae.encode(crop_batch[:, :, :, :3])
            latent_dict = {"samples": latent}
            if noise_mask:
                latent_dict["noise_mask"] = process_mask.unsqueeze(0)

            refined_latent = self._run_ksampler(
                model,
                int(seed) + detection_idx,
                steps,
                cfg,
                sampler_name,
                scheduler,
                positive_cond,
                negative_cond,
                latent_dict,
                denoise,
            )
            enhanced = vae.decode(refined_latent)
            if not isinstance(enhanced, torch.Tensor):
                raise TypeError(
                    "VAE decode must return a BHWC tensor; "
                    f"received type={type(enhanced).__name__}"
                )
            original_decoded_shape = tuple(enhanced.shape)
            while enhanced.ndim > 4 and enhanced.shape[0] == 1:
                enhanced = enhanced[0]
            if enhanced.ndim == 3:
                enhanced = enhanced.unsqueeze(0)
            if enhanced.ndim != 4:
                raise TypeError(
                    "VAE decode must normalize to a BHWC tensor; "
                    f"received shape={original_decoded_shape}, "
                    f"normalized_shape={tuple(enhanced.shape)}"
                )
            if tuple(enhanced.shape) != original_decoded_shape:
                print(
                    f"[HandDetailer] NORMALIZED_VAE_OUTPUT: "
                    f"from={original_decoded_shape}, to={tuple(enhanced.shape)}"
                )
            enhanced = enhanced[:, :, :, :3].clamp(0.0, 1.0)
            if enhanced.shape[1:3] != (actual_height, actual_width):
                enhanced = self._resize_bhwc(
                    enhanced, actual_height, actual_width, mode="bicubic"
                ).clamp(0.0, 1.0)

            alpha = mask.clamp(0.0, 1.0).unsqueeze(-1)
            blend_channels = min(3, channels)
            existing_crop = result[0, crop_y1:crop_y2, crop_x1:crop_x2]
            existing_crop[:, :, :blend_channels] = (
                alpha * enhanced[0, :, :, :blend_channels]
                + (1.0 - alpha) * existing_crop[:, :, :blend_channels]
            )
            full_mask[0, crop_y1:crop_y2, crop_x1:crop_x2] = torch.maximum(
                full_mask[0, crop_y1:crop_y2, crop_x1:crop_x2], mask
            )
            processed_count += 1

        if processed_count == 0:
            print(
                f"[HandDetailer] NO_VALID_HANDS_AFTER_FILTER: "
                f"detected={len(bboxes)}, drop_size={drop_size}"
            )
        return result, full_mask, crop_regions, processed_count

    @staticmethod
    def _is_enabled(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        return str(value).strip().lower() in {"true", "1", "yes", "on"}

    @staticmethod
    def _round_up_to_8(value):
        return ((int(float(value)) + 7) // 8) * 8

    @staticmethod
    def _build_mask(
        height,
        width,
        center_x,
        center_y,
        mask_width,
        mask_height,
        corner_roundness,
        device,
    ):
        half_width = max(1.0, mask_width / 2.0)
        half_height = max(1.0, mask_height / 2.0)
        yy, xx = torch.meshgrid(
            torch.arange(height, dtype=torch.float32, device=device),
            torch.arange(width, dtype=torch.float32, device=device),
            indexing="ij",
        )
        normalized_x = (xx - center_x) / half_width
        normalized_y = (yy - center_y) / half_height
        if corner_roundness <= 0.0:
            return (
                (normalized_x.abs() <= 1.0) & (normalized_y.abs() <= 1.0)
            ).float()
        exponent = 2.0 / max(float(corner_roundness), 0.01)
        return (
            normalized_x.abs().pow(exponent)
            + normalized_y.abs().pow(exponent)
            <= 1.0
        ).float()

    @staticmethod
    def _resize_bhwc(image, height, width, mode):
        nchw = image.permute(0, 3, 1, 2)
        resized = F.interpolate(
            nchw, size=(height, width), mode=mode, align_corners=False
        )
        return resized.permute(0, 2, 3, 1)

    @staticmethod
    def _resize_mask(mask, height, width):
        resized = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        return resized.squeeze(0).squeeze(0)

    @staticmethod
    def _gaussian_blur(mask, sigma):
        if sigma <= 0:
            return mask
        kernel_size = max(3, int(6 * sigma + 1))
        if kernel_size % 2 == 0:
            kernel_size += 1
        coordinates = (
            torch.arange(kernel_size, dtype=torch.float32, device=mask.device)
            - kernel_size // 2
        )
        kernel = torch.exp(-(coordinates ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        pad = kernel_size // 2
        mask_4d = mask.unsqueeze(0).unsqueeze(0)

        horizontal_mode = "reflect" if mask.shape[1] > pad else "replicate"
        padded = F.pad(mask_4d, [pad, pad, 0, 0], mode=horizontal_mode)
        blurred = F.conv2d(padded, kernel.view(1, 1, 1, -1))

        vertical_mode = "reflect" if mask.shape[0] > pad else "replicate"
        padded = F.pad(blurred, [0, 0, pad, pad], mode=vertical_mode)
        blurred = F.conv2d(padded, kernel.view(1, 1, -1, 1))
        return blurred.squeeze(0).squeeze(0).clamp(0.0, 1.0)

    @staticmethod
    def _detect_hands(image, bbox_detector, threshold):
        yolo = getattr(bbox_detector, "bbox_model", None)
        if not callable(yolo):
            raise TypeError(
                "bbox_detector.bbox_model is missing or not callable: "
                f"detector={type(bbox_detector).__name__}"
            )
        image_array = (
            image[0, :, :, :3]
            .detach()
            .clamp(0.0, 1.0)
            .cpu()
            .numpy()
            * 255.0
        ).round().astype(np.uint8)
        predictions = yolo(image_array, verbose=False)
        height, width = image.shape[1:3]
        hands = []
        for prediction in predictions:
            boxes = getattr(prediction, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                confidence = float(box.conf[0])
                if confidence < float(threshold):
                    continue
                x1, y1, x2, y2 = box.xyxy[0].detach().cpu().tolist()
                x1 = min(width, max(0, int(x1)))
                y1 = min(height, max(0, int(y1)))
                x2 = min(width, max(0, int(x2)))
                y2 = min(height, max(0, int(y2)))
                if x2 <= x1 or y2 <= y1:
                    print(
                        f"[HandDetailer] SKIP_INVALID_DETECTION: "
                        f"bbox={(x1, y1, x2, y2)}, confidence={confidence:.4f}"
                    )
                    continue
                hands.append((x1, y1, x2, y2))
        hands.sort(
            key=lambda item: (item[2] - item[0]) * (item[3] - item[1]),
            reverse=True,
        )
        print(
            f"[HandDetailer] DETECTION: accepted={len(hands)}, "
            f"threshold={threshold}"
        )
        return hands

    @staticmethod
    def _encode_conditioning(clip, text):
        from nodes import CLIPTextEncode

        return CLIPTextEncode().encode(clip, text)[0]

    @staticmethod
    def _run_ksampler(
        model,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_dict,
        denoise,
    ):
        from nodes import common_ksampler

        result = common_ksampler(
            model,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_dict,
            denoise=denoise,
        )
        return result[0]["samples"]
