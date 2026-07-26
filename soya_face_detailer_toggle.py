"""Face detailer toggle without an Impact Pack detailer dependency."""

import traceback

import numpy as np
import torch
import torch.nn.functional as F


class SoyaFaceDetailerToggle_mdsoya:
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
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
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
        guide_size,
        bbox_threshold,
        bbox_dilation,
        crop_factor,
        feather,
        noise_mask,
        drop_size,
        bbox_detector,
        cycle,
    ):
        shape = tuple(image.shape) if isinstance(image, torch.Tensor) else None
        try:
            if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[-1] < 3:
                raise TypeError(
                    "image must be a BHWC tensor with at least 3 channels; "
                    f"received type={type(image).__name__}, shape={shape}"
                )

            batch_size, height, width, _ = image.shape
            if not self._is_enabled(enable):
                print(
                    f"[FaceDetailer] DISABLED - bypassing: "
                    f"enable={enable!r}, shape={shape}"
                )
                empty_mask = image.new_zeros(
                    (batch_size, height, width), dtype=torch.float32
                )
                return image, empty_mask

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
                f"[FaceDetailer] ENABLED: shape={shape}, cycles={cycle}, "
                f"threshold={bbox_threshold}, drop_size={drop_size}, "
                f"guide_size={guide_size}, bbox_dilation={bbox_dilation}, "
                f"crop_factor={crop_factor}"
            )

            positive_cond = self._encode_conditioning(clip, positive)
            negative_cond = self._encode_conditioning(clip, negative)
            result_image = image.clone()
            combined_mask = image.new_zeros(
                (batch_size, height, width), dtype=torch.float32
            )
            seed_cursor = int(seed)

            for batch_idx in range(batch_size):
                image_single = result_image[batch_idx:batch_idx + 1]
                batch_mask = image.new_zeros((1, height, width), dtype=torch.float32)

                for cycle_idx in range(int(cycle)):
                    image_single, cycle_mask, processed_count = self._detail_single(
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
                        guide_size,
                        bbox_threshold,
                        bbox_dilation,
                        crop_factor,
                        feather,
                        self._is_enabled(noise_mask),
                        drop_size,
                        bbox_detector,
                    )
                    batch_mask = torch.maximum(batch_mask, cycle_mask)
                    print(
                        f"[FaceDetailer] batch={batch_idx}, cycle={cycle_idx + 1}/"
                        f"{cycle}, processed_faces={processed_count}"
                    )
                    seed_cursor += max(processed_count, 1)

                result_image[batch_idx] = image_single[0]
                combined_mask[batch_idx] = batch_mask[0]

            print(
                f"[FaceDetailer] COMPLETED: batches={batch_size}, "
                f"mask_pixels={(combined_mask > 0).sum().item()}"
            )
            return result_image, combined_mask
        except Exception as exc:
            print(
                f"[FaceDetailer] FAILED: error={exc}, enable={enable!r}, "
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
        guide_size,
        bbox_threshold,
        bbox_dilation,
        crop_factor,
        feather,
        noise_mask,
        drop_size,
        bbox_detector,
    ):
        _, height, width, channels = image.shape
        bboxes = self._detect_faces(image, bbox_detector, bbox_threshold)
        if not bboxes:
            print(
                f"[FaceDetailer] NO_FACES_DETECTED: "
                f"threshold={bbox_threshold}, shape={tuple(image.shape)}"
            )
            return image, image.new_zeros((1, height, width), dtype=torch.float32), 0

        full_mask = image.new_zeros((1, height, width), dtype=torch.float32)
        result = image.clone()
        processed_count = 0

        for detection_idx, (x1, y1, x2, y2) in enumerate(bboxes):
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            if bbox_width < int(drop_size) or bbox_height < int(drop_size):
                print(
                    f"[FaceDetailer] SKIP_SMALL_FACE: "
                    f"bbox={(x1, y1, x2, y2)}, drop_size={drop_size}"
                )
                continue

            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            crop_width_requested = bbox_width * float(crop_factor)
            crop_height_requested = bbox_height * float(crop_factor)

            if float(guide_size) > 0:
                guide_scale = float(guide_size) / max(bbox_width, bbox_height)
                crop_width_requested = bbox_width * max(float(crop_factor), guide_scale)
                crop_height_requested = bbox_height * max(float(crop_factor), guide_scale)

            crop_x1 = max(0, int(center_x - crop_width_requested / 2.0))
            crop_y1 = max(0, int(center_y - crop_height_requested / 2.0))
            crop_x2 = min(width, int(center_x + crop_width_requested / 2.0))
            crop_y2 = min(height, int(center_y + crop_height_requested / 2.0))
            crop_width = crop_x2 - crop_x1
            crop_height = crop_y2 - crop_y1
            if crop_width < 1 or crop_height < 1:
                print(
                    f"[FaceDetailer] SKIP_INVALID_CROP: "
                    f"crop={(crop_x1, crop_y1, crop_x2, crop_y2)}, "
                    f"shape={(height, width)}"
                )
                continue

            # Crop from the progressively updated result so overlapping faces do
            # not erase refinements produced by an earlier detection.
            crop_tensor = result[0, crop_y1:crop_y2, crop_x1:crop_x2].clone()

            dilation = float(bbox_dilation)
            mask_x1 = max(0, int(np.floor(x1 - crop_x1 - dilation)))
            mask_y1 = max(0, int(np.floor(y1 - crop_y1 - dilation)))
            mask_x2 = min(crop_width, int(np.ceil(x2 - crop_x1 + dilation)))
            mask_y2 = min(crop_height, int(np.ceil(y2 - crop_y1 + dilation)))
            if mask_x2 <= mask_x1 or mask_y2 <= mask_y1:
                print(
                    f"[FaceDetailer] SKIP_INVALID_MASK: "
                    f"bbox={(x1, y1, x2, y2)}, dilation={bbox_dilation}, "
                    f"crop={(crop_x1, crop_y1, crop_x2, crop_y2)}"
                )
                continue

            mask = crop_tensor.new_zeros((crop_height, crop_width), dtype=torch.float32)
            mask[mask_y1:mask_y2, mask_x1:mask_x2] = 1.0
            if float(feather) > 0:
                mask = self._gaussian_blur(mask, float(feather))

            if float(guide_size) > 0:
                process_scale = float(guide_size) / max(crop_height, crop_width)
                process_height = max(8, int(crop_height * process_scale) // 8 * 8)
                process_width = max(8, int(crop_width * process_scale) // 8 * 8)
            else:
                process_height = crop_height
                process_width = crop_width

            crop_batch = crop_tensor.unsqueeze(0)
            if (process_height, process_width) != (crop_height, crop_width):
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
            enhanced = self._normalize_decoded_image(enhanced)
            enhanced = enhanced[:, :, :, :3].clamp(0.0, 1.0)
            if enhanced.shape[1:3] != (crop_height, crop_width):
                enhanced = self._resize_bhwc(
                    enhanced, crop_height, crop_width, mode="bicubic"
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
                f"[FaceDetailer] NO_VALID_FACES_AFTER_FILTER: "
                f"detected={len(bboxes)}, drop_size={drop_size}, "
                f"bbox_dilation={bbox_dilation}"
            )
        return result, full_mask, processed_count

    @staticmethod
    def _is_enabled(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        return str(value).strip().lower() in {"true", "1", "yes", "on"}

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
    def _normalize_decoded_image(enhanced):
        if not isinstance(enhanced, torch.Tensor):
            raise TypeError(
                "VAE decode must return a BHWC tensor; "
                f"received type={type(enhanced).__name__}"
            )
        original_shape = tuple(enhanced.shape)
        while enhanced.ndim > 4 and enhanced.shape[0] == 1:
            enhanced = enhanced[0]
        if enhanced.ndim == 3:
            enhanced = enhanced.unsqueeze(0)
        if enhanced.ndim != 4 or enhanced.shape[-1] < 3:
            raise TypeError(
                "VAE decode must normalize to a BHWC tensor with at least 3 channels; "
                f"received shape={original_shape}, normalized_shape={tuple(enhanced.shape)}"
            )
        if tuple(enhanced.shape) != original_shape:
            print(
                f"[FaceDetailer] NORMALIZED_VAE_OUTPUT: "
                f"from={original_shape}, to={tuple(enhanced.shape)}"
            )
        return enhanced

    @staticmethod
    def _detect_faces(image, bbox_detector, threshold):
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
        faces = []
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
                        f"[FaceDetailer] SKIP_INVALID_DETECTION: "
                        f"bbox={(x1, y1, x2, y2)}, confidence={confidence:.4f}"
                    )
                    continue
                faces.append((x1, y1, x2, y2))
        faces.sort(
            key=lambda item: (item[2] - item[0]) * (item[3] - item[1]),
            reverse=True,
        )
        print(
            f"[FaceDetailer] DETECTION: accepted={len(faces)}, "
            f"threshold={threshold}"
        )
        return faces

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
