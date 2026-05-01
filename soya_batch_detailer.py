"""
SoyaBatchDetailer – batch face detailing using KSampler.

Processes SEGS from SoyaProcessCollector, grouping Level 1/2 faces into
batch KSampler calls (max 2 per batch) while Level 0 faces are processed
individually. Uses Collector's batch_groups context to determine grouping.
"""

import gc
import re
import time
import torch
import numpy as np
import torch.nn.functional as F
from collections import namedtuple

# Impact Pack SEG namedtuple compatibility (inlined from soya_process_collector)
SEG = namedtuple("SEG", [
    'cropped_image', 'cropped_mask', 'confidence',
    'crop_region', 'bbox', 'label', 'control_net_wrapper',
], defaults=[None])


class SoyaBatchDetailer_mdsoya:
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
                "noise_mask": ("BOOLEAN", {"default": True}),
                "force_inpaint": ("BOOLEAN", {"default": True}),
                "edge_blur": ("INT", {"default": 5, "min": 0, "max": 100}),
                "edge_feather": ("INT", {"default": 0, "min": 0, "max": 100}),
                "iterations": ("INT", {"default": 1, "min": 1, "max": 10}),
                "color_adjustment": ("COLOR_ADJUST", {"forceInput": True}),
            },
            "optional": {
                "post_color_adjustment": ("COLOR_ADJUST", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "SEGS", "STRING", "IMAGE", "MASK", "IMAGE", "IMAGE", "MASK", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "segs", "info", "before_color_adjusted", "color_mask", "color_adjusted", "enhanced", "post_color_mask", "post_color_adjusted", "eyebrow_overlay", "eyebrow_crop", "eyebrow_blur_result")
    FUNCTION = "detail"
    CATEGORY = "Soya/Detailer"

    def detail(self, image, segs, prompts, context, model, clip, vae,
               negative_prompt, seed, steps, cfg, sampler_name, scheduler,
               denoise, noise_mask, force_inpaint, edge_blur, edge_feather, iterations,
               color_adjustment=None, post_color_adjustment=None):
        t0 = time.time()
        info_lines = ["[Soya Batch Detailer]"]
        info_lines.append(f"  noise_mask={noise_mask} | KSampler에 마스크 전달. 마스크=1 영역만 디노이징, 마스크=0 영역은 원본 유지")
        info_lines.append(f"  force_inpaint={force_inpaint} | 인페인팅 모드 강제 적용")
        info_lines.append(f"  edge_blur={edge_blur} | 마스크 가장자리 부드러움 (가우시안 sigma). 높을수록 전환 곡선이 완만해짐")
        info_lines.append(f"  edge_feather={edge_feather} | 마스크 경계에서 안쪽으로 gradient 폭(px). 0=순수 블러만, 높일수록 가장자리 두꺼워짐")

        # Unpack color adjustment config
        if color_adjustment is None:
            color_adjustment = {
                "enabled": False, "temperature": 0, "tint": 0,
                "saturation": 0, "vibrance": 0, "brightness": 0,
                "contrast": 0, "gamma": 1.0, "mask_sigma": 0.4,
            }
        color_adjust_enabled = color_adjustment.get("enabled", False)
        color_temperature = color_adjustment.get("temperature", 0)
        color_tint = color_adjustment.get("tint", 0)
        color_saturation = color_adjustment.get("saturation", 0)
        color_vibrance = color_adjustment.get("vibrance", 0)
        color_brightness = color_adjustment.get("brightness", 0)
        color_contrast = color_adjustment.get("contrast", 0)
        color_gamma = color_adjustment.get("gamma", 1.0)
        color_mask_sigma = color_adjustment.get("mask_sigma", 0.4)

        _, segs_list = segs

        # Empty SEGS → return original
        if not segs_list:
            info_lines.append("No SEGS to process.")
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
            return (image, segs, "\n".join(info_lines), image, empty_mask, image, image, empty_mask, image, image, image, image)

        # Step 1: Parse prompts → {"1": "prompt1", "2": "prompt2", ...}
        prompt_map = self._parse_prompts(prompts)
        info_lines.append(f"Parsed {len(prompt_map)} face prompts")

        # Debug: show prompt map and SEGS labels
        for i, seg in enumerate(segs_list):
            prompt_for_seg = prompt_map.get(seg.label, "")
            info_lines.append(
                f"  SEGS[{i}]: label={seg.label}, prompt='{prompt_for_seg[:60]}...', "
                f"crop={seg.crop_region}, img_shape={tuple(seg.cropped_image.shape)}"
            )

        # Step 2: Get batch groups from context
        batch_groups = context.get("batch_groups", [[i] for i in range(len(segs_list))])
        info_lines.append(f"Batch groups: {len(batch_groups)}")
        for gi, group in enumerate(batch_groups):
            labels = [segs_list[idx].label for idx in group]
            mode = "batch" if len(group) > 1 else "single"
            info_lines.append(f"  Group {gi + 1}: {mode} ({len(group)}) labels={labels}")

        info_lines.append(f"Iterations: {iterations}")

        # Encode shared negative prompt
        _t = time.time()
        negative_cond = self._encode_conditioning(clip, negative_prompt)
        print(f"[BENCH] negative cond encode: {time.time() - _t:.3f}s")

        # Step 3: Cache positive conditionings (prompts don't change between iterations)
        _t = time.time()
        positive_conds_cache = []
        for seg in segs_list:
            prompt_text = prompt_map.get(seg.label, "")
            cond = self._encode_conditioning(clip, prompt_text)
            positive_conds_cache.append(cond)
        print(f"[BENCH] positive cond encode ({len(segs_list)} faces): {time.time() - _t:.3f}s")

        # Step 3.5: Build gradient masks and apply color adjustment BEFORE KSampler
        _t = time.time()
        crop_grad_masks = []

        for i, seg in enumerate(segs_list):
            enh = seg.cropped_image
            if enh.shape[-1] == 4:
                enh = enh[:, :, :, :3]
            crop_h, crop_w = enh.shape[1], enh.shape[2]

            # Get segment mask (defines shape to follow)
            m = seg.cropped_mask
            if isinstance(m, np.ndarray):
                m = torch.from_numpy(m).float()
            elif isinstance(m, torch.Tensor):
                m = m.float()
            else:
                m = torch.ones(crop_h, crop_w)
            if m.max() > 1.0:
                m = m / 255.0

            if m.shape[0] != crop_h or m.shape[1] != crop_w:
                m = self._resize_2d_gpu(m, crop_h, crop_w)

            # Distance from boundary within the mask shape (GPU)
            binary = (m > 0.5)
            dist = self._distance_transform_gpu(binary)
            max_dist = dist.max()
            if max_dist > 0:
                norm_dist = dist / max_dist
                grad = torch.exp(-((1.0 - norm_dist) ** 2) / (2.0 * color_mask_sigma ** 2))
                grad = grad * binary.float()
            else:
                grad = torch.ones_like(m)

            crop_grad_masks.append(grad)

        # Stack masks (pad to max size if crops differ)
        max_h = max(m.shape[0] for m in crop_grad_masks)
        max_w = max(m.shape[1] for m in crop_grad_masks)
        padded_masks = []
        for m in crop_grad_masks:
            h, w = m.shape
            padded = torch.zeros(max_h, max_w, dtype=torch.float32)
            padded[:h, :w] = m
            padded_masks.append(padded)
        color_mask_batch = torch.stack(padded_masks, dim=0)  # (num_faces, max_h, max_w) MASK

        # Save original crops for before-preview, then apply color adjustment
        original_crops_for_preview = [seg.cropped_image.clone() for seg in segs_list]
        print(f"[BENCH] gradient masks (distance transform x{len(segs_list)}): {time.time() - _t:.3f}s")

        if color_adjust_enabled:
            has_color_change = (
                color_temperature != 0 or color_tint != 0 or
                color_saturation != 0 or color_vibrance != 0 or
                color_brightness != 0 or color_contrast != 0 or
                color_gamma != 1.0
            )
            if has_color_change:
                info_lines.append(f"Color adjustment (pre-KSampler): temp={color_temperature} tint={color_tint} "
                                  f"sat={color_saturation} vib={color_vibrance} "
                                  f"bri={color_brightness} con={color_contrast} gamma={color_gamma} sigma={color_mask_sigma}")
                for i, seg in enumerate(segs_list):
                    enh = seg.cropped_image
                    if enh.shape[-1] == 4:
                        enh = enh[:, :, :, :3]

                    result = self._apply_color_adjust_gpu(
                        enh, crop_grad_masks[i],
                        color_temperature, color_tint, color_saturation,
                        color_vibrance, color_brightness, color_contrast, color_gamma,
                    )

                    # Update seg with color-adjusted crop (will be VAE-encoded for KSampler)
                    segs_list[i] = SEG(
                        cropped_image=result,
                        cropped_mask=seg.cropped_mask,
                        confidence=seg.confidence,
                        crop_region=seg.crop_region,
                        bbox=seg.bbox,
                        label=seg.label,
                    )
            else:
                info_lines.append("Color adjustment enabled but all values are 0")
        else:
            info_lines.append("Color adjustment: disabled")

        # Build previews of original (before) and color-adjusted (after KSampler input) crops
        _t = time.time()
        before_color_adjusted_preview = self._build_enhanced_preview_from_crops(
            original_crops_for_preview)
        color_adjusted_preview = self._build_enhanced_preview_from_crops(
            [seg.cropped_image for seg in segs_list])
        print(f"[BENCH] pre-ksampler previews: {time.time() - _t:.3f}s")

        # Step 4: VAE encode (color-adjusted) cropped images → latents (batched)
        _t = time.time()
        import comfy.model_management as _mm
        # Load VAE alongside model — both fit in VRAM (4.3GB free, VAE ~200MB).
        # This prevents vae.encode() from swapping out the SD model.
        _t_pre = time.time()
        _mm.load_models_gpu([model, vae.patcher])
        print(f"[BENCH] VAE encode preload (model+VAE): {time.time() - _t_pre:.3f}s")
        _encode_groups = {}
        for i, seg in enumerate(segs_list):
            key = (seg.cropped_image.shape[1], seg.cropped_image.shape[2])
            _encode_groups.setdefault(key, []).append(i)
        working_latents = [None] * len(segs_list)
        for indices in _encode_groups.values():
            batch = torch.cat([segs_list[j].cropped_image[:, :, :, :3] for j in indices], dim=0)
            latents = vae.encode(batch)
            for k, idx in enumerate(indices):
                working_latents[idx] = latents[k:k + 1]
        print(f"[BENCH] VAE encode: {time.time() - _t:.3f}s")

        # Step 5: Cache noise masks per group + store for visualization
        _t = time.time()
        group_masks = []
        individual_masks_viz = [None] * len(segs_list)
        original_masks_viz = [None] * len(segs_list)

        for gi, group in enumerate(batch_groups):
            mask_tensor = None
            if noise_mask:
                masks = []
                for idx in group:
                    m = segs_list[idx].cropped_mask
                    if isinstance(m, np.ndarray):
                        m = torch.from_numpy(m).float()
                    elif isinstance(m, torch.Tensor):
                        m = m.float()
                    else:
                        m = torch.ones(segs_list[idx].cropped_image.shape[1],
                                       segs_list[idx].cropped_image.shape[2])
                    if m.max() > 1.0:
                        m = m / 255.0
                    # Save original mask before any processing
                    original_masks_viz[idx] = m.clone()
                    # Feather: create inward gradient from mask boundary (GPU)
                    if edge_feather > 0:
                        m_binary = (m > 0.5)
                        dist_inward = self._distance_transform_gpu(m_binary)
                        m = torch.clamp(dist_inward / max(edge_feather, 1), 0.0, 1.0)
                    # Blur: smooth the transition (GPU)
                    if edge_blur > 0:
                        m = self._gaussian_blur_gpu(m, edge_blur)
                    masks.append(m.unsqueeze(0))
                    individual_masks_viz[idx] = m.clone()
                mask_tensor = torch.cat(masks, dim=0)
            group_masks.append(mask_tensor)
        print(f"[BENCH] noise masks: {time.time() - _t:.3f}s")

        # Step 6: Iteration loop – KSampler only, no encode/decode
        _t = time.time()
        for iteration in range(iterations):
            iter_t0 = time.time()

            for gi, group in enumerate(batch_groups):
                latents = [working_latents[idx] for idx in group]
                batched_latent = torch.cat(latents, dim=0)

                conds = [positive_conds_cache[idx] for idx in group]
                batched_positive = self._batch_conditionings(conds)

                mask_tensor = group_masks[gi]

                refined_latent = self._run_ksampler(
                    model, seed, steps, cfg, sampler_name, scheduler,
                    batched_positive, negative_cond, batched_latent, denoise, mask_tensor,
                )

                for i, seg_idx in enumerate(group):
                    working_latents[seg_idx] = refined_latent[i:i + 1]

                print(f"[BatchDetailer] iter={iteration} group={gi} "
                      f"labels={[segs_list[idx].label for idx in group]} "
                      f"latent_shape={batched_latent.shape}")

            iter_elapsed = time.time() - iter_t0
            info_lines.append(f"  Iteration {iteration + 1}/{iterations}: {iter_elapsed:.2f}\ucd08")

        info_lines.append(f"Sampling completed for {len(segs_list)} faces")
        print(f"[BENCH] KSampler ({iterations} iter, {len(segs_list)} faces): {time.time() - _t:.3f}s")

        # Step 7: VAE decode final latents → images (batched)
        _t = time.time()
        # Reload VAE alongside model (KSampler may have offloaded VAE)
        _t_pre = time.time()
        _mm.load_models_gpu([model, vae.patcher])
        print(f"[BENCH] VAE decode preload (model+VAE): {time.time() - _t_pre:.3f}s")
        _decode_groups = {}
        for i in range(len(segs_list)):
            wl = working_latents[i]
            key = (wl.shape[2], wl.shape[3])
            _decode_groups.setdefault(key, []).append(i)
        decoded = [None] * len(segs_list)
        for indices in _decode_groups.values():
            batch = torch.cat([working_latents[j] for j in indices], dim=0)
            images = vae.decode(batch)
            for k, idx in enumerate(indices):
                decoded[idx] = images[k:k + 1]
        print(f"[BENCH] VAE decode: {time.time() - _t:.3f}s")
        updated_segs = []
        for i, seg in enumerate(segs_list):
            new_seg = SEG(
                cropped_image=decoded[i],
                cropped_mask=seg.cropped_mask,
                confidence=seg.confidence,
                crop_region=seg.crop_region,
                bbox=seg.bbox,
                label=seg.label,
            )
            updated_segs.append(new_seg)

        # Build enhanced preview (KSampler output)
        print(f"[BENCH] VAE decode: {time.time() - _t:.3f}s")
        _t = time.time()
        enhanced_preview = self._build_enhanced_preview(updated_segs)
        print(f"[BENCH] enhanced preview: {time.time() - _t:.3f}s")

        # Step 7.5: Post-enhancement color adjustment (after KSampler, before eyebrow)
        _t = time.time()
        if post_color_adjustment is None:
            post_color_adjustment = {
                "enabled": False, "temperature": 0, "tint": 0,
                "saturation": 0, "vibrance": 0, "brightness": 0,
                "contrast": 0, "gamma": 1.0, "mask_sigma": 0.4,
            }
        post_ca_enabled = post_color_adjustment.get("enabled", False)
        post_ca_temperature = post_color_adjustment.get("temperature", 0)
        post_ca_tint = post_color_adjustment.get("tint", 0)
        post_ca_saturation = post_color_adjustment.get("saturation", 0)
        post_ca_vibrance = post_color_adjustment.get("vibrance", 0)
        post_ca_brightness = post_color_adjustment.get("brightness", 0)
        post_ca_contrast = post_color_adjustment.get("contrast", 0)
        post_ca_gamma = post_color_adjustment.get("gamma", 1.0)
        post_ca_mask_sigma = post_color_adjustment.get("mask_sigma", 0.4)

        post_crop_grad_masks = []

        if post_ca_enabled:
            has_post_change = (
                post_ca_temperature != 0 or post_ca_tint != 0 or
                post_ca_saturation != 0 or post_ca_vibrance != 0 or
                post_ca_brightness != 0 or post_ca_contrast != 0 or
                post_ca_gamma != 1.0
            )

            # Build gradient masks (always, for output visualization)
            for i, seg in enumerate(updated_segs):
                enh = seg.cropped_image
                if enh.shape[-1] == 4:
                    enh = enh[:, :, :, :3]
                crop_h, crop_w = enh.shape[1], enh.shape[2]

                m = seg.cropped_mask
                if isinstance(m, np.ndarray):
                    m = torch.from_numpy(m).float()
                elif isinstance(m, torch.Tensor):
                    m = m.float()
                else:
                    m = torch.ones(crop_h, crop_w)
                if m.max() > 1.0:
                    m = m / 255.0

                if m.shape[0] != crop_h or m.shape[1] != crop_w:
                    m = self._resize_2d_gpu(m, crop_h, crop_w)

                binary = (m > 0.5)
                dist = self._distance_transform_gpu(binary)
                max_dist = dist.max()
                if max_dist > 0:
                    norm_dist = dist / max_dist
                    crop_mask = torch.exp(-((1.0 - norm_dist) ** 2) / (2.0 * post_ca_mask_sigma ** 2))
                    crop_mask = crop_mask * binary.float()
                else:
                    crop_mask = torch.ones_like(m)

                post_crop_grad_masks.append(crop_mask)

            # Apply color adjustment only when values are non-zero
            if has_post_change:
                info_lines.append(f"Post-enhancement color adjust: temp={post_ca_temperature} tint={post_ca_tint} "
                                  f"sat={post_ca_saturation} vib={post_ca_vibrance} "
                                  f"bri={post_ca_brightness} con={post_ca_contrast} gamma={post_ca_gamma} sigma={post_ca_mask_sigma}")
                for i, seg in enumerate(updated_segs):
                    enh = seg.cropped_image
                    if enh.shape[-1] == 4:
                        enh = enh[:, :, :, :3]

                    result = self._apply_color_adjust_gpu(
                        enh, post_crop_grad_masks[i],
                        post_ca_temperature, post_ca_tint, post_ca_saturation,
                        post_ca_vibrance, post_ca_brightness, post_ca_contrast, post_ca_gamma,
                    )

                    updated_segs[i] = SEG(
                        cropped_image=result,
                        cropped_mask=seg.cropped_mask,
                        confidence=seg.confidence,
                        crop_region=seg.crop_region,
                        bbox=seg.bbox,
                        label=seg.label,
                    )
            else:
                info_lines.append("Post-enhancement color adjust enabled but all values are 0")
        else:
            info_lines.append("Post-enhancement color adjust: disabled")
        print(f"[BENCH] post-color adjustment: {time.time() - _t:.3f}s")

        # Build post color mask batch (pad to max size, same as pre-enhancement masks)
        _t = time.time()
        if post_crop_grad_masks:
            post_max_h = max(m.shape[0] for m in post_crop_grad_masks)
            post_max_w = max(m.shape[1] for m in post_crop_grad_masks)
            post_padded_masks = []
            for m in post_crop_grad_masks:
                h, w = m.shape
                padded = torch.zeros(post_max_h, post_max_w, dtype=torch.float32)
                padded[:h, :w] = m
                post_padded_masks.append(padded)
            post_color_mask_batch = torch.stack(post_padded_masks, dim=0)
        else:
            post_color_mask_batch = torch.zeros((1, 64, 64), dtype=torch.float32)

        post_color_adjusted_preview = self._build_enhanced_preview_from_crops(
            [seg.cropped_image for seg in updated_segs])
        print(f"[BENCH] post previews + color masks: {time.time() - _t:.3f}s")

        # Step 7.6: Build eyebrow mask overlay on enhanced crops
        _t = time.time()
        eyebrow_overlay = self._build_eyebrow_overlay_preview(updated_segs, context)

        # Step 7.7: Build tightly cropped eyebrow regions
        eyebrow_crop = self._build_eyebrow_crop_preview(updated_segs, context)
        print(f"[BENCH] eyebrow previews (overlay+crop): {time.time() - _t:.3f}s")

        # Read eyebrow patch settings from context (set via web settings page)
        eyebrow_restore = context.get("eyebrow_restore", False)
        eyebrow_restore_mode = context.get("eyebrow_restore_mode", "hs_preserve")
        eyebrow_blur = context.get("eyebrow_blur", 0)
        eyebrow_hs_percentile = context.get("eyebrow_hs_percentile", 0.0)
        eyebrow_v_range = context.get("eyebrow_v_range", 1.0)
        eyebrow_opacity = context.get("eyebrow_opacity", 0.0)

        # Step 7.8: Restore eyebrow color: blur original eyebrow → extract H,S + V from enhanced
        _t = time.time()
        if eyebrow_restore:
            self._apply_eyebrow_hsv_restore(updated_segs, context, eyebrow_restore_mode, eyebrow_blur, eyebrow_hs_percentile, eyebrow_v_range)

        # Step 7.9: Build blur result preview (blurred eyebrow crop)
        eyebrow_blur_result = self._build_eyebrow_blur_preview(updated_segs, context, eyebrow_blur)
        print(f"[BENCH] eyebrow restore + blur preview: {time.time() - _t:.3f}s")

        # Step 8: Paste-back (hole-punch + fill + eyebrow opacity)
        _t = time.time()
        result_image = self._paste_back(image, updated_segs, context, eyebrow_opacity)
        print(f"[BENCH] paste-back: {time.time() - _t:.3f}s")

        elapsed = time.time() - t0
        info_lines.append(f"Total: {elapsed:.2f}\ucd08")

        # Save detailer results to config for web UI
        _t = time.time()
        self._save_detailer_result(updated_segs, batch_groups, segs_list, elapsed, iterations,
                                   individual_masks_viz, original_masks_viz,
                                   edge_blur, edge_feather)
        print(f"[BENCH] save result to config: {time.time() - _t:.3f}s")

        return (result_image, (segs[0], updated_segs), "\n".join(info_lines), before_color_adjusted_preview, color_mask_batch, color_adjusted_preview, enhanced_preview, post_color_mask_batch, post_color_adjusted_preview, eyebrow_overlay, eyebrow_crop, eyebrow_blur_result)

    # ── Helper methods ───────────────────────────────────────────

    @staticmethod
    def _rgb_to_hsv(rgb):
        """(H, W, 3) float32 RGB [0,1] → HSV [0,1]."""
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.maximum(np.maximum(r, g), b)
        minc = np.minimum(np.minimum(r, g), b)
        delta = maxc - minc

        # Hue [0,1]
        h = np.zeros_like(maxc)
        mask = delta > 0
        m = mask & (maxc == r)
        h[m] = ((g[m] - b[m]) / delta[m]) % 6.0
        m = mask & (maxc == g)
        h[m] = (b[m] - r[m]) / delta[m] + 2.0
        m = mask & (maxc == b)
        h[m] = (r[m] - g[m]) / delta[m] + 4.0
        h = h / 6.0
        h[h < 0] += 1.0

        # Saturation [0,1]
        s = np.where(maxc > 0, delta / maxc, 0.0)

        # Value [0,1]
        v = maxc

        return np.stack([h, s, v], axis=-1)

    @staticmethod
    def _hsv_to_rgb(hsv):
        """(H, W, 3) float32 HSV [0,1] → RGB [0,1]."""
        h = hsv[..., 0] * 6.0
        s = hsv[..., 1]
        v = hsv[..., 2]

        i = np.floor(h).astype(np.int32) % 6
        f = h - np.floor(h)
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))

        r = np.choose(i, [v, q, p, p, t, v])
        g = np.choose(i, [t, v, v, q, p, p])
        b = np.choose(i, [p, p, t, v, v, q])

        return np.stack([r, g, b], axis=-1).clip(0, 1)

    @staticmethod
    def _apply_eyebrow_hsv_restore(updated_segs, context, mode="hs_preserve", eyebrow_blur=0, hs_percentile=0.0, v_range=1.0):
        """Restore eyebrow color using one of two modes:
        - 'hsv_restore': Original mode – blur, percentile filter, V compression, then merge.
        - 'hs_preserve': Simple mode – H,S from original eyebrow, V from enhanced (1:1 pixel map).
        """
        # SEG is already defined at module level

        kept_faces = context.get("kept_faces", [])

        for i, seg in enumerate(updated_segs):
            if i >= len(kept_faces):
                break
            kf = kept_faces[i]
            eb_mask = kf.get("eyebrow_mask")
            if eb_mask is None:
                continue

            enhanced = seg.cropped_image  # (1, H, W, 3)
            original_crop = kf["image"]   # (1, H_orig, W_orig, 3)
            enh_h, enh_w = enhanced.shape[1], enhanced.shape[2]

            # Resize original to match enhanced (GPU, then to numpy for HSV ops)
            if original_crop.shape[1] != enh_h or original_crop.shape[2] != enh_w:
                original_resized = F.interpolate(
                    original_crop.permute(0, 3, 1, 2),
                    size=(enh_h, enh_w), mode='bilinear', align_corners=False,
                ).permute(0, 2, 3, 1)[0].cpu().numpy()
            else:
                original_resized = original_crop[0].cpu().numpy()

            # Resize mask to match enhanced (GPU)
            if eb_mask.shape[0] != enh_h or eb_mask.shape[1] != enh_w:
                eb_mask_t = torch.from_numpy(eb_mask).float() if isinstance(eb_mask, np.ndarray) else eb_mask.float()
                eb_mask = SoyaBatchDetailer_mdsoya._resize_2d_gpu(eb_mask_t, enh_h, enh_w).cpu().numpy()

            threshold = kf.get("eyebrow_threshold", 0.5)
            binary = (eb_mask > threshold).astype(np.float32)

            enhanced_np = enhanced[0].cpu().numpy()

            if mode == "hs_preserve":
                # Simple 1:1 mode: H,S from original, V from enhanced, no extra processing
                orig_hsv = SoyaBatchDetailer_mdsoya._rgb_to_hsv(original_resized)
                enh_hsv = SoyaBatchDetailer_mdsoya._rgb_to_hsv(enhanced_np)

                merged_hsv = enh_hsv.copy()
                merged_hsv[..., 0] = orig_hsv[..., 0]  # H from original
                merged_hsv[..., 1] = orig_hsv[..., 1]  # S from original
                # V stays from enhanced (already in merged_hsv)
                merged_rgb = SoyaBatchDetailer_mdsoya._hsv_to_rgb(merged_hsv)
            else:
                # hsv_restore: original full processing mode
                # Blur original eyebrow area to reduce noise before HS extraction (GPU)
                if eyebrow_blur > 0:
                    orig_t = torch.from_numpy(original_resized).permute(2, 0, 1)  # (3, H, W)
                    blurred = torch.stack([
                        SoyaBatchDetailer_mdsoya._gaussian_blur_gpu(orig_t[c], eyebrow_blur)
                        for c in range(3)
                    ], dim=-1)
                    original_resized = blurred.cpu().numpy()

                # HSV conversion
                orig_hsv = SoyaBatchDetailer_mdsoya._rgb_to_hsv(original_resized)
                enh_hsv = SoyaBatchDetailer_mdsoya._rgb_to_hsv(enhanced_np)

                # Dark HS filter: only use H,S from pixels with V below given percentile
                if hs_percentile > 0:
                    mask_pixels = binary > 0
                    if np.any(mask_pixels):
                        v_values = orig_hsv[..., 2][mask_pixels]
                        threshold_v = np.percentile(v_values, hs_percentile * 100)
                        dark_mask = mask_pixels & (orig_hsv[..., 2] <= threshold_v)

                        if np.any(dark_mask):
                            # Circular mean for H (hue wraps around)
                            dark_h = orig_hsv[..., 0][dark_mask]
                            h_sin = np.mean(np.sin(dark_h * 2 * np.pi))
                            h_cos = np.mean(np.cos(dark_h * 2 * np.pi))
                            avg_h = np.arctan2(h_sin, h_cos) / (2 * np.pi) % 1.0
                            avg_s = np.mean(orig_hsv[..., 1][dark_mask])

                            # Bright pixels (above percentile) get dark-pixel HS
                            bright_mask = mask_pixels & (orig_hsv[..., 2] > threshold_v)
                            orig_hsv[..., 0][bright_mask] = avg_h
                            orig_hsv[..., 1][bright_mask] = avg_s

                # V range compression: push enhanced V toward darker in eyebrow area
                if v_range < 1.0:
                    mask_pixels = binary > 0
                    if np.any(mask_pixels):
                        v_vals = enh_hsv[..., 2][mask_pixels]
                        min_v = v_vals.min()

                        # Compress V toward min: new_V = min + (V - min) * v_range
                        enh_hsv[..., 2] = min_v + (enh_hsv[..., 2] - min_v) * v_range

                # H,S from (filtered) original, compressed V from enhanced
                merged_hsv = orig_hsv.copy()
                merged_hsv[..., 2] = enh_hsv[..., 2]
                merged_rgb = SoyaBatchDetailer_mdsoya._hsv_to_rgb(merged_hsv)

            # Mask blend: eyebrow area gets restored color
            mask_3ch = binary[..., np.newaxis]
            result = mask_3ch * merged_rgb + (1 - mask_3ch) * enhanced_np

            # Replace with new SEG (namedtuple is immutable)
            updated_segs[i] = SEG(
                cropped_image=torch.from_numpy(result.astype(np.float32)).unsqueeze(0),
                cropped_mask=seg.cropped_mask,
                confidence=seg.confidence,
                crop_region=seg.crop_region,
                bbox=seg.bbox,
                label=seg.label,
            )

    @staticmethod
    def _parse_prompts(prompts):
        """Parse [LAB]\\n[1] prompt1\\n[2] prompt2 → {"1": "prompt1", "2": "prompt2"}."""
        result = {}
        if not prompts:
            return result
        in_lab = False
        for line in prompts.strip().split('\n'):
            stripped = line.strip()
            if stripped == '[LAB]':
                in_lab = True
                continue
            if in_lab:
                match = re.match(r'^\[(\d+)\]\s*(.*)', stripped)
                if match:
                    result[match.group(1)] = match.group(2).strip()
        return result

    @staticmethod
    def _encode_conditioning(clip, text):
        """Encode text with CLIP → CONDITIONING."""
        from nodes import CLIPTextEncode
        return CLIPTextEncode().encode(clip, text)[0]

    @staticmethod
    def _batch_conditionings(conds):
        """Batch multiple CONDITIONING into a single batched CONDITIONING.

        Each cond is [[tensor(1, 77, D), {dict}]].
        Returns [[tensor(N, 77, D), batched_dict]].
        """
        tensors = [c[0][0] for c in conds]
        dicts = [c[0][1] for c in conds]

        batched_tensor = torch.cat(tensors, dim=0)

        batched_dict = {}
        for key in dicts[0]:
            val = dicts[0][key]
            if isinstance(val, torch.Tensor):
                batched_dict[key] = torch.cat([d[key] for d in dicts], dim=0)
            else:
                batched_dict[key] = val

        return [[batched_tensor, batched_dict]]

    @staticmethod
    def _run_ksampler(model, seed, steps, cfg, sampler_name, scheduler,
                       positive, negative, latent_samples, denoise, noise_mask=None):
        """Run KSampler with batched inputs."""
        import comfy.sample
        from nodes import common_ksampler

        latent_dict = {"samples": latent_samples}
        if noise_mask is not None:
            latent_dict["noise_mask"] = noise_mask

        result = common_ksampler(
            model, seed, steps, cfg, sampler_name, scheduler,
            positive, negative, latent_dict,
            denoise=denoise,
        )
        return result[0]["samples"]

    @staticmethod
    def _draw_ellipse_on_mask(mask, bbox, value=1.0, offset_x=0, offset_y=0):
        """Draw a filled ellipse at bbox location on the given 2D mask (in-place).

        Args:
            mask: 2D numpy array (H, W) to modify
            bbox: (x1, y1, x2, y2) in original image coordinates
            value: fill value
            offset_x, offset_y: coordinate offset (for local crop masks)
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        mask_h, mask_w = mask.shape
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        a, b = (x2 - x1) / 2.0, (y2 - y1) / 2.0
        if a <= 0 or b <= 0:
            return
        lx1 = max(0, x1 - offset_x)
        ly1 = max(0, y1 - offset_y)
        lx2 = min(mask_w, x2 - offset_x)
        ly2 = min(mask_h, y2 - offset_y)
        if lx1 >= lx2 or ly1 >= ly2:
            return
        yy, xx = np.ogrid[ly1:ly2, lx1:lx2]
        ellipse = ((xx + offset_x - cx) / a) ** 2 + ((yy + offset_y - cy) / b) ** 2 <= 1
        mask[ly1:ly2, lx1:lx2][ellipse] = value

    @staticmethod
    def _paste_back(original_image, updated_segs, context, eyebrow_opacity=0.0):
        """Paste enhanced crops back using hole-punching with Voronoi overlap split.

        1. Create per-face ellipse masks
        2. Resolve overlaps via Voronoi (nearest center) → each face gets its half
        3. Punch holes in original for ALL faces
        4. Fill kept face holes with detailer results
        5. Apply eyebrow opacity (blend eyebrow area with original)
        6. Fill remain face holes with original pixels
        """
        result = original_image.clone()
        img_h, img_w = original_image.shape[1], original_image.shape[2]

        # Collect all face bboxes and centers
        kept_bboxes = [tuple(seg.bbox) for seg in updated_segs]
        remain_bboxes = []
        for rf in context.get("remain_faces", []):
            bbox = rf.get("original_bbox")
            if bbox is not None:
                remain_bboxes.append(tuple(bbox))
        all_bboxes = kept_bboxes + remain_bboxes

        if not all_bboxes:
            return result

        centers = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in all_bboxes]

        # Step 1: Create individual ellipse masks per face
        individual_masks = []
        for bbox in all_bboxes:
            mask = np.zeros((img_h, img_w), dtype=np.float32)
            SoyaBatchDetailer_mdsoya._draw_ellipse_on_mask(mask, bbox, value=1.0)
            individual_masks.append(mask)

        # Step 2: Resolve overlaps with Voronoi (nearest center)
        overlap_count = np.zeros((img_h, img_w), dtype=np.int32)
        for m in individual_masks:
            overlap_count += (m > 0).astype(np.int32)

        overlap_pixels = overlap_count > 1
        if np.any(overlap_pixels):
            ys, xs = np.where(overlap_pixels)
            points = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
            centers_arr = np.array(centers, dtype=np.float64)

            # Squared distances: (N_pixels, N_faces)
            dists = ((points[:, None, :] - centers_arr[None, :, :]) ** 2).sum(axis=2)
            nearest = np.argmin(dists, axis=1)

            # Clear overlap from all masks, assign to nearest center
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

        # Step 4: Fill kept faces with enhanced content
        crop_mode = context.get("crop_mode", "preserve")
        for seg_idx, seg in enumerate(updated_segs):
            enhanced = seg.cropped_image
            crop_region = [int(v) for v in seg.crop_region]
            cx1, cy1, cx2, cy2 = crop_region
            enh_h, enh_w = enhanced.shape[1], enhanced.shape[2]
            uf_raw = context.get("kept_faces", [{}])[seg_idx].get("upscale_passes", 0)
            uf = 2 ** uf_raw if uf_raw > 0 else 1
            # Downscale by exact upscale factor, not by crop_region size
            target_w = enh_w // uf
            target_h = enh_h // uf
            # Recompute crop_region to match exact downscaled size
            cx2 = cx1 + target_w
            cy2 = cy1 + target_h
            crop_w, crop_h = target_w, target_h

            bx1, by1, bx2, by2 = [int(v) for v in seg.bbox]
            bbox_cy = (by1 + by2) / 2
            crop_cy = (cy1 + cy2) / 2
            print(f"[BatchDetailer] paste-back seg[{seg_idx}]: "
                  f"crop=({cx1},{cy1},{cx2},{cy2}) size=({crop_w},{crop_h}) "
                  f"bbox=({bx1},{by1},{bx2},{by2}) "
                  f"enhanced=({enh_w},{enh_h}) upscale=x{uf} "
                  f"target=({target_w},{target_h}) enh%uf=({enh_w%uf},{enh_h%uf})")
            print(f"  [POSITION] crop_center=({(cx1+cx2)/2:.0f},{crop_cy:.0f}) "
                  f"bbox_center=({(bx1+bx2)/2:.0f},{bbox_cy:.0f}) "
                  f"offset=({cx1-bx1},{cy1-by1},{bx2-cx2},{by2-cy2}) "
                  f"(left,top,right,bottom crop溢出)")
            # How much of bbox is covered by crop
            overlap_y1 = max(by1, cy1)
            overlap_y2 = min(by2, cy2)
            overlap_x1 = max(bx1, cx1)
            overlap_x2 = min(bx2, cx2)
            bbox_area = max(1, (bx2-bx1) * (by2-by1))
            if overlap_y2 > overlap_y1 and overlap_x2 > overlap_x1:
                overlap_area = (overlap_x2-overlap_x1) * (overlap_y2-overlap_y1)
                coverage = overlap_area / bbox_area * 100
            else:
                coverage = 0
            print(f"  [COVERAGE] bbox在crop内覆盖率={coverage:.1f}% "
                  f"miss_bottom={max(0,by2-cy2)}px miss_top={max(0,cy1-by1)}px "
                  f"miss_left={max(0,bx1-cx1)}px miss_right={max(0,cx2-bx2)}px")

            if crop_h <= 0 or crop_w <= 0 or enhanced is None:
                continue

            # Resize enhanced by exact downscale factor (GPU bicubic)
            resized = F.interpolate(
                enhanced.permute(0, 3, 1, 2),
                size=(target_h, target_w), mode='bicubic', align_corners=False,
            ).permute(0, 2, 3, 1)[0].clamp(0, 1)  # (target_h, target_w, 3)

            # Read padding offsets from context (zero when crop was within image bounds)
            _kf = context.get("kept_faces", [{}])
            kf_entry = _kf[seg_idx] if seg_idx < len(_kf) else {}
            pad_l = kf_entry.get("crop_pad_left", 0)
            pad_t = kf_entry.get("crop_pad_top", 0)
            pad_r = kf_entry.get("crop_pad_right", 0)
            pad_b = kf_entry.get("crop_pad_bottom", 0)
            # Raw (unclamped) crop coords — needed because clamping + padding
            # shifts the paste position incorrectly when crop goes beyond bounds.
            # cr_y1_raw + pad_t gives the correct image y-coordinate for valid data.
            cr_x1_raw = kf_entry.get("crop_x1_raw", cx1)
            cr_y1_raw = kf_entry.get("crop_y1_raw", cy1)

            # Compute valid sub-region (non-zero-padded portion) of resized image
            valid_x = pad_l
            valid_y = pad_t
            valid_w = target_w - pad_l - pad_r
            valid_h = target_h - pad_t - pad_b
            has_padding = (pad_l + pad_t + pad_r + pad_b) > 0
            if valid_w <= 0 or valid_h <= 0:
                continue

            # Correct paste position for valid data in original image coords.
            # When clamping shifts cy1 (e.g. cr_y1_raw=-40 → cy1=0),
            # cy1 + pad_t would be wrong (0+40=40 vs correct -40+40=0).
            paste_x1 = cr_x1_raw + pad_l
            paste_y1 = cr_y1_raw + pad_t
            paste_x2 = paste_x1 + valid_w
            paste_y2 = paste_y1 + valid_h

            if crop_mode == "maximize_segment_ratio":
                # Maximize: two-step paste-back
                # 1) Restore original in full Voronoi ellipse area (remove blackbox)
                face_voronoi = individual_masks[seg_idx]
                active = face_voronoi > 0
                if np.any(active):
                    rows, cols = np.where(active)
                    fy1, fy2 = rows.min(), rows.max() + 1
                    fx1, fx2 = cols.min(), cols.max() + 1
                    local_mask = face_voronoi[fy1:fy2, fx1:fx2]
                    mask_t = torch.from_numpy(local_mask.astype(np.float32)).unsqueeze(2)
                    result[0, fy1:fy2, fx1:fx2] = (
                        mask_t * original_image[0, fy1:fy2, fx1:fx2]
                        + (1 - mask_t) * result[0, fy1:fy2, fx1:fx2]
                    )

                # 2) Paste enhanced at tight crop_region (Voronoi mask blend)
                if has_padding:
                    # Use only the valid (non-padded) portion
                    resized_valid = resized[valid_y:valid_y+valid_h, valid_x:valid_x+valid_w]
                    voronoi_valid = face_voronoi[paste_y1:paste_y2, paste_x1:paste_x2]
                    fill_t = torch.from_numpy(voronoi_valid.astype(np.float32)).unsqueeze(2)
                    region = result[0, paste_y1:paste_y2, paste_x1:paste_x2, :]
                    result[0, paste_y1:paste_y2, paste_x1:paste_x2, :] = fill_t * resized_valid + (1 - fill_t) * region
                else:
                    voronoi_local = face_voronoi[cy1:cy2, cx1:cx2]
                    fill_t = torch.from_numpy(voronoi_local.astype(np.float32)).unsqueeze(2)
                    region = result[0, cy1:cy2, cx1:cx2, :]
                    result[0, cy1:cy2, cx1:cx2, :] = fill_t * resized + (1 - fill_t) * region

                print(f"[BatchDetailer] paste-back seg[{seg_idx}]: "
                      f"mode=maximize label={seg.label} "
                      f"ellipse=({fx1},{fy1},{fx2},{fy2}) tight=({cx1},{cy1},{cx2},{cy2}) "
                      f"padding=({pad_l},{pad_t},{pad_r},{pad_b}) "
                      f"paste=({paste_x1},{paste_y1},{paste_x2},{paste_y2}) "
                      f"enhanced={enhanced.shape}→resized={resized.shape}")

                # ── Verification: paste-back original crop (before detailer) with same algorithm ──
                try:
                    if kf_entry.get("image") is not None:
                        _orig_crop = kf_entry["image"]  # (1, enh_h, enh_w, 3) before detailer
                        _oc_h, _oc_w = _orig_crop.shape[1], _orig_crop.shape[2]
                        _oc_resized = F.interpolate(
                            _orig_crop.permute(0, 3, 1, 2),
                            size=(target_h, target_w), mode='bicubic', align_corners=False,
                        ).permute(0, 2, 3, 1)[0].clamp(0, 1)
                        _oc_resized_f = _oc_resized.cpu().numpy()
                        if has_padding:
                            # Compare only valid region at corrected paste position
                            _oc_valid = _oc_resized_f[valid_y:valid_y+valid_h, valid_x:valid_x+valid_w]
                            _orig_at = original_image[0, paste_y1:paste_y2,
                                                         paste_x1:paste_x2, :].cpu().numpy()
                        else:
                            _oc_valid = _oc_resized_f
                            _orig_at = original_image[0, cy1:cy1+target_h, cx1:cx1+target_w, :].cpu().numpy()
                        _diff = np.abs(_orig_at - _oc_valid)
                        _max_diff = _diff.max()
                        _mean_diff = _diff.mean()
                        _n_mismatch = int((_diff > 0.001).sum())
                        _total_px = _diff.size
                        print(f"[BatchDetailer VERIFY] seg[{seg_idx}] label={seg.label} "
                              f"orig_crop=({_oc_w}x{_oc_h}) pad=({pad_l},{pad_t},{pad_r},{pad_b}) "
                              f"paste=({paste_x1},{paste_y1}) uf={uf} "
                              f"max_diff={_max_diff:.6f} mean_diff={_mean_diff:.6f} "
                              f"mismatch={_n_mismatch}/{_total_px}")
                        if _max_diff > 0.005:
                            for _c, _cn in enumerate('RGB'):
                                print(f"  [{_cn}] max={_diff[:,:,_c].max():.6f} mean={_diff[:,:,_c].mean():.6f}")
                            if _diff.shape[0] > 4 and _diff.shape[1] > 4:
                                print(f"  [EDGE] top={_diff[0,:,:].mean():.6f} bot={_diff[-1,:,:].mean():.6f} "
                                      f"left={_diff[:,0,:].mean():.6f} right={_diff[:,-1,:].mean():.6f} "
                                      f"center={_diff[2:-2,2:-2,:].mean():.6f}")
                except Exception as _ve:
                    print(f"[BatchDetailer VERIFY] seg[{seg_idx}] failed: {_ve}")
            else:
                # Preserve: use Voronoi-resolved ellipse mask
                if has_padding:
                    resized_valid = resized[valid_y:valid_y+valid_h, valid_x:valid_x+valid_w]
                    face_mask_valid = individual_masks[seg_idx][paste_y1:paste_y2, paste_x1:paste_x2]
                    fill_t = torch.from_numpy(face_mask_valid.astype(np.float32)).unsqueeze(2)
                    region = result[0, paste_y1:paste_y2, paste_x1:paste_x2, :]
                    result[0, paste_y1:paste_y2, paste_x1:paste_x2, :] = fill_t * resized_valid + (1 - fill_t) * region
                else:
                    face_mask = individual_masks[seg_idx][cy1:cy2, cx1:cx2]
                    fill_t = torch.from_numpy(face_mask.astype(np.float32)).unsqueeze(2)
                    region = result[0, cy1:cy2, cx1:cx2, :]
                    result[0, cy1:cy2, cx1:cx2, :] = fill_t * resized + (1 - fill_t) * region

                print(f"[BatchDetailer] paste-back seg[{seg_idx}]: "
                      f"mode=preserve label={seg.label} crop=({cx1},{cy1},{cx2},{cy2}) "
                      f"padding=({pad_l},{pad_t},{pad_r},{pad_b}) "
                      f"enhanced={enhanced.shape}→resized={resized.shape}")

        # Step 5: Apply eyebrow opacity (blend eyebrow area back toward original)
        if eyebrow_opacity > 0:
            kept_faces = context.get("kept_faces", [])

            for seg_idx, seg in enumerate(updated_segs):
                if seg_idx >= len(kept_faces):
                    break
                kf = kept_faces[seg_idx]
                eb_mask = kf.get("eyebrow_mask")
                if eb_mask is None:
                    continue

                crop_region = [int(v) for v in seg.crop_region]
                cx1, cy1 = crop_region[0], crop_region[1]
                enhanced = seg.cropped_image
                enh_h, enh_w = enhanced.shape[1], enhanced.shape[2]
                uf_raw = kf.get("upscale_passes", 0)
                uf = 2 ** uf_raw if uf_raw > 0 else 1
                target_w = enh_w // uf
                target_h = enh_h // uf
                cx2 = cx1 + target_w
                cy2 = cy1 + target_h

                # Resize eyebrow mask to paste area (target_h x target_w) via GPU
                eb_mask_t = torch.from_numpy(eb_mask).float() if isinstance(eb_mask, np.ndarray) else eb_mask.float()
                eb_resized_t = SoyaBatchDetailer_mdsoya._resize_2d_gpu(eb_mask_t, target_h, target_w)
                eb_resized = eb_resized_t.cpu().numpy()

                threshold = kf.get("eyebrow_threshold", 0.5)
                binary = (eb_resized > threshold).astype(np.float32)

                # Clamp to image bounds, accounting for padding
                eb_pad_l = kf.get("crop_pad_left", 0)
                eb_pad_t = kf.get("crop_pad_top", 0)
                eb_pad_r = kf.get("crop_pad_right", 0)
                eb_pad_b = kf.get("crop_pad_bottom", 0)
                eb_valid_x = eb_pad_l
                eb_valid_y = eb_pad_t
                eb_valid_w = target_w - eb_pad_l - eb_pad_r
                eb_valid_h = target_h - eb_pad_t - eb_pad_b
                if eb_valid_w <= 0 or eb_valid_h <= 0:
                    continue

                # Use raw (unclamped) coords to avoid clamping + padding mismatch
                eb_cr_x1_raw = kf.get("crop_x1_raw", cx1)
                eb_cr_y1_raw = kf.get("crop_y1_raw", cy1)
                ty1 = eb_cr_y1_raw + eb_pad_t
                tx1 = eb_cr_x1_raw + eb_pad_l
                ty2 = ty1 + eb_valid_h
                tx2 = tx1 + eb_valid_w
                if ty2 <= ty1 or tx2 <= tx1:
                    continue

                # Use only the valid (non-padded) portion of the mask
                local_binary = binary[eb_valid_y:eb_valid_y+eb_valid_h, eb_valid_x:eb_valid_x+eb_valid_w]

                # Blend: opacity * original + (1 - opacity) * current result
                alpha = torch.from_numpy(local_binary * eyebrow_opacity)[:, :, np.newaxis]
                result[0, ty1:ty2, tx1:tx2] = (
                    alpha * original_image[0, ty1:ty2, tx1:tx2]
                    + (1 - alpha) * result[0, ty1:ty2, tx1:tx2]
                )

        # Step 6: Fill remain faces with original pixels
        for j, bbox in enumerate(remain_bboxes):
            face_idx = len(updated_segs) + j
            face_mask = individual_masks[face_idx]

            # Get bounding box of active region
            active = face_mask > 0
            if not np.any(active):
                continue
            rows, cols = np.where(active)
            ry1, ry2 = rows.min(), rows.max() + 1
            rx1, rx2 = cols.min(), cols.max() + 1

            local_mask = face_mask[ry1:ry2, rx1:rx2]
            mask_t = torch.from_numpy(local_mask).unsqueeze(2)
            result[0, ry1:ry2, rx1:rx2] = (
                mask_t * original_image[0, ry1:ry2, rx1:rx2]
                + (1 - mask_t) * result[0, ry1:ry2, rx1:rx2]
            )

        return result

    @staticmethod
    def _build_enhanced_preview_from_crops(crops_list):
        """Build a vertical stack IMAGE from a list of (1, H, W, 3) tensors."""
        if not crops_list:
            return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        resized = []
        for c in crops_list:
            if c is None:
                continue
            h, w = c.shape[1], c.shape[2]
            new_h = max(1, int(h * 512 / w))
            r = F.interpolate(
                c.permute(0, 3, 1, 2),
                size=(new_h, 512), mode='bilinear', align_corners=False,
            ).permute(0, 2, 3, 1)[0]
            resized.append(r)

        if not resized:
            return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        grid = torch.cat(resized, dim=0)
        return grid.unsqueeze(0)

    @staticmethod
    def _build_eyebrow_overlay_preview(updated_segs, context):
        """Build enhanced crop preview with eyebrow mask overlaid per crop."""
        kept_faces = context.get("kept_faces", [])

        if not updated_segs:
            return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        resized = []
        for i, seg in enumerate(updated_segs):
            if seg.cropped_image is None:
                continue

            enhanced = seg.cropped_image  # (1, H, W, 3)
            h, w = enhanced.shape[1], enhanced.shape[2]
            new_h = max(1, int(h * 512 / w))
            r = F.interpolate(
                enhanced.permute(0, 3, 1, 2),
                size=(new_h, 512), mode='bilinear', align_corners=False,
            ).permute(0, 2, 3, 1)[0]  # (new_h, 512, 3)

            # Overlay eyebrow mask if available
            if i < len(kept_faces) and kept_faces[i].get("eyebrow_mask") is not None:
                eb_mask = kept_faces[i]["eyebrow_mask"]  # (H_mask, W_mask) float32
                threshold = kept_faces[i].get("eyebrow_threshold", 0.5)

                # Resize mask to enhanced crop dimensions (upscale match, GPU)
                mask_h, mask_w = eb_mask.shape[:2]
                if mask_h != h or mask_w != w:
                    eb_mask_t = torch.from_numpy(eb_mask).float() if isinstance(eb_mask, np.ndarray) else eb_mask.float()
                    eb_mask = SoyaBatchDetailer_mdsoya._resize_2d_gpu(eb_mask_t, h, w).cpu().numpy()

                # Resize mask to display size (512 width, GPU)
                eb_mask_t = torch.from_numpy(eb_mask).float() if isinstance(eb_mask, np.ndarray) else eb_mask.float()
                eb_display = SoyaBatchDetailer_mdsoya._resize_2d_gpu(eb_mask_t, new_h, 512).cpu().numpy()

                # Semi-transparent green overlay on enhanced crop
                overlay = r.cpu().numpy().copy()
                alpha = eb_display[:, :, np.newaxis] * 0.4
                green_tint = np.zeros_like(overlay)
                green_tint[:, :, 1] = 200.0 / 255.0
                overlay = overlay * (1 - alpha) + green_tint * alpha

                # Red boundary at mask edge (GPU erosion)
                binary_mask = eb_display > threshold
                binary_t = torch.from_numpy(binary_mask.astype(np.float32))
                eroded_t = -F.max_pool2d(-binary_t.unsqueeze(0).unsqueeze(0), 3, stride=1, padding=1)
                eroded = (eroded_t.squeeze(0).squeeze(0).numpy() > 0.5)
                boundary = binary_mask & ~eroded
                if np.any(boundary):
                    overlay[boundary] = [1.0, 0.15, 0.15]

                r = torch.from_numpy(overlay.astype(np.float32))

            resized.append(r)

        if not resized:
            return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        grid = torch.cat(resized, dim=0)  # (total_h, 512, 3)
        return grid.unsqueeze(0)  # (1, total_h, 512, 3)

    @staticmethod
    def _build_enhanced_preview(updated_segs):
        """Build a vertical stack IMAGE from all enhanced crops for preview."""
        if not updated_segs:
            return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        crops = [seg.cropped_image for seg in updated_segs if seg.cropped_image is not None]
        if not crops:
            return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        # Resize each crop to 512 width, proportional height
        resized = []
        for c in crops:
            h, w = c.shape[1], c.shape[2]
            new_h = max(1, int(h * 512 / w))
            r = F.interpolate(
                c.permute(0, 3, 1, 2),
                size=(new_h, 512), mode='bilinear', align_corners=False,
            ).permute(0, 2, 3, 1)[0]  # (new_h, 512, 3)
            resized.append(r)

        grid = torch.cat(resized, dim=0)  # (total_h, 512, 3)
        return grid.unsqueeze(0)  # (1, total_h, 512, 3)

    @staticmethod
    def _build_eyebrow_crop_preview(updated_segs, context):
        """Same layout as enhanced, but only eyebrow pixels from original crop (black elsewhere)."""
        kept_faces = context.get("kept_faces", [])

        if not updated_segs:
            return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        resized = []
        for i, seg in enumerate(updated_segs):
            if seg.cropped_image is None:
                continue
            if i >= len(kept_faces) or kept_faces[i].get("eyebrow_mask") is None:
                continue

            # Use enhanced crop dimensions as reference (matches enhanced output exactly)
            enh_h, enh_w = seg.cropped_image.shape[1], seg.cropped_image.shape[2]

            original_crop = kept_faces[i]["image"]  # (1, H, W, 3) from collector
            eb_mask = kept_faces[i]["eyebrow_mask"]  # (H_mask, W_mask) float32
            if isinstance(eb_mask, np.ndarray):
                eb_mask = torch.from_numpy(eb_mask).float()
            threshold = kept_faces[i].get("eyebrow_threshold", 0.5)

            # Resize original crop to match enhanced dimensions
            orig_h, orig_w = original_crop.shape[1], original_crop.shape[2]
            if orig_h != enh_h or orig_w != enh_w:
                original_crop = F.interpolate(
                    original_crop.permute(0, 3, 1, 2),
                    size=(enh_h, enh_w), mode='bilinear', align_corners=False,
                ).permute(0, 2, 3, 1)

            # Resize mask to match enhanced dimensions (GPU)
            mask_h, mask_w = eb_mask.shape[:2]
            if mask_h != enh_h or mask_w != enh_w:
                eb_mask = SoyaBatchDetailer_mdsoya._resize_2d_gpu(eb_mask, enh_h, enh_w)

            # Binary mask from threshold → keep only eyebrow pixels
            binary = (eb_mask > threshold).float()
            mask_3ch = binary.unsqueeze(-1)
            masked = original_crop[0] * mask_3ch  # (H, W, 3)

            # Same resize as enhanced: 512 width, using enhanced dimensions
            new_h = max(1, int(enh_h * 512 / enh_w))
            r = F.interpolate(
                masked.unsqueeze(0).permute(0, 3, 1, 2),
                size=(new_h, 512), mode='bilinear', align_corners=False,
            ).permute(0, 2, 3, 1)[0]  # (new_h, 512, 3)

            resized.append(r)

        if not resized:
            return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        grid = torch.cat(resized, dim=0)
        return grid.unsqueeze(0)

    @staticmethod
    def _build_eyebrow_blur_preview(updated_segs, context, sigma):
        """Same as eyebrow_crop but with Gaussian blur applied to original eyebrow pixels."""
        kept_faces = context.get("kept_faces", [])

        if not updated_segs or sigma <= 0:
            return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        resized = []
        for i, seg in enumerate(updated_segs):
            if seg.cropped_image is None:
                continue
            if i >= len(kept_faces) or kept_faces[i].get("eyebrow_mask") is None:
                continue

            enh_h, enh_w = seg.cropped_image.shape[1], seg.cropped_image.shape[2]
            original_crop = kept_faces[i]["image"]
            eb_mask = kept_faces[i]["eyebrow_mask"]
            if isinstance(eb_mask, np.ndarray):
                eb_mask = torch.from_numpy(eb_mask).float()
            threshold = kept_faces[i].get("eyebrow_threshold", 0.5)

            # Resize original to match enhanced
            orig_h, orig_w = original_crop.shape[1], original_crop.shape[2]
            if orig_h != enh_h or orig_w != enh_w:
                original_crop = F.interpolate(
                    original_crop.permute(0, 3, 1, 2),
                    size=(enh_h, enh_w), mode='bilinear', align_corners=False,
                ).permute(0, 2, 3, 1)

            # Resize mask to match enhanced (GPU)
            mask_h, mask_w = eb_mask.shape[:2]
            if mask_h != enh_h or mask_w != enh_w:
                eb_mask = SoyaBatchDetailer_mdsoya._resize_2d_gpu(eb_mask, enh_h, enh_w)

            # Blur original eyebrow pixels (GPU)
            orig_t = original_crop[0].permute(2, 0, 1)  # (3, H, W)
            blurred = torch.stack([
                SoyaBatchDetailer_mdsoya._gaussian_blur_gpu(orig_t[c], sigma)
                for c in range(3)
            ], dim=-1)

            # Apply binary mask → only eyebrow pixels visible
            binary = (eb_mask > threshold).float()
            mask_3ch = binary.unsqueeze(-1)
            masked = blurred * mask_3ch

            # Same resize as enhanced: 512 width
            new_h = max(1, int(enh_h * 512 / enh_w))
            r = F.interpolate(
                masked.unsqueeze(0).permute(0, 3, 1, 2),
                size=(new_h, 512), mode='bilinear', align_corners=False,
            ).permute(0, 2, 3, 1)[0]

            resized.append(r)

        if not resized:
            return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        grid = torch.cat(resized, dim=0)
        return grid.unsqueeze(0)

    @staticmethod
    def _save_detailer_result(updated_segs, batch_groups, segs_list, elapsed, iterations,
                              individual_masks_viz=None, original_masks_viz=None,
                              edge_blur=0, edge_feather=0):
        """Save enhanced crops and mask overlay as base64 to config for web UI display."""
        import io, base64
        from PIL import Image as PILImage
        from .soya_scheduler.config_manager import load_config, save_config

        results = []
        for i, seg in enumerate(updated_segs):
            if seg.cropped_image is None:
                continue
            enhanced_np = (seg.cropped_image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            img = PILImage.fromarray(enhanced_np)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            entry = {
                "label": seg.label,
                "enhanced_b64": b64,
                "enhanced_w": enhanced_np.shape[1],
                "enhanced_h": enhanced_np.shape[0],
            }

            # Build mask overlay visualization
            if (individual_masks_viz and i < len(individual_masks_viz)
                    and individual_masks_viz[i] is not None):
                mask_np = individual_masks_viz[i].numpy()  # blurred/feathered mask
                orig_np = (original_masks_viz[i].numpy()
                           if (original_masks_viz and i < len(original_masks_viz)
                               and original_masks_viz[i] is not None)
                           else mask_np)

                # Resize mask to match enhanced image if needed (GPU)
                h, w = enhanced_np.shape[:2]
                if (mask_np.shape[0], mask_np.shape[1]) != (h, w):
                    mask_t = torch.from_numpy(mask_np).float()
                    mask_np = SoyaBatchDetailer_mdsoya._resize_2d_gpu(mask_t, h, w).cpu().numpy()
                    orig_t = torch.from_numpy(orig_np).float()
                    orig_np = SoyaBatchDetailer_mdsoya._resize_2d_gpu(orig_t, h, w).cpu().numpy()

                overlay = enhanced_np.astype(np.float32).copy()

                # Semi-transparent blue tint for mask area
                alpha = mask_np[:, :, np.newaxis] * 0.35
                blue_tint = np.zeros_like(overlay)
                blue_tint[:, :, 2] = 200  # blue channel
                overlay = overlay * (1 - alpha) + blue_tint * alpha

                # Red outline at original mask boundary (GPU erosion)
                orig_binary = orig_np > 0.5
                orig_binary_t = torch.from_numpy(orig_binary.astype(np.float32))
                eroded_t = -F.max_pool2d(-orig_binary_t.unsqueeze(0).unsqueeze(0), 3, stride=1, padding=1)
                eroded = (eroded_t.squeeze(0).squeeze(0).numpy() > 0.5)
                boundary = orig_binary & ~eroded
                overlay[boundary] = [255, 50, 50]

                overlay = overlay.clip(0, 255).astype(np.uint8)
                buf2 = io.BytesIO()
                PILImage.fromarray(overlay).save(buf2, format="PNG")
                entry["mask_overlay_b64"] = base64.b64encode(buf2.getvalue()).decode("utf-8")

                # Save blurred mask as grayscale
                mask_vis = (mask_np * 255).clip(0, 255).astype(np.uint8)
                buf3 = io.BytesIO()
                PILImage.fromarray(mask_vis).save(buf3, format="PNG")
                entry["mask_b64"] = base64.b64encode(buf3.getvalue()).decode("utf-8")

            results.append(entry)

        config = load_config()
        if not config.get("last_process_result"):
            config["last_process_result"] = {}
        config["last_process_result"]["detailer"] = {
            "results": results,
            "edge_blur": edge_blur,
            "edge_feather": edge_feather,
            "batch_groups": [
                {
                    "indices": group,
                    "type": "\ubc30\uce58" if len(group) > 1 else "\uac1c\ubcc4",
                    "labels": [segs_list[idx].label for idx in group],
                }
                for group in batch_groups
            ],
            "iterations": iterations,
            "total_time": elapsed,
        }
        save_config(config)

    # ── GPU-optimized image processing (replaces PIL & scipy) ────

    @staticmethod
    def _gaussian_blur_gpu(mask, sigma):
        """Gaussian blur via separable convolution — no CPU transfer."""
        ksize = int(6 * sigma + 1)
        if ksize % 2 == 0:
            ksize += 1
        x = torch.arange(ksize, dtype=torch.float32, device=mask.device) - ksize // 2
        kernel = torch.exp(-x ** 2 / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        mask_4d = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        pad = ksize // 2
        padded = F.pad(mask_4d, [pad, pad, 0, 0], mode='reflect')
        blurred = F.conv2d(padded, kernel.view(1, 1, 1, -1))
        padded = F.pad(blurred, [0, 0, pad, pad], mode='reflect')
        blurred = F.conv2d(padded, kernel.view(1, 1, -1, 1))
        return blurred.squeeze(0).squeeze(0)

    @staticmethod
    def _resize_2d_gpu(mask, target_h, target_w):
        """Resize a 2D mask tensor via F.interpolate (no PIL)."""
        m4 = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        m4 = F.interpolate(m4, size=(target_h, target_w), mode='bilinear', align_corners=False)
        return m4.squeeze(0).squeeze(0)

    @staticmethod
    def _distance_transform_gpu(binary_mask):
        """Distance transform via scipy EDT (single-pass O(n), replaces iterative erosion)."""
        from scipy.ndimage import distance_transform_edt
        mask_np = binary_mask.cpu().numpy() if isinstance(binary_mask, torch.Tensor) else binary_mask
        dist = distance_transform_edt(mask_np > 0.5).astype(np.float32)
        return torch.from_numpy(dist).to(binary_mask.device)

    @staticmethod
    def _apply_color_adjust_gpu(image, mask, temperature=0, tint=0, saturation=0,
                                vibrance=0, brightness=0, contrast=0, gamma=1.0):
        """Apply color adjustment using torch GPU operations (no CPU transfer).

        Args:
            image: (1, H, W, 3) float32 tensor
            mask: (H, W) float32 tensor – gradient mask for blending
        Returns:
            (1, H, W, 3) float32 tensor
        """
        color = image[0].clone()  # (H, W, 3)
        m = mask.unsqueeze(-1)    # (H, W, 1)

        # Temperature
        t = temperature * 0.01
        if t != 0:
            color[..., 0] = color[..., 0] + m[..., 0] * t * 0.3
            color[..., 2] = color[..., 2] - m[..., 0] * t * 0.3
            color = color.clamp(0.0, 1.0)

        # Tint
        ti = tint * 0.01
        if ti != 0:
            color[..., 1] = color[..., 1] + m[..., 0] * ti * 0.3
            color[..., 0] = color[..., 0] - m[..., 0] * ti * 0.15
            color[..., 2] = color[..., 2] - m[..., 0] * ti * 0.15
            color = color.clamp(0.0, 1.0)

        # Vibrance
        v = vibrance * 0.01
        if v != 0:
            r_ch, g_ch, b_ch = color[..., 0], color[..., 1], color[..., 2]
            maxC = torch.maximum(torch.maximum(r_ch, g_ch), b_ch)
            minC = torch.minimum(torch.minimum(r_ch, g_ch), b_ch)
            sat = maxC - minC
            luma_w = color.new_tensor([0.299, 0.587, 0.114])
            gray = (color * luma_w).sum(dim=-1)

            if v < 0:
                gray3 = gray.unsqueeze(-1).expand_as(color)
                adjusted = gray3 + (1.0 + v) * (color - gray3)
            else:
                vibranceAmt = v * (1.0 - sat)
                isWarmTone = ((b_ch <= g_ch) & (g_ch <= r_ch)).float()
                warmth = (r_ch - b_ch) / torch.clamp(maxC, min=1e-4)
                skinTone = isWarmTone * warmth * sat * (1.0 - sat)
                vibranceAmt = vibranceAmt * (1.0 - skinTone * 0.5)
                gray3 = gray.unsqueeze(-1).expand_as(color)
                adjusted = gray3 + (1.0 + vibranceAmt.unsqueeze(-1) * 2.0) * (color - gray3)

            color = m * adjusted + (1 - m) * color

        # Saturation
        s = saturation * 0.01
        if s != 0:
            luma_w = color.new_tensor([0.299, 0.587, 0.114])
            gray = (color * luma_w).sum(dim=-1, keepdim=True)
            satMix = 1.0 + (s if s < 0 else s * 2.0)
            adjusted = gray + satMix * (color - gray)
            color = m * adjusted + (1 - m) * color

        # Brightness
        b = brightness / 100.0
        if b != 0:
            bright_factor = 1.0 + b
            brightened = torch.pow(color.clamp(min=1e-6), 1.0 / max(bright_factor, 0.01))
            color = m * brightened + (1 - m) * color

        # Contrast
        c = contrast / 100.0
        if c != 0:
            contrast_factor = 1.0 + c
            mean_val = color.mean()
            contrasted = mean_val + contrast_factor * (color - mean_val)
            color = m * contrasted + (1 - m) * color

        # Gamma
        if gamma != 1.0:
            gamma_corrected = torch.pow(color.clamp(min=1e-6), gamma).clamp(0.0, 1.0)
            color = m * gamma_corrected + (1 - m) * color

        return color.clamp(0.0, 1.0).unsqueeze(0)
