"""
SoyaFaceIDYoloFallback – IPAdapter FaceID with YOLO fallback for face detection.

Supports both FaceID models (ip-adapter-faceid-plusv2) and standard IPA models
(e.g. noobIPAMARK1). Face detection uses insightface + YOLO fallback.

Face Detection Flow:
  1. Insightface progressive loop (640→256) – standard detection
  2. YOLO fallback: detect face bbox → crop → insightface on crop
  3. Both fail → raise error
"""

import os
import sys

import numpy as np
import torch


WEIGHT_TYPES = [
    "linear", "ease in", "ease out", "ease in-out", "reverse in-out",
    "weak input", "weak output", "weak middle", "strong middle",
    "style transfer", "composition", "strong style transfer",
]


def _crop_by_bbox(source, bbox, crop_factor, target_size):
    """Crop region around bbox expanded by crop_factor, resize to target_size.

    Args:
        source: (H, W, 3) uint8 numpy array (BGR)
        bbox: [x1, y1, x2, y2] bounding box
        crop_factor: expansion factor (1.0 = tight, 3.0 = wide)
        target_size: output size (224 or 256)

    Returns:
        (1, target_size, target_size, 3) float32 [0,1] tensor (RGB)
    """
    H, W = source.shape[:2]
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    new_w = bw * crop_factor
    new_h = bh * crop_factor
    nx1 = max(0, int(cx - new_w / 2))
    ny1 = max(0, int(cy - new_h / 2))
    nx2 = min(W, int(cx + new_w / 2))
    ny2 = min(H, int(cy + new_h / 2))

    crop = source[ny1:ny2, nx1:nx2].copy()
    from PIL import Image as PILImage
    crop_rgb = crop[:, :, ::-1]  # BGR → RGB
    pil_img = PILImage.fromarray(crop_rgb)
    pil_img = pil_img.resize((target_size, target_size), PILImage.BILINEAR)
    tensor = torch.from_numpy(np.array(pil_img).astype(np.float32) / 255.0).unsqueeze(0)
    return tensor


def _detect_face_with_yolo_fallback(image_numpy, insightface_model, yolo_model,
                                     yolo_threshold, yolo_crop_factor, is_sdxl,
                                     need_insightface_embed=True):
    """Detect face: insightface first, YOLO crop as fallback.

    Uses face bbox expanded by yolo_crop_factor for the crop,
    so hair, accessories, and head shape are captured.

    Args:
        need_insightface_embed: If False, InsightFace embed extraction is skipped
            when YOLO finds the face. Only the crop image is needed (for non-FaceID
            models that use CLIP Vision instead of InsightFace embeddings).

    Returns:
        (face_embed, face_crop_tensor)
    """
    norm_size = 256 if is_sdxl else 224
    H, W = image_numpy.shape[:2]

    # 1차: insightface progressive loop (640→256)
    for size in range(640, 256, -64):
        insightface_model.det_model.input_size = (size, size)
        face = insightface_model.get(image_numpy)
        if face:
            face_embed = torch.from_numpy(face[0].normed_embedding).unsqueeze(0)
            face_crop = _crop_by_bbox(
                image_numpy, face[0].bbox, yolo_crop_factor, norm_size,
            )
            return face_embed, face_crop

    # 2차: YOLO fallback → crop
    if yolo_model is not None:
        print(f"[Soya:FaceID] InsightFace failed on {W}x{H}, trying YOLO fallback...")
        results = yolo_model(image_numpy, verbose=False)
        best_bbox = None
        best_conf = 0.0
        best_area = 0
        total_detections = 0

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                conf = float(box.conf[0])
                total_detections += 1
                if conf < yolo_threshold:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                area = (x2 - x1) * (y2 - y1)
                if area > best_area:
                    best_area = area
                    best_bbox = (x1, y1, x2, y2)
                    best_conf = conf

        if best_bbox is not None:
            yolo_crop = _crop_by_bbox(
                image_numpy, best_bbox, yolo_crop_factor, norm_size,
            )

            if not need_insightface_embed:
                # Non-FaceID: YOLO crop is sufficient, skip InsightFace embed extraction
                dummy_embed = torch.zeros(1, 512)
                print(f"[Soya:FaceID] YOLO fallback: using crop directly (insightface embed not needed)")
                return dummy_embed, yolo_crop

            # FaceID: must extract InsightFace embedding from crop
            yolo_crop_np = (yolo_crop[0].numpy() * 255).astype(np.uint8)[:, :, ::-1].copy()

            for det_size in range(640, 192, -64):
                insightface_model.det_model.input_size = (det_size, det_size)
                face = insightface_model.get(yolo_crop_np)
                if face:
                    face_embed = torch.from_numpy(face[0].normed_embedding).unsqueeze(0)
                    print(f"[Soya:FaceID] YOLO fallback succeeded, insightface det_size={det_size}")
                    return face_embed, yolo_crop

            print(f"[Soya:FaceID] YOLO found face but insightface failed on crop")

    raise Exception(
        f"InsightFace: No face detected in {W}x{H} image. "
        f"Insightface progressive loop and YOLO fallback both failed."
    )


class SoyaFaceIDYoloFallback_mdsoya:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "ipadapter": ("IPADAPTER",),
                "image": ("IMAGE",),
                "weight": ("FLOAT", {"default": 1.0, "min": -1, "max": 3, "step": 0.05}),
                "weight_faceidv2": ("FLOAT", {"default": 1.0, "min": -1, "max": 5.0, "step": 0.05}),
                "weight_type": (WEIGHT_TYPES,),
                "combine_embeds": (["concat", "add", "subtract", "average", "norm average"],),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "embeds_scaling": (["V only", "K+V", "K+V w/ C penalty", "K+mean(V) w/ C penalty"],),
                "update_model": ("STRING", {"default": "true", "multiline": False}),
                "bbox_detector": ("BBOX_DETECTOR",),
                "yolo_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "yolo_crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.1}),
            },
            "optional": {
                "image_negative": ("IMAGE",),
                "attn_mask": ("MASK",),
                "clip_vision": ("CLIP_VISION",),
                "insightface": ("INSIGHTFACE",),
            },
        }

    RETURN_TYPES = ("MODEL", "IMAGE", "STRING",)
    RETURN_NAMES = ("MODEL", "face_image", "info",)
    FUNCTION = "execute"
    CATEGORY = "ipadapter/faceid"

    def execute(self, model, ipadapter, image, weight, weight_faceidv2, weight_type,
                combine_embeds, start_at, end_at, embeds_scaling,
                update_model, bbox_detector, yolo_threshold, yolo_crop_factor,
                image_negative=None, attn_mask=None, clip_vision=None, insightface=None):

        import comfy.model_management

        # ── Detect IPAdapter model type ──
        is_full = "proj.3.weight" in ipadapter["image_proj"]
        is_portrait = (
            "proj.2.weight" in ipadapter["image_proj"]
            and "proj.3.weight" not in ipadapter["image_proj"]
            and "0.to_q_lora.down.weight" not in ipadapter["ip_adapter"]
        )
        is_portrait_unnorm = "portraitunnorm" in ipadapter
        is_faceid = (
            is_portrait
            or "0.to_q_lora.down.weight" in ipadapter["ip_adapter"]
            or is_portrait_unnorm
        )
        is_plus = (
            is_full
            or "latents" in ipadapter["image_proj"]
            or "perceiver_resampler.proj_in.weight" in ipadapter["image_proj"]
        ) and not is_portrait_unnorm
        is_faceidv2 = "faceidplusv2" in ipadapter
        output_cross_attention_dim = ipadapter["ip_adapter"]["1.to_k_ip.weight"].shape[1]
        is_sdxl = output_cross_attention_dim == 2048

        # ── Build info string ──
        info_lines = [
            f"[Soya FaceID YOLO Fallback]",
            f"  update_model: {update_model}",
            f"  model_type: faceid={is_faceid}, plus={is_plus}, faceidv2={is_faceidv2}",
            f"  arch: {'SDXL' if is_sdxl else 'SD1.5'}",
            f"  weight: {weight}, weight_faceidv2: {weight_faceidv2}",
            f"  weight_type: {weight_type}",
            f"  combine_embeds: {combine_embeds}",
            f"  start_at: {start_at}, end_at: {end_at}",
            f"  embeds_scaling: {embeds_scaling}",
            f"  yolo_threshold: {yolo_threshold}, yolo_crop_factor: {yolo_crop_factor}",
            f"  batch_size: {image.shape[0]}",
        ]
        if attn_mask is not None:
            info_lines.append(f"  attn_mask: provided ({attn_mask.shape})")
        else:
            info_lines.append(f"  attn_mask: none")

        # ── Load insightface if not provided ──
        if insightface is None:
            from .soya_scheduler.model_manager import get_insightface_model
            insightface = get_insightface_model()

        # ── Extract YOLO model from BBOX_DETECTOR ──
        yolo_model = bbox_detector.bbox_model

        # ── Face detection with YOLO fallback ──
        face_cond_embeds = []
        face_images = []

        for i in range(image.shape[0]):
            img_rgb = (image[i].cpu().numpy() * 255).astype(np.uint8)
            img_bgr = img_rgb[:, :, ::-1].copy()
            face_embed, face_crop = _detect_face_with_yolo_fallback(
                img_bgr, insightface, yolo_model,
                yolo_threshold, yolo_crop_factor, is_sdxl,
                need_insightface_embed=is_faceid,
            )
            face_cond_embeds.append(face_embed)
            face_images.append(face_crop)

        face_cond_embeds = torch.cat(face_cond_embeds, dim=0)
        face_image_batch = torch.cat(face_images, dim=0)

        # ── update_model = false: face detection only, skip IPAdapter patching ──
        if update_model == "false":
            # Clear any existing attn2 patches to ensure clean model
            transformer_options = model.model_options.setdefault("transformer_options", {})
            patches_replace = transformer_options.setdefault("patches_replace", {})
            if "attn2" in patches_replace:
                del patches_replace["attn2"]
            info_lines.append(f"  mode: FACE_DETECT_ONLY (no model patch)")
            info_lines.append(f"  faces_detected: {image.shape[0]}")
            info = "\n".join(info_lines)
            print(info)
            return (model, face_image_batch, info)

        # ── update_model = true: apply IPAdapter patches ──
        if clip_vision is None:
            raise ValueError("clip_vision is required when update_model is true.")

        custom_nodes_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if custom_nodes_dir not in sys.path:
            sys.path.append(custom_nodes_dir)

        try:
            from comfyui_ipadapter_plus.IPAdapterPlus import (
                IPAdapter, set_model_patch_replace,
            )
            from comfyui_ipadapter_plus.utils import encode_image_masked
        except ImportError as e:
            raise ImportError(
                f"comfyui_ipadapter_plus import failed: {e}. "
                "Install from: https://github.com/cubiq/ComfyUI_IPAdapter_plus"
            )

        # Clear stale attn2 patches
        transformer_options = model.model_options.setdefault("transformer_options", {})
        patches_replace = transformer_options.setdefault("patches_replace", {})
        if "attn2" in patches_replace:
            del patches_replace["attn2"]

        device = comfy.model_management.get_torch_device()
        dtype = comfy.model_management.unet_dtype()
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            dtype = torch.float16 if comfy.model_management.should_use_fp16() else torch.float32

        comfy.model_management.load_model_gpu(clip_vision.patcher)

        # ── CLIP Vision encode ──
        if is_faceid and not is_plus:
            # Non-plus FaceID: use InsightFace 512-dim embedding directly
            img_cond_embeds = face_cond_embeds.to(device, dtype=dtype)
        else:
            img_cond_embeds = encode_image_masked(clip_vision, face_image_batch, batch_size=0)
            if is_plus:
                img_cond_embeds = img_cond_embeds.penultimate_hidden_states.to(device, dtype=dtype)
            else:
                img_cond_embeds = img_cond_embeds.image_embeds.to(device, dtype=dtype)

        # ── Unconditional embeddings ──
        if image_negative is not None and not is_faceid:
            img_uncond_embeds = encode_image_masked(clip_vision, image_negative, batch_size=0)
            if is_plus:
                img_uncond_embeds = img_uncond_embeds.penultimate_hidden_states.to(device, dtype=dtype)
            else:
                img_uncond_embeds = img_uncond_embeds.image_embeds.to(device, dtype=dtype)
        else:
            img_uncond_embeds = torch.zeros_like(img_cond_embeds)

        # ── Create IPAdapter instance ──
        cross_attention_dim = 1280 if (is_plus and is_sdxl and not is_faceid) or is_portrait_unnorm else output_cross_attention_dim
        clip_extra_context_tokens = 16 if (is_plus and not is_faceid) or is_portrait or is_portrait_unnorm else 4

        ipa = IPAdapter(
            ipadapter,
            cross_attention_dim=cross_attention_dim,
            output_cross_attention_dim=output_cross_attention_dim,
            clip_embeddings_dim=img_cond_embeds.shape[-1],
            clip_extra_context_tokens=clip_extra_context_tokens,
            is_sdxl=is_sdxl,
            is_plus=is_plus,
            is_full=is_full,
            is_faceid=is_faceid,
            is_portrait_unnorm=is_portrait_unnorm,
        ).to(device, dtype=dtype)

        # ── IPAdapter projection ──
        if is_faceid and is_plus:
            face_embeds_dev = face_cond_embeds.to(device, dtype=dtype)
            cond = ipa.get_image_embeds_faceid_plus(
                face_embeds_dev, img_cond_embeds,
                weight_faceidv2, is_faceidv2, 0,
            ).to(device, dtype=dtype)
            uncond = ipa.get_image_embeds_faceid_plus(
                torch.zeros_like(face_embeds_dev), img_uncond_embeds,
                weight_faceidv2, is_faceidv2, 0,
            ).to(device, dtype=dtype)
        else:
            cond, uncond = ipa.get_image_embeds(img_cond_embeds, img_uncond_embeds, 0)

        # ── Combine embeddings ──
        if combine_embeds != "concat" and cond.shape[0] > 1:
            cond_list = [cond[i] for i in range(cond.shape[0])]
            if combine_embeds == "add":
                cond = torch.stack(cond_list).sum(0).unsqueeze(0)
            elif combine_embeds == "subtract":
                cond = cond_list[0].unsqueeze(0) - torch.stack(cond_list[1:]).sum(0).unsqueeze(0)
            elif combine_embeds == "average":
                cond = torch.stack(cond_list).mean(0).unsqueeze(0)
            elif combine_embeds == "norm average":
                cond = torch.stack(cond_list).mean(0).unsqueeze(0)
                cond = cond / torch.norm(cond, dim=-1, keepdim=True)
            uncond = uncond[:1]

        # ── Sigma range ──
        sigma_start = model.get_model_object("model_sampling").percent_to_sigma(start_at)
        sigma_end = model.get_model_object("model_sampling").percent_to_sigma(end_at)

        # ── Prepare attention mask ──
        if attn_mask is not None:
            attn_mask = attn_mask.to(device, dtype=dtype)

        # ── Build patch kwargs ──
        patch_kwargs = {
            "ipadapter": ipa,
            "weight": weight,
            "cond": cond,
            "cond_alt": None,
            "uncond": uncond,
            "weight_type": weight_type,
            "mask": attn_mask,
            "sigma_start": sigma_start,
            "sigma_end": sigma_end,
            "unfold_batch": False,
            "embeds_scaling": embeds_scaling,
        }

        # ── Install patches on cross-attention layers ──
        number = 0
        if not is_sdxl:
            for idx in [1, 2, 4, 5, 7, 8]:
                patch_kwargs["module_key"] = str(number * 2 + 1)
                set_model_patch_replace(model, patch_kwargs, ("input", idx))
                number += 1
            for idx in [3, 4, 5, 6, 7, 8, 9, 10, 11]:
                patch_kwargs["module_key"] = str(number * 2 + 1)
                set_model_patch_replace(model, patch_kwargs, ("output", idx))
                number += 1
            patch_kwargs["module_key"] = str(number * 2 + 1)
            set_model_patch_replace(model, patch_kwargs, ("middle", 0))
        else:
            for idx in [4, 5, 7, 8]:
                block_indices = range(2) if idx in [4, 5] else range(10)
                for index in block_indices:
                    patch_kwargs["module_key"] = str(number * 2 + 1)
                    set_model_patch_replace(model, patch_kwargs, ("input", idx, index))
                    number += 1
            for idx in range(6):
                block_indices = range(2) if idx in [3, 4, 5] else range(10)
                for index in block_indices:
                    patch_kwargs["module_key"] = str(number * 2 + 1)
                    set_model_patch_replace(model, patch_kwargs, ("output", idx, index))
                    number += 1
            for index in range(10):
                patch_kwargs["module_key"] = str(number * 2 + 1)
                set_model_patch_replace(model, patch_kwargs, ("middle", 0, index))
                number += 1

        info_lines.append(f"  mode: FULL (model patched with IPAdapter)")
        info_lines.append(f"  faces_detected: {image.shape[0]}")
        info_lines.append(f"  patch_layers: {number}")
        info = "\n".join(info_lines)
        print(info)

        return (model, face_image_batch, info)
