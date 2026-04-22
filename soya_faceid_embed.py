"""
SoyaFaceIDMaskPatcher – ComfyUI node for FaceID Plus V2 with pre-computed embeddings.

Pipeline:
  1. YOLO face detection on input image
  2. Load characters, filter by prompt
  3. Load pre-computed faceid embeddings for matched characters
  4. CLIP Vision matching (Hungarian algorithm) to assign characters to detected faces
  5. Generate per-character attention masks from bboxes
  6. Load IPAdapter model + companion LoRA
  7. For each matched character: encode face_crop → CLIP → project → patch UNet attention
  8. Return patched model + combined mask
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment


_DIR = os.path.dirname(os.path.abspath(__file__))
_NODE_INFO_PATH = os.path.join(_DIR, "soya_scheduler", "node_info.json")


def _load_node_info():
    import json
    if os.path.exists(_NODE_INFO_PATH):
        with open(_NODE_INFO_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"settings": {}}


def _filter_characters_by_prompt(characters, prompt):
    """Return characters whose name appears in the prompt."""
    prompt_norm = prompt.lower().replace('_', ' ')
    matched = []
    for ch in characters:
        name = ch["name"].lower().replace('_', ' ')
        if name in prompt_norm:
            matched.append(ch)
    return matched


def _detect_faces_yolo(image_tensor, yolo_model, threshold=0.5):
    """Detect faces in a BHWC [0,1] tensor. Returns list of dicts with bbox/crop."""
    # image_tensor: (1, H, W, 3) or (H, W, 3)
    if image_tensor.dim() == 4:
        image_tensor = image_tensor[0]
    img_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
    H, W = img_np.shape[:2]

    results = yolo_model(img_np, verbose=False)
    faces = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            conf = float(box.conf[0])
            if conf < threshold:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            faces.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": conf,
            })

    # sort by bbox size descending
    for f in faces:
        f["area"] = (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1])
    faces.sort(key=lambda f: f["area"], reverse=True)
    # re-sort left to right
    faces.sort(key=lambda f: f["bbox"][0])
    return faces


def _match_characters_hungarian(query_embeds, ref_embeds, char_names):
    """Hungarian algorithm matching. Returns (names, scores) lists."""
    M = query_embeds.shape[0]
    N = ref_embeds.shape[0]
    if N == 0:
        return ["unknown"] * M, [0.0] * M

    sim_matrix = F.cosine_similarity(query_embeds.unsqueeze(1), ref_embeds.unsqueeze(0), dim=2)
    cost_matrix = -sim_matrix.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    final_names = ["unknown"] * M
    final_scores = [0.0] * M
    for r, c in zip(row_ind, col_ind):
        final_names[r] = char_names[c]
        final_scores[r] = float(sim_matrix[r, c].item())
    return final_names, final_scores


def _encode_images_for_matching(clip_vision, images_list):
    """Encode a list of (H,W,3) uint8 numpy arrays using CLIP Vision for matching.
    Returns (N, embed_dim) tensor."""
    import comfy.model_management

    tensors = []
    for img_np in images_list:
        img = Image.fromarray(img_np.astype(np.uint8)).resize((224, 224), Image.BILINEAR)
        t = torch.from_numpy(np.array(img).astype(np.float32) / 255.0)
        tensors.append(t)
    batch = torch.stack(tensors)

    _orig = comfy.model_management.load_model_gpu
    comfy.model_management.load_model_gpu = lambda *a, **kw: None
    try:
        with torch.no_grad():
            output = clip_vision.encode_image(batch, crop="center")
    finally:
        comfy.model_management.load_model_gpu = _orig

    embeds = output.image_embeds
    if embeds.dim() > 2:
        embeds = embeds.view(embeds.shape[0], -1)
    return embeds.detach().cpu()


def _make_attention_mask(bbox, H, W, sigma_factor=0.3):
    """Create a soft attention mask from a bbox with Gaussian-like falloff."""
    x1, y1, x2, y2 = bbox
    # Create coordinate grid
    ys = torch.linspace(0, 1, H)
    xs = torch.linspace(0, 1, W)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')

    # bbox center and size in normalized coords
    cx = ((x1 + x2) / 2.0) / W
    cy = ((y1 + y2) / 2.0) / H
    bw = max((x2 - x1) / W, 0.01)
    bh = max((y2 - y1) / H, 0.01)

    # Gaussian mask
    sigma_x = bw * sigma_factor * 2
    sigma_y = bh * sigma_factor * 2
    mask = torch.exp(-0.5 * (((grid_x - cx) / sigma_x) ** 2 + ((grid_y - cy) / sigma_y) ** 2))
    return mask.unsqueeze(0)  # (1, H, W)


class SoyaFaceIDMaskPatcher_mdsoya:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"forceInput": True}),
                "ipadapter_file": (cls._get_ipadapter_files(),),
                "clip_vision": ("CLIP_VISION",),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "weight_faceidv2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "embeds_scaling": (["V only", "K+V", "K+V+Biased"], {"default": "V only"}),
            },
        }

    RETURN_TYPES = ("MODEL", "IMAGE", "MASK")
    RETURN_NAMES = ("model", "image", "mask")
    FUNCTION = "execute"
    CATEGORY = "soya"

    @staticmethod
    def _get_ipadapter_files():
        try:
            import folder_paths
            files = folder_paths.get_filename_list("ipadapter")
            if files:
                return files
        except Exception:
            pass
        return [""]

    def execute(self, model, image, prompt, ipadapter_file, clip_vision,
                lora_strength, weight, weight_faceidv2, start_at, end_at, embeds_scaling):
        # ── Import IPAdapter dependencies ──
        try:
            from comfyui_ipadapter_plus.IPAdapterPlus import (
                IPAdapter, ipadapter_execute, set_model_patch_replace,
            )
            from comfyui_ipadapter_plus.CrossAttentionPatch import ipadapter_attention
            from comfyui_ipadapter_plus.utils import (
                ipadapter_model_loader, encode_image_masked, insightface_loader,
                get_lora_file, image_to_tensor,
            )
        except ImportError:
            raise ImportError(
                "comfyui_ipadapter_plus is required for SoyaFaceIDMaskPatcher. "
                "Install it from: https://github.com/cubiq/ComfyUI_IPAdapter_plus"
            )

        import comfy.model_management
        import folder_paths
        from safetensors.torch import load_file as load_safetensors

        # Clear stale attn2 patches from previous runs to prevent accumulation
        transformer_options = model.model_options.setdefault("transformer_options", {})
        patches_replace = transformer_options.setdefault("patches_replace", {})
        if "attn2" in patches_replace:
            del patches_replace["attn2"]

        device = comfy.model_management.get_torch_device()
        dtype = comfy.model_management.unet_dtype()
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            dtype = torch.float16 if comfy.model_management.should_use_fp16() else torch.float32

        # ── Load settings ──
        node_info = _load_node_info()
        settings = node_info.get("settings", {})
        base_path = settings.get("base_path", "")
        bbox_model_name = settings.get("bbox_model", "face_yolov8m.pt")
        threshold = settings.get("bbox_threshold", 0.5)
        mask_sigma_factor = float(settings.get("faceid_mask_sigma_factor", 0.4))
        crop_zoom = float(settings.get("faceid_crop_zoom", 1.0))

        if not base_path:
            raise ValueError("base_path not configured in Soya Scheduler settings")

        # ── Load YOLO model ──
        from ultralytics import YOLO
        import folder_paths as fp
        bbox_path = fp.get_full_path("ultralytics_bbox", bbox_model_name)
        if bbox_path is None:
            raise FileNotFoundError(f"YOLO model not found: {bbox_model_name}")
        yolo = YOLO(bbox_path)

        # ── Detect faces ──
        faces = _detect_faces_yolo(image, yolo, threshold)
        if not faces:
            # No faces detected, return unmodified
            B, H, W = image.shape[:3]
            empty_mask = torch.ones(B, H, W, dtype=torch.float32)
            return (model, image, empty_mask)

        # ── Load and filter characters ──
        from .soya_scheduler.config_manager import load_characters, find_faceid_embed, find_image_file
        from .soya_scheduler.ray_worker import _filter_characters_by_prompt as _filter_chars

        all_chars = load_characters(base_path)
        matched_chars = _filter_chars(all_chars, prompt)

        # Only keep characters that have a faceid embed
        chars_with_embed = []
        for ch in matched_chars:
            embed_path = find_faceid_embed(base_path, ch["name"])
            if embed_path:
                ch["_embed_path"] = embed_path
                chars_with_embed.append(ch)

        if not chars_with_embed:
            # No characters with faceid embeds matched
            B, H, W = image.shape[:3]
            empty_mask = torch.ones(B, H, W, dtype=torch.float32)
            return (model, image, empty_mask)

        # ── Load pre-computed embeddings and build reference crops for matching ──
        ref_embeds_list = []
        ref_crops_list = []
        char_names = []

        for ch in chars_with_embed:
            st = load_safetensors(ch["_embed_path"])
            face_crop = st["face_crop"]  # (3, 256, 256) float32
            face_crop_np = (face_crop.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            ref_crops_list.append(face_crop_np)
            char_names.append(ch["name"])
            del st

        # ── Extract face crops from detected faces for CLIP matching ──
        if image.dim() == 4:
            img_tensor = image[0]  # (H, W, 3)
        else:
            img_tensor = image
        img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
        H, W = img_np.shape[:2]
        pil_img = Image.fromarray(img_np)

        det_crops = []
        for f in faces:
            x1, y1, x2, y2 = f["bbox"]
            crop = pil_img.crop((x1, y1, x2, y2))
            det_crops.append(np.array(crop))

        # ── CLIP Vision matching ──
        # Ensure clip_vision is loaded on device
        comfy.model_management.load_model_gpu(clip_vision.patcher)

        query_embeds = _encode_images_for_matching(clip_vision, det_crops)
        ref_embeds = _encode_images_for_matching(clip_vision, ref_crops_list)
        assignments, scores = _match_characters_hungarian(query_embeds, ref_embeds, char_names)

        # ── Generate attention masks per character ──
        char_masks = {}
        for i, (face, name) in enumerate(zip(faces, assignments)):
            if name == "unknown":
                continue
            mask = _make_attention_mask(face["bbox"], H, W, sigma_factor=mask_sigma_factor)
            if name not in char_masks:
                char_masks[name] = mask
            else:
                # Multiple faces assigned to same character: take max
                char_masks[name] = torch.max(char_masks[name], mask)

        # Combine all character masks into a single mask for the output
        combined_mask = torch.zeros(1, H, W)
        for m in char_masks.values():
            combined_mask = torch.max(combined_mask, m)

        # Expand to batch
        B = image.shape[0]
        combined_mask = combined_mask.expand(B, -1, -1).clone()

        # ── Load IPAdapter model ──
        ipadapter_path = folder_paths.get_full_path("ipadapter", ipadapter_file)
        if ipadapter_path is None:
            raise FileNotFoundError(f"IPAdapter file not found: {ipadapter_file}")

        ipadapter = ipadapter_model_loader(ipadapter_path)

        # Detect model type (match official IPAdapter Plus detection)
        is_full = "proj.3.weight" in ipadapter["image_proj"]
        is_portrait = "proj.2.weight" in ipadapter["image_proj"] and not "proj.3.weight" in ipadapter["image_proj"] and not "0.to_q_lora.down.weight" in ipadapter["ip_adapter"]
        is_portrait_unnorm = "portraitunnorm" in ipadapter
        is_faceid = is_portrait or "0.to_q_lora.down.weight" in ipadapter["ip_adapter"] or is_portrait_unnorm
        is_plus = (is_full or "latents" in ipadapter["image_proj"] or "perceiver_resampler.proj_in.weight" in ipadapter["image_proj"]) and not is_portrait_unnorm
        is_faceidv2 = "faceidplusv2" in ipadapter
        output_cross_attention_dim = ipadapter["ip_adapter"]["1.to_k_ip.weight"].shape[1]
        is_sdxl = output_cross_attention_dim == 2048

        if not (is_faceid and is_plus):
            raise ValueError(
                f"IPAdapter model {ipadapter_file} is not a FaceID Plus model. "
                "Please use a FaceID Plus V2 model (e.g. ip-adapter-faceid-plusv2)."
            )

        cross_attention_dim = output_cross_attention_dim
        clip_extra_context_tokens = 4

        # ── Apply companion LoRA ──
        # Auto-detect companion LoRA: search both models/ipadapter/ and models/loras/
        lora_file = None
        basename = os.path.splitext(os.path.basename(ipadapter_file))[0]
        lora_candidates = [
            basename + "_lora.safetensors",  # e.g. ip-adapter-faceid-plusv2_sdxl_lora.safetensors
            basename + ".lora.safetensors",   # alternate naming
        ]
        # Check models/ipadapter/ first
        for candidate in lora_candidates:
            path = folder_paths.get_full_path("ipadapter", candidate)
            if path is not None:
                lora_file = path
                break
        # Fall back to models/loras/
        if lora_file is None:
            import re
            for candidate in lora_candidates:
                lora_list = folder_paths.get_filename_list("loras")
                match = [e for e in lora_list if e == candidate]
                if match:
                    lora_file = folder_paths.get_full_path("loras", match[0])
                    break

        if lora_file is not None and lora_strength > 0:
            from comfy.sd import load_lora_for_models
            lora_model = comfy.utils.load_torch_file(lora_file, safe_load=True)
            model, _ = load_lora_for_models(model, None, lora_model, lora_strength, 0)
            print(f"[Soya:FaceID] Loaded companion LoRA: {lora_file}")
        elif lora_file is None:
            print(f"[Soya:FaceID] Warning: companion LoRA not found for {ipadapter_file}")

        # ── Encode uncond first to get clip_embeddings_dim ──
        img_uncond = torch.zeros([1, 224, 224, 3])
        img_uncond_embeds = encode_image_masked(
            clip_vision, img_uncond, batch_size=0
        ).penultimate_hidden_states.to(device, dtype=dtype)

        # ── Create single IPAdapter instance (shared weights) ──
        ipa = IPAdapter(
            ipadapter,
            cross_attention_dim=cross_attention_dim,
            output_cross_attention_dim=output_cross_attention_dim,
            clip_embeddings_dim=img_uncond_embeds.shape[-1],
            clip_extra_context_tokens=clip_extra_context_tokens,
            is_sdxl=is_sdxl,
            is_plus=is_plus,
            is_full=False,
            is_faceid=is_faceid,
            is_portrait_unnorm=False,
        ).to(device, dtype=dtype)

        # ── Sigma range ──
        sigma_start = model.get_model_object("model_sampling").percent_to_sigma(start_at)
        sigma_end = model.get_model_object("model_sampling").percent_to_sigma(end_at)

        # ── For each matched character, create and install patches ──
        for char_name, char_mask in char_masks.items():
            # Find the character's embed
            ch = next(c for c in chars_with_embed if c["name"] == char_name)
            st = load_safetensors(ch["_embed_path"])
            face_embed = st["face_embed"].to(device, dtype=dtype)  # (1, 512)
            face_crop_t = st["face_crop"].unsqueeze(0)  # (1, 3, 256, 256)
            del st

            # face_crop is (3, 256, 256) → convert to BHWC for encode_image_masked
            face_crop_bhwc = face_crop_t.squeeze(0).permute(1, 2, 0)  # (256, 256, 3)
            face_crop_bhwc = face_crop_bhwc.unsqueeze(0)  # (1, 256, 256, 3)

            # Apply crop zoom to exclude hair area from CLIP encoding
            if crop_zoom > 1.0:
                import torch.nn.functional as F
                _, fc_h, fc_w, _ = face_crop_bhwc.shape
                new_h = int(fc_h / crop_zoom)
                new_w = int(fc_w / crop_zoom)
                top = (fc_h - new_h) // 2
                left = (fc_w - new_w) // 2
                face_crop_clipped = face_crop_bhwc[:, top:top+new_h, left:left+new_w, :]
                face_crop_clipped = face_crop_clipped.permute(0, 3, 1, 2)
                face_crop_clipped = F.interpolate(face_crop_clipped, size=(256, 256), mode='bilinear', align_corners=False)
                face_crop_clipped = face_crop_clipped.permute(0, 2, 3, 1)
                face_crop_bhwc = face_crop_clipped

            # CLIP Vision encode
            img_cond_embeds = encode_image_masked(
                clip_vision, face_crop_bhwc, batch_size=0
            ).penultimate_hidden_states.to(device, dtype=dtype)

            # Project through IPAdapter
            cond = ipa.get_image_embeds_faceid_plus(
                face_embed, img_cond_embeds, weight_faceidv2, is_faceidv2, 0
            ).to(device, dtype=dtype)

            uncond = ipa.get_image_embeds_faceid_plus(
                torch.zeros_like(face_embed), img_uncond_embeds, weight_faceidv2, is_faceidv2, 0
            ).to(device, dtype=dtype)

            # Prepare attention mask for this character
            attn_mask = char_mask.to(device, dtype=dtype)

            # Build patch kwargs
            patch_kwargs = {
                "ipadapter": ipa,
                "weight": weight,
                "cond": cond,
                "cond_alt": None,
                "uncond": uncond,
                "weight_type": "standard",
                "mask": attn_mask,
                "sigma_start": sigma_start,
                "sigma_end": sigma_end,
                "unfold_batch": False,
                "embeds_scaling": embeds_scaling,
            }

            # Install patches on all cross-attention layers
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

        return (model, image, combined_mask)


# ── Split-pipeline node: lightweight model patcher ──────────────────

class SoyaFaceIDModelPatcher_mdsoya:
    """Takes FACEID_PATCH from SoyaProcessCollectorEmbedimg and a MODEL,
    applies companion LoRA + IPAdapter attention patches, returns patched MODEL.

    This node is intentionally lightweight — all heavy computation (CLIP encoding,
    IPAdapter projection, mask generation) happens in the collector node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "faceid_patch": ("FACEID_PATCH",),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "Soya/Scheduler"

    def patch(self, model, faceid_patch):
        if faceid_patch is None:
            return (model,)

        from comfyui_ipadapter_plus.IPAdapterPlus import set_model_patch_replace
        from comfyui_ipadapter_plus.CrossAttentionPatch import ipadapter_attention

        # Clear stale attn2 patches from previous runs to prevent accumulation
        transformer_options = model.model_options.setdefault("transformer_options", {})
        patches_replace = transformer_options.setdefault("patches_replace", {})
        if "attn2" in patches_replace:
            del patches_replace["attn2"]

        import comfy.model_management

        device = comfy.model_management.get_torch_device()
        dtype = comfy.model_management.unet_dtype()

        ipa = faceid_patch["ipadapter"]
        char_patches = faceid_patch["char_patches"]
        is_sdxl = faceid_patch["is_sdxl"]
        weight = faceid_patch["weight"]
        start_at = faceid_patch["start_at"]
        end_at = faceid_patch["end_at"]
        embeds_scaling = faceid_patch["embeds_scaling"]
        lora_strength = faceid_patch["lora_strength"]
        lora_file = faceid_patch.get("lora_file")

        # ── Apply companion LoRA ──
        if lora_file is not None and lora_strength > 0:
            import comfy.utils
            from comfy.sd import load_lora_for_models
            lora_model = comfy.utils.load_torch_file(lora_file, safe_load=True)
            model, _ = load_lora_for_models(model, None, lora_model, lora_strength, 0)

        # ── Sigma range ──
        sigma_start = model.get_model_object("model_sampling").percent_to_sigma(start_at)
        sigma_end = model.get_model_object("model_sampling").percent_to_sigma(end_at)

        # ── Install patches per character ──
        for char_idx, cp in enumerate(char_patches):
            patch_kwargs = {
                "ipadapter": ipa,
                "weight": weight,
                "cond": cp["cond"],
                "cond_alt": None,
                "uncond": cp["uncond"],
                "weight_type": "standard",
                "mask": cp["mask"].to(device, dtype=dtype),
                "sigma_start": sigma_start,
                "sigma_end": sigma_end,
                "unfold_batch": False,
                "embeds_scaling": embeds_scaling,
            }

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

        # ── Verify patches were accumulated correctly ──
        attn2 = model.model_options.get("transformer_options", {}).get("patches_replace", {}).get("attn2", {})
        if attn2:
            sample_key = list(attn2.keys())[0]
            n_callbacks = len(attn2[sample_key].callback)
            print(f"[Soya:FaceIDModelPatcher] {len(char_patches)} characters, {len(attn2)} layers, {n_callbacks} callbacks per layer (expected {len(char_patches)})")
        else:
            print(f"[Soya:FaceIDModelPatcher] WARNING: no attn2 patches found after install!")
        return (model,)
