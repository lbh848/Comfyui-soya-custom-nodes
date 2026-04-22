"""
SoyaProcessCollectorEmbedimg – awaits Ray worker result, loads pre-computed
FaceID embeddings (face_embed + clip_embed), projects through IPAdapter,
and outputs FACEID_PATCH data (per-character cond/uncond/mask).

Inputs: task_ref + LATENT only. All settings come from Soya Scheduler config.
CLIP Vision is NOT needed – clip_embed is pre-computed in the safetensors file.
"""

import os
import torch

from .soya_scheduler.task_store import pop
from .soya_scheduler.config_manager import load_config, find_faceid_embed


def _make_attention_mask(bbox, H, W, sigma_factor=0.4):
    """Create a soft attention mask from a bbox with Gaussian-like falloff."""
    x1, y1, x2, y2 = bbox
    ys = torch.linspace(0, 1, H)
    xs = torch.linspace(0, 1, W)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')

    cx = ((x1 + x2) / 2.0) / W
    cy = ((y1 + y2) / 2.0) / H
    bw = max((x2 - x1) / W, 0.01)
    bh = max((y2 - y1) / H, 0.01)

    sigma_x = bw * sigma_factor * 2
    sigma_y = bh * sigma_factor * 2
    mask = torch.exp(-0.5 * (((grid_x - cx) / sigma_x) ** 2 + ((grid_y - cy) / sigma_y) ** 2))
    return mask.unsqueeze(0)  # (1, H, W)


class SoyaProcessCollectorEmbedimg_mdsoya:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task_ref": ("STRING", {"forceInput": True}),
                "latent": ("LATENT", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("FACEID_PATCH", "MASK", "LATENT")
    RETURN_NAMES = ("faceid_patch", "mask", "latent")
    FUNCTION = "collect_embed"
    CATEGORY = "Soya/Scheduler"

    def collect_embed(self, task_ref, latent):
        # ── Import IPAdapter dependencies ──
        try:
            from comfyui_ipadapter_plus.IPAdapterPlus import IPAdapter
            from comfyui_ipadapter_plus.utils import (
                ipadapter_model_loader, get_lora_file,
            )
        except ImportError:
            raise ImportError(
                "comfyui_ipadapter_plus is required. "
                "Install from: https://github.com/cubiq/ComfyUI_IPAdapter_plus"
            )

        import comfy.model_management
        import folder_paths
        from safetensors.torch import load_file as load_safetensors

        # ── Read all settings from config ──
        config = load_config()
        settings = config.get("settings", {})
        base_path = settings.get("base_path", "")
        if not base_path:
            raise ValueError("base_path not configured in Soya Scheduler settings")

        ipadapter_file = settings.get("faceid_ipadapter_file", "")
        weight = float(settings.get("faceid_weight", 1.0))
        weight_faceidv2 = float(settings.get("faceid_weight_faceidv2", 1.0))
        start_at = float(settings.get("faceid_start_at", 0.0))
        end_at = float(settings.get("faceid_end_at", 1.0))
        embeds_scaling = settings.get("faceid_embeds_scaling", "V only")
        lora_strength = float(settings.get("faceid_lora_strength", 1.0))
        mask_sigma_factor = float(settings.get("faceid_mask_sigma_factor", 0.4))

        # Auto-detect IPAdapter file if not set
        if not ipadapter_file:
            ipadapter_files = folder_paths.get_filename_list("ipadapter")
            faceid_files = [f for f in ipadapter_files if "faceid" in f.lower()]
            if faceid_files:
                ipadapter_file = faceid_files[0]
            elif ipadapter_files:
                ipadapter_file = ipadapter_files[0]
            else:
                raise FileNotFoundError("No IPAdapter model files found")

        # ── Await Ray result ──
        task = pop(task_ref)
        if task is None:
            return (None, torch.ones(1, 64, 64, dtype=torch.float32), latent)

        import ray
        future = task["future"]
        result = ray.get(future)

        kept_faces = result.get("kept_faces", [])
        if not kept_faces:
            return (None, torch.ones(1, 64, 64, dtype=torch.float32), latent)

        # ── Filter kept faces: only those with faceid embeds ──
        faces_with_embeds = []
        for f in kept_faces:
            name = f.get("assignment", "unknown")
            if name == "unknown":
                continue
            embed_path = find_faceid_embed(base_path, name)
            if embed_path:
                faces_with_embeds.append({
                    "name": name,
                    "bbox": f["bbox"],
                    "embed_path": embed_path,
                })

        if not faces_with_embeds:
            return (None, torch.ones(1, 64, 64, dtype=torch.float32), latent)

        device = comfy.model_management.get_torch_device()
        dtype = comfy.model_management.unet_dtype()
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            dtype = torch.float16 if comfy.model_management.should_use_fp16() else torch.float32

        # ── Load IPAdapter model ──
        ipadapter_path = folder_paths.get_full_path("ipadapter", ipadapter_file)
        if ipadapter_path is None:
            raise FileNotFoundError(f"IPAdapter file not found: {ipadapter_file}")
        ipadapter = ipadapter_model_loader(ipadapter_path)

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
                "Please use a FaceID Plus V2 model."
            )

        cross_attention_dim = output_cross_attention_dim
        clip_extra_context_tokens = 4

        # ── Get uncond clip embed shape from first faceid embed ──
        first_st = load_safetensors(faces_with_embeds[0]["embed_path"])
        ref_ipa = first_st.get("clip_embed_ipa")
        if ref_ipa is None:
            raise ValueError(
                "clip_embed_ipa not found in faceid embed. "
                "Set 'FaceID CLIP Vision Model' in Soya Scheduler settings and regenerate embeds."
            )
        clip_embeddings_dim = ref_ipa.shape[-1]
        uncond_clip_embed = torch.zeros_like(ref_ipa).to(device, dtype=dtype)
        del first_st

        # ── Create IPAdapter instance (shared weights) ──
        ipa = IPAdapter(
            ipadapter,
            cross_attention_dim=cross_attention_dim,
            output_cross_attention_dim=output_cross_attention_dim,
            clip_embeddings_dim=clip_embeddings_dim,
            clip_extra_context_tokens=clip_extra_context_tokens,
            is_sdxl=is_sdxl,
            is_plus=is_plus,
            is_full=False,
            is_faceid=is_faceid,
            is_portrait_unnorm=False,
        ).to(device, dtype=dtype)

        # ── Latent dimensions for masks ──
        latent_samples = latent["samples"]  # (B, C, H_latent, W_latent)
        _, _, H_lat, W_lat = latent_samples.shape
        # Convert latent dims to pixel-space (approximate, assuming 8x downscale)
        H = H_lat * 8
        W = W_lat * 8
        B = latent_samples.shape[0]

        # ── Per-character patch computation ──
        char_patches = []
        combined_mask = torch.zeros(1, H, W)

        for face_info in faces_with_embeds:
            st = load_safetensors(face_info["embed_path"])
            face_embed = st["face_embed"].to(device, dtype=dtype)  # (1, 512)
            clip_embed_ipa = st.get("clip_embed_ipa")
            if clip_embed_ipa is None:
                print(f"[Soya:CollectorEmbedimg] Warning: {face_info['name']} has no clip_embed_ipa, skipping")
                del st
                continue
            clip_embed_ipa = clip_embed_ipa.to(device, dtype=dtype)
            del st

            # Project through IPAdapter (pre-computed, no CLIP Vision at runtime)
            cond = ipa.get_image_embeds_faceid_plus(
                face_embed, clip_embed_ipa, weight_faceidv2, is_faceidv2, 0
            ).to(device, dtype=dtype)

            uncond = ipa.get_image_embeds_faceid_plus(
                torch.zeros_like(face_embed), uncond_clip_embed, weight_faceidv2, is_faceidv2, 0
            ).to(device, dtype=dtype)

            # Attention mask from bbox (pixel-space)
            attn_mask = _make_attention_mask(face_info["bbox"], H, W, sigma_factor=mask_sigma_factor)
            combined_mask = torch.max(combined_mask, attn_mask)

            char_patches.append({
                "cond": cond,
                "uncond": uncond,
                "mask": attn_mask,
            })

        if not char_patches:
            return (None, torch.ones(B, H, W, dtype=torch.float32), latent)

        # Expand mask to batch
        combined_mask = combined_mask.expand(B, -1, -1).clone()

        # ── Detect companion LoRA ──
        lora_file = None
        basename = os.path.splitext(os.path.basename(ipadapter_file))[0]
        lora_candidates = [
            basename + "_lora.safetensors",
            basename + ".lora.safetensors",
        ]
        for candidate in lora_candidates:
            path = folder_paths.get_full_path("ipadapter", candidate)
            if path is not None:
                lora_file = path
                break
        if lora_file is None:
            for candidate in lora_candidates:
                lora_list = folder_paths.get_filename_list("loras")
                match = [e for e in lora_list if e == candidate]
                if match:
                    lora_file = folder_paths.get_full_path("loras", match[0])
                    break
        if lora_file is not None:
            print(f"[Soya:CollectorEmbedimg] Found companion LoRA: {lora_file}")
        else:
            print(f"[Soya:CollectorEmbedimg] Warning: companion LoRA not found for {ipadapter_file}")

        # ── Package FACEID_PATCH ──
        faceid_patch = {
            "ipadapter": ipa,
            "char_patches": char_patches,
            "is_sdxl": is_sdxl,
            "weight": weight,
            "start_at": start_at,
            "end_at": end_at,
            "embeds_scaling": embeds_scaling,
            "lora_strength": lora_strength,
            "lora_file": lora_file,
        }

        print(f"[Soya:CollectorEmbedimg] Computed {len(char_patches)} character patches")
        return (faceid_patch, combined_mask, latent)
