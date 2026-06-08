"""
SoyaIPAApplyFacePatches – Apply per-face IP-Adapter patches using face context.

Takes IPA_FACE_CONTEXT (from IPA Patch Maker) + MODEL + IPADAPTER + CLIP_VISION,
creates feathered masks from each face's bounding box, and applies IP-Adapter
patches cumulatively. Each face only influences its own masked region.

Workflow:
  [IPA Patch Maker] → face_context → [This Node] → patched MODEL → [KSampler]
  [Model] [IPAdapter] [CLIP Vision] ────────↑
"""

import os
import sys
import torch
import numpy as np
from PIL import Image, ImageFilter

WEIGHT_TYPES = [
    "linear", "ease in", "ease out", "ease in-out", "reverse in-out",
    "weak input", "weak output", "weak middle", "strong middle",
    "style transfer", "composition", "strong style transfer",
]


class SoyaIPAApplyFacePatches_mdsoya:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "face_context": ("IPA_FACE_CONTEXT",),
                "model": ("MODEL",),
                "ipadapter": ("IPADAPTER",),
                "clip_vision": ("CLIP_VISION",),
                "feather_radius": ("INT", {"default": 20, "min": 0, "max": 100, "step": 1}),
                "mask_expansion": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 3.0, "step": 0.1}),
                "weight_type": (WEIGHT_TYPES,),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "embeds_scaling": (["V only", "K+V", "K+V w/ C penalty", "K+mean(V) w/ C penalty"],),
            },
        }

    RETURN_TYPES = ("MODEL", "MASK", "MASK", "STRING", "STRING")
    RETURN_NAMES = ("model", "face_masks", "combined_mask", "face_names", "info")
    OUTPUT_IS_LIST = (False, True, False, True, False)
    FUNCTION = "process"
    CATEGORY = "Soya/IPA"

    def process(self, face_context, model, ipadapter, clip_vision,
                feather_radius, mask_expansion,
                weight_type, start_at, end_at, embeds_scaling):

        if not face_context or not face_context.get("matches"):
            return (model, [], torch.zeros(1, 1, 1), [], "No face context provided.")

        matches = face_context["matches"]
        ipa_embeds = face_context.get("ipa_embeds", {})
        img_H = face_context.get("img_H", 0)
        img_W = face_context.get("img_W", 0)

        if img_H == 0 or img_W == 0:
            return (model, [], torch.zeros(1, 1, 1), [], "Invalid image dimensions in context.")

        # ── Import IPAdapterEmbeds ──
        custom_nodes_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if custom_nodes_dir not in sys.path:
            sys.path.append(custom_nodes_dir)
        try:
            from ComfyUI_IPAdapter_plus.IPAdapterPlus import IPAdapterEmbeds
        except ImportError:
            from comfyui_ipadapter_plus.IPAdapterPlus import IPAdapterEmbeds

        ipa_node = IPAdapterEmbeds()

        # ── Create masks + apply patches per matched face ──
        masks = []
        matched_names = []
        info_lines = []
        patched_model = model

        for i, match in enumerate(matches):
            name = match["name"]
            bbox = match["bbox"]
            score = match["score"]

            if name == "unknown":
                info_lines.append(f"  Face {i+1}: unknown (score: {score:.4f}) -- SKIPPED")
                continue

            if name not in ipa_embeds:
                info_lines.append(f"  Face {i+1}: {name} (score: {score:.4f}) -- NO IPA CACHE")
                continue

            bx1, by1, bx2, by2 = bbox
            cx = (bx1 + bx2) / 2
            cy = (by1 + by2) / 2
            bw = bx2 - bx1
            bh = by2 - by1

            ex1 = max(0, int(cx - bw * mask_expansion / 2))
            ey1 = max(0, int(cy - bh * mask_expansion / 2))
            ex2 = min(img_W, int(cx + bw * mask_expansion / 2))
            ey2 = min(img_H, int(cy + bh * mask_expansion / 2))

            mask_np = np.zeros((img_H, img_W), dtype=np.float32)
            mask_np[ey1:ey2, ex1:ex2] = 1.0

            if feather_radius > 0:
                mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
                mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=feather_radius))
                mask_np = np.array(mask_pil).astype(np.float32) / 255.0

            mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)  # [1, H, W]

            entry = ipa_embeds[name]
            result = ipa_node.apply_ipadapter(
                patched_model, ipadapter, entry["embeds"], entry["strength"],
                weight_type, start_at, end_at, None, mask_tensor, clip_vision,
                embeds_scaling,
            )
            patched_model = result[0]

            masks.append(mask_tensor)
            matched_names.append(name)
            info_lines.append(
                f"  Face {i+1}: {name} (score: {score:.4f}, weight: {entry['strength']:.2f})"
            )

        if not masks:
            info = f"Matched {len(matches)} faces but 0 IPA patches applied.\n" + "\n".join(info_lines)
            return (model, [], torch.zeros(1, img_H, img_W), [], info)

        # Combine all face masks into one (max = OR)
        combined = torch.stack(masks, dim=0).max(dim=0).values  # [1, H, W]

        info = f"Applied {len(masks)} IPA patches\n" + "\n".join(info_lines)
        print(f"[IPAApplyFacePatches] {info}")

        return (patched_model, masks, combined, matched_names, info)
