"""
SoyaIPAdapterEmbedCache – Encode images from a directory, combine embeds, cache result.

Flow:
  1. Check if cache file exists at <path>/cach.SDC
     - If exists: load raw embeds from cache, combine with current method, return
  2. Otherwise: load images → CLIP Vision encode → save raw embeds to cache → combine → return
"""

import os
import sys
from collections import Counter

import numpy as np
import torch
import folder_paths
from PIL import Image, ImageOps
import node_helpers

SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff', '.tif', '.gif'}
CACHE_FILENAME = "cache.ipadpt"

COMBINE_METHODS = ["average", "norm average", "concat", "add", "subtract", "max", "min"]


def _detect_is_plus(ipadapter):
    """Auto-detect plus model from ipadapter checkpoint keys."""
    image_proj = ipadapter.get("image_proj", {})
    return (
        "proj.3.weight" in image_proj
        or "latents" in image_proj
        or "perceiver_resampler.proj_in.weight" in image_proj
    )


def _combine_embeds(embeds, method):
    """Combine a batch of embeds [N, ...] along dim=0 using the specified method."""
    if method == "concat":
        return embeds
    elif method == "add":
        return torch.sum(embeds, dim=0).unsqueeze(0)
    elif method == "subtract":
        return (embeds[0] - torch.mean(embeds[1:], dim=0)).unsqueeze(0)
    elif method == "average":
        return torch.mean(embeds, dim=0).unsqueeze(0)
    elif method == "norm average":
        return torch.mean(
            embeds / torch.norm(embeds, dim=0, keepdim=True), dim=0,
        ).unsqueeze(0)
    elif method == "max":
        return torch.max(embeds, dim=0).values.unsqueeze(0)
    elif method == "min":
        return torch.min(embeds, dim=0).values.unsqueeze(0)
    return embeds


def _import_encode_image_masked():
    """Import encode_image_masked from IPAdapter Plus (handles both folder name cases)."""
    custom_nodes_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if custom_nodes_dir not in sys.path:
        sys.path.append(custom_nodes_dir)
    try:
        from ComfyUI_IPAdapter_plus.utils import encode_image_masked
    except ImportError:
        from comfyui_ipadapter_plus.utils import encode_image_masked
    return encode_image_masked


class SoyaIPAdapterEmbedCache_mdsoya:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": "", "multiline": False}),
                "ipadapter": ("IPADAPTER",),
                "clip_vision": ("CLIP_VISION",),
                "combine_method": (COMBINE_METHODS,),
            },
        }

    RETURN_TYPES = ("EMBEDS",)
    RETURN_NAMES = ("embeds",)
    FUNCTION = "execute"
    CATEGORY = "ipadapter/faceid"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def execute(self, path, ipadapter, clip_vision, combine_method):
        import comfy.model_management

        # ── Resolve path ──
        path = path.strip()
        if not os.path.isabs(path):
            input_dir = folder_paths.get_input_directory()
            path = os.path.join(input_dir, path)
        if not os.path.isdir(path):
            raise ValueError(f"Directory not found: {path}")

        cache_path = os.path.join(path, CACHE_FILENAME)

        # ── Cache hit: load raw embeds, combine, return ──
        if os.path.isfile(cache_path):
            print(f"[Soya:EmbedCache] Loading cached embeds from {cache_path}")
            raw_embeds = torch.load(cache_path).cpu()
            combined = _combine_embeds(raw_embeds, combine_method)
            print(f"[Soya:EmbedCache] Combined ({combine_method}): {raw_embeds.shape[0]} images → {combined.shape}")
            return (combined,)

        # ── Cache miss: encode images ──
        encode_image_masked = _import_encode_image_masked()
        is_plus = _detect_is_plus(ipadapter)

        # Load images
        files = sorted([
            f for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f))
            and os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
        ])
        if not files:
            raise ValueError(f"No supported image files found in: {path}")

        loaded = []
        for filename in files:
            filepath = os.path.join(path, filename)
            try:
                img = node_helpers.pillow(Image.open, filepath)
                img = ImageOps.exif_transpose(img)
                if img.mode == 'I':
                    img = img.point(lambda i: i * (1 / 255))
                img = img.convert("RGB")
                loaded.append(img)
            except Exception:
                continue

        if not loaded:
            raise ValueError(f"No valid images could be loaded from: {path}")

        # Resize to majority size
        size_counts = Counter(img.size for img in loaded)
        target_w, target_h = size_counts.most_common(1)[0][0]

        images = []
        for img in loaded:
            if img.size != (target_w, target_h):
                img = img.resize((target_w, target_h), Image.LANCZOS)
            tensor = torch.from_numpy(np.array(img).astype(np.float32) / 255.0)[None,]
            images.append(tensor)

        batch = torch.cat(images, dim=0)

        # CLIP Vision encode
        comfy.model_management.load_model_gpu(clip_vision.patcher)
        encoded = encode_image_masked(clip_vision, batch, batch_size=0)

        if is_plus:
            raw_embeds = encoded.penultimate_hidden_states
        else:
            raw_embeds = encoded.image_embeds

        # Save raw embeds to cache (before combine, so combine_method changes don't need re-encoding)
        torch.save(raw_embeds, cache_path)
        print(f"[Soya:EmbedCache] Saved raw embeds to {cache_path}")
        print(f"[Soya:EmbedCache] Images: {len(loaded)}, Plus: {is_plus}, Raw shape: {raw_embeds.shape}")

        # Combine and return
        combined = _combine_embeds(raw_embeds, combine_method)
        print(f"[Soya:EmbedCache] Combined ({combine_method}): {combined.shape}")

        return (combined,)
