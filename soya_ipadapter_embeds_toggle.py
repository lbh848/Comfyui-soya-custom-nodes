"""
SoyaIPAdapterEmbedsToggle – Wrapper around IPAdapterEmbeds with enable/disable toggle.

enable="true"  → delegate to IPAdapterEmbeds.apply_ipadapter (model patched)
enable="false" → return model as-is (passthrough)

Note: Use IPAdapter Patch Cleaner (Soya) to clear stale patches before this node.
"""

import os
import sys


WEIGHT_TYPES = [
    "linear", "ease in", "ease out", "ease in-out", "reverse in-out",
    "weak input", "weak output", "weak middle", "strong middle",
    "style transfer", "composition", "strong style transfer",
]


class SoyaIPAdapterEmbedsToggle_mdsoya:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enable": ("STRING", {"default": "true", "multiline": False}),
                "model": ("MODEL",),
                "ipadapter": ("IPADAPTER",),
                "pos_embed": ("EMBEDS",),
                "weight": ("FLOAT", {"default": 1.0, "min": -1, "max": 3, "step": 0.05}),
                "weight_type": (WEIGHT_TYPES,),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "embeds_scaling": (["V only", "K+V", "K+V w/ C penalty", "K+mean(V) w/ C penalty"],),
            },
            "optional": {
                "neg_embed": ("EMBEDS",),
                "attn_mask": ("MASK",),
                "clip_vision": ("CLIP_VISION",),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "execute"
    CATEGORY = "ipadapter/faceid"

    def execute(self, enable, model, ipadapter, pos_embed, weight, weight_type,
                start_at, end_at, embeds_scaling,
                neg_embed=None, attn_mask=None, clip_vision=None):
        use = enable.strip().lower() in ("true", "1", "yes")

        if not use:
            print("[Soya:EmbedsToggle] DISABLED — skipping IPAdapter patch")
            return (model,)

        # Delegate to original IPAdapterEmbeds
        custom_nodes_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if custom_nodes_dir not in sys.path:
            sys.path.append(custom_nodes_dir)

        try:
            from ComfyUI_IPAdapter_plus.IPAdapterPlus import IPAdapterEmbeds
        except ImportError:
            from comfyui_ipadapter_plus.IPAdapterPlus import IPAdapterEmbeds

        node = IPAdapterEmbeds()
        return node.apply_ipadapter(
            model, ipadapter, pos_embed, weight, weight_type,
            start_at, end_at, neg_embed, attn_mask, clip_vision,
            embeds_scaling,
        )
