"""
SoyaSegModelProvider – Loads ISNet eye/eyebrow segmentation models.

Scans models/soya_seg/ for .ckpt/.pth/.safetensors files and provides
dropdown selection. An absolute path can be supplied via the optional
*_path inputs to override the dropdown — useful when the checkpoint
lives outside models/soya_seg/. Outputs are passed to Soya Simple Eye
Collector.
"""

import os
import torch


class SoyaSegModelProvider_mdsoya:
    @classmethod
    def INPUT_TYPES(cls):
        from .soya_scheduler.config_manager import get_available_models

        models = get_available_models("soya_seg")
        options = ["(skip)"] + models if models else ["(skip)"]

        try:
            devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] + ["cpu"]
        except Exception:
            devices = ["cuda:0", "cpu"]

        return {
            "required": {
                "eye_seg_model": (options,),
                "eyebrow_seg_model": (options,),
                "device": (devices, {"default": "cuda:0"}),
            },
            "optional": {
                "eye_seg_model_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Absolute path to an eye ISNet checkpoint. When set and the file exists, this overrides eye_seg_model.",
                    },
                ),
                "eyebrow_seg_model_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Absolute path to an eyebrow ISNet checkpoint. When set and the file exists, this overrides eyebrow_seg_model.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("SOYA_SEG_MODEL", "SOYA_SEG_MODEL")
    RETURN_NAMES = ("eye_model", "eyebrow_model")
    FUNCTION = "load"
    CATEGORY = "Soya/FaceDetailer"

    def load(
        self,
        eye_seg_model,
        eyebrow_seg_model,
        device,
        eye_seg_model_path="",
        eyebrow_seg_model_path="",
    ):
        from .soya_scheduler.model_manager import get_eye_seg_model, get_eyebrow_model

        eye = None
        if eye_seg_model_path and os.path.isfile(eye_seg_model_path):
            eye = get_eye_seg_model(eye_seg_model_path, device)
        elif eye_seg_model != "(skip)":
            eye = get_eye_seg_model(eye_seg_model, device)

        eyebrow = None
        if eyebrow_seg_model_path and os.path.isfile(eyebrow_seg_model_path):
            eyebrow = get_eyebrow_model(eyebrow_seg_model_path, device)
        elif eyebrow_seg_model != "(skip)":
            eyebrow = get_eyebrow_model(eyebrow_seg_model, device)

        return (
            {"model": eye, "device": device},
            {"model": eyebrow, "device": device},
        )
