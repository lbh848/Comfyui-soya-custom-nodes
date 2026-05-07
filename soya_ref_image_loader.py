"""
SoyaRefImageLoader – Reference image loader with built-in fallback.

Drop-in replacement for ComfyUI's built-in LoadImage node.
If no image is specified (or the image doesn't exist), generates a
solid-color fallback image instead of raising an error.

Use _meta title "레퍼런스이미지로드" so the hooking server can inject
the image filename at runtime.
"""

import os
import torch
import numpy as np
from PIL import Image
import folder_paths


class SoyaRefImageLoader_mdsoya:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "doit"
    CATEGORY = "Soya"

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = []
        if os.path.isdir(input_dir):
            for f in os.listdir(input_dir):
                if os.path.isfile(os.path.join(input_dir, f)):
                    files.append(f)
        return {
            "required": {
                "image": (sorted(files) or ["(none)"],),
                "fallback_width": ("INT", {"default": 1024, "min": 64, "max": 4096}),
                "fallback_height": ("INT", {"default": 1024, "min": 64, "max": 4096}),
            },
        }

    def doit(self, image, fallback_width, fallback_height):
        if image and image != "(none)":
            image_path = folder_paths.get_annotated_filepath(image)
            if os.path.isfile(image_path):
                print(f"[SoyaRefImageLoader] Loading: {image}")
                img = Image.open(image_path).convert("RGB")
                img_array = np.array(img).astype(np.float32) / 255.0
                tensor = torch.from_numpy(img_array).unsqueeze(0)
                return (tensor,)

        # Fallback: generate solid neutral gray image
        print(f"[SoyaRefImageLoader] No valid image, using {fallback_width}x{fallback_height} fallback")
        img = Image.new("RGB", (fallback_width, fallback_height), (128, 128, 128))
        img_array = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_array).unsqueeze(0)
        return (tensor,)
