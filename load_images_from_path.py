import os
from collections import Counter
import numpy as np
import torch
from PIL import Image, ImageOps
import node_helpers


SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff', '.tif', '.gif'}


class LoadImagesFromPath_mdsoya:
    """
    Loads all images from a directory path and outputs them as a batch.
    Also outputs the filenames as a string batch.
    Images with different sizes are resized to the majority size.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "filenames")
    FUNCTION = "load_images"
    CATEGORY = "Soya/Image"

    @classmethod
    def IS_CHANGED(cls, path):
        return float("nan")
    OUTPUT_IS_LIST = (False, True)

    def load_images(self, path):
        path = path.strip()
        if not os.path.isdir(path):
            raise ValueError(f"Directory not found: {path}")

        files = sorted([
            f for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f))
            and os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
        ])

        if not files:
            raise ValueError(f"No supported image files found in: {path}")

        # 1차: 모든 이미지를 로드하여 (filename, PIL Image) 수집
        loaded = []
        for filename in files:
            filepath = os.path.join(path, filename)
            try:
                img = node_helpers.pillow(Image.open, filepath)
                img = ImageOps.exif_transpose(img)
                if img.mode == 'I':
                    img = img.point(lambda i: i * (1 / 255))
                img = img.convert("RGB")
                loaded.append((filename, img))
            except Exception:
                continue

        if not loaded:
            raise ValueError(f"No valid images could be loaded from: {path}")

        # 가장 대다수인 사이즈 찾기
        size_counts = Counter(img.size for _, img in loaded)
        target_w, target_h = size_counts.most_common(1)[0][0]

        # 2차: 대상 사이즈에 맞게 리사이즈 후 텐서 변환
        images = []
        filenames = []
        for filename, img in loaded:
            w, h = img.size
            if (w, h) != (target_w, target_h):
                img = img.resize((target_w, target_h), Image.LANCZOS)

            image = np.array(img).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            images.append(image)
            filenames.append(filename)

        batch = torch.cat(images, dim=0)
        return (batch, filenames)
