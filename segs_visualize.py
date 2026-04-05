import torch
import numpy as np


class SegsVisualize_mdsoya:
    """
    지정한 label에 해당하는 SEGS의 bbox와 crop_region를 이미지 위에 시각화합니다.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "fallback_image": ("IMAGE",),
                "segs": ("SEGS",),
                "label": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("visualized_image",)
    FUNCTION = "doit"
    CATEGORY = "Soya/Debug"
    OUTPUT_NODE = True

    def doit(self, fallback_image, segs, label):
        img = fallback_image[0].clone()  # [H, W, C]
        H, W = img.shape[0], img.shape[1]

        found = False
        for seg in segs[1]:
            if str(seg.label) != label:
                continue
            found = True

            # crop_region: 녹색 사각형
            cr = seg.crop_region
            self._draw_rect(img, cr[0], cr[1], cr[2], cr[3], color=(0.0, 1.0, 0.0), thickness=2)

            # bbox: 빨간색 사각형
            bb = seg.bbox
            self._draw_rect(img, bb[0], bb[1], bb[2], bb[3], color=(1.0, 0.0, 0.0), thickness=2)

        if not found:
            print(f"[SegsVisualize] label '{label}' not found in segs")

        return (img.unsqueeze(0),)

    @staticmethod
    def _draw_rect(img, x1, y1, x2, y2, color, thickness=2):
        """이미지 텐서에 직사각형을 그립니다."""
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        H, W = img.shape[0], img.shape[1]
        t = thickness
        c = torch.tensor(color, dtype=img.dtype, device=img.device)

        # 수평선
        for dy in range(t):
            row_top = min(y1 + dy, H - 1)
            row_bot = min(y2 - 1 - dy, H - 1)
            img[row_top, max(0, x1):min(W, x2)] = c
            img[row_bot, max(0, x1):min(W, x2)] = c

        # 수직선
        for dx in range(t):
            col_left = min(x1 + dx, W - 1)
            col_right = min(x2 - 1 - dx, W - 1)
            img[max(0, y1):min(H, y2), col_left] = c
            img[max(0, y1):min(H, y2), col_right] = c
