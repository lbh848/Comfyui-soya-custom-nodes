"""
SoyaCannyFaceOverlay – Multiplies image by soft mask, extracts Canny edges,
overlays them back to thicken facial feature lines for expression preservation
during denoising.
"""

import numpy as np
import torch


class SoyaCannyFaceOverlay_mdsoya:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "low_threshold": ("FLOAT", {
                    "default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01
                }),
                "high_threshold": ("FLOAT", {
                    "default": 0.2, "min": 0.01, "max": 1.0, "step": 0.01
                }),
                "edge_strength": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01
                }),
                "dilate": ("INT", {
                    "default": 1, "min": 0, "max": 20, "step": 1
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply"
    CATEGORY = "Soya/Mask"

    def apply(self, image, mask, low_threshold=0.1, high_threshold=0.2, edge_strength=0.3, dilate=1):
        import cv2

        B, H, W, C = image.shape

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        if mask.shape[0] == 1 and B > 1:
            mask = mask.expand(B, -1, -1)

        result = image.clone()

        for i in range(B):
            img_np = image[i].cpu().numpy()          # (H, W, 3) float [0,1]
            m = mask[i].cpu().numpy()                 # (H, W)   float [0,1]

            # Soft-masked face region
            masked = img_np * m[:, :, np.newaxis]

            # Grayscale for Canny
            gray = (np.mean(masked, axis=2) * 255).clip(0, 255).astype(np.uint8)

            edges = cv2.Canny(
                gray,
                int(low_threshold * 255),
                int(high_threshold * 255),
            )

            # Dilate edges to make lines thicker
            if dilate > 0:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1)
                )
                edges = cv2.dilate(edges, kernel)

            edges_f = edges.astype(np.float32) / 255.0 * edge_strength

            # Darken original where edges exist → thicker face lines
            result[i] = torch.from_numpy(
                (img_np * (1.0 - edges_f[:, :, np.newaxis])).clip(0, 1).astype(np.float32)
            )

        return (result,)
