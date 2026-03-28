import torch
import numpy as np
import scipy.ndimage


class MaskShapeExpand:
    """
    Expands a mask while preserving its actual shape using morphological dilation.
    Unlike bbox-based scaling, this respects the mask's contours and irregular shapes.
    The dilation radius is calculated proportionally from the mask's bounding box size.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "scale_factor": ("FLOAT", {"default": 2.0, "min": 0.0001, "max": 10.0, "step": 0.0001}),
                "minimum_size": ("INT", {"default": 1, "min": 1, "max": 8192, "step": 1}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "expand"
    CATEGORY = "Soya"

    def expand(self, mask, scale_factor, minimum_size):
        B, H, W = mask.shape
        out_masks = []

        for i in range(B):
            m = mask[i]

            # Find bounding box of the active mask region
            active = torch.nonzero(m > 0)
            if active.numel() == 0:
                out_masks.append(m.unsqueeze(0))
                continue

            y_min = active[:, 0].min().item()
            y_max = active[:, 0].max().item()
            x_min = active[:, 1].min().item()
            x_max = active[:, 1].max().item()

            w_old = x_max - x_min + 1
            h_old = y_max - y_min + 1
            avg_dim = (w_old + h_old) / 2.0

            # Calculate target dimension: at least minimum_size
            target_dim = max(avg_dim * scale_factor, float(minimum_size))

            # Radius of expansion (how many pixels to dilate outward)
            radius = (target_dim - avg_dim) / 2.0

            if radius <= 0:
                out_masks.append(m.unsqueeze(0))
                continue

            r_int = max(1, int(round(radius)))

            # Circular structuring element for shape-preserving dilation
            y_k, x_k = np.ogrid[-r_int:r_int + 1, -r_int:r_int + 1]
            kernel = (x_k * x_k + y_k * y_k) <= (r_int * r_int)

            m_np = m.numpy().astype(np.float64)
            result = scipy.ndimage.grey_dilation(m_np, footprint=kernel)

            out_masks.append(
                torch.from_numpy(result.astype(m.numpy().dtype)).unsqueeze(0)
            )

        return (torch.cat(out_masks, dim=0),)
