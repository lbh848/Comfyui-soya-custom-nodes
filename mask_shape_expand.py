import torch
import numpy as np
import scipy.ndimage


class MaskShapeExpand_mdsoya:
    """
    Expands or shrinks a mask while preserving its actual shape using morphological operations.
    Unlike bbox-based scaling, this respects the mask's contours and irregular shapes.
    scale_factor > 1.0: expands (dilation)
    scale_factor < 1.0: shrinks (erosion)
    scale_factor = 1.0: no change
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

            if radius == 0:
                out_masks.append(m.unsqueeze(0))
                continue

            if radius < 0:
                # Erosion: shrink the mask
                shrink_r = abs(radius)
                r_int = max(1, int(round(shrink_r)))

                binary = (m.numpy() > 0)
                dist = scipy.ndimage.distance_transform_edt(binary)
                result = (dist >= shrink_r).astype(np.float64)

                out = torch.from_numpy(result.astype(m.numpy().dtype))
                out_masks.append(out.unsqueeze(0))
                continue

            r_int = max(1, int(round(radius)))

            # bbox + radius 영역만 크롭해서 처리 (이미지 전체 대신)
            cy1 = max(0, y_min - r_int)
            cy2 = min(H, y_max + r_int + 1)
            cx1 = max(0, x_min - r_int)
            cx2 = min(W, x_max + r_int + 1)

            cropped = m[cy1:cy2, cx1:cx2]
            cropped_np = cropped.numpy()
            binary = cropped_np > 0

            # Distance transform: O(n), radius와 무관하게 일정 속도
            dist = scipy.ndimage.distance_transform_edt(~binary)
            expanded = dist <= radius

            # 기존 마스크 값 유지, 팽창 영역은 1.0
            result = np.where(binary, cropped_np, expanded.astype(np.float64))

            out = m.clone()
            out[cy1:cy2, cx1:cx2] = torch.from_numpy(result.astype(cropped_np.dtype))
            out_masks.append(out.unsqueeze(0))

        return (torch.cat(out_masks, dim=0),)
