import torch
import numpy as np
import json

try:
    from impact.core import SEG
except ImportError:
    from collections import namedtuple
    SEG = namedtuple("SEG", ['cropped_image', 'cropped_mask', 'confidence', 'crop_region', 'bbox', 'label', 'control_net_wrapper'], defaults=[None])


class ConditionalImageSegsSwitch_mdsoya:
    """
    SEGS 크기에 따라 크롭/업스케일 파라미터를 계산합니다.
    Level 0: bbox 최장변 > crop_size OR area > level1_threshold → bbox × scale 크롭 + 업스케일 1x
    Level 1: level2_threshold < area <= level1_threshold → crop_size/2 크롭 + 업스케일 1x
    Level 2: area <= level2_threshold → crop_size/4 크롭 + 업스케일 2x (2회)
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "segs": ("SEGS",),
                "segs_areas": ("STRING", {"forceInput": True}),
                "crop_size": ("INT", {"default": 1024, "min": 4, "max": 8192, "step": 4}),
                "level1_threshold": ("INT", {"default": 50000, "min": 0, "max": 99999999, "step": 1}),
                "level2_threshold": ("INT", {"default": 10000, "min": 0, "max": 99999999, "step": 1}),
                "level0_crop_scale": ("FLOAT", {"default": 1.2, "min": 1.0, "max": 4.0, "step": 0.1}),
                "wildcard": ("STRING", {"multiline": True, "forceInput": True}),
            }
        }

    RETURN_TYPES = ("SEGS", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("modified_segs", "bbox_list", "crop_params", "info", "wildcard")
    FUNCTION = "doit"
    CATEGORY = "Soya/SEGS"
    OUTPUT_NODE = True

    def doit(self, image, segs, segs_areas, crop_size, level1_threshold, level2_threshold, level0_crop_scale, wildcard):
        try:
            areas = json.loads(segs_areas)
        except (json.JSONDecodeError, TypeError):
            areas = []

        H, W = image.shape[1], image.shape[2]

        modified_segs_list = []
        bbox_entries = []
        crop_params_list = []
        info_lines = []

        for i, seg in enumerate(segs[1]):
            area = areas[i] if i < len(areas) else 0
            bx1, by1, bx2, by2 = seg.bbox
            ox1, oy1, ox2, oy2 = seg.crop_region

            bbox_w = bx2 - bx1
            bbox_h = by2 - by1
            bbox_max_side = max(bbox_w, bbox_h)

            if bbox_max_side > crop_size or area > level1_threshold:
                # Level 0: bbox × scale 크롭 + 업스케일 1x
                level = 0
                crop_dim = int(bbox_max_side * level0_crop_scale)

                if crop_dim >= min(W, H):
                    cx1, cy1, cx2, cy2 = 0, 0, W, H
                else:
                    cx1, cy1, cx2, cy2 = self._compute_crop(
                        bx1, by1, bx2, by2, crop_dim, crop_dim, W, H
                    )

            elif level2_threshold < area <= level1_threshold:
                # Level 1: crop_size/2 크롭 + 업스케일 1x
                level = 1
                crop_dim = crop_size // 2
                cx1, cy1, cx2, cy2 = self._compute_crop(
                    bx1, by1, bx2, by2, crop_dim, crop_dim, W, H
                )

            else:
                # Level 2: crop_size/4 크롭 + 업스케일 2x (2회)
                level = 2
                crop_dim = crop_size // 4
                cx1, cy1, cx2, cy2 = self._compute_crop(
                    bx1, by1, bx2, by2, crop_dim, crop_dim, W, H
                )

            # 원본 크롭 공간 좌표 (Starter에서 업스케일 비율에 맞게 조정)
            new_crop_region = (
                max(0, ox1 - cx1),
                max(0, oy1 - cy1),
                min(cx2 - cx1, ox2 - cx1),
                min(cy2 - cy1, oy2 - cy1),
            )
            new_bbox = (
                max(0, bx1 - cx1),
                max(0, by1 - cy1),
                min(cx2 - cx1, bx2 - cx1),
                min(cy2 - cy1, by2 - cy1),
            )
            bbox_entry = (cx1, cy1, cx2, cy2)
            crop_params_list.append({
                "level": level,
                "cx1": cx1, "cy1": cy1, "cx2": cx2, "cy2": cy2,
            })

            # Crop mask from original mask at intersection offset (NOT resize)
            cropped_mask = seg.cropped_mask
            if cropped_mask is not None:
                if isinstance(cropped_mask, np.ndarray):
                    cropped_mask = torch.from_numpy(cropped_mask).float()

                new_cr_h = new_crop_region[3] - new_crop_region[1]
                new_cr_w = new_crop_region[2] - new_crop_region[0]
                if new_cr_h > 0 and new_cr_w > 0:
                    mask_y1 = max(0, cy1 - oy1)
                    mask_x1 = max(0, cx1 - ox1)
                    mask_y2 = min(cropped_mask.shape[-2], mask_y1 + new_cr_h)
                    mask_x2 = min(cropped_mask.shape[-1], mask_x1 + new_cr_w)

                    if len(cropped_mask.shape) == 2:
                        cropped_mask = cropped_mask[mask_y1:mask_y2, mask_x1:mask_x2]
                    elif len(cropped_mask.shape) == 3:
                        cropped_mask = cropped_mask[:, mask_y1:mask_y2, mask_x1:mask_x2]

            new_seg = SEG(
                cropped_image=None,
                cropped_mask=cropped_mask,
                confidence=seg.confidence,
                crop_region=new_crop_region,
                bbox=new_bbox,
                label=seg.label,
                control_net_wrapper=seg.control_net_wrapper,
            )
            modified_segs_list.append(new_seg)
            bbox_entries.append(bbox_entry)

            info_lines.append(
                f"seg[{i}] label={seg.label} area={area} level={level} "
                f"crop={bbox_entry} new_crop_region={new_crop_region} new_bbox={new_bbox}"
            )

        if not modified_segs_list:
            return (((crop_size, crop_size), []), "", "", "No segs found", wildcard)

        bbox_str = ";".join(f"{x1},{y1},{x2},{y2}" for x1, y1, x2, y2 in bbox_entries)
        crop_params_str = json.dumps(crop_params_list)
        info = "\n".join(info_lines)

        modified_segs = ((crop_size, crop_size), modified_segs_list)

        return (modified_segs, bbox_str, crop_params_str, info, wildcard)

    @staticmethod
    def _compute_crop(bx1, by1, bx2, by2, crop_w, crop_h, img_w, img_h):
        center_x = (bx1 + bx2) / 2
        center_y = (by1 + by2) / 2

        cx1 = int(center_x - crop_w / 2)
        cy1 = int(center_y - crop_h / 2)
        cx2 = cx1 + crop_w
        cy2 = cy1 + crop_h

        if cx1 < 0:
            cx2 += (-cx1)
            cx1 = 0
        if cy1 < 0:
            cy2 += (-cy1)
            cy1 = 0
        if cx2 > img_w:
            cx1 -= (cx2 - img_w)
            cx2 = img_w
        if cy2 > img_h:
            cy1 -= (cy2 - img_h)
            cy2 = img_h

        cx1 = max(0, cx1)
        cy1 = max(0, cy1)
        cx2 = min(img_w, cx2)
        cy2 = min(img_h, cy2)

        return cx1, cy1, cx2, cy2
