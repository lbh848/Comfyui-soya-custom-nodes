import torch
import numpy as np
import json


def _calc_seg_areas(segs):
    """SEGS의 각 세그먼트 실제 픽셀 수(area)를 계산합니다."""
    areas = []
    for seg in segs[1]:
        mask = seg.cropped_mask
        if mask is not None:
            if isinstance(mask, np.ndarray):
                area = int(np.count_nonzero(mask))
            elif isinstance(mask, torch.Tensor):
                area = int(torch.count_nonzero(mask).item())
            else:
                area = 0
        else:
            # fallback: bbox area
            x1, y1, x2, y2 = seg.bbox
            area = int((x2 - x1) * (y2 - y1))
        areas.append(area)
    return areas


class SegsAreaInfo_mdsoya:
    """
    SEGS의 각 세그먼트별 실제 픽셀 수(area)를 출력합니다.
    출력1: 디버그 문자열
    출력2: 수치 데이터 (JSON 배열 문자열, 예: "[100, 200, 300]")
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "segs": ("SEGS",),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("debug_info", "areas_data")
    FUNCTION = "doit"
    CATEGORY = "Soya/SEGS"
    OUTPUT_NODE = True

    def doit(self, segs):
        areas = _calc_seg_areas(segs)

        # Debug string output
        lines = []
        for i, (area, seg) in enumerate(zip(areas, segs[1])):
            label = seg.label or f"seg_{i}"
            mask = seg.cropped_mask
            bx1, by1, bx2, by2 = seg.bbox
            bbox_w, bbox_h = bx2 - bx1, by2 - by1
            cr = seg.crop_region
            cr_w, cr_h = cr[2] - cr[0], cr[3] - cr[1]
            lines.append(f"{label}: {area} px [crop_region:{cr_w}x{cr_h}, bbox:{bbox_w}x{bbox_h}]")

        total = sum(areas)
        avg = int(total / len(areas)) if areas else 0
        lines.append(f"total: {total} px")
        lines.append(f"avg: {avg} px")
        debug_str = "\n".join(lines)

        # Numeric data output (JSON array)
        areas_json = json.dumps(areas)

        return (debug_str, areas_json)
