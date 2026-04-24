import torch
import numpy as np

try:
    from impact.core import SEG
except ImportError:
    from collections import namedtuple
    SEG = namedtuple("SEG", ['cropped_image', 'cropped_mask', 'confidence', 'crop_region', 'bbox', 'label', 'control_net_wrapper'], defaults=[None])


class MaskAndSegsBBox_mdsoya:
    """
    SEGS의 bbox 영역 내 마스크 픽셀만 유지합니다 (AND 연산).
    bbox 바깥의 마스크 픽셀은 0으로 지웁니다.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "segs": ("SEGS",),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("filtered_mask",)
    FUNCTION = "doit"
    CATEGORY = "Soya/Mask"

    def doit(self, mask, segs):
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        result = torch.zeros_like(mask)

        for seg in segs[1]:
            bx1, by1, bx2, by2 = seg.bbox
            # bbox 내의 mask만 유지 (AND)
            for b in range(mask.shape[0]):
                result[b, by1:by2, bx1:bx2] = torch.max(
                    result[b, by1:by2, bx1:bx2],
                    mask[b, by1:by2, bx1:bx2],
                )

        return (result,)
