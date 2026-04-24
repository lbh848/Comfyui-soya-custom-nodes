import torch
import numpy as np

try:
    from impact.core import SEG
except ImportError:
    from collections import namedtuple
    SEG = namedtuple("SEG", ['cropped_image', 'cropped_mask', 'confidence', 'crop_region', 'bbox', 'label', 'control_net_wrapper'], defaults=[None])


class SegsScaleBy_mdsoya:
    """
    SEGS의 모든 좌표(bbox, crop_region), mask, cropped_image를 지정한 배율로 스케일합니다.
    업스케일 모델로 이미지를 키운 후, 동일 배율로 SEGS를 맞출 때 사용합니다.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "segs": ("SEGS",),
                "scale_by": ("FLOAT", {"default": 2.0, "min": 0.01, "max": 100.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("SEGS",)
    FUNCTION = "doit"
    CATEGORY = "Soya/SEGS"

    def doit(self, segs, scale_by):
        h, w = segs[0]
        new_h = int(h * scale_by)
        new_w = int(w * scale_by)

        new_segs = []
        for seg in segs[1]:
            cropped_image = seg.cropped_image
            cropped_mask = seg.cropped_mask
            x1, y1, x2, y2 = seg.crop_region
            bx1, by1, bx2, by2 = seg.bbox

            crop_region = (int(x1 * scale_by), int(y1 * scale_by),
                           int(x2 * scale_by), int(y2 * scale_by))
            bbox = (int(bx1 * scale_by), int(by1 * scale_by),
                    int(bx2 * scale_by), int(by2 * scale_by))

            crop_w = crop_region[2] - crop_region[0]
            crop_h = crop_region[3] - crop_region[1]

            # mask resize
            if isinstance(cropped_mask, np.ndarray):
                cropped_mask = torch.from_numpy(cropped_mask)

            if cropped_mask is not None and crop_h > 0 and crop_w > 0:
                if len(cropped_mask.shape) == 2:
                    cropped_mask = torch.nn.functional.interpolate(
                        cropped_mask.unsqueeze(0).unsqueeze(0),
                        size=(crop_h, crop_w), mode='bilinear', align_corners=False
                    ).squeeze(0).squeeze(0)
                elif len(cropped_mask.shape) == 3:
                    cropped_mask = torch.nn.functional.interpolate(
                        cropped_mask.unsqueeze(0),
                        size=(crop_h, crop_w), mode='bilinear', align_corners=False
                    ).squeeze(0)

            # cropped_image resize
            if cropped_image is not None and crop_h > 0 and crop_w > 0:
                if isinstance(cropped_image, np.ndarray):
                    cropped_image = torch.from_numpy(cropped_image)
                cropped_image = torch.nn.functional.interpolate(
                    cropped_image.permute(0, 3, 1, 2),
                    size=(crop_h, crop_w), mode='bilinear', align_corners=False
                ).permute(0, 2, 3, 1)
                cropped_image = cropped_image.numpy()

            new_seg = SEG(cropped_image, cropped_mask, seg.confidence,
                          crop_region, bbox, seg.label, seg.control_net_wrapper)
            new_segs.append(new_seg)

        return (((new_h, new_w), new_segs),)
