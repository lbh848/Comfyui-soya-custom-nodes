import torch
from collections import namedtuple

try:
    from impact.core import SEG
except ImportError:
    SEG = namedtuple("SEG", ['cropped_image', 'cropped_mask', 'confidence', 'crop_region', 'bbox', 'label', 'control_net_wrapper'], defaults=[None])


class SegsLabelTransfer_mdsoya:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "original_segs": ("SEGS",),
                "modified_segs": ("SEGS",),
            }
        }

    RETURN_TYPES = ("SEGS",)
    FUNCTION = "doit"
    CATEGORY = "Soya/SEGS"

    def doit(self, original_segs, modified_segs):
        orig_size, orig_segs = original_segs
        mod_size, mod_segs = modified_segs

        labels = [seg.label for seg in orig_segs]

        new_segs = []
        for i, seg in enumerate(mod_segs):
            label = labels[i] if i < len(labels) else seg.label
            new_seg = SEG(
                seg.cropped_image,
                seg.cropped_mask,
                seg.confidence,
                seg.crop_region,
                seg.bbox,
                label,
                seg.control_net_wrapper,
            )
            new_segs.append(new_seg)

        return ((mod_size, new_segs),)


NODE_CLASS_MAPPINGS = {
    "SegsLabelTransfer_mdsoya": SegsLabelTransfer_mdsoya,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SegsLabelTransfer_mdsoya": "SEGS Label Transfer (Soya)",
}
