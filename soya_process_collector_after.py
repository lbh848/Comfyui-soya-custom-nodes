"""
SoyaProcessCollectorAfter – lightweight passthrough node with the same output
signature as SoyaProcessCollector. Returns empty defaults except for the image.
"""

from collections import namedtuple

SEG = namedtuple("SEG", [
    'cropped_image', 'cropped_mask', 'confidence',
    'crop_region', 'bbox', 'label', 'control_net_wrapper',
], defaults=[None])


class SoyaProcessCollectorAfter_mdsoya:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "main_image": ("IMAGE", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE", "STRING", "SEGS", "CONTEXT")
    RETURN_NAMES = ("prompts", "main_image", "info", "segs", "context")
    FUNCTION = "passthrough"
    CATEGORY = "Soya/Scheduler"

    def passthrough(self, main_image):
        empty_segs = ((64, 64), [])
        empty_ctx = {}
        return ("", main_image, "", empty_segs, empty_ctx)
