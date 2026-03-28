"""
Aligns SEGS batch with mask batch by inserting minimal SEG at empty mask positions.
"""
import torch
import numpy as np

try:
    from impact.core import SEG as ImpactSEG
except ImportError:
    ImpactSEG = None


class AlignSegsToMaskBatch_mdsoya:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_batch": ("MASK",),
                "segs": ("SEGS",),
            }
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("SEGS", "INT")
    RETURN_NAMES = ("aligned_segs", "segs_count")
    FUNCTION = "align"
    CATEGORY = "Soya/Segs"

    @staticmethod
    def _make_minimal_seg(segs_type):
        minimal_cropped_image = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
        minimal_cropped_mask = np.zeros((1, 1), dtype=np.float32)
        minimal_bbox = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        minimal_crop_region = [0, 0, 1, 1]

        if ImpactSEG is not None:
            return ImpactSEG(
                cropped_image=minimal_cropped_image,
                cropped_mask=minimal_cropped_mask,
                confidence=0.0,
                crop_region=minimal_crop_region,
                bbox=minimal_bbox,
                label="empty",
                control_net_wrapper=None,
            )

        if str(segs_type) == "<class 'list'>" or segs_type is None:
            class DummySEG:
                def __init__(self, **kwargs):
                    self.__dict__.update(kwargs)
                def _replace(self, **kwargs):
                    return DummySEG(**{**self.__dict__, **kwargs})
            segs_type = DummySEG

        try:
            minimal_seg = segs_type(
                cropped_image=minimal_cropped_image,
                cropped_mask=minimal_cropped_mask,
                confidence=0.0,
                crop_region=minimal_crop_region,
                bbox=minimal_bbox,
                label="empty",
                control_net_wrapper=None,
            )
        except:
            class DummySEG:
                def __init__(self, **kwargs):
                    self.__dict__.update(kwargs)
                def _replace(self, **kwargs):
                    return DummySEG(**{**self.__dict__, **kwargs})
            minimal_seg = DummySEG(
                cropped_image=minimal_cropped_image,
                cropped_mask=minimal_cropped_mask,
                confidence=0.0,
                crop_region=minimal_crop_region,
                bbox=minimal_bbox,
                label="empty",
                control_net_wrapper=None,
            )
        return minimal_seg

    def align(self, mask_batch, segs):
        mask_list = []
        for masks in mask_batch:
            if isinstance(masks, torch.Tensor):
                if masks.dim() == 2:
                    mask_list.append(masks)
                else:
                    for i in range(masks.shape[0]):
                        mask_list.append(masks[i])
            else:
                 mask_list.append(masks)
        
        num_masks = len(mask_list)

        valid_segments = []
        ref_size = (64, 64)
        
        for s in segs:
            if isinstance(s, tuple) and len(s) == 2:
                ref_size = s[0]
                valid_segments.extend(s[1])
            elif isinstance(s, list):
                for item in s:
                    if isinstance(item, tuple) and len(item) == 2:
                        ref_size = item[0]
                        valid_segments.extend(item[1])

        segs_type = type(valid_segments[0]) if valid_segments else None
        
        minimal_seg = self._make_minimal_seg(segs_type)
        
        aligned_segments = []
        seg_iter = iter(valid_segments)

        for i, m in enumerate(mask_list):
            if isinstance(m, torch.Tensor):
                is_empty = m.max().item() == 0.0
            else:
                is_empty = False
                
            if is_empty:
                aligned_segments.append(minimal_seg)
            else:
                try:
                    seg = next(seg_iter)
                    aligned_segments.append(seg)
                except StopIteration:
                    aligned_segments.append(minimal_seg)

        aligned_segs = (ref_size, aligned_segments)

        # Do NOT wrap in lists! Match the format of MergeSegs
        return (aligned_segs, len(aligned_segments))

