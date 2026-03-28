from .sort_batch_by_segs import SortBatchBySegsBBox_mdsoya
from .character_identifier import IdentifyCharacters_mdsoya
from .bbox_debug import BBoxDebug_mdsoya
from .mask_bitwise import MaskBitwiseAnd_mdsoya, MaskBitwiseOr_mdsoya, MaskBitwiseAndBatch_mdsoya, MaskBitwiseOrBatch_mdsoya
from .merge_segs import MergeSegs_mdsoya
from .eye_detector import MediaPipeEyeDetector_mdsoya
from .mask_proportional_expand import MaskProportionalExpand_mdsoya
from .mask_shape_expand import MaskShapeExpand_mdsoya
from .align_segs_to_mask import AlignSegsToMaskBatch_mdsoya
from .load_images_from_path import LoadImagesFromPath_mdsoya
from .filter_batch_by_prompt import FilterBatchByPrompt_mdsoya
from .assign_characters_clip import AssignCharactersCLIP_mdsoya
from .filter_and_assign_characters import FilterAndAssignCharacters_mdsoya
from .debug_print import DebugPrint_mdsoya
from .anime_blink_detector import AnimeBlinkDetector_mdsoya
from .split_person_segment import SplitPersonSegment_mdsoya
from .join_strings import JoinStringBatch_mdsoya
from .florence2_caption_filter import Florence2CaptionFilter_mdsoya
from .zip_strings import ZipStringBatch_mdsoya

NODE_CLASS_MAPPINGS = {
    "SortBatchBySegsBBox_mdsoya": SortBatchBySegsBBox_mdsoya,
    "IdentifyCharacters_mdsoya": IdentifyCharacters_mdsoya,
    "BBoxDebug_mdsoya": BBoxDebug_mdsoya,
    "MaskBitwiseAnd_mdsoya": MaskBitwiseAnd_mdsoya,
    "MaskBitwiseOr_mdsoya": MaskBitwiseOr_mdsoya,
    "MaskBitwiseAndBatch_mdsoya": MaskBitwiseAndBatch_mdsoya,
    "MaskBitwiseOrBatch_mdsoya": MaskBitwiseOrBatch_mdsoya,
    "MergeSegs_mdsoya": MergeSegs_mdsoya,
    "MediaPipeEyeDetector_mdsoya": MediaPipeEyeDetector_mdsoya,
    "MaskProportionalExpand_mdsoya": MaskProportionalExpand_mdsoya,
    "MaskShapeExpand_mdsoya": MaskShapeExpand_mdsoya,
    "AlignSegsToMaskBatch_mdsoya": AlignSegsToMaskBatch_mdsoya,
    "LoadImagesFromPath_mdsoya": LoadImagesFromPath_mdsoya,
    "FilterBatchByPrompt_mdsoya": FilterBatchByPrompt_mdsoya,
    "AssignCharactersCLIP_mdsoya": AssignCharactersCLIP_mdsoya,
    "FilterAndAssignCharacters_mdsoya": FilterAndAssignCharacters_mdsoya,
    "DebugPrint_mdsoya": DebugPrint_mdsoya,
    "AnimeBlinkDetector_mdsoya": AnimeBlinkDetector_mdsoya,
    "SplitPersonSegment_mdsoya": SplitPersonSegment_mdsoya,
    "JoinStringBatch_mdsoya": JoinStringBatch_mdsoya,
    "Florence2CaptionFilter_mdsoya": Florence2CaptionFilter_mdsoya,
    "ZipStringBatch_mdsoya": ZipStringBatch_mdsoya,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SortBatchBySegsBBox_mdsoya": "Sort Batch by Segs BBox (Soya)",
    "IdentifyCharacters_mdsoya": "Identify Characters (Soya)",
    "BBoxDebug_mdsoya": "BBox Debug (Soya)",
    "MaskBitwiseAnd_mdsoya": "Mask Bitwise And (Soya)",
    "MaskBitwiseOr_mdsoya": "Mask Bitwise Or (Soya)",
    "MaskBitwiseAndBatch_mdsoya": "Mask Bitwise And Batch (Soya)",
    "MaskBitwiseOrBatch_mdsoya": "Mask Bitwise Or Batch (Soya)",
    "MergeSegs_mdsoya": "Merge Segs (Soya)",
    "MediaPipeEyeDetector_mdsoya": "MediaPipe Eye Detector (Soya)",
    "MaskProportionalExpand_mdsoya": "Mask Proportional Expand (Soya)",
    "MaskShapeExpand_mdsoya": "Mask Shape Expand (Soya)",
    "AlignSegsToMaskBatch_mdsoya": "Align Segs to Mask Batch (Soya)",
    "LoadImagesFromPath_mdsoya": "Load Images From Path (Soya)",
    "FilterBatchByPrompt_mdsoya": "Filter Batch by Prompt (Soya)",
    "AssignCharactersCLIP_mdsoya": "Assign Characters by CLIP Vision (Soya)",
    "FilterAndAssignCharacters_mdsoya": "Filter & Assign Characters (Soya)",
    "DebugPrint_mdsoya": "Debug Print (Soya)",
    "AnimeBlinkDetector_mdsoya": "Anime Blink Detector (Soya)",
    "SplitPersonSegment_mdsoya": "Split Person Segment (Soya)",
    "JoinStringBatch_mdsoya": "Join String Batch (Soya)",
    "Florence2CaptionFilter_mdsoya": "Florence2 Caption Filter (Soya)",
    "ZipStringBatch_mdsoya": "Zip String Batch (Soya)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
