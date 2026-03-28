from .sort_batch_by_segs import SortBatchBySegsBBox
from .character_identifier import IdentifyCharacters
from .bbox_debug import BBoxDebug
from .mask_bitwise import MaskBitwiseAnd, MaskBitwiseOr, MaskBitwiseAndBatch, MaskBitwiseOrBatch
from .merge_segs import MergeSegs
from .eye_detector import MediaPipeEyeDetector
from .mask_proportional_expand import MaskProportionalExpand
from .mask_shape_expand import MaskShapeExpand
from .align_segs_to_mask import AlignSegsToMaskBatch
from .load_images_from_path import LoadImagesFromPath
from .filter_batch_by_prompt import FilterBatchByPrompt
from .assign_characters_clip import AssignCharactersCLIP
from .filter_and_assign_characters import FilterAndAssignCharacters
from .debug_print import DebugPrint
from .anime_blink_detector import AnimeBlinkDetector

NODE_CLASS_MAPPINGS = {
    "SortBatchBySegsBBox": SortBatchBySegsBBox,
    "IdentifyCharacters": IdentifyCharacters,
    "BBoxDebug": BBoxDebug,
    "MaskBitwiseAnd": MaskBitwiseAnd,
    "MaskBitwiseOr": MaskBitwiseOr,
    "MaskBitwiseAndBatch": MaskBitwiseAndBatch,
    "MaskBitwiseOrBatch": MaskBitwiseOrBatch,
    "MergeSegs": MergeSegs,
    "MediaPipeEyeDetector": MediaPipeEyeDetector,
    "MaskProportionalExpand": MaskProportionalExpand,
    "MaskShapeExpand": MaskShapeExpand,
    "AlignSegsToMaskBatch": AlignSegsToMaskBatch,
    "LoadImagesFromPath": LoadImagesFromPath,
    "FilterBatchByPrompt": FilterBatchByPrompt,
    "AssignCharactersCLIP": AssignCharactersCLIP,
    "FilterAndAssignCharacters": FilterAndAssignCharacters,
    "DebugPrint": DebugPrint,
    "AnimeBlinkDetector": AnimeBlinkDetector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SortBatchBySegsBBox": "Sort Batch by Segs BBox (Soya)",
    "IdentifyCharacters": "Identify Characters (Soya)",
    "BBoxDebug": "BBox Debug (Soya)",
    "MaskBitwiseAnd": "Mask Bitwise And (Soya)",
    "MaskBitwiseOr": "Mask Bitwise Or (Soya)",
    "MaskBitwiseAndBatch": "Mask Bitwise And Batch (Soya)",
    "MaskBitwiseOrBatch": "Mask Bitwise Or Batch (Soya)",
    "MergeSegs": "Merge Segs (Soya)",
    "MediaPipeEyeDetector": "MediaPipe Eye Detector (Soya)",
    "MaskProportionalExpand": "Mask Proportional Expand (Soya)",
    "MaskShapeExpand": "Mask Shape Expand (Soya)",
    "AlignSegsToMaskBatch": "Align Segs to Mask Batch (Soya)",
    "LoadImagesFromPath": "Load Images From Path (Soya)",
    "FilterBatchByPrompt": "Filter Batch by Prompt (Soya)",
    "AssignCharactersCLIP": "Assign Characters by CLIP Vision (Soya)",
    "FilterAndAssignCharacters": "Filter & Assign Characters (Soya)",
    "DebugPrint": "Debug Print (Soya)",
    "AnimeBlinkDetector": "Anime Blink Detector (Soya)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
