from .sort_batch_by_segs import SortBatchBySegsBBox_mdsoya
from .character_identifier import IdentifyCharacters_mdsoya
from .mask_bitwise import MaskBitwiseAnd_mdsoya, MaskBitwiseOr_mdsoya, MaskBitwiseAndBatch_mdsoya, MaskBitwiseOrBatch_mdsoya
from .merge_segs import MergeSegs_mdsoya
from .mask_proportional_expand import MaskProportionalExpand_mdsoya
from .mask_shape_expand import MaskShapeExpand_mdsoya
from .align_segs_to_mask import AlignSegsToMaskBatch_mdsoya
from .load_images_from_path import LoadImagesFromPath_mdsoya
from .filter_and_assign_characters import FilterAndAssignCharacters_mdsoya
from .filter_closed_eyes import FilterClosedEyes_mdsoya
from .conditional_lora_loader import ConditionalLoraLoader_mdsoya
from .segs_area import SegsAreaInfo_mdsoya
from .conditional_image_segs_switch import ConditionalImageSegsSwitch_mdsoya
from .detailer_distributor_starter import DetailerDistributorStarter_mdsoya
from .detailer_distributor_pipe import DetailerDistributorPipe_mdsoya
from .soya_batch_detailer import SoyaBatchDetailer_mdsoya
from .mask_and_segs_bbox import MaskAndSegsBBox_mdsoya
from .segs_label_transfer import SegsLabelTransfer_mdsoya
from .soya_color_adjust_config import SoyaColorAdjustConfig_mdsoya
from .soya_faceid_yolo_fallback import SoyaFaceIDYoloFallback_mdsoya
from .soya_conditioning_switch import SoyaConditioningSwitch_mdsoya
from .soya_pose_json_converter import SoyaPoseJsonConverter_mdsoya
from .soya_string_to_float import SoyaStringToFloat_mdsoya
from .soya_face_detailer_toggle import SoyaFaceDetailerToggle_mdsoya
from .soya_simple_eye_collector import SoyaSimpleEyeCollector_mdsoya
from .soya_seg_model_provider import SoyaSegModelProvider_mdsoya
from .soya_passthrough import SoyaPassthrough_mdsoya
from .soya_faceid_model_switch import SoyaFaceIDModelSwitch_mdsoya

NODE_CLASS_MAPPINGS = {
    "SortBatchBySegsBBox_mdsoya": SortBatchBySegsBBox_mdsoya,
    "IdentifyCharacters_mdsoya": IdentifyCharacters_mdsoya,
    "MaskBitwiseAnd_mdsoya": MaskBitwiseAnd_mdsoya,
    "MaskBitwiseOr_mdsoya": MaskBitwiseOr_mdsoya,
    "MaskBitwiseAndBatch_mdsoya": MaskBitwiseAndBatch_mdsoya,
    "MaskBitwiseOrBatch_mdsoya": MaskBitwiseOrBatch_mdsoya,
    "MergeSegs_mdsoya": MergeSegs_mdsoya,
    "MaskProportionalExpand_mdsoya": MaskProportionalExpand_mdsoya,
    "MaskShapeExpand_mdsoya": MaskShapeExpand_mdsoya,
    "AlignSegsToMaskBatch_mdsoya": AlignSegsToMaskBatch_mdsoya,
    "LoadImagesFromPath_mdsoya": LoadImagesFromPath_mdsoya,
    "FilterAndAssignCharacters_mdsoya": FilterAndAssignCharacters_mdsoya,
    "FilterClosedEyes_mdsoya": FilterClosedEyes_mdsoya,
    "ConditionalLoraLoader_mdsoya": ConditionalLoraLoader_mdsoya,
    "SegsAreaInfo_mdsoya": SegsAreaInfo_mdsoya,
    "ConditionalImageSegsSwitch_mdsoya": ConditionalImageSegsSwitch_mdsoya,
    "DetailerDistributorStarter_mdsoya": DetailerDistributorStarter_mdsoya,
    "DetailerDistributorPipe_mdsoya": DetailerDistributorPipe_mdsoya,
    "SoyaBatchDetailer_mdsoya": SoyaBatchDetailer_mdsoya,
    "MaskAndSegsBBox_mdsoya": MaskAndSegsBBox_mdsoya,
    "SegsLabelTransfer_mdsoya": SegsLabelTransfer_mdsoya,
    "SoyaColorAdjustConfig_mdsoya": SoyaColorAdjustConfig_mdsoya,
    "SoyaFaceIDYoloFallback_mdsoya": SoyaFaceIDYoloFallback_mdsoya,
    "SoyaConditioningSwitch_mdsoya": SoyaConditioningSwitch_mdsoya,
    "SoyaPoseJsonConverter_mdsoya": SoyaPoseJsonConverter_mdsoya,
    "SoyaStringToFloat_mdsoya": SoyaStringToFloat_mdsoya,
    "SoyaFaceDetailerToggle_mdsoya": SoyaFaceDetailerToggle_mdsoya,
    "SoyaSimpleEyeCollector_mdsoya": SoyaSimpleEyeCollector_mdsoya,
    "SoyaSegModelProvider_mdsoya": SoyaSegModelProvider_mdsoya,
    "SoyaPassthrough_mdsoya": SoyaPassthrough_mdsoya,
    "SoyaFaceIDModelSwitch_mdsoya": SoyaFaceIDModelSwitch_mdsoya,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SortBatchBySegsBBox_mdsoya": "Sort Batch by Segs BBox (Soya)",
    "IdentifyCharacters_mdsoya": "Identify Characters (Soya)",
    "MaskBitwiseAnd_mdsoya": "Mask Bitwise And (Soya)",
    "MaskBitwiseOr_mdsoya": "Mask Bitwise Or (Soya)",
    "MaskBitwiseAndBatch_mdsoya": "Mask Bitwise And Batch (Soya)",
    "MaskBitwiseOrBatch_mdsoya": "Mask Bitwise Or Batch (Soya)",
    "MergeSegs_mdsoya": "Merge Segs (Soya)",
    "MaskProportionalExpand_mdsoya": "Mask Proportional Expand (Soya)",
    "MaskShapeExpand_mdsoya": "Mask Shape Expand (Soya)",
    "AlignSegsToMaskBatch_mdsoya": "Align Segs to Mask Batch (Soya)",
    "LoadImagesFromPath_mdsoya": "Load Images From Path (Soya)",
    "FilterAndAssignCharacters_mdsoya": "Filter & Assign Characters (Soya)",
    "FilterClosedEyes_mdsoya": "Filter Closed Eyes (Soya)",
    "ConditionalLoraLoader_mdsoya": "Conditional LoRA Loader (Soya)",
    "SegsAreaInfo_mdsoya": "SEGS Area Info (Soya)",
    "ConditionalImageSegsSwitch_mdsoya": "Conditional Image/Segs Switch (Soya)",
    "DetailerDistributorStarter_mdsoya": "Detailer Distributor Starter (Soya)",
    "DetailerDistributorPipe_mdsoya": "Detailer Distributor Pipe (Soya)",
    "SoyaBatchDetailer_mdsoya": "Soya Batch Detailer (Soya)",
    "MaskAndSegsBBox_mdsoya": "Mask AND Segs BBox (Soya)",
    "SegsLabelTransfer_mdsoya": "SEGS Label Transfer (Soya)",
    "SoyaColorAdjustConfig_mdsoya": "Color Adjust Config (Soya)",
    "SoyaFaceIDYoloFallback_mdsoya": "Soya FaceID YOLO Fallback (Soya)",
    "SoyaConditioningSwitch_mdsoya": "Conditioning Switch (Soya)",
    "SoyaPoseJsonConverter_mdsoya": "Pose JSON Converter (Soya)",
    "SoyaStringToFloat_mdsoya": "String to Float (Soya)",
    "SoyaFaceDetailerToggle_mdsoya": "Face Detailer Toggle (Soya)",
    "SoyaSimpleEyeCollector_mdsoya": "Soya Simple Eye Collector (Soya)",
    "SoyaSegModelProvider_mdsoya": "Soya Seg Model Provider (Soya)",
    "SoyaPassthrough_mdsoya": "Passthrough (Soya)",
    "SoyaFaceIDModelSwitch_mdsoya": "Soya FaceID Model Switch (Soya)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
