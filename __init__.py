# v3.1 shared nodes
from .sort_batch_by_segs import SortBatchBySegsBBox_mdsoya
from .character_identifier import IdentifyCharacters_mdsoya
from .mask_bitwise import MaskBitwiseAnd_mdsoya
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
from .soya_faceid_model_switch import SoyaFaceIDModelSwitch_mdsoya
from .soya_passthrough import SoyaPassthrough_mdsoya
from .soya_ipadapter_patch_cleaner import SoyaIPAdapterPatchCleaner_mdsoya
from .soya_hiresfix_toggle import SoyaHiresfixToggle_mdsoya
from .soya_upscale_toggle import SoyaUpscaleToggle_mdsoya

# main-only nodes (for 배포예정_삽화_V3 + ray scheduler)
from .soya_process_divider import SoyaProcessDivider_mdsoya
from .soya_process_collector import SoyaProcessCollector_mdsoya
from .soya_color_adjust import SoyaColorAdjust_mdsoya
from .soya_mask_range_adjust import SoyaMaskRangeAdjust_mdsoya
from .soya_mask_brightness import SoyaMaskBrightness_mdsoya
from .execution_timer import TimeStart_mdsoya, TimeEnd_mdsoya

NODE_CLASS_MAPPINGS = {
    # v3.1 shared
    "SortBatchBySegsBBox_mdsoya": SortBatchBySegsBBox_mdsoya,
    "IdentifyCharacters_mdsoya": IdentifyCharacters_mdsoya,
    "MaskBitwiseAnd_mdsoya": MaskBitwiseAnd_mdsoya,
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
    "SoyaFaceIDModelSwitch_mdsoya": SoyaFaceIDModelSwitch_mdsoya,
    "SoyaPassthrough_mdsoya": SoyaPassthrough_mdsoya,
    "SoyaIPAdapterPatchCleaner_mdsoya": SoyaIPAdapterPatchCleaner_mdsoya,
    "SoyaHiresfixToggle_mdsoya": SoyaHiresfixToggle_mdsoya,
    "SoyaUpscaleToggle_mdsoya": SoyaUpscaleToggle_mdsoya,
    # main-only (V3 + ray)
    "SoyaProcessDivider_mdsoya": SoyaProcessDivider_mdsoya,
    "SoyaProcessCollector_mdsoya": SoyaProcessCollector_mdsoya,
    "SoyaColorAdjust_mdsoya": SoyaColorAdjust_mdsoya,
    "SoyaMaskRangeAdjust_mdsoya": SoyaMaskRangeAdjust_mdsoya,
    "SoyaMaskBrightness_mdsoya": SoyaMaskBrightness_mdsoya,
    "TimeStart_mdsoya": TimeStart_mdsoya,
    "TimeEnd_mdsoya": TimeEnd_mdsoya,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # v3.1 shared
    "SortBatchBySegsBBox_mdsoya": "Sort Batch by Segs BBox (Soya)",
    "IdentifyCharacters_mdsoya": "Identify Characters (Soya)",
    "MaskBitwiseAnd_mdsoya": "Mask Bitwise And (Soya)",
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
    "SoyaFaceIDModelSwitch_mdsoya": "Soya FaceID Model Switch (Soya)",
    "SoyaPassthrough_mdsoya": "Passthrough (Soya)",
    "SoyaIPAdapterPatchCleaner_mdsoya": "IPAdapter Patch Cleaner (Soya)",
    "SoyaHiresfixToggle_mdsoya": "Hiresfix Toggle (Soya)",
    "SoyaUpscaleToggle_mdsoya": "Upscale Toggle (Soya)",
    # main-only (V3 + ray)
    "SoyaProcessDivider_mdsoya": "Soya Process Divider (Soya)",
    "SoyaProcessCollector_mdsoya": "Soya Process Collector (Soya)",
    "SoyaColorAdjust_mdsoya": "Color Adjust (Soya)",
    "SoyaMaskRangeAdjust_mdsoya": "Mask Range Adjust (Soya)",
    "SoyaMaskBrightness_mdsoya": "Mask Brightness (Soya)",
    "TimeStart_mdsoya": "Time Start (Soya)",
    "TimeEnd_mdsoya": "Time End (Soya)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

WEB_DIRECTORY = "./soya_scheduler/web"

# Soya Model Manager – web extension
from .soya_model_manager import WEB_DIRECTORY as _mm_web  # noqa: F401

# Set up API routes
try:
    from .soya_scheduler.server import setup_routes
    setup_routes()
except Exception as e:
    print(f"[Soya:Scheduler] Failed to setup routes: {e}")

# Soya Model Manager – API routes
try:
    from .soya_model_manager import setup_routes as _mm_setup
    _mm_setup()
except Exception as e:
    print(f"[Soya:ModelManager] Failed to setup routes: {e}")
