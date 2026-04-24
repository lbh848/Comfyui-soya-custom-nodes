from .soya_process_divider import SoyaProcessDivider_mdsoya
from .soya_process_collector import SoyaProcessCollector_mdsoya
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
from .filter_closed_eyes import FilterClosedEyes_mdsoya
from .conditional_lora_loader import ConditionalLoraLoader_mdsoya
from .expression_tag_extractor import ExpressionTagExtractor_mdsoya
from .expression_tag_integrator import ExpressionTagIntegrator_mdsoya
from .segs_scale_by import SegsScaleBy_mdsoya
from .segs_area import SegsAreaInfo_mdsoya
from .conditional_image_segs_switch import ConditionalImageSegsSwitch_mdsoya
from .detailer_distributor_starter import DetailerDistributorStarter_mdsoya
from .detailer_distributor_pipe import DetailerDistributorPipe_mdsoya
from .soya_batch_detailer import SoyaBatchDetailer_mdsoya
from .segs_visualize import SegsVisualize_mdsoya
from .mask_and_segs_bbox import MaskAndSegsBBox_mdsoya
from .segs_label_transfer import SegsLabelTransfer_mdsoya
from .soya_scheduler_test import SoyaSchedulerTest_mdsoya
from .execution_timer import TimeStart_mdsoya, TimeEnd_mdsoya
from .soya_color_adjust import SoyaColorAdjust_mdsoya
from .soya_canny_face_overlay import SoyaCannyFaceOverlay_mdsoya
from .soya_face_detailer import SoyaFaceDetailer_mdsoya
from .soya_face_pasteback import SoyaFacePasteback_mdsoya
from .soya_process_collector2 import SoyaProcessCollector2_mdsoya
from .soya_eye_extractor import SoyaEyeExtractor_mdsoya
from .soya_mask_range_adjust import SoyaMaskRangeAdjust_mdsoya
from .soya_mask_brightness import SoyaMaskBrightness_mdsoya
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
# [FaceID disabled - low performance]
# from .soya_faceid_embed import SoyaFaceIDMaskPatcher_mdsoya, SoyaFaceIDModelPatcher_mdsoya
# from .soya_process_collector_after import SoyaProcessCollectorAfter_mdsoya
# from .soya_process_collector_embedimg import SoyaProcessCollectorEmbedimg_mdsoya

NODE_CLASS_MAPPINGS = {
    "SoyaProcessDivider_mdsoya": SoyaProcessDivider_mdsoya,
    "SoyaProcessCollector_mdsoya": SoyaProcessCollector_mdsoya,
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
    "FilterClosedEyes_mdsoya": FilterClosedEyes_mdsoya,
    "ConditionalLoraLoader_mdsoya": ConditionalLoraLoader_mdsoya,
    "ExpressionTagExtractor_mdsoya": ExpressionTagExtractor_mdsoya,
    "ExpressionTagIntegrator_mdsoya": ExpressionTagIntegrator_mdsoya,
    "SegsScaleBy_mdsoya": SegsScaleBy_mdsoya,
    "SegsAreaInfo_mdsoya": SegsAreaInfo_mdsoya,
    "ConditionalImageSegsSwitch_mdsoya": ConditionalImageSegsSwitch_mdsoya,
    "DetailerDistributorStarter_mdsoya": DetailerDistributorStarter_mdsoya,
    "DetailerDistributorPipe_mdsoya": DetailerDistributorPipe_mdsoya,
    "SoyaBatchDetailer_mdsoya": SoyaBatchDetailer_mdsoya,
    "SegsVisualize_mdsoya": SegsVisualize_mdsoya,
    "MaskAndSegsBBox_mdsoya": MaskAndSegsBBox_mdsoya,
    "SegsLabelTransfer_mdsoya": SegsLabelTransfer_mdsoya,
    "SoyaSchedulerTest_mdsoya": SoyaSchedulerTest_mdsoya,
    "TimeStart_mdsoya": TimeStart_mdsoya,
    "TimeEnd_mdsoya": TimeEnd_mdsoya,
    "SoyaColorAdjust_mdsoya": SoyaColorAdjust_mdsoya,
    "SoyaCannyFaceOverlay_mdsoya": SoyaCannyFaceOverlay_mdsoya,
    "SoyaFaceDetailer_mdsoya": SoyaFaceDetailer_mdsoya,
    "SoyaFacePasteback_mdsoya": SoyaFacePasteback_mdsoya,
    "SoyaProcessCollector2_mdsoya": SoyaProcessCollector2_mdsoya,
    "SoyaEyeExtractor_mdsoya": SoyaEyeExtractor_mdsoya,
    "SoyaMaskRangeAdjust_mdsoya": SoyaMaskRangeAdjust_mdsoya,
    "SoyaMaskBrightness_mdsoya": SoyaMaskBrightness_mdsoya,
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
    # [FaceID disabled - low performance]
    # "SoyaFaceIDMaskPatcher_mdsoya": SoyaFaceIDMaskPatcher_mdsoya,
    # "SoyaFaceIDModelPatcher_mdsoya": SoyaFaceIDModelPatcher_mdsoya,
    # "SoyaProcessCollectorAfter_mdsoya": SoyaProcessCollectorAfter_mdsoya,
    # "SoyaProcessCollectorEmbedimg_mdsoya": SoyaProcessCollectorEmbedimg_mdsoya,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SoyaProcessDivider_mdsoya": "Soya Process Divider (Soya)",
    "SoyaProcessCollector_mdsoya": "Soya Process Collector (Soya)",
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
    "FilterClosedEyes_mdsoya": "Filter Closed Eyes (Soya)",
    "ConditionalLoraLoader_mdsoya": "Conditional LoRA Loader (Soya)",
    "ExpressionTagExtractor_mdsoya": "Expression Tag Extractor (Soya)",
    "ExpressionTagIntegrator_mdsoya": "Expression Tag Integrator (Soya)",
    "SegsScaleBy_mdsoya": "SEGS Scale By (Soya)",
    "SegsAreaInfo_mdsoya": "SEGS Area Info (Soya)",
    "ConditionalImageSegsSwitch_mdsoya": "Conditional Image/Segs Switch (Soya)",
    "DetailerDistributorStarter_mdsoya": "Detailer Distributor Starter (Soya)",
    "DetailerDistributorPipe_mdsoya": "Detailer Distributor Pipe (Soya)",
    "SoyaBatchDetailer_mdsoya": "Soya Batch Detailer (Soya)",
    "SegsVisualize_mdsoya": "SEGS Visualize (Soya)",
    "MaskAndSegsBBox_mdsoya": "Mask AND Segs BBox (Soya)",
    "SegsLabelTransfer_mdsoya": "SEGS Label Transfer (Soya)",
    "SoyaSchedulerTest_mdsoya": "Soya Scheduler Test (Soya)",
    "TimeStart_mdsoya": "Time Start (Soya)",
    "TimeEnd_mdsoya": "Time End (Soya)",
    "SoyaColorAdjust_mdsoya": "Color Adjust (Soya)",
    "SoyaCannyFaceOverlay_mdsoya": "Canny Face Overlay (Soya)",
    "SoyaFaceDetailer_mdsoya": "Soya Face Detailer (Soya)",
    "SoyaFacePasteback_mdsoya": "Soya Face Paste-back (Soya)",
    "SoyaProcessCollector2_mdsoya": "Soya Process Collector2 (Soya)",
    "SoyaEyeExtractor_mdsoya": "Soya Eye Extractor (Soya)",
    "SoyaMaskRangeAdjust_mdsoya": "Mask Range Adjust (Soya)",
    "SoyaMaskBrightness_mdsoya": "Mask Brightness (Soya)",
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
    # [FaceID disabled - low performance]
    # "SoyaFaceIDMaskPatcher_mdsoya": "Soya FaceID Mask Patcher (Soya)",
    # "SoyaFaceIDModelPatcher_mdsoya": "Soya FaceID Model Patcher (Soya)",
    # "SoyaProcessCollectorAfter_mdsoya": "Soya Process Collector After (Soya)",
    # "SoyaProcessCollectorEmbedimg_mdsoya": "Soya Process Collector Embedimg (Soya)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

WEB_DIRECTORY = "./soya_scheduler/web"

# Set up API routes
try:
    from .soya_scheduler.server import setup_routes
    setup_routes()
except Exception as e:
    print(f"[Soya:Scheduler] Failed to setup routes: {e}")
