import torch

try:
    from impact.impact_pack import FaceDetailer
    HAS_IMPACT = True
except ImportError:
    HAS_IMPACT = False


class SoyaFaceDetailerToggle_mdsoya:
    """
    FaceDetailer 래퍼 — enable=False면 연산 없이 원본 이미지를 그대로 반환합니다.
    enable=True면 Impact Pack FaceDetailer를 호출합니다.
    """

    @classmethod
    def INPUT_TYPES(cls):
        if not HAS_IMPACT:
            return {"required": {}}

        inputs = FaceDetailer.INPUT_TYPES()
        # enable을 맨 앞에 추가
        inputs["required"] = {
            "enable": ("STRING", {"default": "true"}),
            **inputs["required"],
        }
        return inputs

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK", "DETAILER_PIPE", "IMAGE")
    RETURN_NAMES = ("image", "cropped_refined", "cropped_enhanced_alpha", "mask", "detailer_pipe", "cnet_images")
    OUTPUT_IS_LIST = (False, True, True, False, False, True)
    FUNCTION = "doit"
    CATEGORY = "Soya"

    def doit(self, enable, image, model, clip, vae, guide_size, guide_size_for, max_size,
             seed, steps, cfg, sampler_name, scheduler,
             positive, negative, denoise, feather, noise_mask, force_inpaint,
             bbox_threshold, bbox_dilation, bbox_crop_factor,
             sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion,
             sam_mask_hint_threshold, sam_mask_hint_use_negative,
             drop_size, bbox_detector, wildcard, cycle=1,
             sam_model_opt=None, segm_detector_opt=None, detailer_hook=None,
             inpaint_model=False, noise_mask_feather=0,
             scheduler_func_opt=None, tiled_encode=False, tiled_decode=False):

        use = enable.strip().lower() in ("true", "1", "yes")

        if not use:
            print("[SoyaFaceDetailerToggle] DISABLED — bypassing FaceDetailer")
            B, H, W, C = image.shape
            empty_mask = torch.zeros((B, H, W), dtype=torch.float32)
            empty_img = [torch.zeros((1, 64, 64, 3), dtype=torch.float32)]
            pipe = (model, clip, vae, positive, negative, wildcard,
                    bbox_detector, segm_detector_opt, sam_model_opt, detailer_hook,
                    None, None, None, None)
            return (image, empty_img, empty_img, empty_mask, pipe, empty_img)

        print("[SoyaFaceDetailerToggle] ENABLED — running FaceDetailer")
        fd = FaceDetailer()
        return fd.doit(
            image, model, clip, vae, guide_size, guide_size_for, max_size,
            seed, steps, cfg, sampler_name, scheduler,
            positive, negative, denoise, feather, noise_mask, force_inpaint,
            bbox_threshold, bbox_dilation, bbox_crop_factor,
            sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion,
            sam_mask_hint_threshold, sam_mask_hint_use_negative,
            drop_size, bbox_detector, wildcard, cycle,
            sam_model_opt=sam_model_opt, segm_detector_opt=segm_detector_opt,
            detailer_hook=detailer_hook, inpaint_model=inpaint_model,
            noise_mask_feather=noise_mask_feather, scheduler_func_opt=scheduler_func_opt,
            tiled_encode=tiled_encode, tiled_decode=tiled_decode,
        )
