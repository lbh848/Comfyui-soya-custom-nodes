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

    def doit(self, **kwargs):

        enable = kwargs.get("enable", "true")
        use = enable.strip().lower() in ("true", "1", "yes")

        if not use:
            print("[SoyaFaceDetailerToggle] DISABLED — bypassing FaceDetailer")
            image = kwargs["image"]
            model = kwargs["model"]
            clip = kwargs["clip"]
            vae = kwargs["vae"]
            positive = kwargs.get("positive")
            negative = kwargs.get("negative")
            wildcard = kwargs.get("wildcard")
            bbox_detector = kwargs.get("bbox_detector")
            segm_detector_opt = kwargs.get("segm_detector_opt")
            sam_model_opt = kwargs.get("sam_model_opt")
            detailer_hook = kwargs.get("detailer_hook")

            B, H, W, C = image.shape
            empty_mask = torch.zeros((B, H, W), dtype=torch.float32)
            empty_img = [torch.zeros((1, 64, 64, 3), dtype=torch.float32)]
            pipe = (model, clip, vae, positive, negative, wildcard,
                    bbox_detector, segm_detector_opt, sam_model_opt, detailer_hook,
                    None, None, None, None)
            return (image, empty_img, empty_img, empty_mask, pipe, empty_img)

        print("[SoyaFaceDetailerToggle] ENABLED — running FaceDetailer")
        fd = FaceDetailer()
        # Pass all kwargs except 'enable' to FaceDetailer
        fd_kwargs = {k: v for k, v in kwargs.items() if k != "enable"}
        return fd.doit(**fd_kwargs)
