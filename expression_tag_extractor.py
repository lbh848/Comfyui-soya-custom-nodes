import torch
import numpy as np
from PIL import Image


EXPRESSION_POOL = {
    # 기본 표정
    "smile", "light_smile", "false_smile", "nervous_smile",
    "evil_smile", "seductive_smile", "smirk", "grin", "evil_grin",
    "frown", "light_frown", "scowl", "pout", "pouty_lips",
    "expressionless", "laughing",
    # 입/이 관련
    "open_mouth", "closed_mouth", "parted_lips", "lip_biting",
    "licking_lips", "tongue", "tongue_out", "long_tongue",
    "clenched_teeth", "teeth", "upper_teeth", "lower_teeth",
    "sharp_teeth", "fang", "fang_out", "fangs", "fangs_out",
    "puckered_lips", "pursed_lips", "sideways_mouth",
    "triangle_mouth", "split_mouth", "rectangular_mouth", "wavy_mouth",
    "mouth_pull",
    # 감정
    "angry", "annoyed", "embarrassed", "surprised", "scared",
    "crying", "crying_with_eyes_open", "sad", "happy", "happy_tears",
    "serious", "nervous", "shy", "bored", "tired", "sleepy",
    "pain", "jealous", "worried", "confused", "panicking",
    # 얼굴 붉힘/땀/눈물
    "blush", "light_blush", "full-face_blush", "blush_stickers",
    "nose_blush", "ear_blush", "body_blush",
    "sweatdrop", "flying_sweatdrops", "sweat", "sweating_profusely",
    "tears", "teardrop", "streaming_tears", "tearing_up",
    "happy_tears", "wiping_tears",
    # 기타 표정 요소
    "anger_vein", "spoken_anger_vein", "spoken_blush",
    "spoken_sweatdrop", "spoken_ellipsis",
    "jitome", "crazy_smile", "ahegao",
    "screaming", "shouting", "sigh", "yawning",
    "defeat", "gloom_(expression)",
    "raised_eyebrow", "raised_eyebrows",
}


class ExpressionTagExtractor_mdsoya:
    """
    애니메이션 얼굴 이미지에서 DeepDanbooru ONNX 모델로
    표정 관련 태그만 추출하는 노드.
    """

    _model = None
    _threshold = None

    @classmethod
    def _get_model(cls, threshold):
        if cls._model is None or cls._threshold != threshold:
            from deepdanbooru_onnx import DeepDanbooru
            print("[Soya: ExpressionTag] DeepDanbooru 모델 로딩 중...")
            cls._model = DeepDanbooru(threshold=threshold)
            cls._threshold = threshold
        return cls._model

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.1, "max": 0.95, "step": 0.05},
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tags",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "extract_tags"
    CATEGORY = "Soya"

    def extract_tags(self, image, threshold):
        model = self._get_model(threshold)
        results = []

        for i, img in enumerate(image):
            img_np = (255.0 * img.cpu().numpy()).astype(np.uint8)
            pil_img = Image.fromarray(img_np)

            tags_dict = model(pil_img)

            extracted = [tag for tag in tags_dict if tag in EXPRESSION_POOL]

            tag_str = ", ".join(tag.replace("_", " ") for tag in extracted)
            print(
                f"[Soya: ExpressionTag] Batch {i}: "
                f"total={len(tags_dict)}, expression={len(extracted)} -> {tag_str}"
            )
            results.append(tag_str)

        return (results,)
