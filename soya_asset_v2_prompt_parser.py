from .soya_prompt_parser import SoyaPromptParser_mdsoya


class SoyaAssetV2PromptParser_mdsoya(SoyaPromptParser_mdsoya):
    """Parses a tagged text block (asset v2 format) and outputs each field with the correct type."""

    _FIELDS = {
        # STRING fields
        "ANIMA_QUALITY": "STRING",
        "ANIMA_ARTIST": "STRING",
        "ANIMA": "STRING",
        "SDXL_QUALITY": "STRING",
        "SDXL_ARTIST": "STRING",
        "SDXL": "STRING",
        "CHAR_LIST": "STRING",
        "FACE_ID_ACTIVATE": "STRING",
        "FACE_ID_DIR": "STRING",
        "STYLE_ACTIVATE": "STRING",
        "STYLE_DIR": "STRING",
        "LORA_ACTIVATE": "STRING",
        "LORA_DATA": "STRING",
        "FACE_LORA_ACTIVATE": "STRING",
        "FACE_LORA_DATA": "STRING",
        "STYLE_LORA_ACTIVATE": "STRING",
        "STYLE_LORA_DATA": "STRING",
        "CHAR_FACE_TAG_INFORM": "STRING",
        "POSE_ACTIVATE": "STRING",
        "POSE_DATA": "STRING",
        "HRF_ACTIVATE": "STRING",
        "ANIMA_HRF_ACTIVATE": "STRING",
        "HRF_RESTORE_SIZE": "STRING",
        "HRF_CONTROL_NET": "STRING",
        "ANIMA_FD_ACTIVATE": "STRING",
        "ANIMA_HD_ACTIVATE": "STRING",
        "ANIMA_ED_ACTIVATE": "STRING",
        "FD_ACTIVATE": "STRING",
        "HD_ACTIVATE": "STRING",
        "ED_ACTIVATE": "STRING",
        # FLOAT fields
        "FACE_ID_STR": "FLOAT",
        "STYLE_STR": "FLOAT",
        "FACE_CROP_TOP": "FLOAT",
        "FACE_CROP_BOTTOM": "FLOAT",
        "HRF_SIZE": "FLOAT",
        # INT fields
        "IMG_W": "INT",
        "IMG_H": "INT",
        "SEED": "INT",
        # Preserve all existing output indices.
        "SNAPREF": "STRING",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = tuple(t for t in _FIELDS.values())
    RETURN_NAMES = tuple(_FIELDS.keys())
    FUNCTION = "parse"
    CATEGORY = "Soya/Util"
    OUTPUT_NODE = True
