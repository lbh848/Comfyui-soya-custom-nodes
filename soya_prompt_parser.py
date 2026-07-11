class SoyaPromptParser_mdsoya:
    """Parses a tagged text block and outputs each field with the correct type."""

    _FIELDS = {
        # STRING fields
        "ANIMA_QUALITY": "STRING",
        "ANIMA_ARTIST": "STRING",
        "ANIMA_CONTENT": "STRING",
        "ANIMA_ALL": "STRING",
        "SDXL_QUALITY": "STRING",
        "SDXL_ARTIST": "STRING",
        "SDXL": "STRING",
        "CHAR_LIST": "STRING",
        "CACHE_PATH": "STRING",
        "FACE_ID_ACTIVATE": "STRING",
        "FACE_ID_DIR": "STRING",
        "LORA_ACTIVATE": "STRING",
        "FACE_LORA_ACTIVATE": "STRING",
        "LORA_DATA": "STRING",
        "FACE_LORA_DATA": "STRING",
        "STYLE_LORA_ACTIVATE": "STRING",
        "STYLE_LORA_DATA": "STRING",
        "CHAR_FACE_TAG_INFORM": "STRING",
        "HRF_ACTIVATE": "STRING",
        "ANIMA_HRF_ACTIVATE": "STRING",
        "HRF_RESTORE_SIZE": "STRING",
        "ANIMA_FD_ACTIVATE": "STRING",
        "ANIMA_HD_ACTIVATE": "STRING",
        "ANIMA_ED_ACTIVATE": "STRING",
        "FD_ACTIVATE": "STRING",
        "HD_ACTIVATE": "STRING",
        "ED_ACTIVATE": "STRING",
        # FLOAT fields
        "FACE_ID_STR": "FLOAT",
        "FACE_CROP_TOP": "FLOAT",
        "FACE_CROP_BOTTOM": "FLOAT",
        "HRF_SIZE": "FLOAT",
        # INT fields
        "IMG_W": "INT",
        "IMG_H": "INT",
        "SEED": "INT",
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

    def parse(self, text):
        parsed = {}
        current_key = None
        current_lines = []

        for line in text.split("\n"):
            line = line.rstrip("\r")
            stripped = line.strip()

            bracket = stripped.startswith("[") and stripped.endswith("]")
            if bracket:
                if current_key is not None:
                    parsed[current_key] = "\n".join(current_lines).strip()
                current_key = stripped[1:-1]
                current_lines = []
            else:
                if current_key is not None:
                    current_lines.append(line)

        if current_key is not None:
            parsed[current_key] = "\n".join(current_lines).strip()

        results = []
        for key, dtype in self._FIELDS.items():
            raw = parsed.get(key, "")
            if dtype == "INT":
                try:
                    results.append(int(raw))
                except (ValueError, TypeError):
                    results.append(0)
            elif dtype == "FLOAT":
                try:
                    results.append(float(raw))
                except (ValueError, TypeError):
                    results.append(0.0)
            else:
                results.append(raw)

        return tuple(results)
