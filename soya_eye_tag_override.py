import json


class SoyaEyeTagOverride_mdsoya:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "names": ("STRING",),
                "overlap_ratios": ("FLOAT",),
                "character_data": ("STRING", {
                    "default": '{"list": []}',
                    "multiline": True,
                }),
                "threshold": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                }),
                "override_tag": ("STRING", {
                    "default": "closed eyes",
                }),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("character_data",)
    FUNCTION = "override_eye_tags"
    CATEGORY = "Soya/FaceMatch"

    def override_eye_tags(self, names, overlap_ratios, character_data,
                          threshold, override_tag):
        th = threshold[0] if isinstance(threshold, list) else threshold
        tag = override_tag[0] if isinstance(override_tag, list) else override_tag
        data_str = character_data[0] if isinstance(character_data, list) else character_data

        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            return (data_str,)

        char_list = data.get("list", [])

        for i, name in enumerate(names):
            if i >= len(overlap_ratios):
                break
            if overlap_ratios[i] > th:
                for entry in char_list:
                    if entry.get("CHAR") == name:
                        entry["EYE_TAGS"] = tag

        return (json.dumps(data, ensure_ascii=False),)
