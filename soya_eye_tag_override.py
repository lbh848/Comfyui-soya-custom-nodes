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
                "on_eye_closed_additional": ("STRING", {
                    "default": "",
                }),
                "eye_context": ("EYE_CONTEXT",),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("character_data",)
    FUNCTION = "override_eye_tags"
    CATEGORY = "Soya/FaceMatch"

    def override_eye_tags(self, names, overlap_ratios, character_data,
                          threshold, override_tag, on_eye_closed_additional,
                          eye_context):
        th = threshold[0] if isinstance(threshold, list) else threshold
        tag = override_tag[0] if isinstance(override_tag, list) else override_tag
        additional = on_eye_closed_additional[0] if isinstance(on_eye_closed_additional, list) else on_eye_closed_additional
        data_str = character_data[0] if isinstance(character_data, list) else character_data

        # Extract per-eye ratios from eye_context
        if isinstance(eye_context, list) and len(eye_context) > 0:
            ctx = eye_context[0] if isinstance(eye_context[0], dict) else {}
        elif isinstance(eye_context, dict):
            ctx = eye_context
        else:
            ctx = {}

        faces = ctx.get("faces", []) if isinstance(ctx, dict) else []
        face_per_eye = {}
        for face in faces:
            face_per_eye[face["name"]] = face.get("per_eye_ratios", [])

        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            return (data_str,)

        char_list = data.get("list", [])

        for i, name in enumerate(names):
            if i >= len(overlap_ratios):
                break

            per_eye = face_per_eye.get(name, [])

            if per_eye:
                closed_count = sum(1 for r in per_eye if r > th)
                if closed_count == 0:
                    continue  # all eyes open
                elif closed_count == len(per_eye):
                    # all eyes closed
                    applied_tag = tag
                else:
                    # wink: some eyes closed, some open
                    existing_tag = ""
                    for entry in char_list:
                        if entry.get("CHAR") == name:
                            existing_tag = entry.get("EYE_TAGS", "")
                            break
                    parts = [p for p in [existing_tag, additional] if p]
                    applied_tag = ", ".join(parts)
            else:
                # fallback to overall overlap_ratio
                if overlap_ratios[i] <= th:
                    continue
                applied_tag = tag

            for entry in char_list:
                if entry.get("CHAR") == name:
                    entry["EYE_TAGS"] = applied_tag

        return (json.dumps(data, ensure_ascii=False),)