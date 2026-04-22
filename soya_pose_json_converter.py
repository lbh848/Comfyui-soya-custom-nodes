import json


class SoyaPoseJsonConverter_mdsoya:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_json": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("POSE_KEYPOINT", "STRING")
    RETURN_NAMES = ("pose_keypoint", "info")
    FUNCTION = "convert"
    CATEGORY = "Soya"
    OUTPUT_NODE = True

    def convert(self, pose_json):
        data = json.loads(pose_json.strip())

        people = data.get("people", [])
        canvas_h = data.get("canvas_height", 512)
        canvas_w = data.get("canvas_width", 512)
        num_people = len(people)

        info_lines = [f"[Pose JSON Converter]"]
        info_lines.append(f"  canvas: {canvas_w}x{canvas_h}")
        info_lines.append(f"  people: {num_people}")

        for i, person in enumerate(people):
            parts = []
            for key in ["pose_keypoints_2d", "face_keypoints_2d",
                        "hand_left_keypoints_2d", "hand_right_keypoints_2d"]:
                arr = person.get(key, [])
                n_kp = len(arr) // 3 if arr else 0
                if n_kp > 0:
                    parts.append(f"{key}: {n_kp}pts")
            info_lines.append(f"  person[{i}]: {', '.join(parts) if parts else 'no keypoints'}")

        info = "\n".join(info_lines)
        print(info)

        return ([data], info)
