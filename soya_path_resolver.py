import os
import folder_paths


class SoyaPathResolver_mdsoya:
    """
    Load Images From Path와 Batch LoRA Loader가 사용하는 기준 경로를 출력합니다.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("paths",)
    FUNCTION = "resolve"
    CATEGORY = "Soya/Debug"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def resolve(self):
        lines = []
        lines.append("[input]")
        lines.append(folder_paths.get_input_directory())
        lines.append("")
        lines.append("[lora]")
        lora_bases = folder_paths.get_folder_paths("loras")
        if lora_bases:
            for p in lora_bases:
                lines.append(p)
        else:
            lines.append("(no lora paths registered)")
        return ("\n".join(lines),)
