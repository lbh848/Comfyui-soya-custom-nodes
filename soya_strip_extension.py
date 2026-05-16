import os


class SoyaStripExtension_mdsoya:
    """파일명 목록에서 지정한 접미사(예: .webp)를 제거합니다."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_names": ("STRING", {"forceInput": True}),
                "strip_suffix": ("STRING", {"default": ".webp", "multiline": False}),
            }
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_names",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "strip_extension"
    CATEGORY = "Soya/Utility"

    def strip_extension(self, file_names, strip_suffix):
        suffix = strip_suffix[0] if strip_suffix else ""
        return ([name.removesuffix(suffix) for name in file_names],)
