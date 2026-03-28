class JoinStringBatch_mdsoya:
    """
    Batch 형태의 문자열 리스트를 지정된 구분자로 하나의 문자열로 결합합니다.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "strings": ("STRING",),
                "delimiter": ("STRING", {"default": ","}),
            }
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    OUTPUT_IS_LIST = (False,)
    FUNCTION = "join_strings"
    CATEGORY = "Soya/Utils"

    def join_strings(self, strings, delimiter):
        delim = delimiter[0] if delimiter else ","
        result = delim.join(str(s) for s in strings)
        return (result,)
