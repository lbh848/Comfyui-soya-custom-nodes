import ast


class ZipStringBatch_mdsoya:
    """
    두 개의 문자열 배치를 받아 배치별로 결합하여 배치로 반환합니다.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string1": ("STRING", {"forceInput": True}),
                "string2": ("STRING", {"forceInput": True}),
                "delimiter": ("STRING", {"default": ", "}),
            }
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "zip_strings"
    CATEGORY = "Soya/Utils"

    def _flatten(self, items):
        """문자열 리스트 표현, 실제 리스트, 일반 문자열 모두 평탄화."""
        flat = []
        for item in items:
            # 이미 리스트인 경우
            if isinstance(item, list):
                flat.extend(str(x) for x in item)
                continue
            s = str(item).strip()
            # 문자열이 리스트 형태인 경우 파싱
            if s.startswith('[') and s.endswith(']'):
                try:
                    parsed = ast.literal_eval(s)
                    if isinstance(parsed, list):
                        flat.extend(str(x) for x in parsed)
                        continue
                except (ValueError, SyntaxError):
                    pass
            flat.append(s)
        return flat

    def zip_strings(self, string1, string2, delimiter):
        delim = delimiter[0] if delimiter else ", "

        string1 = self._flatten(string1)
        string2 = self._flatten(string2)

        # Broadcast
        if len(string1) == 1 and len(string2) > 1:
            string1 = string1 * len(string2)
        elif len(string2) == 1 and len(string1) > 1:
            string2 = string2 * len(string1)

        result = []
        for a, b in zip(string1, string2):
            result.append(a + delim + b)
        return (result,)
