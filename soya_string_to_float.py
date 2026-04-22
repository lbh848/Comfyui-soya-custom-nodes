class SoyaStringToFloat_mdsoya:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"default": "0.0", "multiline": False}),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float",)
    FUNCTION = "convert"
    CATEGORY = "Soya"

    def convert(self, text):
        return (float(text.strip()),)
