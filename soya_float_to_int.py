class SoyaFloatToInt_mdsoya:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "float_value": ("FLOAT", {"default": 0.0}),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int",)
    FUNCTION = "convert"
    CATEGORY = "Soya"

    def convert(self, float_value):
        return (int(float_value),)
