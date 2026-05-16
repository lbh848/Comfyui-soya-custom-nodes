class SoyaMultiply_mdsoya:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "a": ("FLOAT", {"default": 1.0}),
                "b": ("FLOAT", {"default": 1.0}),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("result",)
    FUNCTION = "multiply"
    CATEGORY = "Soya"

    def multiply(self, a, b):
        return (a * b,)
