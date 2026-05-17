class SoyaToString_mdsoya:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("FLOAT", {"forceInput": True}),
            },
            "optional": {
                "value_int": ("INT", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "doit"
    CATEGORY = "Soya"

    def doit(self, value=None, value_int=None):
        v = value if value_int is None else value_int
        return (str(v),)
