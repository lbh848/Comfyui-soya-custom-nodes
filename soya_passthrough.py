class SoyaPassthrough_mdsoya:
    """Passes through any input unchanged. Useful for organizing node wires."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("*",),
            }
        }

    RETURN_TYPES = ("*",)
    FUNCTION = "passthrough"
    CATEGORY = "Soya/Util"
    OUTPUT_NODE = False

    def passthrough(self, input):
        return (input,)
