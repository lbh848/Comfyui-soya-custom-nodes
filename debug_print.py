class DebugPrint_mdsoya:
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "print_text"
    CATEGORY = "Soya"

    def print_text(self, text):
        print(f"[DebugPrint] {text}")
        return ()
