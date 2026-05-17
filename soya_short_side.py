class SoyaShortSide_mdsoya:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("short_side",)
    FUNCTION = "doit"
    CATEGORY = "Soya"

    def doit(self, image):
        _, h, w, _ = image.shape
        return (min(h, w),)
