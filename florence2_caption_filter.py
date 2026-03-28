class Florence2CaptionFilter_mdsoya:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "caption": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "filter_caption"
    CATEGORY = "Soya"

    def filter_caption(self, image, caption):
        batch_size = image.shape[0]

        if batch_size == 1:
            return (caption,)

        captions = caption if isinstance(caption, list) else [caption]
        filtered = []

        for i in range(0, len(captions), 2):
            if i + 1 < len(captions):
                a, b = captions[i], captions[i + 1]
                filtered.append(a if len(a.split()) >= len(b.split()) else b)
            else:
                filtered.append(captions[i])

        return (filtered,)
