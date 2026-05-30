import torch


class FilterImagesByName_mdsoya:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filenames": ("STRING", {"forceInput": True}),
                "filter_names": ("STRING", {"default": "", "multiline": True}),
                "mode": (["include", "exclude"],),
            }
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "filenames")
    OUTPUT_IS_LIST = (False, True)
    FUNCTION = "filter_images"
    CATEGORY = "Soya/Image"

    def filter_images(self, images, filenames, filter_names, mode):
        batch = images[0]
        filter_str = filter_names[0]
        mode_val = mode[0]

        filter_set = set(name.strip() for name in filter_str.split(",") if name.strip())

        if not filter_set:
            return (batch, filenames)

        indices = []
        for i, name in enumerate(filenames):
            match = any(kw in name for kw in filter_set)
            if mode_val == "include" and match:
                indices.append(i)
            elif mode_val == "exclude" and not match:
                indices.append(i)

        if not indices:
            raise ValueError("No images matched the filter condition")

        filtered_batch = batch[indices]
        filtered_names = [filenames[i] for i in indices]

        return (filtered_batch, filtered_names)
