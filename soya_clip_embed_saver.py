import os
import torch
import folder_paths
import comfy.model_management


class SoyaClipEmbedSaver_mdsoya:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "path": ("STRING", {"default": "", "multiline": False}),
                "clip_vision": ("CLIP_VISION",),
            }
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "save_embeds"
    CATEGORY = "Soya/Image"

    @classmethod
    def IS_CHANGED(cls, images, path, clip_vision):
        return float("nan")

    def save_embeds(self, images, path, clip_vision):
        path_str = path[0].strip()
        if not os.path.isabs(path_str):
            input_dir = folder_paths.get_input_directory()
            path_str = os.path.join(input_dir, path_str)
        if not os.path.isdir(path_str):
            raise ValueError(f"Directory not found: {path_str}")

        comfy.model_management.load_model_gpu(clip_vision[0].patcher)

        embeds = []
        for img in images:
            if img.dim() == 3:
                img = img.unsqueeze(0)
            encoded = clip_vision[0].encode_image(img)
            embeds.append(encoded.image_embeds)

        cache_path = os.path.join(path_str, "cache.pt")
        torch.save({"embeds": torch.cat(embeds, dim=0)}, cache_path)
        print(f"[SoyaClipEmbedSaver] Saved {len(embeds)} embeds → {cache_path}")

        return (images,)
