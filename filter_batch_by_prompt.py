import torch


class FilterBatchByPrompt_mdsoya:
    """
    prompt에 포함된 이름만 batch_names와 batch_images에서 필터링하여 반환합니다.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "batch_names": ("STRING", {"multiline": True, "default": ""}),
                "batch_images": ("IMAGE",),
            }
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("matched_names", "matched_images")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "filter_by_prompt"
    CATEGORY = "Soya/Character"

    def filter_by_prompt(self, prompt, batch_names, batch_images):
        # prompt가 비어있으면 빈 결과 반환
        if not prompt or (isinstance(prompt, list) and (len(prompt) == 0 or not prompt[0])):
            return ([], torch.zeros((0, 64, 64, 3), dtype=torch.float32))

        prompt_text = prompt[0] if isinstance(prompt, list) else prompt

        def preprocess(text):
            return text.lower().replace("_", " ").replace("-", " ")

        prompt_processed = preprocess(prompt_text)

        matched_names = []
        matched_images = []

        for i, name in enumerate(batch_names):
            name_processed = preprocess(name)
            if name_processed in prompt_processed:
                matched_names.append(name)
                if i < len(batch_images):
                    matched_images.append(batch_images[i])

        if matched_images:
            batch = torch.cat(matched_images, dim=0)
        else:
            batch = torch.zeros((0, 64, 64, 3), dtype=torch.float32)

        return (matched_names, batch)
