import math
import traceback

import comfy.utils
import node_helpers


class SoyaQwenEditConditioning_mdsoya:
    """Qwen Image Edit conditioning aligned to the masked target latent.

    The target-latent sizing follows the v2 encoder recommended with
    Phr00t/Qwen-Image-Edit-Rapid-AIO, without replacing ComfyUI core files.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": True,
                    },
                ),
            },
            "optional": {
                "vae": ("VAE",),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "target_latent": ("LATENT",),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "Soya/Qwen"

    @staticmethod
    def _target_dimensions(target_latent):
        if target_latent is None:
            return None
        if not isinstance(target_latent, dict) or "samples" not in target_latent:
            print(
                "[QWEN_EDIT_CONDITIONING] target_latent 형식 오류: "
                f"type={type(target_latent).__name__}, value={target_latent!r}"
            )
            raise ValueError("target_latent에 samples가 없습니다")
        samples = target_latent["samples"]
        if not hasattr(samples, "shape") or len(samples.shape) < 4:
            print(
                "[QWEN_EDIT_CONDITIONING] target_latent samples 차원 오류: "
                f"shape={getattr(samples, 'shape', None)!r}"
            )
            raise ValueError("target_latent samples는 4차원이어야 합니다")
        return int(samples.shape[-1]) * 8, int(samples.shape[-2]) * 8

    def encode(
        self,
        clip,
        prompt,
        vae=None,
        image1=None,
        image2=None,
        image3=None,
        image4=None,
        target_latent=None,
    ):
        try:
            if not isinstance(prompt, str):
                print(
                    "[QWEN_EDIT_CONDITIONING] 프롬프트 형식 오류: "
                    f"type={type(prompt).__name__}, value={prompt!r}"
                )
                raise TypeError("Qwen Edit 프롬프트는 문자열이어야 합니다")
            if not prompt.strip():
                print("[QWEN_EDIT_CONDITIONING] 프롬프트가 비어 있습니다")

            target_dimensions = self._target_dimensions(target_latent)
            images_vl = []
            reference_latents = []
            image_prompt_parts = []

            for index, image in enumerate(
                (image1, image2, image3, image4),
                start=1,
            ):
                if image is None:
                    continue
                if not hasattr(image, "shape") or len(image.shape) != 4:
                    print(
                        "[QWEN_EDIT_CONDITIONING] 이미지 차원 오류: "
                        f"index={index}, shape={getattr(image, 'shape', None)!r}"
                    )
                    raise ValueError(
                        f"Qwen Edit 참조 이미지 {index}는 4차원이어야 합니다"
                    )

                samples = image.movedim(-1, 1)
                source_height = int(samples.shape[2])
                source_width = int(samples.shape[3])
                if source_width <= 0 or source_height <= 0:
                    print(
                        "[QWEN_EDIT_CONDITIONING] 이미지 크기 오류: "
                        f"index={index}, width={source_width}, height={source_height}"
                    )
                    raise ValueError(
                        f"Qwen Edit 참조 이미지 {index} 크기가 올바르지 않습니다"
                    )

                vision_scale = math.sqrt(
                    (384 * 384) / (source_width * source_height)
                )
                vision_width = max(1, round(source_width * vision_scale))
                vision_height = max(1, round(source_height * vision_scale))
                vision_samples = comfy.utils.common_upscale(
                    samples,
                    vision_width,
                    vision_height,
                    "lanczos",
                    "center",
                )
                images_vl.append(vision_samples.movedim(1, -1))

                if vae is not None:
                    latent_samples = samples
                    if target_dimensions is not None:
                        latent_samples = comfy.utils.common_upscale(
                            samples,
                            target_dimensions[0],
                            target_dimensions[1],
                            "lanczos",
                            "center",
                        )
                    reference_latents.append(
                        vae.encode(
                            latent_samples.movedim(1, -1)[:, :, :, :3]
                        )
                    )

                image_prompt_parts.append(
                    "Picture "
                    f"{index}: <|vision_start|><|image_pad|><|vision_end|>"
                )

            llama_template = (
                "<|im_start|>system\n"
                "Describe key details of the input image, including objects, "
                "characters, poses, facial features, clothing, setting, textures, "
                "and style. Then follow the user's instruction to alter, modify, "
                "or recreate the image while preserving details that were not "
                "requested to change."
                "<|im_end|>\n<|im_start|>user\n{}"
                "<|im_end|>\n<|im_start|>assistant\n"
            )
            full_prompt = "".join(image_prompt_parts) + prompt
            tokens = clip.tokenize(
                full_prompt,
                images=images_vl,
                llama_template=llama_template,
            )
            conditioning = clip.encode_from_tokens_scheduled(tokens)
            if reference_latents:
                conditioning = node_helpers.conditioning_set_values(
                    conditioning,
                    {"reference_latents": reference_latents},
                    append=True,
                )

            print(
                "[QWEN_EDIT_CONDITIONING] 인코딩 완료: "
                f"images={len(images_vl)}, "
                f"reference_latents={len(reference_latents)}, "
                f"target_dimensions={target_dimensions}"
            )
            return (conditioning,)
        except Exception as exc:
            print(
                "[QWEN_EDIT_CONDITIONING] 인코딩 실패: "
                f"type={type(exc).__name__}, error={exc}, "
                f"prompt_preview={str(prompt)[:300]!r}"
            )
            traceback.print_exc()
            raise
