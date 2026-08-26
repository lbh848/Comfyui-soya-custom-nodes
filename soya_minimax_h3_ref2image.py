import math
import time
import traceback

import torch

import comfy.model_management
import comfy.nested_tensor
import comfy.utils
import node_helpers
import nodes


CANVAS_MULTIPLE = 32
LATENT_DOWNSCALE = 16
VIDEO_LATENT_CHANNELS = 24
AUDIO_LATENT_CHANNELS = 32
MAX_REFERENCE_IMAGES = 9
REFERENCE_MAX_SHORT_EDGE = 2048
WARN_TARGET_PIXELS = 4 * 1024 * 1024


def _shape_tuple(value):
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return tuple(int(item) for item in shape)


def _resize_image(image, width, height):
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(
        samples,
        width,
        height,
        "lanczos",
        "disabled",
    )
    return samples.movedim(1, -1)


def _aligned_reference_size(source_width, source_height, target_width, target_height, mode):
    if mode == "match":
        scale = min(
            1.0,
            math.sqrt(
                (target_width * target_height) / (source_width * source_height)
            ),
        )
    elif mode == "max":
        scale = min(1.0, REFERENCE_MAX_SHORT_EDGE / min(source_width, source_height))
    else:
        print(f"[H3_REF2IMAGE] 지원하지 않는 ref_image_size: value={mode!r}")
        raise ValueError("ref_image_size는 match 또는 max여야 합니다")

    scaled_width = source_width * scale
    scaled_height = source_height * scale
    width = max(
        CANVAS_MULTIPLE,
        round(scaled_width / CANVAS_MULTIPLE) * CANVAS_MULTIPLE,
    )
    height = max(
        CANVAS_MULTIPLE,
        round(scaled_height / CANVAS_MULTIPLE) * CANVAS_MULTIPLE,
    )
    return width, height


class SoyaMiniMaxH3ReferenceToImage_mdsoya:
    """Experimental MiniMax H3 REF2I conditioning with a true T=1 latent."""

    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            f"ref_image_{index}": (
                "IMAGE",
                {
                    "tooltip": (
                        f"Reference image {index}; address it as <Picture {index}> "
                        "in the prompt. Reference slots must be contiguous."
                    )
                },
            )
            for index in range(2, MAX_REFERENCE_IMAGES + 1)
        }
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": (
                    "VAE",
                    {
                        "tooltip": (
                            "Use the MiniMax H3 single-image VAE. The normal video "
                            "VAE can run, but is not the intended quality path."
                        )
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "default": "Create a new still image using <Picture 1> as the character reference.",
                        "multiline": True,
                        "dynamicPrompts": True,
                    },
                ),
                "width": (
                    "INT",
                    {
                        "default": 1344,
                        "min": CANVAS_MULTIPLE,
                        "max": nodes.MAX_RESOLUTION,
                        "step": CANVAS_MULTIPLE,
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 768,
                        "min": CANVAS_MULTIPLE,
                        "max": nodes.MAX_RESOLUTION,
                        "step": CANVAS_MULTIPLE,
                    },
                ),
                "ref_image_size": (
                    ["match", "max"],
                    {
                        "default": "match",
                        "tooltip": (
                            "match limits each reference to the target pixel area; "
                            "max keeps up to a 2048px short edge and can require much more VRAM."
                        ),
                    },
                ),
                "ref_image_1": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "First reference image; address it as <Picture 1> in the prompt."
                        )
                    },
                ),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("positive", "latent", "diagnostics")
    FUNCTION = "prepare"
    CATEGORY = "Soya/MiniMax H3 (Experimental)"
    DESCRIPTION = (
        "Builds MiniMax H3 reference conditioning for 1-9 images and an empty "
        "single-frame video latent with a zero-length audio stream. Decode the "
        "sampled latent with the standard VAE Decode node and the H3 image VAE."
    )

    @staticmethod
    def _validate_target(width, height):
        for name, value in (("width", width), ("height", height)):
            if isinstance(value, bool) or not isinstance(value, int):
                print(
                    f"[H3_REF2IMAGE] 대상 크기 형식 오류: {name}={value!r}, "
                    f"type={type(value).__name__}"
                )
                raise TypeError(f"{name}는 정수여야 합니다")
            if value < CANVAS_MULTIPLE or value > nodes.MAX_RESOLUTION:
                print(
                    f"[H3_REF2IMAGE] 대상 크기 범위 오류: {name}={value}, "
                    f"allowed={CANVAS_MULTIPLE}..{nodes.MAX_RESOLUTION}"
                )
                raise ValueError(f"{name}가 허용 범위를 벗어났습니다")
            if value % CANVAS_MULTIPLE != 0:
                print(
                    f"[H3_REF2IMAGE] 대상 크기 정렬 오류: {name}={value}, "
                    f"multiple={CANVAS_MULTIPLE}"
                )
                raise ValueError(f"{name}는 {CANVAS_MULTIPLE}의 배수여야 합니다")

    @staticmethod
    def _validate_reference(image, index):
        shape = _shape_tuple(image)
        if shape is None or len(shape) != 4:
            print(
                f"[H3_REF2IMAGE] 레퍼런스 차원 오류: index={index}, shape={shape!r}"
            )
            raise ValueError(f"ref_image_{index}는 [B,H,W,C] 4차원 이미지여야 합니다")
        batch, height, width, channels = shape
        if batch < 1 or height < 1 or width < 1 or channels < 3:
            print(
                "[H3_REF2IMAGE] 레퍼런스 크기 오류: "
                f"index={index}, shape={shape!r}"
            )
            raise ValueError(f"ref_image_{index}의 크기 또는 채널 수가 올바르지 않습니다")
        if batch > 1:
            print(
                "[H3_REF2IMAGE] 레퍼런스 배치에서 첫 이미지만 사용: "
                f"index={index}, batch={batch}"
            )
        return width, height

    @staticmethod
    def _reference_sequence(ref_image_1, optional_images):
        references = [ref_image_1]
        missing_index = None
        for index in range(2, MAX_REFERENCE_IMAGES + 1):
            image = optional_images.get(f"ref_image_{index}")
            if image is None:
                if missing_index is None:
                    missing_index = index
                continue
            if missing_index is not None:
                print(
                    "[H3_REF2IMAGE] 레퍼런스 슬롯 간격 오류: "
                    f"ref_image_{missing_index} is empty, ref_image_{index} is connected"
                )
                raise ValueError(
                    "MiniMax H3 Picture 번호 보존을 위해 레퍼런스 슬롯을 1번부터 연속으로 연결하세요"
                )
            references.append(image)
        return references

    @staticmethod
    def _validate_reference_latent(latent, index):
        shape = _shape_tuple(latent)
        if (
            shape is None
            or len(shape) != 5
            or shape[0] != 1
            or shape[1] != VIDEO_LATENT_CHANNELS
            or shape[2] != 1
        ):
            print(
                "[H3_REF2IMAGE] H3 VAE latent 형식 오류: "
                f"index={index}, shape={shape!r}, expected=[1,24,1,H,W]"
            )
            raise ValueError(
                "레퍼런스 VAE 출력이 MiniMax H3 단일 프레임 latent 형식이 아닙니다"
            )

    def prepare(
        self,
        clip,
        vae,
        prompt,
        width,
        height,
        ref_image_size,
        ref_image_1,
        **optional_images,
    ):
        started_at = time.perf_counter()
        try:
            if not isinstance(prompt, str):
                print(
                    "[H3_REF2IMAGE] 프롬프트 형식 오류: "
                    f"type={type(prompt).__name__}, value={prompt!r}"
                )
                raise TypeError("MiniMax H3 프롬프트는 문자열이어야 합니다")
            if not prompt.strip():
                print("[H3_REF2IMAGE] 프롬프트가 비어 있습니다")
                raise ValueError("MiniMax H3 프롬프트는 비어 있을 수 없습니다")

            self._validate_target(width, height)
            if width * height > WARN_TARGET_PIXELS:
                print(
                    "[H3_REF2IMAGE] 고해상도 VRAM 경고: "
                    f"width={width}, height={height}, pixels={width * height}"
                )

            references = self._reference_sequence(ref_image_1, optional_images)
            ref_items = []
            ref_blocks = []
            reference_diagnostics = []
            reference_token_count = 0

            for index, image in enumerate(references, start=1):
                source_width, source_height = self._validate_reference(image, index)
                ref_width, ref_height = _aligned_reference_size(
                    source_width,
                    source_height,
                    width,
                    height,
                    ref_image_size,
                )
                resized = _resize_image(image[:1], ref_width, ref_height)
                latent = vae.encode(resized)
                self._validate_reference_latent(latent, index)

                ref_items.append({"type": "image", "data": resized})
                ref_blocks.append(
                    {
                        "kind": "image",
                        "latent_h": ref_height // LATENT_DOWNSCALE,
                        "latent_w": ref_width // LATENT_DOWNSCALE,
                        "latent": latent,
                    }
                )
                tokens = (ref_height // CANVAS_MULTIPLE) * (
                    ref_width // CANVAS_MULTIPLE
                )
                reference_token_count += tokens
                reference_diagnostics.append(
                    f"Picture {index}: {source_width}x{source_height}->{ref_width}x{ref_height} ({tokens} tokens)"
                )

            encoded_tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
            conditioning = clip.encode_from_tokens_scheduled(encoded_tokens)
            conditioning = node_helpers.conditioning_set_values(
                conditioning,
                {"minimax_refs": ref_blocks},
            )

            device = comfy.model_management.intermediate_device()
            video = torch.zeros(
                (
                    1,
                    VIDEO_LATENT_CHANNELS,
                    1,
                    height // LATENT_DOWNSCALE,
                    width // LATENT_DOWNSCALE,
                ),
                device=device,
            )
            audio = torch.zeros(
                (1, AUDIO_LATENT_CHANNELS, 2, 0),
                device=device,
            )
            latent = {
                "samples": comfy.nested_tensor.NestedTensor((video, audio)),
            }

            target_tokens = (height // CANVAS_MULTIPLE) * (
                width // CANVAS_MULTIPLE
            )
            elapsed = time.perf_counter() - started_at
            diagnostic_lines = [
                "MiniMax H3 REF2I experimental preparation",
                f"target: {width}x{height}, T=1, target visual tokens={target_tokens}",
                f"references: {len(references)}, ref visual tokens={reference_token_count}",
                *reference_diagnostics,
                f"video latent: {tuple(video.shape)}",
                f"audio latent: {tuple(audio.shape)}",
                f"prepare elapsed: {elapsed:.3f}s",
            ]
            diagnostics = "\n".join(diagnostic_lines)
            print(f"[H3_REF2IMAGE] 준비 완료\n{diagnostics}")
            return conditioning, latent, diagnostics
        except torch.cuda.OutOfMemoryError as exc:
            print(
                "[H3_REF2IMAGE] CUDA OOM: "
                f"width={width!r}, height={height!r}, ref_image_size={ref_image_size!r}, "
                f"error={exc}. 해상도 또는 레퍼런스 크기/개수를 줄이세요."
            )
            traceback.print_exc()
            raise
        except Exception as exc:
            print(
                "[H3_REF2IMAGE] 준비 실패: "
                f"type={type(exc).__name__}, error={exc}, width={width!r}, "
                f"height={height!r}, ref_image_size={ref_image_size!r}, "
                f"prompt_preview={str(prompt)[:300]!r}"
            )
            traceback.print_exc()
            raise
