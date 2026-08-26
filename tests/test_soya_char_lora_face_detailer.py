import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch


MODULE_PATH = Path(__file__).parents[1] / "soya_char_lora_face_detailer.py"


def load_module():
    comfy_module = types.ModuleType("comfy")
    comfy_module.__path__ = []
    comfy_sd = types.ModuleType("comfy.sd")
    comfy_utils = types.ModuleType("comfy.utils")
    comfy_module.sd = comfy_sd
    comfy_module.utils = comfy_utils

    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_full_path = lambda *_args: None
    folder_paths.get_folder_paths = lambda *_args: []

    spec = importlib.util.spec_from_file_location(
        "soya_char_lora_face_detailer_for_test", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules,
        {
            "comfy": comfy_module,
            "comfy.sd": comfy_sd,
            "comfy.utils": comfy_utils,
            "folder_paths": folder_paths,
        },
    ):
        spec.loader.exec_module(module)
    return module


MODULE = load_module()


class CroppingVAE:
    def __init__(self, compression):
        self.compression = compression
        self.encoded_shapes = []

    def spacial_compression_encode(self):
        return self.compression

    def encode(self, image):
        height = (image.shape[1] // self.compression) * self.compression
        width = (image.shape[2] // self.compression) * self.compression
        cropped = image[:, :height, :width, :]
        self.encoded_shapes.append(tuple(cropped.shape))
        return cropped

    def decode(self, latent):
        return latent


class CharLoraFaceDetailerCropTests(unittest.TestCase):
    def test_bottom_edge_crop_shifts_inward_and_uses_actual_vae_multiple(self):
        vae = CroppingVAE(compression=16)
        node = MODULE.SoyaCharLoraFaceDetailer_mdsoya()
        image = torch.zeros((1, 100, 100, 3), dtype=torch.float32)
        face_context = {
            "matches": [
                {
                    "name": "alice",
                    "crop": [20, 70, 44, 100],
                    "score": 1.0,
                }
            ]
        }
        char_tags = {
            "list": [
                {
                    "CHAR": "alice",
                    "FACE_TAGS": "",
                    "EYE_TAGS": "",
                    "POSITIVE": "",
                }
            ]
        }

        with (
            mock.patch.object(MODULE, "_encode_conditioning", return_value=None),
            mock.patch.object(
                MODULE,
                "_run_ksampler",
                side_effect=lambda *_args: _args[-2]["samples"],
            ),
        ):
            result, mask, crop_preview, info = node.execute(
                enable="true",
                image=image,
                model=object(),
                clip=object(),
                vae=vae,
                face_context=face_context,
                char_tags=json.dumps(char_tags),
                quality_tags="",
                artist_tags="",
                negative="",
                lora_list='{"list":[]}',
                base_model="anima",
                crop_expand_factor=3.0,
                upscale_factor=1.0,
                seed=1,
                steps=1,
                cfg=1.0,
                sampler_name="euler",
                scheduler="simple",
                denoise=0.3,
                feather=0,
                corner_roundness=0.0,
                noise_mask=True,
            )

        self.assertEqual(tuple(result.shape), (1, 100, 100, 3))
        self.assertEqual(tuple(mask.shape), (1, 100, 100))
        self.assertEqual(tuple(crop_preview.shape), (1, 100, 100, 3))
        self.assertEqual(vae.encoded_shapes, [(1, 96, 80, 3)])
        self.assertIn("process_crop:(0,4,80,100)", info)
        self.assertIn("crop: 80x96", info)

    def test_crop_larger_than_non_aligned_image_uses_largest_aligned_window(self):
        cx1, cy1, cx2, cy2 = MODULE._fit_vae_crop_bounds(
            10,
            900,
            500,
            1284,
            image_w=1284,
            image_h=1284,
            crop_expand_factor=4.0,
            spatial_compression=8,
        )

        self.assertEqual((cx2 - cx1) % 8, 0)
        self.assertEqual((cy2 - cy1) % 8, 0)
        self.assertEqual((cx2 - cx1, cy2 - cy1), (1280, 1280))
        self.assertEqual(cy2, 1284)


if __name__ == "__main__":
    unittest.main(verbosity=2)
