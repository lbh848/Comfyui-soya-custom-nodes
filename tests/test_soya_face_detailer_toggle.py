import contextlib
import importlib.util
import io
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch


MODULE_PATH = Path(__file__).parents[1] / "soya_face_detailer_toggle.py"
SPEC = importlib.util.spec_from_file_location("soya_face_detailer_toggle", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
FaceDetailer = MODULE.SoyaFaceDetailerToggle_mdsoya


class DummyModel:
    def get_model_object(self, _name):
        return None


class DummyDetector:
    def bbox_model(self, _image, verbose=False):
        return []


class IdentityVAE:
    def encode(self, image):
        return image

    def decode(self, latent):
        return latent


class ExtraSingletonVAE(IdentityVAE):
    def decode(self, latent):
        return latent.unsqueeze(0)


def call_node(node, **overrides):
    arguments = {
        "enable": "true",
        "image": torch.zeros((1, 8, 8, 3), dtype=torch.float32),
        "model": DummyModel(),
        "clip": object(),
        "vae": IdentityVAE(),
        "positive": "face",
        "negative": "bad face",
        "seed": 10,
        "steps": 8,
        "cfg": 1.0,
        "sampler_name": "euler",
        "scheduler": "simple",
        "denoise": 0.3,
        "guide_size": 0.0,
        "bbox_threshold": 0.5,
        "bbox_dilation": 0,
        "crop_factor": 1.0,
        "feather": 0,
        "noise_mask": True,
        "drop_size": 1,
        "bbox_detector": DummyDetector(),
        "cycle": 1,
    }
    arguments.update(overrides)
    return node.doit(**arguments)


def add_point_one(
    _model,
    _seed,
    _steps,
    _cfg,
    _sampler,
    _scheduler,
    _positive,
    _negative,
    latent_dict,
    _denoise,
):
    return latent_dict["samples"] + 0.1


class FaceDetailerTests(unittest.TestCase):
    def test_input_and_output_layout_remain_workflow_compatible(self):
        comfy_module = types.ModuleType("comfy")
        comfy_module.__path__ = []
        samplers_module = types.ModuleType("comfy.samplers")

        class KSampler:
            SAMPLERS = ("euler",)
            SCHEDULERS = ("simple",)

        samplers_module.KSampler = KSampler
        comfy_module.samplers = samplers_module
        with mock.patch.dict(
            sys.modules,
            {"comfy": comfy_module, "comfy.samplers": samplers_module},
        ):
            required = FaceDetailer.INPUT_TYPES()["required"]

        self.assertEqual(
            list(required),
            [
                "enable",
                "image",
                "model",
                "clip",
                "vae",
                "positive",
                "negative",
                "seed",
                "steps",
                "cfg",
                "sampler_name",
                "scheduler",
                "denoise",
                "guide_size",
                "bbox_threshold",
                "bbox_dilation",
                "crop_factor",
                "feather",
                "noise_mask",
                "drop_size",
                "bbox_detector",
                "cycle",
            ],
        )
        self.assertEqual(FaceDetailer.RETURN_TYPES, ("IMAGE", "MASK"))
        self.assertEqual(required["feather"][0], "INT")
        self.assertEqual(required["noise_mask"][0], "BOOLEAN")
        self.assertEqual(required["drop_size"][0], "INT")

    def test_disabled_node_bypasses_with_boolean_false(self):
        node = FaceDetailer()
        image = torch.rand((1, 8, 8, 3), dtype=torch.float32)

        result, mask = call_node(
            node,
            enable=False,
            image=image,
            model=None,
            bbox_detector=None,
        )

        self.assertIs(result, image)
        self.assertEqual(torch.count_nonzero(mask).item(), 0)

    def test_overlapping_faces_use_progressively_updated_result(self):
        node = FaceDetailer()
        with (
            mock.patch.object(
                node,
                "_detect_faces",
                return_value=[(2, 2, 6, 6), (2, 2, 6, 6)],
            ),
            mock.patch.object(node, "_run_ksampler", side_effect=add_point_one),
        ):
            result, mask, processed = node._detail_single(
                torch.zeros((1, 8, 8, 3), dtype=torch.float32),
                DummyModel(),
                IdentityVAE(),
                None,
                None,
                10,
                8,
                1.0,
                "euler",
                "simple",
                0.3,
                0.0,
                0.5,
                0,
                1.0,
                0,
                True,
                1,
                DummyDetector(),
            )

        self.assertAlmostEqual(result[0, 4, 4, 0].item(), 0.2, places=6)
        self.assertAlmostEqual(mask[0, 4, 4].item(), 1.0, places=6)
        self.assertEqual(processed, 2)

    def test_positive_bbox_dilation_expands_face_mask(self):
        node = FaceDetailer()
        with (
            mock.patch.object(node, "_detect_faces", return_value=[(2, 2, 6, 6)]),
            mock.patch.object(node, "_run_ksampler", side_effect=add_point_one),
        ):
            _result, mask, processed = node._detail_single(
                torch.zeros((1, 8, 8, 3), dtype=torch.float32),
                DummyModel(),
                IdentityVAE(),
                None,
                None,
                10,
                8,
                1.0,
                "euler",
                "simple",
                0.3,
                0.0,
                0.5,
                1,
                2.0,
                0,
                True,
                1,
                DummyDetector(),
            )

        self.assertEqual(mask[0, 1, 1].item(), 1.0)
        self.assertEqual(mask[0, 0, 0].item(), 0.0)
        self.assertEqual(processed, 1)

    def test_anima_vae_extra_singleton_dimension_is_normalized(self):
        node = FaceDetailer()
        stdout = io.StringIO()
        with (
            mock.patch.object(node, "_detect_faces", return_value=[(2, 2, 6, 6)]),
            mock.patch.object(node, "_run_ksampler", side_effect=add_point_one),
            contextlib.redirect_stdout(stdout),
        ):
            result, _mask, processed = node._detail_single(
                torch.zeros((1, 8, 8, 3), dtype=torch.float32),
                DummyModel(),
                ExtraSingletonVAE(),
                None,
                None,
                10,
                8,
                1.0,
                "euler",
                "simple",
                0.3,
                0.0,
                0.5,
                0,
                1.0,
                0,
                True,
                1,
                DummyDetector(),
            )

        self.assertAlmostEqual(result[0, 4, 4, 0].item(), 0.1, places=6)
        self.assertEqual(processed, 1)
        self.assertIn("NORMALIZED_VAE_OUTPUT", stdout.getvalue())

    def test_multiple_cycles_accumulate_masks_and_results(self):
        node = FaceDetailer()
        call_count = 0

        def fake_detail_single(image, *_args):
            nonlocal call_count
            mask = torch.zeros((1, 8, 8), dtype=torch.float32)
            mask[0, call_count, call_count] = 1.0
            call_count += 1
            return image + 0.1, mask, 1

        with (
            mock.patch.object(node, "_encode_conditioning", return_value=None),
            mock.patch.object(node, "_detail_single", side_effect=fake_detail_single),
        ):
            result, mask = call_node(node, cycle=2)

        self.assertAlmostEqual(result[0, 0, 0, 0].item(), 0.2, places=6)
        self.assertEqual(mask[0, 0, 0].item(), 1.0)
        self.assertEqual(mask[0, 1, 1].item(), 1.0)
        self.assertEqual(call_count, 2)

    def test_enabled_failure_logs_context_and_full_traceback(self):
        node = FaceDetailer()
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            mock.patch.object(
                node,
                "_encode_conditioning",
                side_effect=RuntimeError("conditioning failed"),
            ),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
            self.assertRaisesRegex(RuntimeError, "conditioning failed"),
        ):
            call_node(node)

        self.assertIn("[FaceDetailer] FAILED", stdout.getvalue())
        self.assertIn("conditioning failed", stdout.getvalue())
        self.assertIn("Traceback (most recent call last)", stderr.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
