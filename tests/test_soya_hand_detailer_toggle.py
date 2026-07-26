import contextlib
import importlib.util
import io
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch


MODULE_PATH = Path(__file__).parents[1] / "soya_hand_detailer_toggle.py"
SPEC = importlib.util.spec_from_file_location("soya_hand_detailer_toggle", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
HandDetailer = MODULE.SoyaHandDetailerToggle_mdsoya


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
        "positive": "hand",
        "negative": "bad hand",
        "seed": 10,
        "steps": 8,
        "cfg": 1.0,
        "sampler_name": "euler",
        "scheduler": "simple",
        "denoise": 0.3,
        "bbox_threshold": 0.4,
        "mask_expand": 1.0,
        "crop_expand": 1.0,
        "upscale_factor": 1.0,
        "feather": 0.0,
        "corner_roundness": 0.0,
        "noise_mask": True,
        "drop_size": 1,
        "bbox_detector": DummyDetector(),
        "cycle": 1,
    }
    arguments.update(overrides)
    return node.doit(**arguments)


class HandDetailerTests(unittest.TestCase):
    def test_input_layout_is_v2_compatible_and_uses_typed_hand_defaults(self):
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
            required = HandDetailer.INPUT_TYPES()["required"]

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
                "bbox_threshold",
                "mask_expand",
                "crop_expand",
                "upscale_factor",
                "feather",
                "corner_roundness",
                "noise_mask",
                "drop_size",
                "bbox_detector",
                "cycle",
            ],
        )
        self.assertEqual(required["feather"][0], "FLOAT")
        self.assertEqual(required["noise_mask"][0], "BOOLEAN")
        self.assertEqual(required["drop_size"][0], "INT")

    def test_disabled_node_bypasses_without_requiring_model_or_detector(self):
        node = HandDetailer()
        image = torch.rand((1, 8, 8, 3), dtype=torch.float32)

        result, mask, preview = call_node(
            node,
            enable="false",
            image=image,
            model=None,
            bbox_detector=None,
        )

        self.assertIs(result, image)
        self.assertEqual(torch.count_nonzero(mask).item(), 0)
        self.assertEqual(torch.count_nonzero(preview).item(), 0)

    def test_overlapping_hands_use_progressively_updated_result(self):
        node = HandDetailer()

        def fake_ksampler(
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

        with (
            mock.patch.object(
                node,
                "_detect_hands",
                return_value=[(2, 2, 6, 6), (2, 2, 6, 6)],
            ),
            mock.patch.object(node, "_run_ksampler", side_effect=fake_ksampler),
        ):
            result, mask, crop_regions, processed = node._detail_single(
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
                0.4,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                True,
                1,
                DummyDetector(),
            )

        self.assertAlmostEqual(result[0, 4, 4, 0].item(), 0.2, places=6)
        self.assertAlmostEqual(mask[0, 4, 4].item(), 1.0, places=6)
        self.assertEqual(processed, 2)
        self.assertEqual(len(crop_regions), 2)

    def test_anima_vae_extra_singleton_dimension_is_normalized(self):
        node = HandDetailer()

        def fake_ksampler(
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

        stdout = io.StringIO()
        with (
            mock.patch.object(
                node,
                "_detect_hands",
                return_value=[(2, 2, 6, 6)],
            ),
            mock.patch.object(node, "_run_ksampler", side_effect=fake_ksampler),
            contextlib.redirect_stdout(stdout),
        ):
            result, _mask, _crop_regions, processed = node._detail_single(
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
                0.4,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                True,
                1,
                DummyDetector(),
            )

        self.assertAlmostEqual(result[0, 4, 4, 0].item(), 0.1, places=6)
        self.assertEqual(processed, 1)
        self.assertIn("NORMALIZED_VAE_OUTPUT", stdout.getvalue())

    def test_multiple_cycles_accumulate_masks_and_results(self):
        node = HandDetailer()
        call_count = 0

        def fake_detail_single(image, *_args):
            nonlocal call_count
            mask = torch.zeros((1, 8, 8), dtype=torch.float32)
            mask[0, call_count, call_count] = 1.0
            call_count += 1
            return image + 0.1, mask, [(0, 0, 8, 8)], 1

        with (
            mock.patch.object(node, "_encode_conditioning", return_value=None),
            mock.patch.object(node, "_detail_single", side_effect=fake_detail_single),
        ):
            result, mask, _preview = call_node(node, cycle=2)

        self.assertAlmostEqual(result[0, 0, 0, 0].item(), 0.2, places=6)
        self.assertEqual(mask[0, 0, 0].item(), 1.0)
        self.assertEqual(mask[0, 1, 1].item(), 1.0)
        self.assertEqual(call_count, 2)

    def test_enabled_failure_logs_context_and_full_traceback(self):
        node = HandDetailer()
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

        self.assertIn("[HandDetailer] FAILED", stdout.getvalue())
        self.assertIn("conditioning failed", stdout.getvalue())
        self.assertIn("Traceback (most recent call last)", stderr.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
