import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch


MODULE_PATH = Path(__file__).parents[1] / "soya_hiresfix_spectrum_toggle.py"
PACKAGE_NAME = "_soya_hiresfix_spectrum_test_package"
MODULE_NAME = f"{PACKAGE_NAME}.soya_hiresfix_spectrum_toggle"


def _load_module():
    package = types.ModuleType(PACKAGE_NAME)
    package.__path__ = [str(MODULE_PATH.parent)]

    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    comfy.samplers = types.ModuleType("comfy.samplers")

    class KSampler:
        SAMPLERS = ("euler",)
        SCHEDULERS = ("simple",)

    comfy.samplers.KSampler = KSampler

    modules = {
        PACKAGE_NAME: package,
        "comfy": comfy,
        "comfy.samplers": comfy.samplers,
    }
    with mock.patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[MODULE_NAME] = module
        spec.loader.exec_module(module)
    return module


MODULE = _load_module()
HiresfixSpectrum = MODULE.SoyaHiresfixSpectrumToggle_mdsoya


class HiresfixSpectrumTests(unittest.TestCase):
    def _inputs(self, **overrides):
        values = {
            "enable": "true",
            "image": torch.zeros((1, 2, 2, 4)),
            "model": mock.sentinel.model,
            "positive": mock.sentinel.positive,
            "negative": mock.sentinel.negative,
            "vae": mock.Mock(),
            "clip": mock.sentinel.clip,
            "seed": 99,
            "steps": 12,
            "cfg": 3.5,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 0.25,
            "tiled_vae": "false",
            "tile_size": 512,
            "spectrum_options": mock.sentinel.spectrum_options,
        }
        values.update(overrides)
        return values

    def test_exposes_shared_spectrum_options_without_legacy_widgets(self):
        input_types = HiresfixSpectrum.INPUT_TYPES()

        self.assertEqual(
            input_types["optional"]["spectrum_options"][0],
            "SOYA_SPECTRUM_MOD_GUIDANCE_OPTIONS",
        )
        self.assertNotIn("quality_tags", input_types["required"])
        self.assertNotIn("mod_w_profile", input_types["required"])
        self.assertNotIn("dcw", input_types["required"])

    def test_disabled_bypasses_vae_and_sampler(self):
        values = self._inputs(enable="false")

        with mock.patch.object(MODULE, "sample_spectrum_mod_guidance") as sampler:
            result = HiresfixSpectrum().doit(**values)

        self.assertIs(result[0], values["image"])
        values["vae"].encode.assert_not_called()
        values["vae"].decode.assert_not_called()
        sampler.assert_not_called()

    def test_enabled_uses_shared_sampler_and_decodes_result(self):
        values = self._inputs()
        latent = mock.sentinel.latent
        sampled = mock.sentinel.sampled
        decoded = torch.tensor([[[[-1.0, 0.5, 2.0]]]])
        values["vae"].encode.return_value = latent
        values["vae"].decode.return_value = decoded

        with mock.patch.object(
            MODULE,
            "sample_spectrum_mod_guidance",
            return_value=({"samples": sampled},),
        ) as sampler:
            result = HiresfixSpectrum().doit(**values)

        encoded_image = values["vae"].encode.call_args.args[0]
        self.assertTrue(torch.equal(encoded_image, values["image"][:, :, :, :3]))
        sampler.assert_called_once_with(
            values["model"],
            values["clip"],
            99,
            12,
            3.5,
            "euler",
            "simple",
            values["positive"],
            values["negative"],
            {"samples": latent},
            0.25,
            values["spectrum_options"],
            log_prefix="[SoyaHiresfixSpectrumToggle]",
        )
        values["vae"].decode.assert_called_once_with(sampled)
        self.assertTrue(torch.equal(result[0], torch.tensor([[[[0.0, 0.5, 1.0]]]])))

    def test_failure_logs_traceback_and_reraises(self):
        values = self._inputs()
        values["vae"].encode.return_value = mock.sentinel.latent

        with (
            mock.patch.object(
                MODULE,
                "sample_spectrum_mod_guidance",
                side_effect=RuntimeError("sample failed"),
            ),
            mock.patch.object(MODULE.traceback, "print_exc") as print_exc,
            mock.patch("builtins.print") as print_log,
            self.assertRaisesRegex(RuntimeError, "sample failed"),
        ):
            HiresfixSpectrum().doit(**values)

        print_exc.assert_called_once_with()
        self.assertTrue(
            any("hires.fix 실패" in str(call) for call in print_log.call_args_list)
        )


if __name__ == "__main__":
    unittest.main()
