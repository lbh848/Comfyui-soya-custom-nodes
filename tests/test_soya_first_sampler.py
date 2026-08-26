import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).parents[1] / "soya_first_sampler.py"
PACKAGE_NAME = "_soya_first_sampler_test_package"
MODULE_NAME = f"{PACKAGE_NAME}.soya_first_sampler"


def _load_module():
    package = types.ModuleType(PACKAGE_NAME)
    package.__path__ = [str(MODULE_PATH.parent)]

    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    comfy.hooks = types.ModuleType("comfy.hooks")
    comfy.samplers = types.ModuleType("comfy.samplers")
    comfy.sd = types.ModuleType("comfy.sd")
    comfy.utils = types.ModuleType("comfy.utils")

    class KSampler:
        SAMPLERS = ("euler",)
        SCHEDULERS = ("simple",)

    comfy.samplers.KSampler = KSampler

    folder_paths = types.ModuleType("folder_paths")
    regional = types.ModuleType(f"{PACKAGE_NAME}.anima_regional_conditioning")
    regional.AnimaConditioningRegionChain = object
    regional.ApplyAnimaRegionalConditioningPatch = object

    modules = {
        PACKAGE_NAME: package,
        "comfy": comfy,
        "comfy.hooks": comfy.hooks,
        "comfy.samplers": comfy.samplers,
        "comfy.sd": comfy.sd,
        "comfy.utils": comfy.utils,
        "folder_paths": folder_paths,
        f"{PACKAGE_NAME}.anima_regional_conditioning": regional,
    }
    with mock.patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[MODULE_NAME] = module
        spec.loader.exec_module(module)
        spectrum_core = sys.modules[f"{PACKAGE_NAME}.soya_spectrum_mod_guidance"]
    return module, spectrum_core


MODULE, SPECTRUM_CORE = _load_module()
FirstSampler = MODULE.SoyaFirstSampler_mdsoya
SpectrumOptions = MODULE.SoyaSpectrumModGuidanceOptions_mdsoya


class FirstSamplerTests(unittest.TestCase):
    def _dispatch(self, node, sampler_mode, spectrum_options=None):
        return node._sample_selected(
            sampler_mode,
            "model",
            1,
            28,
            4.0,
            "euler",
            "simple",
            "positive",
            "negative",
            "clip",
            "latent",
            1.0,
            "single",
            0.5,
            0.7,
            0.1,
            spectrum_options,
        )

    def test_sampler_mode_is_optional_and_defaults_to_fast(self):
        input_types = FirstSampler.INPUT_TYPES()

        self.assertNotIn("sampler_mode", input_types["required"])
        self.assertEqual(
            input_types["optional"]["sampler_mode"][0],
            ["KSampler", "FAST", "SpectrumKSamplerModGuidance"],
        )
        self.assertEqual(
            input_types["optional"]["sampler_mode"][1]["default"],
            "FAST",
        )
        self.assertEqual(
            input_types["optional"]["spectrum_options"][0],
            "SOYA_SPECTRUM_MOD_GUIDANCE_OPTIONS",
        )

    def test_lora_path_normalizes_windows_separators_for_linux(self):
        raw = r"SOYA_CHAR_LORA\SOYA_BOT_LORA\character\model.safetensors"
        normalized = "SOYA_CHAR_LORA/SOYA_BOT_LORA/character/model.safetensors"
        resolved = f"/loras/{normalized}"

        with (
            mock.patch.object(MODULE.os.path, "isfile", side_effect=lambda path: path == resolved),
            mock.patch.object(MODULE.os.path, "realpath", side_effect=lambda path: path),
            mock.patch.object(
                MODULE.folder_paths,
                "get_full_path",
                return_value=resolved,
                create=True,
            ) as get_full_path,
        ):
            result = MODULE._resolve_lora_path(raw)

        self.assertEqual(result, resolved)
        get_full_path.assert_called_once_with("loras", normalized)

    def test_spectrum_options_node_exposes_current_defaults(self):
        input_types = SpectrumOptions.INPUT_TYPES()["required"]

        self.assertEqual(
            input_types["positive"][1]["default"],
            "masterpiece, best quality, highres, absurdres, very aesthetic",
        )
        self.assertEqual(
            input_types["negative"][1]["default"],
            (
                "score_1, score_2, score_3, worst quality, lowres, old, "
                "bad hands, bad anatomy"
            ),
        )
        self.assertEqual(
            input_types["mod_w_profile"][0],
            ["off", "step_i8_skip27", "step_i14", "uniform_w3"],
        )
        self.assertEqual(input_types["refresh_ratio"][1]["default"], -1.0)
        self.assertEqual(input_types["adaptive_smc_alpha"][1]["default"], 0.1)
        self.assertEqual(
            SpectrumOptions.RETURN_TYPES,
            ("SOYA_SPECTRUM_MOD_GUIDANCE_OPTIONS",),
        )

    def test_spectrum_options_node_builds_typed_bundle(self):
        result = SpectrumOptions().build(
            "quality positive",
            "quality negative",
            "step_i14",
            0.25,
            0.3,
        )

        self.assertEqual(result, ({
            "positive": "quality positive",
            "negative": "quality negative",
            "mod_w_profile": "step_i14",
            "refresh_ratio": 0.25,
            "adaptive_smc_alpha": 0.3,
        },))

    def test_ksampler_mode_dispatches_to_stock_sampler(self):
        node = FirstSampler()

        with (
            mock.patch.object(
                node,
                "_sample_stock_padded",
                return_value=("stock",),
            ) as stock,
            mock.patch.object(node, "_sample_spectrum") as fast,
            mock.patch.object(node, "_sample_spectrum_mod_guidance") as mod_guidance,
        ):
            result = self._dispatch(node, "KSampler")

        self.assertEqual(result, ("stock",))
        stock.assert_called_once()
        fast.assert_not_called()
        mod_guidance.assert_not_called()

    def test_fast_mode_dispatches_to_spectrum_sampler(self):
        node = FirstSampler()

        with (
            mock.patch.object(node, "_sample_stock_padded") as stock,
            mock.patch.object(
                node,
                "_sample_spectrum",
                return_value=("fast",),
            ) as fast,
            mock.patch.object(node, "_sample_spectrum_mod_guidance") as mod_guidance,
        ):
            result = self._dispatch(node, "FAST")

        self.assertEqual(result, ("fast",))
        stock.assert_not_called()
        fast.assert_called_once()
        mod_guidance.assert_not_called()

    def test_mod_guidance_mode_dispatches_to_embedded_implementation(self):
        node = FirstSampler()

        with (
            mock.patch.object(node, "_sample_stock_padded") as stock,
            mock.patch.object(node, "_sample_spectrum") as fast,
            mock.patch.object(
                node,
                "_sample_spectrum_mod_guidance",
                return_value=("mod-guidance",),
            ) as mod_guidance,
        ):
            result = self._dispatch(node, "SpectrumKSamplerModGuidance")

        self.assertEqual(result, ("mod-guidance",))
        stock.assert_not_called()
        fast.assert_not_called()
        mod_guidance.assert_called_once_with(
            "model",
            "clip",
            1,
            28,
            4.0,
            "euler",
            "simple",
            "positive",
            "negative",
            "latent",
            1.0,
            None,
        )

    def test_embedded_mod_guidance_uses_fixed_personal_profile(self):
        node = FirstSampler()
        model = mock.Mock()
        mod_model = mock.sentinel.mod_model
        model.clone.return_value = mod_model
        setup_mod_guidance = mock.Mock()
        spectrum_sample = mock.Mock(return_value=("sampled",))

        with mock.patch.object(
            SPECTRUM_CORE,
            "_spectrum_mod_guidance_runtime",
            return_value=(setup_mod_guidance, spectrum_sample),
        ):
            result = node._sample_spectrum_mod_guidance(
                model,
                "clip",
                1,
                28,
                4.0,
                "euler",
                "simple",
                "positive",
                "negative",
                "latent",
                1.0,
            )

        self.assertEqual(result, ("sampled",))
        setup_mod_guidance.assert_called_once_with(
            mod_model,
            "clip",
            "positive",
            "negative",
            None,
            "masterpiece, best quality, highres, absurdres, very aesthetic",
            3.0,
            quality_neg=(
                "score_1, score_2, score_3, worst quality, lowres, old, "
                "bad hands, bad anatomy"
            ),
            start_layer=8,
            end_layer=27,
            taper=0,
            taper_scale=0.25,
            final_w=0.0,
        )
        spectrum_sample.assert_called_once_with(
            mod_model,
            1,
            28,
            4.0,
            "euler",
            "simple",
            "positive",
            "negative",
            "latent",
            1.0,
            window_size=2.0,
            flex_window=0.25,
            warmup_steps=6,
            blend_w=0.3,
            cheby_degree=3,
            ridge_lambda=0.1,
            dcw_mode="off",
            smc_cfg_alpha=0.10,
            smc_cfg_lambda=5.0,
            schedule="window",
            refresh_ratio=-1.0,
        )

    def test_embedded_mod_guidance_uses_connected_options_and_sea_schedule(self):
        node = FirstSampler()
        model = mock.Mock()
        mod_model = mock.sentinel.mod_model
        model.clone.return_value = mod_model
        setup_mod_guidance = mock.Mock()
        spectrum_sample = mock.Mock(return_value=("sampled",))
        options = {
            "positive": "quality positive",
            "negative": "quality negative",
            "mod_w_profile": "step_i14",
            "refresh_ratio": 0.25,
            "adaptive_smc_alpha": 0.3,
        }

        with mock.patch.object(
            SPECTRUM_CORE,
            "_spectrum_mod_guidance_runtime",
            return_value=(setup_mod_guidance, spectrum_sample),
        ):
            result = node._sample_spectrum_mod_guidance(
                model,
                "clip",
                1,
                28,
                4.0,
                "euler",
                "simple",
                "positive",
                "negative",
                "latent",
                1.0,
                options,
            )

        self.assertEqual(result, ("sampled",))
        setup_mod_guidance.assert_called_once_with(
            mod_model,
            "clip",
            "positive",
            "negative",
            None,
            "quality positive",
            3.0,
            quality_neg="quality negative",
            start_layer=14,
            end_layer=-1,
            taper=0,
            taper_scale=0.25,
            final_w=0.0,
        )
        self.assertEqual(spectrum_sample.call_args.kwargs["schedule"], "sea")
        self.assertEqual(spectrum_sample.call_args.kwargs["refresh_ratio"], 0.25)
        self.assertEqual(spectrum_sample.call_args.kwargs["smc_cfg_alpha"], 0.3)

    def test_mod_profile_off_skips_guidance_but_still_samples(self):
        node = FirstSampler()
        model = mock.Mock()
        setup_mod_guidance = mock.Mock()
        spectrum_sample = mock.Mock(return_value=("sampled",))
        options = {
            "positive": "",
            "negative": "",
            "mod_w_profile": "off",
            "refresh_ratio": -0.5,
            "adaptive_smc_alpha": 0.0,
        }

        with mock.patch.object(
            SPECTRUM_CORE,
            "_spectrum_mod_guidance_runtime",
            return_value=(setup_mod_guidance, spectrum_sample),
        ):
            result = node._sample_spectrum_mod_guidance(
                model,
                None,
                1,
                28,
                4.0,
                "euler",
                "simple",
                "positive",
                "negative",
                "latent",
                1.0,
                options,
            )

        self.assertEqual(result, ("sampled",))
        model.clone.assert_not_called()
        setup_mod_guidance.assert_not_called()
        self.assertIs(spectrum_sample.call_args.args[0], model)
        self.assertEqual(spectrum_sample.call_args.kwargs["schedule"], "window")
        self.assertEqual(spectrum_sample.call_args.kwargs["refresh_ratio"], -1.0)

    def test_unknown_sampler_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "지원하지 않는 sampler_mode"):
            self._dispatch(FirstSampler(), "unknown")


if __name__ == "__main__":
    unittest.main()
