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
    return module


MODULE = _load_module()
FirstSampler = MODULE.SoyaFirstSampler_mdsoya


class FirstSamplerTests(unittest.TestCase):
    def _dispatch(self, node, sampler_mode):
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
            "latent",
            1.0,
            "single",
            0.5,
            0.7,
            0.1,
        )

    def test_sampler_mode_is_optional_and_defaults_to_fast(self):
        input_types = FirstSampler.INPUT_TYPES()

        self.assertNotIn("sampler_mode", input_types["required"])
        self.assertEqual(
            input_types["optional"]["sampler_mode"][0],
            ["KSampler", "FAST"],
        )
        self.assertEqual(
            input_types["optional"]["sampler_mode"][1]["default"],
            "FAST",
        )

    def test_ksampler_mode_dispatches_to_stock_sampler(self):
        node = FirstSampler()

        with (
            mock.patch.object(
                node,
                "_sample_stock_padded",
                return_value=("stock",),
            ) as stock,
            mock.patch.object(node, "_sample_spectrum") as fast,
        ):
            result = self._dispatch(node, "KSampler")

        self.assertEqual(result, ("stock",))
        stock.assert_called_once()
        fast.assert_not_called()

    def test_fast_mode_dispatches_to_spectrum_sampler(self):
        node = FirstSampler()

        with (
            mock.patch.object(node, "_sample_stock_padded") as stock,
            mock.patch.object(
                node,
                "_sample_spectrum",
                return_value=("fast",),
            ) as fast,
        ):
            result = self._dispatch(node, "FAST")

        self.assertEqual(result, ("fast",))
        stock.assert_not_called()
        fast.assert_called_once()

    def test_unknown_sampler_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "지원하지 않는 sampler_mode"):
            self._dispatch(FirstSampler(), "unknown")


if __name__ == "__main__":
    unittest.main()
