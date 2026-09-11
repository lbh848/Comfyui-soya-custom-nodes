"""CPU-only tests using native Comfy hooks and small, synthetic linear layers."""

import contextlib
import importlib.util
import io
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F


COMFY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(COMFY_ROOT))
import comfy.cli_args

# Set CPU mode before importing model management. Never load a checkpoint or
# initialize CUDA in this test; use the Comfy environment for its dependencies.
comfy.cli_args.args.cpu = True
import comfy.hooks
import comfy.lora
import comfy.model_patcher
import comfy.patcher_extension
import comfy.sampler_helpers

SPEC = importlib.util.spec_from_file_location(
    "_character_lora_residual_test", Path(__file__).resolve().parents[1] / "soya_character_lora.py"
)
residual = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(residual)


class _ToyModel(torch.nn.Module):
    def __init__(self, dtype=torch.float64):
        super().__init__()
        self.diffusion_model = torch.nn.Sequential(
            torch.nn.Linear(32, 24, dtype=dtype),
            torch.nn.Tanh(),
            torch.nn.Linear(24, 16, dtype=dtype),
        )

    def get_dtype_inference(self):
        return self.diffusion_model[0].weight.dtype


KEYS = ["diffusion_model.0.weight", "diffusion_model.2.weight"]
KEY_MAP = {f"test_layer_{i}": key for i, key in enumerate(KEYS)}


def _lora(rank=2, *, dtype=torch.float64):
    data = {}
    for i, (out_features, in_features) in enumerate(((24, 32), (16, 24))):
        data[f"test_layer_{i}.lora_up.weight"] = torch.randn(out_features, rank, dtype=dtype) * 0.05
        data[f"test_layer_{i}.lora_down.weight"] = torch.randn(rank, in_features, dtype=dtype) * 0.05
        data[f"test_layer_{i}.alpha"] = torch.tensor(float(rank), dtype=dtype)
    return data


def _load(patcher, data, strength):
    with mock.patch.object(comfy.lora, "model_lora_keys_unet", return_value=KEY_MAP):
        return residual.load_character_lora(patcher, data, strength)


def _reference(model, x, loras):
    for i, layer in enumerate((model.diffusion_model[0], model.diffusion_model[2])):
        weight = layer.weight.clone()
        for data, strength in loras:
            up = data[f"test_layer_{i}.lora_up.weight"].to(weight)
            down = data[f"test_layer_{i}.lora_down.weight"].to(weight)
            alpha = data[f"test_layer_{i}.alpha"].item()
            weight = weight + (strength * alpha / down.shape[0]) * (up @ down)
        x = F.linear(x, weight, layer.bias)
        if i == 0:
            x = torch.tanh(x)
    return x


def _execute(patcher, groups, sample):
    conds = {
        "positive": [{"hooks": group} if group is not None else {} for group in groups],
        "negative": [{"hooks": group} if group is not None else {} for group in groups],
    }
    options = {"transformer_options": {}}
    comfy.sampler_helpers.prepare_model_patcher(patcher, conds, options)
    registered = options["registered_hooks"]
    for group in groups:
        if group is not None:
            for hook in group.hooks:
                assert registered.contains(hook)
    guider = types.SimpleNamespace(model_patcher=patcher, conds=conds)
    executor = comfy.patcher_extension.WrapperExecutor.new_class_executor(
        sample, guider, [residual._sample_with_character_loras]
    )
    return executor.execute()


class CharacterLoRAResidualTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(80324)
        self.model = _ToyModel()
        self.patcher = comfy.model_patcher.ModelPatcher(
            self.model, load_device=torch.device("cpu"), offload_device=torch.device("cpu")
        )

    def _assert_clean(self, patcher):
        self.assertIsNone(patcher.get_injections(residual._INJECTION_KEY))
        for layer in patcher.model.modules():
            self.assertFalse(layer._forward_hooks)

    def test_two_characters_and_background_match_merged_math_without_weight_copies(self):
        data_a, data_b = _lora(), _lora(rank=3)
        patcher, group_a = _load(self.patcher, data_a, 0.8)
        patcher, group_b = _load(patcher, data_b, 0.9)
        originals = {key: value.clone() for key, value in self.model.state_dict().items()}
        pointers = {key: value.data_ptr() for key, value in self.model.state_dict().items()}
        x = torch.randn(2, 7, 32, dtype=torch.float64)
        expected = [_reference(self.model, x, [(data_a, 0.8)]),
                    _reference(self.model, x, [(data_b, 0.9)]),
                    _reference(self.model, x, [])]
        self.assertFalse(patcher.hook_patches)
        self.assertEqual(len(patcher.get_wrappers(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, residual._INJECTION_KEY
        )), 1)

        def sample():
            patcher.inject_model()
            for index in (0, 1, 2, 1, 0, 2):
                group = (group_a, group_b, None)[index]
                patcher.apply_hooks(group)
                actual = self.model.diffusion_model(x)
                torch.testing.assert_close(actual, expected[index], rtol=1e-10, atol=1e-10)
                self.assertFalse(patcher.hook_backup)
                self.assertFalse(patcher.cached_hook_patches)
                for key, value in self.model.state_dict().items():
                    self.assertTrue(torch.equal(value, originals[key]))
                    self.assertEqual(value.data_ptr(), pointers[key])

        with mock.patch.object(patcher, "patch_hook_weight_to_device", side_effect=AssertionError("merged weight")):
            _execute(patcher, [group_a, group_b, None], sample)
        self._assert_clean(patcher)
        torch.testing.assert_close(self.model.diffusion_model(x), expected[2])

    def test_multiple_loras_and_cloned_reordered_groups_keep_strength_and_batch_rows(self):
        data_a, data_extra, data_b = _lora(1), _lora(2), _lora(3)
        patcher, group_a = _load(self.patcher, data_a, 0.65)
        patcher, group_extra = _load(patcher, data_extra, -0.2)
        patcher, group_b = _load(patcher, data_b, 0.4)
        combined = group_a.clone_and_combine(group_extra)
        combined = combined.clone()
        combined.hooks[0].hook_keyframe._current_keyframe = comfy.hooks.HookKeyframe(strength=0.5)
        x = torch.randn(4, 3, 32, dtype=torch.float64)
        expected_a = _reference(self.model, x, [(data_a, 0.65 * 0.5), (data_extra, -0.2)])
        expected_b = _reference(self.model, x, [(data_b, 0.4)])

        def sample():
            patcher.inject_model()
            for group, expected in ((group_b, expected_b), (combined, expected_a), (group_b, expected_b)):
                patcher.apply_hooks(group)
                torch.testing.assert_close(self.model.diffusion_model(x), expected, rtol=1e-10, atol=1e-10)

        _execute(patcher, [group_b, combined, None], sample)
        self._assert_clean(patcher)

    def test_cached_output_layer_calls_still_follow_active_character(self):
        data = _lora()
        patcher, group = _load(self.patcher, data, 0.7)
        layer = self.model.diffusion_model[2]
        x = torch.randn(2, 5, 24, dtype=torch.float64)
        delta = data["test_layer_1.lora_up.weight"] @ data["test_layer_1.lora_down.weight"]
        expected = F.linear(x, layer.weight + 0.7 * delta, layer.bias)
        background = F.linear(x, layer.weight, layer.bias)

        def sample():
            patcher.inject_model()
            patcher.apply_hooks(group)
            torch.testing.assert_close(layer(input=x), expected, rtol=1e-10, atol=1e-10)
            patcher.apply_hooks(None)
            torch.testing.assert_close(layer(x), background)

        _execute(patcher, [group, None], sample)
        self._assert_clean(patcher)

    def test_device_storage_contains_only_lora_factors_and_survives_temporary_ejection(self):
        data = _lora()
        patcher, group = _load(self.patcher, data, 0.8)
        runtime = residual._CharacterLoRARuntime(group.hooks)
        patcher.set_injections(residual._INJECTION_KEY, [comfy.patcher_extension.PatcherInjection(
            inject=runtime.inject, eject=runtime.eject
        )])
        try:
            patcher.inject_model()
            state = runtime.states[patcher]
            adapters = [entry[1] for entries in state["layers"].values() for entry in entries]
            expected_count = sum(value.numel() for value in data.values()) - len(KEYS)
            # Native load_lora converts alpha tensors to scalars.
            self.assertEqual(sum(value.numel() for adapter in adapters for value in adapter.weights
                                 if torch.is_tensor(value)), expected_count)
            self.assertLess(expected_count, sum(layer.weight.numel() for layer in
                                               (self.model.diffusion_model[0], self.model.diffusion_model[2])))
            patcher.apply_hooks(group)
            patcher.apply_hooks(None)
            self.assertIs(runtime.states[patcher], state)
            self.assertIs(next(iter(state["layers"].values()))[0][1], adapters[0])
        finally:
            runtime.close(patcher)
        self.assertFalse(runtime.states)
        self._assert_clean(patcher)

    def test_compute_dtype_cast_preserves_base_storage(self):
        model = _ToyModel(dtype=torch.bfloat16)
        patcher = comfy.model_patcher.ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
        patcher, group = _load(patcher, _lora(dtype=torch.float32), 0.75)
        original = model.diffusion_model[0].weight.clone()

        def sample():
            patcher.inject_model()
            patcher.apply_hooks(group)
            out = model.diffusion_model(torch.randn(2, 32, dtype=torch.bfloat16))
            self.assertEqual(out.dtype, torch.bfloat16)
            self.assertTrue(torch.isfinite(out).all())
            self.assertTrue(torch.equal(model.diffusion_model[0].weight, original))

        _execute(patcher, [group, None], sample)
        self._assert_clean(patcher)

    def test_native_dora_diff_and_non_linear_paths_are_preserved(self):
        for kind in ("dora", "diff", "conv"):
            with self.subTest(kind=kind):
                model = _ToyModel()
                patcher = comfy.model_patcher.ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
                data = _lora()
                if kind == "dora":
                    data["test_layer_0.dora_scale"] = torch.ones(24, dtype=torch.float64)
                elif kind == "diff":
                    data = {"test_layer_0.diff": torch.zeros(24, 32, dtype=torch.float64)}
                else:
                    model.diffusion_model[0] = torch.nn.Conv2d(32, 24, 1)
                with contextlib.redirect_stdout(io.StringIO()) as log:
                    patched, group = _load(patcher, data, 0.8)
                self.assertIn("Native weight path", log.getvalue())
                self.assertTrue(patched.hook_patches)
                self.assertIsInstance(group.hooks[0], comfy.hooks.WeightHook)
                self.assertFalse(patched.get_wrappers(
                    comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, residual._INJECTION_KEY
                ))

    def test_sampler_failure_cleans_hooks_and_preserves_other_injections(self):
        patcher, group = _load(self.patcher, _lora(), 0.8)
        events = []
        patcher.set_injections("other", [comfy.patcher_extension.PatcherInjection(
            inject=lambda _: events.append("inject"), eject=lambda _: events.append("eject")
        )])

        def sample():
            patcher.inject_model()
            patcher.apply_hooks(group)
            raise RuntimeError("synthetic sampler failure")

        with contextlib.redirect_stdout(io.StringIO()) as log, contextlib.redirect_stderr(io.StringIO()) as error:
            with self.assertRaisesRegex(RuntimeError, "synthetic sampler failure"):
                _execute(patcher, [group, None], sample)
        self.assertIn("Sampling failed", log.getvalue())
        self.assertIn("Traceback", error.getvalue())
        self._assert_clean(patcher)
        self.assertIsNotNone(patcher.get_injections("other"))
        self.assertEqual(events[-1], "inject")
        patcher.eject_model()

    def test_partial_injection_failure_cleans_already_registered_layer_hooks(self):
        patcher, group = _load(self.patcher, _lora(), 0.8)

        def sample():
            with mock.patch.object(self.model.diffusion_model[2], "register_forward_hook",
                                   side_effect=RuntimeError("synthetic injection failure")):
                patcher.inject_model()

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "synthetic injection failure"):
                _execute(patcher, [group, None], sample)
        self._assert_clean(patcher)


if __name__ == "__main__":
    unittest.main()
