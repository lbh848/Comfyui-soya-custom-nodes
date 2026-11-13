"""CPU regression checks for character swaps that retain fixed model weights."""
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import comfy.cli_args
comfy.cli_args.args.cpu = True
import comfy.model_patcher
import comfy.sd

package = ModuleType("soya_character_test")
package.__path__ = [str(Path(__file__).resolve().parents[1])]
sys.modules[package.__name__] = package
character = importlib.import_module(package.__name__ + ".soya_character_lora")


class CharacterCacheTests(unittest.TestCase):
    def setUp(self):
        model = torch.nn.Module()
        model.diffusion_model = torch.nn.Module()
        model.diffusion_model.proj = torch.nn.Linear(2, 3, bias=False)
        model.model_config = SimpleNamespace(unet_config={})
        model.get_dtype_inference = lambda: torch.float32
        self.base = comfy.model_patcher.ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
        self.layer = model.diffusion_model.proj
        self.lora = {"diffusion_model.proj.lora_up.weight": torch.ones(3, 1),
                     "diffusion_model.proj.lora_down.weight": torch.ones(1, 2)}
        self.x = torch.tensor([[[1., 2.], [3., -1.]]])

    def run_model(self, model, fail=False):
        layer, x = self.layer, self.x
        class Executor:
            class_obj = SimpleNamespace(model_patcher=model, conds={})
            def __call__(self):
                model.inject_model()
                result = layer(x)
                if fail:
                    raise RuntimeError("sampling failure")
                return result
        return character._sample_with_character_loras(Executor())

    def test_character_swap_and_combination_keep_style_uuid_and_base(self):
        fixed, _ = comfy.sd.load_lora_for_models(self.base, None, self.lora, .25, 0.)
        first = character.load_character_lora_unmerged(fixed, self.lora, .6)
        second = character.load_character_lora_unmerged(fixed, self.lora, -.3)
        combined = character.load_character_lora_unmerged(second, self.lora, .8)
        before = self.layer.weight.detach().clone()
        for model in (first, second, combined):
            self.assertIs(model.model, fixed.model)
            self.assertEqual(model.patches_uuid, fixed.patches_uuid)
            self.assertEqual(model.patches, fixed.patches)
        self.assertNotIn(character._INJECTION_KEY, fixed.model_options["transformer_options"])
        baseline = self.layer(self.x)
        for model, strength in ((first, .6), (second, -.3), (combined, .5)):
            expected = baseline + self.x.sum(-1, keepdim=True).expand_as(baseline) * strength
            torch.testing.assert_close(self.run_model(model), expected)
            self.assertFalse(self.layer._forward_hooks)
            self.assertNotIn(character._INJECTION_KEY, model.injections)
        torch.testing.assert_close(self.layer.weight, before)

    def test_failure_releases_hooks_and_device_adapter_references(self):
        model = character.load_character_lora_unmerged(self.base, self.lora, .6)
        runtimes = []
        runtime_type = character._CharacterLoRARuntime
        def capture(hooks):
            runtime = runtime_type(hooks)
            runtimes.append(runtime)
            return runtime
        with patch.object(character, "_CharacterLoRARuntime", side_effect=capture):
            with self.assertRaisesRegex(RuntimeError, "sampling failure"):
                self.run_model(model, fail=True)
        self.assertEqual(runtimes[0].states, {})
        self.assertFalse(self.layer._forward_hooks)
        self.assertNotIn(character._INJECTION_KEY, model.injections)
        replacement = character.load_character_lora_unmerged(self.base, self.lora, -.2)
        expected = self.layer(self.x) - self.x.sum(-1, keepdim=True).expand(1, 2, 3) * .2
        torch.testing.assert_close(self.run_model(replacement), expected)


if __name__ == "__main__":
    unittest.main()
