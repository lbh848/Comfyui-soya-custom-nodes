"""CPU regression checks for character swaps that retain fixed model weights."""
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import numpy as np
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

    def test_sdxl_convolution_swaps_match_merged_weights(self):
        torch.manual_seed(42)
        for kernel in (1, 3):
            for flattened in (False, True):
                with self.subTest(kernel=kernel, flattened=flattened):
                    self.layer = torch.nn.Conv2d(2, 3, kernel, stride=2, padding=kernel // 2,
                                                 dilation=2, bias=True)
                    self.base.model.diffusion_model.proj = self.layer
                    self.x = torch.randn(1, 2, 11, 13)
                    up, down = torch.randn(3, 2, 1, 1), torch.randn(2, 2, kernel, kernel)
                    lora = {"diffusion_model.proj.lora_up.weight": up.flatten(1) if flattened else up,
                            "diffusion_model.proj.lora_down.weight": down.flatten(1) if flattened else down,
                            "diffusion_model.proj.alpha": torch.tensor(1.)}
                    before = self.layer.weight.detach().clone()
                    for strength in (.6, -.3, .6):
                        model = character.load_character_lora_unmerged(self.base, lora, strength)
                        self.assertEqual(model.patches_uuid, self.base.patches_uuid)
                        self.assertEqual(model.patches, self.base.patches)
                        delta = (up.flatten(1) @ down.flatten(1)).reshape_as(before) * (strength / 2)
                        expected = torch.nn.functional.conv2d(
                            self.x, before + delta, self.layer.bias, stride=self.layer.stride,
                            padding=self.layer.padding, dilation=self.layer.dilation,
                        )
                        torch.testing.assert_close(self.run_model(model), expected)
                        self.assertFalse(self.layer._forward_hooks)
                        self.assertNotIn(character._INJECTION_KEY, model.injections)
                    torch.testing.assert_close(self.layer.weight, before)

    def test_ilxl_face_and_eye_paths_retain_style_across_character_swaps(self):
        fixed, _ = comfy.sd.load_lora_for_models(self.base, None, self.lora, .25, 0.)
        baseline = self.layer(self.x)
        image = torch.zeros(1, 32, 32, 3)
        vae = SimpleNamespace(encode=lambda x: x, decode=lambda x: x, spacial_compression_encode=lambda: 8)
        for kind in ("face", "eye"):
            for suffix in ("", "_v2_core"):
                with self.subTest(kind=kind, suffix=suffix):
                    module = importlib.import_module(package.__name__ + f".soya_char_lora_{kind}_detailer{suffix}")
                    node = getattr(module, f"SoyaCharLora{kind.title()}Detailer_mdsoya")()
                    for name, strength in (("Alice", .6), ("Bob", -.3), ("Alice", .6)):
                        kwargs = dict(enable="true", image=image, model=fixed, clip=object(), vae=vae,
                                      face_context={"matches": [{"name": name, "crop": [8, 8, 24, 24],
                                                                 "bbox": [8, 8, 24, 24], "score": 1.}]},
                                      char_tags=json.dumps({"list": [{"CHAR": name, "FACE_TAGS": "face", "EYE_TAGS": "eyes"}]}),
                                      quality_tags="", artist_tags="", negative="", base_model="sdxl",
                                      lora_list=json.dumps({"list": [{"BASE": "sdxl", "CHAR": name,
                                                                     "lora_path": name, "str": strength}]}),
                                      crop_expand_factor=1., upscale_factor=1., seed=42, steps=2, cfg=1.,
                                      sampler_name="euler", scheduler="simple", denoise=.2, feather=0., noise_mask=True)
                        if kind == "face":
                            kwargs["corner_roundness"] = 0.
                        else:
                            kwargs.update(eye_context={"faces": [{"eye_mask": np.ones((16, 16), dtype=np.uint8),
                                                                  "bbox_clamped": [8, 8, 24, 24]}]},
                                          eyebrow_restore=False, eyebrow_opacity=0., eyebrow_th=.5)
                        sampled = []
                        def sample(model, *args):
                            self.assertIs(model.model, fixed.model)
                            self.assertEqual(model.patches_uuid, fixed.patches_uuid)
                            self.assertEqual(model.patches, fixed.patches)
                            sampled.append(self.run_model(model))
                            return args[7]["samples"]
                        with patch.object(module, "_resolve_lora_path", return_value=name), \
                             patch.object(character, "load_character_lora_file", return_value=self.lora), \
                             patch.object(module, "_encode_conditioning", return_value=[]), \
                             patch.object(module, "_run_ksampler", side_effect=sample):
                            node.execute(**kwargs)
                        self.assertEqual(len(sampled), 1)
                        expected = baseline + self.x.sum(-1, keepdim=True).expand_as(baseline) * strength
                        torch.testing.assert_close(sampled[0], expected)
                        self.assertFalse(self.layer._forward_hooks)
        self.assertNotIn(character._INJECTION_KEY, fixed.model_options["transformer_options"])


if __name__ == "__main__":
    unittest.main()
