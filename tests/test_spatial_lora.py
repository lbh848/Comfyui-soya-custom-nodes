"""CPU math and real Anima/Cosmos block tests; no production data writes."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))
import comfy.cli_args
comfy.cli_args.args.cpu = True
import comfy.model_patcher
import comfy.ops
from comfy.ldm.cosmos.predict2 import MiniTrainDIT
from comfy.weight_adapter.lora import LoRAAdapter

PACKAGE = "_spatial_test_package"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
spec = importlib.util.spec_from_file_location(f"{PACKAGE}.soya_spatial_lora", ROOT / "soya_spatial_lora.py")
spatial = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = spatial
spec.loader.exec_module(spatial)


def adapter(strength=1.0):
    result = LoRAAdapter(set(), (torch.eye(4), torch.eye(4), 4.0, None, None, None))
    result.multiplier = strength
    return result


class SpatialMathTests(unittest.TestCase):
    def test_two_and_three_regions_overlap_budget_and_uncovered_background(self):
        for count in (2, 3):
            masks = [torch.zeros(1, 4, 8) for _ in range(count)]
            masks[0][:, :, :4] = 1
            masks[1][:, :, 2:6] = 1
            if count == 3:
                masks[2][:, 2:, 2:4] = 1
            weights = spatial._token_weights(masks, (1, 4, 8), 2, 1)
            torch.testing.assert_close(weights.sum(0), torch.ones(8))
            self.assertEqual(weights[0, -1], 1)
            self.assertLessEqual(weights[1:, 1].sum(), 1)
            self.assertEqual(weights[1, 0], 1)
            self.assertEqual(weights[2, 0], 0)

    def test_odd_sizes_temporal_and_resolution_switch(self):
        for shape in ((1, 5, 7), (2, 4, 6), (1, 108, 157), (1, 54, 78)):
            weights = spatial._token_weights([torch.ones(1, 5, 7)], shape, 2, 1)
            expected = shape[0] * ((shape[1] + 1) // 2) * ((shape[2] + 1) // 2)
            self.assertEqual(weights.shape, (2, expected))
            torch.testing.assert_close(weights[1], torch.ones(expected))
            self.assertFalse(weights[0].any())

    def test_image_residual_is_zero_outside_region_in_flat_and_grid_shapes(self):
        weights = torch.tensor([[0., 0., 1., 1.], [1., 0., 0., 0.], [0., 1., 0., 0.]])
        for shape in ((4, 4, 4), (4, 1, 2, 2, 4)):
            x = torch.ones(shape)
            base = torch.full(shape, 7.)  # Includes the common style model output.
            out = spatial._route_residual(x, base, adapter(.8), "image", 1, weights, [2, 2, 2])
            delta = (out - base).reshape(4, 4, 4)
            torch.testing.assert_close(delta[:, 0], torch.full((4, 4), .8))
            self.assertFalse(delta[:, 1:].any())
            torch.testing.assert_close(base, torch.full(shape, 7.))

    def test_text_kv_routes_to_own_slot_in_all_cfg_rows(self):
        x = torch.ones(4, 9, 4)
        base = torch.full_like(x, 5)
        out = spatial._route_residual(x, base, adapter(.9), "text", 2, None, [2, 3, 4])
        torch.testing.assert_close(out[:, :5], base[:, :5])
        torch.testing.assert_close(out[:, 5:], base[:, 5:] + .9)
        torch.testing.assert_close(base, torch.full_like(x, 5))

    def test_negative_context_repeated_and_long_prompts_not_truncated(self):
        positive = [torch.ones(1, n, 4) * i for i, n in enumerate((3, 7, 2))]
        raw = torch.stack([torch.full((5, 4), float(i)) for i in (10, 11, 20, 21)])
        context, lengths = spatial._unified_context(raw, positive, [1, 0])
        self.assertEqual(lengths, [5, 7, 5])
        for start in (0, 5, 12):
            torch.testing.assert_close(context[:2, start:start + 5], raw[:2])
        torch.testing.assert_close(context[2:, 5:12], positive[1].expand(2, -1, -1))

    def test_attention_blocks_foreign_prompt_but_allows_background(self):
        weights = torch.eye(3)
        bias = spatial._build_flux_cross_attention_bias(weights.bool(), [2, 2, 2], "uncovered_only", torch.device("cpu"), torch.float32)
        self.assertTrue(torch.isneginf(bias[0, 0, 1, :2]).all())
        self.assertTrue(torch.isneginf(bias[0, 0, 1, 4:]).all())
        self.assertTrue((bias[0, 0, 1, 2:4] == 0).all())
        self.assertTrue((bias[0, 0, 0, :2] == 0).all())


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.diffusion_model = MiniTrainDIT(
            16, 16, 1, 4, 4, 2, 1, concat_padding_mask=False,
            model_channels=64, num_blocks=1, num_heads=4,
            crossattn_emb_channels=16, pos_emb_cls="rope3d", image_model="anima",
            rope_enable_fps_modulation=False, operations=comfy.ops.disable_weight_init,
            dtype=torch.float32, device=torch.device("cpu"),
        )
        for parameter in self.parameters():
            torch.nn.init.normal_(parameter, std=.05)

    def get_dtype_inference(self):
        return torch.float32


class SpatialAnimaBlockTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(175753)
        self.model = ToyModel()
        self.patcher = comfy.model_patcher.ModelPatcher(self.model, torch.device("cpu"), torch.device("cpu"))
        self.masks = [torch.zeros(1, 4, 6), torch.zeros(1, 4, 6)]
        self.masks[0][:, :, :2] = 1
        self.masks[1][:, :, 2:4] = 1
        self.conditions = [[[torch.randn(1, 4, 16), {}]] for _ in range(2)]
        self.background = [[torch.randn(1, 4, 16), {}]]

    def _build(self):
        key_map, loras = {}, []
        for i, (path, module) in enumerate(self.model.diffusion_model.named_modules()):
            if module in spatial._layer_roles(self.model.diffusion_model):
                key_map[f"layer{i}"] = f"diffusion_model.{path}.weight"
        for char in range(2):
            data = {}
            for name, path in key_map.items():
                module = self.patcher.get_model_object(path[:-7])
                data[f"{name}.lora_up.weight"] = torch.randn(module.out_features, 2) * .05
                data[f"{name}.lora_down.weight"] = torch.randn(2, module.in_features) * .05
            loras.append((char, data, (.8, .9)[char]))
        with mock.patch.object(comfy.lora, "model_lora_keys_unet", return_value=key_map):
            return spatial.apply_spatial_loras(self.patcher, loras, self.masks, self.conditions, self.background)

    def test_real_block_single_shared_forward_cfg_and_no_weight_or_hook_leak(self):
        patched = self._build()
        runtime = patched.model_options["transformer_options"][spatial._KEY]
        dit = self.model.diffusion_model
        original = {k: v.clone() for k, v in dit.state_dict().items()}
        original_attention = dit.blocks[0].cross_attn.attn_op
        for flags, batch in (([0, 1], 2), ([1, 0], 4), ([0], 1), ([1], 1)):
            opts = {spatial._KEY: runtime, "cond_or_uncond": flags}
            result = dit(torch.randn(batch, 4, 1, 4, 6), torch.ones(batch), torch.randn(batch, 4, 16),
                         transformer_options={**opts, "wrappers": {"diffusion_model": {"test": [spatial._diffusion_wrapper]}}})
            self.assertEqual(result.shape, (batch, 4, 1, 4, 6))
            self.assertTrue(torch.isfinite(result).all())
            self.assertIs(dit.blocks[0].cross_attn.attn_op, original_attention)
            for module in dit.modules():
                self.assertFalse(module._forward_hooks)
        self.assertEqual(runtime.forward_count, 4)
        for key, value in dit.state_dict().items():
            torch.testing.assert_close(value, original[key], rtol=0, atol=0)
        self.assertNotIn(spatial._KEY, self.patcher.model_options["transformer_options"])
        self.assertFalse(patched.hook_patches)

    def test_failure_restores_attention_hooks_and_sample_releases_caches(self):
        patched = self._build()
        runtime = patched.model_options["transformer_options"][spatial._KEY]
        dit = self.model.diffusion_model
        original_attention = dit.blocks[0].cross_attn.attn_op
        class Broken:
            class_obj = dit
            def __call__(self, *args, **kwargs):
                raise RuntimeError("test-only forward failure")
        with self.assertRaisesRegex(RuntimeError, "test-only"):
            spatial._diffusion_wrapper(Broken(), torch.randn(1, 4, 1, 4, 6), torch.ones(1), torch.randn(1, 4, 16),
                                       transformer_options={spatial._KEY: runtime, "cond_or_uncond": [0]})
        self.assertIs(dit.blocks[0].cross_attn.attn_op, original_attention)
        self.assertFalse(any(m._forward_hooks for m in dit.modules()))
        class SampleBroken:
            class_obj = types.SimpleNamespace(model_patcher=patched)
            def __call__(self):
                runtime.prepared["test"] = torch.ones(1)
                raise RuntimeError("test-only sample failure")
        with self.assertRaisesRegex(RuntimeError, "test-only"):
            spatial._sample_wrapper(SampleBroken())
        self.assertFalse(runtime.prepared)
        self.assertFalse(runtime.adapters)
        self.assertFalse(runtime.layouts)

    def test_one_owned_canvas_matches_native_lora_math(self):
        # Opposite case: no cross-region interaction. Routing must not weaken
        # or drop any image/text LoRA layer versus its native merged reference.
        self.masks[0].fill_(1)
        self.masks[1].zero_()
        patched = self._build()
        runtime = patched.model_options["transformer_options"][spatial._KEY]
        dit = self.model.diffusion_model
        x, t = torch.randn(1, 4, 1, 4, 6), torch.ones(1)
        negative = torch.randn(1, 4, 16)
        options = {spatial._KEY: runtime, "cond_or_uncond": [0],
                   "wrappers": {"diffusion_model": {"test": [spatial._diffusion_wrapper]}}}
        routed = dit(x, t, negative, transformer_options=options)
        originals = {k: v.clone() for k, v in dit.state_dict().items()}
        try:
            with torch.no_grad():
                for path, entries in runtime.patches.items():
                    weight = dit.get_submodule(path).weight
                    for slot, _role, strength, source in entries:
                        if slot == 1:
                            up, down, alpha, *_ = source.weights
                            scale = 1. if alpha is None else alpha / down.shape[0]
                            weight.add_(strength * scale * (up @ down))
            native = dit(x, t, self.conditions[0][0][0], transformer_options={})
            torch.testing.assert_close(routed, native, rtol=3e-5, atol=3e-6)
        finally:
            dit.load_state_dict(originals)

    def test_unsupported_adapter_does_not_fall_back_to_global_weights(self):
        key = "diffusion_model.t_embedder.1.linear_1.weight"
        module = self.patcher.get_model_object(key[:-7])
        lora = {"unsupported.lora_up.weight": torch.ones(module.out_features, 1),
                "unsupported.lora_down.weight": torch.ones(1, module.in_features)}
        original = module.weight.clone()
        with mock.patch.object(comfy.lora, "model_lora_keys_unet", return_value={"unsupported": key}):
            with self.assertRaisesRegex(RuntimeError, "no spatial/text routing"):
                spatial.apply_spatial_loras(self.patcher, [(0, lora, .8)], self.masks, self.conditions, self.background)
        torch.testing.assert_close(module.weight, original, rtol=0, atol=0)
        self.assertFalse(self.patcher.hook_patches)


if __name__ == "__main__":
    unittest.main()
