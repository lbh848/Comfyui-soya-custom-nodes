"""CPU contract checks; run with ComfyUI's Python from the ComfyUI directory."""
import importlib
import copy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import comfy.cli_args
comfy.cli_args.args.cpu = True
import comfy.ops
from comfy.weight_adapter import LoRAAdapter

PACKAGE = Path(__file__).resolve().parents[1]
package = ModuleType("soya_spatial_test")
package.__path__ = [str(PACKAGE)]
sys.modules[package.__name__] = package
spatial = importlib.import_module("soya_spatial_test.soya_moss_spatial_lora")


def adapter(inputs=2, outputs=2):
    return LoRAAdapter(set(), (torch.ones(outputs, 1), torch.ones(1, inputs), 1., None, None, None))


class SpatialLoraTests(unittest.TestCase):
    def test_fixed_lora_reuses_weights_across_character_and_bbox_changes(self):
        import comfy.model_patcher
        import comfy.supported_models
        import comfy.utils
        import folder_paths
        transport = importlib.import_module("soya_spatial_test.soya_illustration_json_v2")
        config = comfy.supported_models.MOSSImage(dict(dim=32, ffn_dim=32, num_heads=2,
                    num_layers=1, text_dim=16, latent_channels=8, fused_qkv=True, dtype=torch.float32))
        config.custom_operations = comfy.ops.manual_cast
        base = comfy.model_patcher.ModelPatcher(config.get_model({}, device=torch.device("cpu")),
                    torch.device("cpu"), torch.device("cpu"))
        lora = {"diffusion_model.layers.0.proj.q.lora_up.weight": torch.ones(32, 1),
                "diffusion_model.layers.0.proj.q.lora_down.weight": torch.ones(1, 32)}
        # Use an existing file for the real file-stat cache key; only its tensor
        # decoding is mocked. No model files or GPU are needed for this check.
        path = str(Path(__file__).resolve())
        payload = {"moss": {"bbox_coord_space": "abs_centralized", "bbox_format": "yxyx",
                   "items": [{"label": "character", "name": "Alice", "bbox": [-16, -16, 16, 0]}]},
                   "workflow": {"width": 32, "height": 32, "loras": {"activate": True,
                   "list": [{"lora_path": path, "str": .4},
                            {"CHAR": "Alice", "lora_path": path, "str": .5}]}}}
        regional = spatial.SoyaMossSpatialLoraV2_mdsoya()
        fixed = transport.SoyaMossOptionalLoraV2_mdsoya()
        with patch.object(comfy.utils, "load_torch_file", return_value=lora) as read, \
             patch.object(folder_paths, "get_full_path_or_raise", return_value=path):
            routed, _, _ = regional.load_loras(base, None, payload)
            first, = fixed.load(routed, "fixed", 1., "first JSON")
            first_runtime = first.model_options["transformer_options"][spatial._KEY]
            fixed_base = first.get_attachment(spatial._BASE_KEY)
            self.assertEqual(read.call_count, 3)  # style, character, fixed
            changed = copy.deepcopy(payload)
            changed["moss"]["items"][0].update(name="Bob", bbox=[-16, 0, 16, 16])
            changed["workflow"]["loras"]["list"][1].update(CHAR="Bob", str=-.25)
            routed, _, _ = regional.load_loras(base, None, changed)
            second, = fixed.load(routed, "fixed", 1., "different JSON")
            second_runtime = second.model_options["transformer_options"][spatial._KEY]
            self.assertEqual(read.call_count, 4)  # only the new character
            self.assertIs(first.model, second.model)
            self.assertEqual(first.patches_uuid, second.patches_uuid)
            self.assertIs(second.get_attachment(spatial._BASE_KEY), fixed_base)
            self.assertIsNot(first_runtime, second_runtime)
            self.assertEqual(first_runtime.boxes, [[(0., 0., 1., .5)]])
            self.assertEqual(second_runtime.boxes, [[(0., .5, 1., 1.)]])
            self.assertEqual(second_runtime.patches["layers.0.proj.qkv"][0][2], -.25)
            self.assertFalse(base.patches)
            # Removing all characters must also reuse the same fixed weights.
            changed["workflow"]["loras"]["list"].pop()
            routed, _, _ = regional.load_loras(base, None, changed)
            third, = fixed.load(routed, "fixed", 1., "no character adapters")
            self.assertIs(third, fixed_base)
            self.assertEqual(third.patches_uuid, first.patches_uuid)
            self.assertNotIn(spatial._KEY, third.model_options["transformer_options"])
            self.assertEqual(read.call_count, 4)
            # A real fixed-style change must still invalidate weight patches.
            changed["workflow"]["loras"]["list"][0]["str"] = .8
            routed, _, _ = regional.load_loras(base, None, changed)
            fourth, = fixed.load(routed, "fixed", 1., "changed style")
            self.assertNotEqual(fourth.patches_uuid, third.patches_uuid)
            self.assertEqual(read.call_count, 6)

    def test_combined_character_adapters_keep_base_and_other_qkv_rows(self):
        runtime = spatial._Runtime([], {})
        runtime.active = True
        x, base = torch.ones(1, 4, 2), torch.full((1, 4, 6), 3.)
        entries = [(0, slice(0, 2), .5, adapter()), (1, slice(2, 4), -.25, adapter())]
        output = runtime.layer_hook(entries, torch.tensor([[1., 0.], [0., 1.]]),
                                    4, None, (x,), {}, base)
        expected = base.clone()
        expected[:, 0, :2] += 1.
        expected[:, 1, 2:4] -= .5
        torch.testing.assert_close(output, expected)
        torch.testing.assert_close(base, torch.full_like(base, 3.))

    def test_node_with_native_moss_split_qkv_mapping_and_reference_forward(self):
        import comfy.model_patcher
        import comfy.supported_models
        config = comfy.supported_models.MOSSImage(dict(dim=32, ffn_dim=32, num_heads=2,
                    num_layers=1, text_dim=16, latent_channels=8, fused_qkv=True, dtype=torch.float32))
        config.custom_operations = comfy.ops.manual_cast
        model = config.get_model({}, device=torch.device("cpu"))
        # INT8 ComfyUI Linear inherits nn.Module + MixedPrecisionOp, not
        # nn.Linear. Exercise that actual class rather than a mock subclass.
        original = model.diffusion_model.layers[0].proj["o"]
        quant_linear = comfy.ops.mixed_precision_ops(compute_dtype=torch.float32).Linear(32, 32, bias=False)
        quant_linear.load_state_dict(original.state_dict())
        model.diffusion_model.layers[0].proj["o"] = quant_linear
        self.assertNotIsInstance(quant_linear, torch.nn.Linear)
        patcher = comfy.model_patcher.ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
        lora = {"diffusion_model.layers.0.proj.q.lora_up.weight": torch.ones(32, 1),
                "diffusion_model.layers.0.proj.q.lora_down.weight": torch.ones(1, 32),
                "diffusion_model.layers.0.proj.o.lora_up.weight": torch.ones(32, 1),
                "diffusion_model.layers.0.proj.o.lora_down.weight": torch.ones(1, 32)}
        payload = {"moss": {"bbox_coord_space": "abs_centralized", "bbox_format": "yxyx",
                   "items": [{"label": "character", "name": "Alice", "bbox": [-16, -16, 16, 0]}]},
                   "workflow": {"width": 32, "height": 32,
                   "loras": {"activate": True, "list": [{"CHAR": "Alice", "lora_path": "a", "str": .5}]}}}
        with patch.object(spatial.SoyaBatchLoraLoader_mdsoya, "_resolve_lora_path", return_value="a"), \
             patch.object(comfy.utils, "load_torch_file", return_value=lora):
            routed, _, info = spatial.SoyaMossSpatialLoraV2_mdsoya().load_loras(patcher, None, payload)
        self.assertIn("BBox: Alice", info)
        self.assertFalse(patcher.patches)
        runtime = routed.model_options["transformer_options"][spatial._KEY]
        self.assertIn("layers.0.proj.o", runtime.patches)
        entry = runtime.patches["layers.0.proj.qkv"][0]
        self.assertEqual((entry[1].start, entry[1].stop), (0, 32))
        # A tiny native MOSS forward catches actual positional wrapper arguments
        # and both appended-reference and isolated-reference token ordering.
        with torch.no_grad(), torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            for parameter in model.parameters():
                parameter.uniform_(-.1, .1)
            previous = comfy.model_management.in_training
            comfy.model_management.in_training = True
            try:
                for method in ("index", "index_timestep_zero_isolated"):
                    out = model.diffusion_model(torch.randn(1, 8, 2, 2), torch.ones(1), torch.randn(1, 3, 16),
                            transformer_options=routed.model_options["transformer_options"],
                            ref_latents=[torch.randn(1, 8, 1, 2)], ref_latents_method=method)
                    self.assertEqual(out.shape, (1, 8, 2, 2))
                    self.assertTrue(torch.isfinite(out).all())
            finally:
                comfy.model_management.in_training = previous

    def test_name_mapping_and_repeated_panels(self):
        layout = {"bbox_coord_space": "abs_centralized", "bbox_format": "yxyx",
                  "items": [{"label": "character", "name": "Bob", "bbox": [-50, 0, 50, 50]},
                            {"label": "text", "name": "Alice", "bbox": [-50, -50, 50, 50]},
                            {"label": "character", "name": "Alice", "bbox": [-50, -50, 0, 0]},
                            {"label": "character", "name": "Alice", "bbox": [0, -50, 50, 0]}]}
        boxes = spatial.character_boxes(layout, {"width": 100, "height": 100})
        weights = spatial.token_weights([boxes["Alice"], boxes["Bob"]], 2, 2, "cpu", torch.float32)
        torch.testing.assert_close(weights, torch.tensor([[1., 0., 1., 0.], [0., 1., 0., 1.]]))

    def test_overlap_is_normalized_and_repeated_boxes_do_not_double_strength(self):
        box = (0., 0., 1., .5)
        weights = spatial.token_weights([[box, box], [box]], 1, 4, "cpu", torch.float32)
        torch.testing.assert_close(weights, torch.tensor([[.5, .5, 0., 0.], [.5, .5, 0., 0.]]))

    def test_delta_is_zero_outside_bbox_and_on_text_reference_tokens(self):
        runtime = spatial._Runtime([], {})
        runtime.active = True
        weights = torch.tensor([[1., 0., 1., 0.]])
        entries = [(0, slice(None), .5, adapter())]
        x, base = torch.ones(2, 7, 2), torch.full((2, 7, 2), 3.)
        output = runtime.layer_hook(entries, weights, 7, None, (x,), {}, base)
        expected = base.clone()
        expected[:, [0, 2]] += 1.
        torch.testing.assert_close(output, expected)
        torch.testing.assert_close(base, torch.full_like(base, 3.))

    def test_split_qkv_rows_are_local_and_negative_strength_is_preserved(self):
        runtime = spatial._Runtime([], {})
        runtime.active = True
        x, base = torch.ones(1, 4, 2), torch.zeros(1, 4, 6)
        output = runtime.layer_hook([(0, slice(2, 4), -.5, adapter())], torch.tensor([[1., 0.]]),
                                    4, None, (x,), {}, base)
        expected = torch.zeros_like(base)
        expected[:, 0, 2:4] = -1.
        torch.testing.assert_close(output, expected)

    def test_wrapper_restores_hooks_on_error_and_excludes_reference_preprocessing(self):
        dit = torch.nn.Module()
        dit.projection = torch.nn.Linear(2, 2, bias=False)
        torch.nn.init.zeros_(dit.projection.weight)
        original = lambda: dit.projection(torch.ones(1, 3, 2))
        dit.preprocess_refs = original
        runtime = spatial._Runtime([[(0., 0., 1., .5)]],
                                   {"projection": [(0, slice(None), 1., adapter())]})

        def forward(*args, **kwargs):
            self.assertTrue(dit.projection.comfy_force_forward)
            torch.testing.assert_close(dit.preprocess_refs(), torch.zeros(1, 3, 2))
            output = dit.projection(torch.ones(1, 3, 2))
            torch.testing.assert_close(output, torch.tensor([[[2., 2.], [0., 0.], [0., 0.]]]))
            raise RuntimeError("sampling interrupted")

        class Executor:
            class_obj = dit
            __call__ = staticmethod(forward)

        with self.assertRaisesRegex(RuntimeError, "sampling interrupted"):
            spatial._diffusion_wrapper(Executor(), torch.zeros(1, 2, 1, 2), None,
                                       torch.zeros(1, 1, 2), {spatial._KEY: runtime})
        self.assertIs(dit.preprocess_refs, original)
        self.assertFalse(dit.projection._forward_hooks)
        self.assertNotIn("comfy_force_forward", dit.projection.__dict__)
        self.assertFalse(runtime.active)
        torch.testing.assert_close(dit.projection(torch.ones(1, 3, 2)), torch.zeros(1, 3, 2))

    def test_fused_activation_linear_uses_activated_input(self):
        module = torch.nn.Linear(2, 2, bias=False)
        torch.nn.init.zeros_(module.weight)
        module.comfy_force_forward = True
        runtime = spatial._Runtime([], {})
        runtime.active = True
        from functools import partial
        handle = module.register_forward_hook(partial(runtime.layer_hook,
                    [(0, slice(None), 1., adapter())], torch.tensor([[1., 0.]]), 3), with_kwargs=True)
        try:
            x = torch.ones(1, 3, 4)
            output = comfy.ops.linear_input_act(module, x, "situ")
            activated = comfy.ops._eager_input_act(x, "situ", None, 0.)
            expected = torch.zeros(1, 3, 2)
            expected[:, :1] = activated[:, :1].sum(-1, keepdim=True)
            torch.testing.assert_close(output, expected)
        finally:
            handle.remove()

    def test_sample_cleanup_releases_device_adapter_cache(self):
        runtime = spatial._Runtime([], {})
        runtime.adapters["cached"] = torch.ones(1)

        class Executor:
            class_obj = SimpleNamespace(model_patcher=SimpleNamespace(
                model_options={"transformer_options": {spatial._KEY: runtime}}))

            def __call__(self):
                raise RuntimeError("cancelled")

        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            spatial._sample_wrapper(Executor())
        self.assertFalse(runtime.adapters)


if __name__ == "__main__":
    unittest.main()
