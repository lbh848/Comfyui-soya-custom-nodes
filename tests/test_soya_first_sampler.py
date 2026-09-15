import ast
import importlib.util
import inspect
import json
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

    # Legacy conditioning hooks must not be used by either sampler path.
    comfy.hooks.set_conds_props = None
    comfy.hooks.set_default_conds_and_combine = None

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


class _FakeMask:
    def __init__(self, channel):
        self.channel = channel


class _FakeClip:
    def __init__(self):
        self.prompts = []

    def tokenize(self, prompt):
        self.prompts.append(prompt)
        return prompt

    def encode_from_tokens_scheduled(self, tokens):
        return [[tokens, {"prompt": tokens}]]


def _multi_char_payload(names, *, infos=None, triggers=None, background="shared background"):
    infos = infos or [f"{name} profile" for name in names]
    triggers = triggers or [[f"{name} trigger"] for name in names]
    return json.dumps(
        {
            "enable": True,
            "char_num": len(names),
            "char_name_list": names,
            "char_inform": infos,
            "char_trigger_list": triggers,
            "background_trigger_list": ["background trigger"],
            "shared_tag": {"before_char": ["shared before"], "after_char": ["shared after"]},
            "background_prompt": background,
            "composition_prompt": "two subjects in a clear composition",
            "mask_fingerprint": "0" * 64,
        }
    )


def _lora_data(entries):
    return json.dumps(
        {
            "list": [
                {
                    "BASE": "Anima",
                    "CHAR": character,
                    "lora_path": filename,
                    "str": strength,
                }
                for character, filename, strength in entries
            ]
        }
    )


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

    def _run_multi_sample(self, names, entries, *, sampler_mode="KSampler"):
        node = FirstSampler()
        input_model, routed_model, sampled = object(), object(), object()
        clip = _FakeClip()
        masks = [_FakeMask(channel) for channel in "RGB"[:len(names)]]
        negative = [["BASE_NEG", {"source": "base-negative"}]]
        spatial = types.ModuleType(f"{PACKAGE_NAME}.soya_spatial_lora")
        spatial.apply_spatial_loras = mock.Mock(return_value=routed_model)
        selected = mock.Mock(return_value=(sampled,))
        with (
            mock.patch.dict(sys.modules, {spatial.__name__: spatial}),
            mock.patch.object(node, "_load_lora", side_effect=lambda entry: (
                {"test_filename": entry["lora_path"]}, f"/test-only/{entry['lora_path']}",
            )) as load_lora,
            mock.patch.object(node, "_apply_global_loras", side_effect=AssertionError(
                "multi-character sampling must not stack character LoRAs globally")),
            mock.patch.object(MODULE.comfy.hooks, "set_conds_props", side_effect=AssertionError(
                "multi-character sampling must not create per-character prediction branches")),
            mock.patch.object(MODULE.comfy.hooks, "set_default_conds_and_combine", side_effect=AssertionError(
                "multi-character sampling must share the model forward")),
            mock.patch.object(MODULE, "_load_channel_masks", return_value=masks) as load_masks,
            mock.patch.object(node, "_sample_selected", selected),
        ):
            result = node.sample(
                model=input_model, positive=[["BASE_POS", {}]], negative=negative, clip=clip,
                multi_char=_multi_char_payload(names), seed=7, latent_image={"samples": object()},
                LORA_ACT="true", LORA_DATA=_lora_data(entries), mask_location="test-only-masks",
                steps=28, cfg=4.0, sampler_name="euler", scheduler="simple", denoise=1.0,
                split_mode="single", spd_scale=0.5, spd_sigma=0.7, adaptive_smc_alpha=0.1,
                sampler_mode=sampler_mode,
            )
        load_masks.assert_called_once_with("test-only-masks", len(names))
        self.assertEqual(load_lora.call_count, len(entries))
        spatial.apply_spatial_loras.assert_called_once()
        selected.assert_called_once()
        self.assertIs(selected.call_args.args[1], routed_model)
        self.assertEqual(result, (sampled, input_model))
        return {
            "input_model": input_model, "clip": clip, "masks": masks,
            "spatial": spatial.apply_spatial_loras, "selected": selected, "negative": negative,
        }

    def _assert_multi_char_conditioning(self, case, names, entries):
        model, loaded, masks, regions, background = case["spatial"].call_args.args
        self.assertIs(model, case["input_model"])
        self.assertIs(masks, case["masks"])
        self.assertEqual(loaded, [
            (names.index(name), {"test_filename": path}, strength)
            for name, path, strength in entries
        ])
        self.assertEqual(len(regions), len(names))
        for index, name in enumerate(names):
            with self.subTest(character=name):
                self.assertEqual(len(regions[index]), 1)
                prompt = regions[index][0][0]
                self.assertIn(f"{name} profile", prompt)
                self.assertIn(f"{name} trigger", prompt)
                for shared in ("shared before", "shared after", "two subjects in a clear composition"):
                    self.assertIn(shared, prompt)
                for other in names:
                    if other != name:
                        self.assertNotIn(f"{other} profile", prompt)
                        self.assertNotIn(f"{other} trigger", prompt)
        self.assertEqual(len(background), 1)
        self.assertIn("shared background", background[0][0])
        self.assertIn("background trigger", background[0][0])
        for name in names:
            self.assertNotIn(f"{name} profile", background[0][0])
            self.assertNotIn(f"{name} trigger", background[0][0])
        # A single shared CFG pair reaches all three sampler modes; masks and
        # per-character prompt/adapter ownership live on the routed model.
        self.assertIs(case["selected"].call_args.args[7], background)
        self.assertIs(case["selected"].call_args.args[8], case["negative"])
        self.assertIs(case["selected"].call_args.args[9], case["clip"])

    def test_multi_char_hibiki_hoshino_routes_loras_and_prompts_per_character(self):
        names = ["Hibiki", "Hoshino"]
        entries = [
            ("Hibiki", "test_hibiki_lora.safetensors", 0.8),
            ("Hoshino", "test_hoshino_lora.safetensors", 0.9),
        ]
        case = self._run_multi_sample(names, entries)
        self.assertEqual(case["selected"].call_args.args[0], "KSampler")
        self._assert_multi_char_conditioning(case, names, entries)

    def test_multi_char_matches_by_name_when_order_and_lora_count_differ(self):
        names = ["Hoshino", "Hibiki"]
        entries = [
            ("Hibiki", "test_hibiki_face.safetensors", 0.8),
            ("Hibiki", "test_hibiki_style.safetensors", 0.2),
            ("Hoshino", "test_hoshino_lora.safetensors", 0.9),
        ]
        case = self._run_multi_sample(names, entries, sampler_mode="FAST")
        self._assert_multi_char_conditioning(case, names, entries)

    def test_multi_char_keeps_masked_prompt_when_one_character_has_no_lora(self):
        names = ["Mina", "Niko"]
        entries = [("Mina", "test_mina_lora.safetensors", 0.65)]
        self._assert_multi_char_conditioning(self._run_multi_sample(names, entries), names, entries)

    def test_multi_char_with_no_loras_still_routes_all_prompts(self):
        names = ["Aster", "Briar"]
        self._assert_multi_char_conditioning(self._run_multi_sample(names, []), names, [])

    def test_multi_char_passes_shared_conditioning_to_all_sampler_modes(self):
        names = ["North", "South"]
        entries = [
            ("North", "test_north_lora.safetensors", 0.4),
            ("South", "test_south_lora.safetensors", 0.6),
        ]
        for mode in ("KSampler", "FAST", "SpectrumKSamplerModGuidance"):
            with self.subTest(sampler_mode=mode):
                case = self._run_multi_sample(names, entries, sampler_mode=mode)
                self.assertEqual(case["selected"].call_args.args[0], mode)
                self._assert_multi_char_conditioning(case, names, entries)

    def test_disabled_multi_char_preserves_single_preset_global_lora_behavior(self):
        node = FirstSampler()
        input_model = object()
        global_model = object()
        sampled = object()
        positive = [["BASE_POS", {"source": "base-positive"}]]
        negative = [["BASE_NEG", {"source": "base-negative"}]]
        selected = mock.Mock(return_value=(sampled,))
        spatial = types.ModuleType(f"{PACKAGE_NAME}.soya_spatial_lora")
        spatial.apply_spatial_loras = mock.Mock(side_effect=AssertionError("single path must not apply spatial routing"))
        with (
            mock.patch.dict(sys.modules, {spatial.__name__: spatial}),
            mock.patch.object(node, "_apply_global_loras", return_value=global_model) as apply_global,
            mock.patch.object(MODULE, "_load_channel_masks", side_effect=AssertionError("single path must not load RGB masks"), create=True),
            mock.patch.object(MODULE.comfy.hooks, "set_conds_props", side_effect=AssertionError("single path must keep conditioning unchanged")),
            mock.patch.object(node, "_sample_selected", selected),
        ):
            result = node.sample(
                model=input_model,
                positive=positive,
                negative=negative,
                clip=None,
                multi_char="",
                seed=7,
                latent_image={"samples": object()},
                LORA_ACT="true",
                LORA_DATA=_lora_data([("Hibiki", "test_single_lora.safetensors", 0.8)]),
                mask_location="unused",
                steps=28,
                cfg=4.0,
                sampler_name="euler",
                scheduler="simple",
                denoise=1.0,
                split_mode="single",
                spd_scale=0.5,
                spd_sigma=0.7,
                adaptive_smc_alpha=0.1,
                sampler_mode="KSampler",
            )

        apply_global.assert_called_once()
        self.assertIs(apply_global.call_args.args[0], input_model)
        self.assertIs(selected.call_args.args[1], global_model)
        self.assertIs(selected.call_args.args[7], positive)
        self.assertIs(selected.call_args.args[8], negative)
        self.assertEqual(result[0], sampled)
        self.assertIs(result[1], global_model)

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

    def test_existing_node_contract_is_preserved_without_a_separate_spatial_node(self):
        required = (
            "model", "positive", "negative", "clip", "multi_char", "seed", "latent_image",
            "LORA_ACT", "LORA_DATA", "mask_location", "steps", "cfg", "sampler_name",
            "scheduler", "denoise", "split_mode", "spd_scale", "spd_sigma", "adaptive_smc_alpha",
        )
        optional = ("sampler_mode", "spectrum_options")
        self.assertEqual(tuple(FirstSampler.INPUT_TYPES()["required"]), required)
        self.assertEqual(tuple(FirstSampler.INPUT_TYPES()["optional"]), optional)
        self.assertEqual(tuple(inspect.signature(FirstSampler.sample).parameters), ("self",) + required + optional)
        self.assertEqual(FirstSampler.RETURN_TYPES, ("LATENT", "MODEL"))
        self.assertEqual(FirstSampler.RETURN_NAMES, ("output", "model"))
        self.assertEqual(FirstSampler.FUNCTION, "sample")
        self.assertEqual(FirstSampler.CATEGORY, "sampling")
        self.assertFalse(hasattr(MODULE, "SoyaFirstSamplerSpatialLoRA_mdsoya"))
        tree = ast.parse((MODULE_PATH.parent / "__init__.py").read_text(encoding="utf-8"))
        mappings = {
            target.id: node.value
            for node in tree.body if isinstance(node, ast.Assign)
            for target in node.targets if isinstance(target, ast.Name)
        }
        classes = mappings["NODE_CLASS_MAPPINGS"]
        displays = mappings["NODE_DISPLAY_NAME_MAPPINGS"]
        sampler_classes = {
            key.value: value.id for key, value in zip(classes.keys, classes.values)
            if "FirstSampler" in key.value
        }
        self.assertEqual(sampler_classes, {"SoyaFirstSampler_mdsoya": "SoyaFirstSampler_mdsoya"})
        sampler_displays = {
            key.value: value.value for key, value in zip(displays.keys, displays.values)
            if "FirstSampler" in key.value
        }
        self.assertEqual(sampler_displays, {"SoyaFirstSampler_mdsoya": "1st sampler"})

    def test_spatial_sampler_routes_by_slot_without_hook_or_global_merge(self):
        spatial = types.ModuleType(f"{PACKAGE_NAME}.soya_spatial_lora")
        for names in (("Hibiki", "Hoshino"), ("Aster", "Briar"), ("North", "South", "West")):
            with self.subTest(names=names):
                node = FirstSampler()
                model, patched, sampled = object(), object(), object()
                spatial.apply_spatial_loras = mock.Mock(return_value=patched)
                masks = [_FakeMask(i) for i in range(len(names))]
                clip = _FakeClip()
                entries = [(name, f"test_{i}.safetensors", .4 + i * .1) for i, name in reversed(list(enumerate(names)))]
                # Two adapters on one character and a different file order
                # must not change the relationship to the mask and prompt.
                entries.append((names[0], "test_extra.safetensors", -.2))
                with (
                    mock.patch.dict(sys.modules, {spatial.__name__: spatial}),
                    mock.patch.object(node, "_load_lora", side_effect=lambda entry: (entry["lora_path"], entry["lora_path"])),
                    mock.patch.object(node, "_apply_global_loras", side_effect=AssertionError("shared path must not merge character weights")),
                    mock.patch.object(MODULE, "_load_channel_masks", return_value=masks),
                    mock.patch.object(node, "_sample_selected", return_value=(sampled,)) as selected,
                ):
                    result = node.sample(
                        model=model, positive=[["base", {}]], negative=[["negative", {}]], clip=clip,
                        multi_char=_multi_char_payload(names), seed=7, latent_image={"samples": object()},
                        LORA_ACT="true", LORA_DATA=_lora_data(entries), mask_location="test-only", steps=30,
                        cfg=5, sampler_name="euler", scheduler="simple", denoise=1., split_mode="single",
                        spd_scale=.5, spd_sigma=.7, adaptive_smc_alpha=0., sampler_mode="KSampler",
                    )
                applied = spatial.apply_spatial_loras.call_args.args
                self.assertIs(applied[0], model)
                self.assertEqual(applied[1], [(names.index(name), path, strength) for name, path, strength in entries])
                self.assertIs(applied[2], masks)
                for index, name in enumerate(names):
                    self.assertIn(f"{name} profile", applied[3][index][0][0])
                self.assertIs(selected.call_args.args[1], patched)
                self.assertEqual(len(selected.call_args.args[7]), 1)
                self.assertEqual(len(selected.call_args.args[8]), 1)
                self.assertIs(result[0], sampled)
                self.assertIs(result[1], model)


if __name__ == "__main__":
    unittest.main()
