import importlib.util
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

    # The sampler's multi-character path attaches hooks and masks through this
    # ComfyUI boundary.  The behavior is supplied per test so the assertions
    # observe the conditioning that would reach a real sampler.
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
    character_lora = types.ModuleType(f"{PACKAGE_NAME}.soya_character_lora")
    character_lora.load_character_lora = None

    modules = {
        PACKAGE_NAME: package,
        "comfy": comfy,
        "comfy.hooks": comfy.hooks,
        "comfy.samplers": comfy.samplers,
        "comfy.sd": comfy.sd,
        "comfy.utils": comfy.utils,
        "folder_paths": folder_paths,
        f"{PACKAGE_NAME}.anima_regional_conditioning": regional,
        f"{PACKAGE_NAME}.soya_character_lora": character_lora,
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


class _FakeHookGroup:
    def __init__(self, labels):
        self.labels = tuple(labels)

    def clone_and_combine(self, other):
        return _FakeHookGroup(self.labels + other.labels)


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


def _conditioning_records(conditions):
    return [
        (entry[0], entry[1])
        for entry in conditions
        if isinstance(entry, (list, tuple))
        and len(entry) >= 2
        and isinstance(entry[1], dict)
    ]


def _records_for_mask(conditions, mask):
    return [
        (value, metadata)
        for value, metadata in _conditioning_records(conditions)
        if metadata.get("_test_mask") is mask
    ]


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
        input_model = object()
        clip = _FakeClip()
        masks = [
            _FakeMask("R"),
            _FakeMask("G"),
            _FakeMask("B"),
        ][: len(names)]
        hook_calls = []
        registered_models = []
        sampled = object()

        def load_lora(entry):
            filename = entry["lora_path"]
            return {"test_filename": filename}, f"/test-only/{filename}"

        def load_character_lora(current_model, lora, strength_model):
            hook_calls.append(
                (current_model, None, lora["test_filename"], strength_model, 0.0)
            )
            registered_model = object()
            registered_models.append(registered_model)
            hooks = _FakeHookGroup([(lora["test_filename"], strength_model)])
            return registered_model, hooks

        def set_conds_props(
            conds,
            strength=1.0,
            set_cond_area="default",
            mask=None,
            hooks=None,
            timesteps_range=None,
            append_hooks=True,
        ):
            result = []
            for conditioning in conds:
                copied = []
                for value, metadata in conditioning:
                    metadata = dict(metadata)
                    metadata.update(
                        {
                            "_test_hooks": hooks,
                            "_test_mask": mask,
                            "_test_area": set_cond_area,
                        }
                    )
                    copied.append([value, metadata])
                result.append(copied)
            return result

        def set_default_conds_and_combine(conds, new_conds, hooks=None, timesteps_range=None):
            result = []
            for conditioning, default_conditioning in zip(conds, new_conds):
                copied_default = []
                for value, metadata in default_conditioning:
                    metadata = dict(metadata)
                    metadata.update(
                        {
                            "default": True,
                            "_test_hooks": hooks,
                            "_test_mask": None,
                            "_test_area": "default",
                        }
                    )
                    copied_default.append([value, metadata])
                result.append(list(conditioning) + copied_default)
            return result

        selected = mock.Mock(return_value=(sampled,))
        multi_char = _multi_char_payload(names)
        with (
            mock.patch.object(node, "_load_lora", side_effect=load_lora),
            mock.patch.object(
                MODULE,
                "load_character_lora",
                side_effect=load_character_lora,
            ),
            mock.patch.object(MODULE.comfy.hooks, "set_conds_props", side_effect=set_conds_props),
            mock.patch.object(
                MODULE.comfy.hooks,
                "set_default_conds_and_combine",
                side_effect=set_default_conds_and_combine,
            ),
            mock.patch.object(
                node,
                "_apply_global_loras",
                side_effect=AssertionError("multi-character sampling must not stack character LoRAs globally"),
            ),
            mock.patch.object(
                MODULE,
                "ApplyAnimaRegionalConditioningPatch",
                side_effect=AssertionError("multi-character sampling must not use regional patching"),
                create=True,
            ),
            mock.patch.object(
                MODULE,
                "AnimaConditioningRegionChain",
                side_effect=AssertionError("multi-character sampling must not build regional chains"),
                create=True,
            ),
            mock.patch.object(MODULE, "_load_channel_masks", return_value=masks) as load_masks,
            mock.patch.object(node, "_sample_selected", selected),
        ):
            result = node.sample(
                model=input_model,
                positive=[["BASE_POS", {"source": "base-positive"}]],
                negative=[["BASE_NEG", {"source": "base-negative"}]],
                clip=clip,
                multi_char=multi_char,
                seed=7,
                latent_image={"samples": object()},
                LORA_ACT="true",
                LORA_DATA=_lora_data(entries),
                mask_location="test-only-masks",
                steps=28,
                cfg=4.0,
                sampler_name="euler",
                scheduler="simple",
                denoise=1.0,
                split_mode="single",
                spd_scale=0.5,
                spd_sigma=0.7,
                adaptive_smc_alpha=0.1,
                sampler_mode=sampler_mode,
            )

        return {
            "node": node,
            "input_model": input_model,
            "clip": clip,
            "masks": masks,
            "hook_calls": hook_calls,
            "registered_models": registered_models,
            "selected": selected,
            "result": result,
        }

    def _assert_multi_char_conditioning(self, case, names, expected_labels):
        selected_call = case["selected"].call_args
        positive = selected_call.args[7]
        negative = selected_call.args[8]

        self.assertEqual(len(_conditioning_records(positive)), len(names) + 1)
        self.assertEqual(len(_conditioning_records(negative)), len(names) + 1)

        for index, name in enumerate(names):
            with self.subTest(character=name):
                positive_records = _records_for_mask(positive, case["masks"][index])
                negative_records = _records_for_mask(negative, case["masks"][index])
                self.assertEqual(len(positive_records), 1)
                self.assertEqual(len(negative_records), 1)

                positive_value, positive_metadata = positive_records[0]
                _negative_value, negative_metadata = negative_records[0]
                positive_hooks = positive_metadata["_test_hooks"]
                negative_hooks = negative_metadata["_test_hooks"]
                labels = expected_labels[index]
                if labels is None:
                    self.assertIsNone(positive_hooks)
                else:
                    self.assertEqual(positive_hooks.labels, labels)
                self.assertIs(negative_hooks, positive_hooks)
                self.assertIs(positive_metadata["_test_mask"], case["masks"][index])
                self.assertIs(negative_metadata["_test_mask"], case["masks"][index])
                self.assertEqual(positive_metadata["_test_area"], "default")
                self.assertEqual(negative_metadata["_test_area"], "default")

                # Each region receives its own prompt context.  Checking every
                # other profile catches accidental prompt concatenation between
                # characters even when masks and hooks look correct.
                self.assertIn(f"{name} profile", str(positive_value))
                for other_name in names:
                    if other_name != name:
                        self.assertNotIn(f"{other_name} profile", str(positive_value))

        background_positive = _records_for_mask(positive, None)
        background_negative = _records_for_mask(negative, None)
        self.assertEqual(len(background_positive), 1)
        self.assertEqual(len(background_negative), 1)
        self.assertIsNone(background_positive[0][1]["_test_hooks"])
        self.assertIsNone(background_negative[0][1]["_test_hooks"])
        self.assertIsNone(background_positive[0][1]["_test_mask"])
        self.assertIsNone(background_negative[0][1]["_test_mask"])
        self.assertEqual(background_positive[0][1]["_test_area"], "default")
        self.assertEqual(background_negative[0][1]["_test_area"], "default")

    def test_multi_char_hibiki_hoshino_uses_one_hook_and_mask_per_character(self):
        case = self._run_multi_sample(
            ["Hibiki", "Hoshino"],
            [
                ("Hibiki", "test_hibiki_lora.safetensors", 0.8),
                ("Hoshino", "test_hoshino_lora.safetensors", 0.9),
            ],
        )
        self.assertEqual(case["selected"].call_args.args[0], "KSampler")
        self.assertIs(case["selected"].call_args.args[1], case["registered_models"][-1])
        self.assertEqual(case["result"][0], case["selected"].return_value[0])
        self.assertIs(case["result"][1], case["input_model"])
        self.assertEqual(case["clip"].prompts.count(""), 0)
        self._assert_multi_char_conditioning(
            case,
            ["Hibiki", "Hoshino"],
            [
                (("test_hibiki_lora.safetensors", 0.8),),
                (("test_hoshino_lora.safetensors", 0.9),),
            ],
        )

        self.assertEqual(
            [(path, strength) for _model, _clip, path, strength, _clip_strength in case["hook_calls"]],
            [
                ("test_hibiki_lora.safetensors", 0.8),
                ("test_hoshino_lora.safetensors", 0.9),
            ],
        )
        self.assertIs(case["selected"].call_args.args[9], case["clip"])
        self.assertTrue(any("Hibiki profile" in prompt for prompt in case["clip"].prompts))
        self.assertTrue(any("Hoshino profile" in prompt for prompt in case["clip"].prompts))

    def test_multi_char_matches_hooks_by_name_when_order_and_lora_count_differ(self):
        case = self._run_multi_sample(
            ["Hoshino", "Hibiki"],
            [
                ("Hibiki", "test_hibiki_face.safetensors", 0.8),
                ("Hibiki", "test_hibiki_style.safetensors", 0.2),
                ("Hoshino", "test_hoshino_lora.safetensors", 0.9),
            ],
            sampler_mode="FAST",
        )
        self.assertEqual(case["selected"].call_args.args[0], "FAST")
        self._assert_multi_char_conditioning(
            case,
            ["Hoshino", "Hibiki"],
            [
                (("test_hoshino_lora.safetensors", 0.9),),
                (
                    ("test_hibiki_face.safetensors", 0.8),
                    ("test_hibiki_style.safetensors", 0.2),
                ),
            ],
        )
        self.assertEqual(
            [call[2:] for call in case["hook_calls"]],
            [
                ("test_hibiki_face.safetensors", 0.8, 0.0),
                ("test_hibiki_style.safetensors", 0.2, 0.0),
                ("test_hoshino_lora.safetensors", 0.9, 0.0),
            ],
        )

    def test_multi_char_keeps_masked_prompt_when_one_character_has_no_lora(self):
        case = self._run_multi_sample(
            ["Mina", "Niko"],
            [("Mina", "test_mina_lora.safetensors", 0.65)],
        )
        self._assert_multi_char_conditioning(
            case,
            ["Mina", "Niko"],
            [(("test_mina_lora.safetensors", 0.65),), None],
        )

        self.assertEqual(len(case["hook_calls"]), 1)

    def test_multi_char_with_no_loras_keeps_all_regions_masked_without_hooks(self):
        names = ["Aster", "Briar"]
        case = self._run_multi_sample(names, [])

        self._assert_multi_char_conditioning(case, names, [None, None])
        self.assertEqual(case["hook_calls"], [])
        self.assertIs(case["selected"].call_args.args[1], case["input_model"])
        self.assertIs(case["result"][1], case["input_model"])

    def test_multi_char_passes_hooked_conditioning_to_stock_and_spectrum_modes(self):
        for sampler_mode in ("KSampler", "FAST"):
            with self.subTest(sampler_mode=sampler_mode):
                case = self._run_multi_sample(
                    ["North", "South"],
                    [
                        ("North", "test_north_lora.safetensors", 0.4),
                        ("South", "test_south_lora.safetensors", 0.6),
                    ],
                    sampler_mode=sampler_mode,
                )
                self.assertEqual(case["selected"].call_args.args[0], sampler_mode)
                self.assertIs(case["result"][1], case["input_model"])
                self.assertEqual(len(_conditioning_records(case["selected"].call_args.args[7])), 3)

    def test_disabled_multi_char_preserves_single_preset_global_lora_behavior(self):
        node = FirstSampler()
        input_model = object()
        global_model = object()
        sampled = object()
        positive = [["BASE_POS", {"source": "base-positive"}]]
        negative = [["BASE_NEG", {"source": "base-negative"}]]
        selected = mock.Mock(return_value=(sampled,))
        with (
            mock.patch.object(node, "_apply_global_loras", return_value=global_model) as apply_global,
            mock.patch.object(node, "_build_character_hooks", side_effect=AssertionError("single path must not build per-character hooks")),
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


if __name__ == "__main__":
    unittest.main()
