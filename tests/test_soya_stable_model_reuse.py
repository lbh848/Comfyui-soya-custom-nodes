import importlib.util
import hashlib
import sys
import types
import uuid
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).parents[1] / "soya_stable_model_reuse.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "_soya_stable_model_reuse_test_module",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


class FakeModelPatcher:
    def __init__(self, base_model, patch_value, *, patch_uuid=None):
        self.model = base_model
        self.clone_base_uuid = base_model.clone_base_uuid
        self.patches_uuid = patch_uuid or uuid.uuid4()
        self.load_device = "cuda:0"
        self.offload_device = "cpu"
        self.weight_inplace_update = False
        self.patches = {
            "diffusion_model.block.weight": [
                (0.75, ("lora", (torch.tensor([patch_value, 2.0]),)), 1.0, None, None)
            ]
        }
        self.model_options = {"transformer_options": {"sage": True}}
        self.object_patches = {}
        self.weight_wrapper_patches = {}
        self.wrappers = {}
        self.callbacks = {}
        self.injections = {}
        self.attachments = {}
        self.additional_models = {}


def setup_function():
    MODULE.clear_stable_model_reuse_cache()


def _reuse(node, model, scope, configuration_a, configuration_b):
    output = node.reuse(model, scope, configuration_a, configuration_b)
    assert set(output) == {"ui", "result"}
    return output["result"][0], output["ui"]["stable_model_reuse"][0]


def _legacy_signature(model, configuration_a, configuration_b):
    hasher = hashlib.sha256()
    MODULE._hash_token(hasher, "soya-stable-model-reuse-v1")
    MODULE._hash_token(hasher, f"base_object:{id(model.model)}")
    MODULE._hash_token(hasher, f"clone_base_uuid:{model.clone_base_uuid}")
    MODULE._hash_token(hasher, f"load_device:{model.load_device}")
    MODULE._hash_token(hasher, f"offload_device:{model.offload_device}")
    MODULE._hash_token(
        hasher,
        f"weight_inplace_update:{model.weight_inplace_update}",
    )
    MODULE._update_signature(hasher, model.patches)
    MODULE._update_signature(hasher, model.model_options)
    MODULE._update_signature(hasher, model.object_patches)
    MODULE._update_signature(hasher, model.weight_wrapper_patches)
    MODULE._update_signature(hasher, model.wrappers)
    MODULE._update_signature(hasher, model.callbacks)
    MODULE._update_signature(hasher, model.injections)
    MODULE._update_signature(hasher, model.attachments)
    MODULE._update_signature(hasher, {})
    MODULE._hash_token(
        hasher,
        MODULE._canonical_configuration(configuration_a),
    )
    MODULE._hash_token(
        hasher,
        MODULE._canonical_configuration(configuration_b),
    )
    return hasher.hexdigest()


def test_observability_preserves_original_cache_signature():
    base = types.SimpleNamespace(clone_base_uuid=uuid.uuid4())
    model = FakeModelPatcher(base, 3.0)
    configuration_a = '{"style":["soft", 0.6]}'
    configuration_b = "character-lora"

    assert MODULE.model_reuse_signature(
        model,
        configuration_a,
        configuration_b,
    ) == _legacy_signature(model, configuration_a, configuration_b)


def test_semantically_identical_new_patcher_reuses_first_instance():
    base = types.SimpleNamespace(clone_base_uuid=uuid.uuid4())
    first = FakeModelPatcher(base, 1.0)
    rebuilt = FakeModelPatcher(base, 1.0)
    node = MODULE.SoyaStableModelPatcherReuse_mdsoya()

    first_result, first_trace = _reuse(
        node, first, "asset", '{"list":[1]}', ""
    )
    rebuilt_result, rebuilt_trace = _reuse(
        node, rebuilt, "asset", '{ "list": [1] }', ""
    )

    assert first_result is first
    assert rebuilt_result is first
    assert first_trace["cache_state"] == "miss"
    assert rebuilt_trace["cache_state"] == "hit"
    assert rebuilt_trace["incoming_patches_uuid"] == str(rebuilt.patches_uuid)
    assert rebuilt_trace["chosen_patches_uuid"] == str(first.patches_uuid)
    assert rebuilt_trace["incoming_is_chosen"] is False
    assert rebuilt_trace["changed_components"] == []
    assert rebuilt_trace["signature"] == first_trace["signature"]


def test_patch_content_change_replaces_cached_model_even_with_same_shape():
    base = types.SimpleNamespace(clone_base_uuid=uuid.uuid4())
    first = FakeModelPatcher(base, 1.0)
    changed = FakeModelPatcher(base, 9.0)
    node = MODULE.SoyaStableModelPatcherReuse_mdsoya()

    first_result, _first_trace = _reuse(node, first, "asset", "same", "same")
    changed_result, changed_trace = _reuse(
        node, changed, "asset", "same", "same"
    )

    assert first_result is first
    assert changed_result is changed
    assert changed_trace["cache_state"] == "replace"
    assert changed_trace["changed_components"] == ["patches"]


def test_configuration_change_replaces_cached_model():
    base = types.SimpleNamespace(clone_base_uuid=uuid.uuid4())
    first = FakeModelPatcher(base, 1.0)
    rebuilt = FakeModelPatcher(base, 1.0)
    node = MODULE.SoyaStableModelPatcherReuse_mdsoya()

    first_result, _first_trace = _reuse(
        node, first, "illustration", "character-a", ""
    )
    rebuilt_result, rebuilt_trace = _reuse(
        node, rebuilt, "illustration", "character-b", ""
    )

    assert first_result is first
    assert rebuilt_result is rebuilt
    assert rebuilt_trace["cache_state"] == "replace"
    assert rebuilt_trace["changed_components"] == ["configuration_a"]


def test_scope_keeps_only_current_configuration():
    base = types.SimpleNamespace(clone_base_uuid=uuid.uuid4())
    first_a = FakeModelPatcher(base, 1.0)
    model_b = FakeModelPatcher(base, 2.0)
    second_a = FakeModelPatcher(base, 1.0)
    node = MODULE.SoyaStableModelPatcherReuse_mdsoya()

    first_result, _first_trace = _reuse(
        node, first_a, "illustration", "a", ""
    )
    model_b_result, _model_b_trace = _reuse(
        node, model_b, "illustration", "b", ""
    )
    second_a_result, second_a_trace = _reuse(
        node, second_a, "illustration", "a", ""
    )

    assert first_result is first_a
    assert model_b_result is model_b
    assert second_a_result is second_a
    assert second_a_trace["cache_state"] == "replace"


def test_empty_scope_is_rejected():
    base = types.SimpleNamespace(clone_base_uuid=uuid.uuid4())
    model = FakeModelPatcher(base, 1.0)
    node = MODULE.SoyaStableModelPatcherReuse_mdsoya()

    try:
        node.reuse(model, "", "", "")
    except ValueError as exc:
        assert "cache_scope" in str(exc)
    else:
        raise AssertionError("empty cache_scope must fail")
