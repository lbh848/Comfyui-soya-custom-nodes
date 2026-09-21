import importlib.util
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


def test_semantically_identical_new_patcher_reuses_first_instance():
    base = types.SimpleNamespace(clone_base_uuid=uuid.uuid4())
    first = FakeModelPatcher(base, 1.0)
    rebuilt = FakeModelPatcher(base, 1.0)
    node = MODULE.SoyaStableModelPatcherReuse_mdsoya()

    assert node.reuse(first, "asset", '{"list":[1]}', "")[0] is first
    assert node.reuse(rebuilt, "asset", '{ "list": [1] }', "")[0] is first


def test_patch_content_change_replaces_cached_model_even_with_same_shape():
    base = types.SimpleNamespace(clone_base_uuid=uuid.uuid4())
    first = FakeModelPatcher(base, 1.0)
    changed = FakeModelPatcher(base, 9.0)
    node = MODULE.SoyaStableModelPatcherReuse_mdsoya()

    assert node.reuse(first, "asset", "same", "same")[0] is first
    assert node.reuse(changed, "asset", "same", "same")[0] is changed


def test_configuration_change_replaces_cached_model():
    base = types.SimpleNamespace(clone_base_uuid=uuid.uuid4())
    first = FakeModelPatcher(base, 1.0)
    rebuilt = FakeModelPatcher(base, 1.0)
    node = MODULE.SoyaStableModelPatcherReuse_mdsoya()

    assert node.reuse(first, "illustration", "character-a", "")[0] is first
    assert node.reuse(rebuilt, "illustration", "character-b", "")[0] is rebuilt


def test_scope_keeps_only_current_configuration():
    base = types.SimpleNamespace(clone_base_uuid=uuid.uuid4())
    first_a = FakeModelPatcher(base, 1.0)
    model_b = FakeModelPatcher(base, 2.0)
    second_a = FakeModelPatcher(base, 1.0)
    node = MODULE.SoyaStableModelPatcherReuse_mdsoya()

    assert node.reuse(first_a, "illustration", "a", "")[0] is first_a
    assert node.reuse(model_b, "illustration", "b", "")[0] is model_b
    assert node.reuse(second_a, "illustration", "a", "")[0] is second_a


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
