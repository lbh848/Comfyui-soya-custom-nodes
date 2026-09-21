"""Configuration-aware reuse of fully patched ComfyUI ModelPatchers.

ComfyUI assigns a new ``patches_uuid`` whenever an otherwise identical LoRA
stack is rebuilt. On some warm, resident GPU paths that forces the same base
model to be unpatched and patched again for every image. This node keeps one
completed ModelPatcher per workflow scope and returns it again only when the
effective patch/configuration signature is unchanged.

The cache deliberately holds a single entry per scope. Asset batches benefit
from long runs with one character, while illustration character changes replace
the old entry instead of retaining many LoRA tensors in RAM.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import threading
import traceback
from dataclasses import dataclass
from typing import Any

import torch


_LOG_PREFIX = "[Soya:StableModelReuse]"
_TENSOR_SAMPLE_COUNT = 16
_MAX_SIGNATURE_DEPTH = 10


@dataclass
class _CachedModel:
    signature: str
    components: dict[str, str]
    model: Any


_MODEL_CACHE: dict[str, _CachedModel] = {}
_MODEL_CACHE_LOCK = threading.RLock()


def _hash_token(hasher, value: Any) -> None:
    hasher.update(str(value).encode("utf-8", errors="backslashreplace"))
    hasher.update(b"\0")


def _canonical_configuration(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parsed = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return text
    return json.dumps(
        parsed,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _tensor_sample_bytes(value: torch.Tensor) -> bytes:
    detached = value.detach()
    count = int(detached.numel())
    if count == 0:
        return b""
    flat = detached.reshape(-1)
    sample_count = min(_TENSOR_SAMPLE_COUNT, count)
    if sample_count == 1:
        indices = torch.zeros(1, dtype=torch.long, device=flat.device)
    else:
        indices = torch.linspace(
            0,
            count - 1,
            steps=sample_count,
            dtype=torch.float64,
            device=flat.device,
        ).round().to(dtype=torch.long)
    sampled = flat.index_select(0, indices)
    if sampled.is_complex():
        sampled = torch.view_as_real(sampled).reshape(-1)
    if sampled.dtype == torch.bool:
        sampled = sampled.to(dtype=torch.uint8)
    elif sampled.is_floating_point():
        sampled = sampled.to(dtype=torch.float64)
    else:
        sampled = sampled.to(dtype=torch.int64)
    return sampled.to(device="cpu").contiguous().numpy().tobytes()


def _update_signature(
    hasher,
    value: Any,
    *,
    depth: int = 0,
    seen: set[int] | None = None,
) -> None:
    if seen is None:
        seen = set()
    if depth > _MAX_SIGNATURE_DEPTH:
        _hash_token(hasher, f"depth:{type(value).__module__}.{type(value).__qualname__}")
        return
    if value is None or isinstance(value, (bool, int, str)):
        _hash_token(hasher, f"{type(value).__name__}:{value}")
        return
    if isinstance(value, float):
        normalized = value if math.isfinite(value) else str(value)
        _hash_token(hasher, f"float:{normalized!r}")
        return
    if isinstance(value, bytes):
        _hash_token(hasher, f"bytes:{len(value)}")
        hasher.update(value)
        return
    if isinstance(value, torch.Tensor):
        _hash_token(
            hasher,
            f"tensor:{tuple(value.shape)}:{value.dtype}:{value.layout}:{value.requires_grad}",
        )
        hasher.update(_tensor_sample_bytes(value))
        return

    object_id = id(value)
    if object_id in seen:
        _hash_token(hasher, f"cycle:{type(value).__module__}.{type(value).__qualname__}")
        return

    if isinstance(value, dict):
        seen.add(object_id)
        _hash_token(hasher, f"dict:{len(value)}")
        for key in sorted(value, key=lambda item: str(item)):
            _update_signature(hasher, key, depth=depth + 1, seen=seen)
            _update_signature(hasher, value[key], depth=depth + 1, seen=seen)
        seen.remove(object_id)
        return
    if isinstance(value, (list, tuple)):
        seen.add(object_id)
        _hash_token(hasher, f"{type(value).__name__}:{len(value)}")
        for item in value:
            _update_signature(hasher, item, depth=depth + 1, seen=seen)
        seen.remove(object_id)
        return
    if isinstance(value, (set, frozenset)):
        seen.add(object_id)
        child_hashes = []
        for item in value:
            child = hashlib.sha256()
            _update_signature(child, item, depth=depth + 1, seen=seen)
            child_hashes.append(child.hexdigest())
        _hash_token(hasher, f"{type(value).__name__}:{len(child_hashes)}")
        for child_hash in sorted(child_hashes):
            _hash_token(hasher, child_hash)
        seen.remove(object_id)
        return
    if isinstance(value, functools.partial):
        seen.add(object_id)
        _hash_token(hasher, "partial")
        _update_signature(hasher, value.func, depth=depth + 1, seen=seen)
        _update_signature(hasher, value.args, depth=depth + 1, seen=seen)
        _update_signature(hasher, value.keywords or {}, depth=depth + 1, seen=seen)
        seen.remove(object_id)
        return
    if isinstance(value, torch.nn.Module):
        _hash_token(hasher, f"module:{type(value).__module__}.{type(value).__qualname__}")
        return
    if callable(value):
        _hash_token(
            hasher,
            "callable:"
            f"{getattr(value, '__module__', type(value).__module__)}."
            f"{getattr(value, '__qualname__', type(value).__qualname__)}",
        )
        closure = getattr(value, "__closure__", None)
        if closure:
            seen.add(object_id)
            for cell in closure:
                try:
                    cell_value = cell.cell_contents
                except ValueError:
                    cell_value = "<empty>"
                _update_signature(hasher, cell_value, depth=depth + 1, seen=seen)
            seen.remove(object_id)
        return

    state = getattr(value, "__dict__", None)
    if isinstance(state, dict):
        seen.add(object_id)
        _hash_token(hasher, f"object:{type(value).__module__}.{type(value).__qualname__}")
        _update_signature(hasher, state, depth=depth + 1, seen=seen)
        seen.remove(object_id)
        return
    _hash_token(hasher, f"type:{type(value).__module__}.{type(value).__qualname__}")


def model_reuse_fingerprint(
    model: Any,
    configuration_a: Any = "",
    configuration_b: Any = "",
) -> tuple[str, dict[str, str]]:
    """Fingerprint the effective patch stack and its observable components.

    The main hash is intentionally fed in the same order and with the same
    values as the original v1 signature. Component hashes are calculated in
    parallel and are diagnostic only; they never participate in cache choice.
    """
    base_model = getattr(model, "model", None)
    hasher = hashlib.sha256()
    components: dict[str, str] = {}
    _hash_token(hasher, "soya-stable-model-reuse-v1")

    class _TeeHasher:
        def __init__(self, *targets):
            self.targets = targets

        def update(self, payload: bytes) -> None:
            for target in self.targets:
                target.update(payload)

    def add_token_component(name: str, token: Any) -> None:
        component_hasher = hashlib.sha256()
        _hash_token(_TeeHasher(hasher, component_hasher), token)
        components[name] = component_hasher.hexdigest()

    def add_value_component(name: str, value: Any) -> None:
        component_hasher = hashlib.sha256()
        _update_signature(_TeeHasher(hasher, component_hasher), value)
        components[name] = component_hasher.hexdigest()

    add_token_component("base_object", f"base_object:{id(base_model)}")
    add_token_component(
        "clone_base_uuid",
        f"clone_base_uuid:{getattr(model, 'clone_base_uuid', None)}",
    )
    add_token_component(
        "load_device",
        f"load_device:{getattr(model, 'load_device', None)}",
    )
    add_token_component(
        "offload_device",
        f"offload_device:{getattr(model, 'offload_device', None)}",
    )
    add_token_component(
        "weight_inplace_update",
        f"weight_inplace_update:{getattr(model, 'weight_inplace_update', None)}",
    )
    add_value_component("patches", getattr(model, "patches", {}) or {})
    add_value_component("model_options", getattr(model, "model_options", {}) or {})
    add_value_component("object_patches", getattr(model, "object_patches", {}) or {})
    add_value_component(
        "weight_wrapper_patches",
        getattr(model, "weight_wrapper_patches", {}) or {},
    )
    add_value_component("wrappers", getattr(model, "wrappers", {}) or {})
    add_value_component("callbacks", getattr(model, "callbacks", {}) or {})
    add_value_component("injections", getattr(model, "injections", {}) or {})
    add_value_component("attachments", getattr(model, "attachments", {}) or {})
    additional_models = getattr(model, "additional_models", {}) or {}
    additional_identity = {
        str(key): [str(getattr(item, "clone_base_uuid", None)) for item in items]
        for key, items in additional_models.items()
    }
    add_value_component("additional_models", additional_identity)
    add_token_component(
        "configuration_a",
        _canonical_configuration(configuration_a),
    )
    add_token_component(
        "configuration_b",
        _canonical_configuration(configuration_b),
    )
    return hasher.hexdigest(), components


def model_reuse_signature(
    model: Any,
    configuration_a: Any = "",
    configuration_b: Any = "",
) -> str:
    """Fingerprint the effective patch stack without using ``patches_uuid``."""
    signature, _components = model_reuse_fingerprint(
        model,
        configuration_a,
        configuration_b,
    )
    return signature


def clear_stable_model_reuse_cache() -> None:
    """Clear all product cache entries. Intended for shutdown/tests."""
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE.clear()


class SoyaStableModelPatcherReuse_mdsoya:
    """Reuse one fully patched ModelPatcher while its configuration is stable."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "cache_scope": ("STRING", {"default": "soya_model"}),
                "configuration_a": ("STRING", {"default": "", "multiline": True}),
                "configuration_b": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "reuse"
    CATEGORY = "Soya/Model"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def reuse(self, model, cache_scope, configuration_a, configuration_b):
        scope = str(cache_scope or "").strip()
        if not scope:
            error = ValueError("cache_scope is empty")
            print(f"{_LOG_PREFIX} reuse failed: scope={scope!r}, error={error}")
            raise error
        try:
            signature, components = model_reuse_fingerprint(
                model,
                configuration_a,
                configuration_b,
            )
            incoming_uuid = str(getattr(model, "patches_uuid", None))
            with _MODEL_CACHE_LOCK:
                cached = _MODEL_CACHE.get(scope)
                if cached is not None and cached.signature == signature:
                    chosen = cached.model
                    cache_state = "hit"
                else:
                    chosen = model
                    cache_state = "miss" if cached is None else "replace"
                    _MODEL_CACHE[scope] = _CachedModel(signature, components, model)
                previous_signature = cached.signature if cached is not None else ""
                changed_components = (
                    [
                        name
                        for name, digest in components.items()
                        if cached.components.get(name) != digest
                    ]
                    if cached is not None
                    else []
                )
            chosen_uuid = str(getattr(chosen, "patches_uuid", None))
            trace = {
                "schema_version": 1,
                "scope": scope,
                "cache_state": cache_state,
                "signature": signature,
                "previous_signature": previous_signature,
                "incoming_patches_uuid": incoming_uuid,
                "chosen_patches_uuid": chosen_uuid,
                "incoming_is_chosen": chosen is model,
                "changed_components": changed_components,
                "components": components,
            }
            print(
                f"{_LOG_PREFIX} scope={scope!r}, cache={cache_state}, "
                f"signature={signature[:16]}, incoming_uuid={incoming_uuid}, "
                f"chosen_uuid={chosen_uuid}, "
                f"changed_components={changed_components or ['none']}"
            )
            return {
                "ui": {"stable_model_reuse": [trace]},
                "result": (chosen,),
            }
        except Exception as exc:
            print(
                f"{_LOG_PREFIX} reuse failed: scope={scope!r}, "
                f"incoming_uuid={getattr(model, 'patches_uuid', None)}, "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise
