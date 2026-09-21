"""Low-interference probes used by the production image-corruption diagnostic.

The probe nodes only observe and pass through their inputs.  They intentionally
avoid wrapping the diffusion model or replacing the sampler implementation so
the diagnostic keeps the same execution path as normal asset generation.
"""

from __future__ import annotations

import hashlib
import functools
import json
import math
import sys
import threading
import time
import traceback
from collections import deque
from typing import Any

import torch

import comfy.utils


_LOG_PREFIX = "[Soya:ImageDiagnosticProbe] "
_STABLE_MODELS: dict[str, dict[str, Any]] = {}
_STABLE_MODELS_LOCK = threading.Lock()
_LIFECYCLE_LOCK = threading.RLock()
_LIFECYCLE_PRELUDE: deque[dict[str, Any]] = deque(maxlen=512)
_LIFECYCLE_RUN_KEY: str | None = None
_LIFECYCLE_ACTIVE = False
_LIFECYCLE_ARMED = False
_LIFECYCLE_SEQUENCE = 0
_LIFECYCLE_HOOKS_INSTALLED = False
_LIFECYCLE_WRAPPER_MARKER = "_soya_image_diagnostic_lifecycle_wrapper"


def _uuid_text(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _json_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return str(value)
    return repr(value)


def _emit(event: dict[str, Any]) -> None:
    print(_LOG_PREFIX + json.dumps(event, ensure_ascii=False, sort_keys=True))


def _diagnostic_trace_failure(
    context: str,
    exc: BaseException,
    state: Any = None,
) -> None:
    print(
        "[Soya:ImageDiagnosticProbe] lifecycle trace failed: "
        f"context={context}, state={state!r}, "
        f"error={type(exc).__name__}: {exc}"
    )
    traceback.print_exc()


def _mapping_key_digest(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    encoded = "\0".join(sorted(str(key) for key in value)).encode(
        "utf-8", errors="backslashreplace"
    )
    return hashlib.sha256(encoded).hexdigest()


def _parent_chain(model: Any, *, limit: int = 16) -> list[dict[str, Any]]:
    output = []
    seen = set()
    current = model
    while current is not None and len(output) < limit:
        marker = id(current)
        if marker in seen:
            output.append({"patcher_id": marker, "cycle": True})
            break
        seen.add(marker)
        output.append(
            {
                "patcher_id": marker,
                "patches_uuid": _uuid_text(getattr(current, "patches_uuid", None)),
            }
        )
        current = getattr(current, "parent", None)
    if current is not None and len(output) >= limit:
        output.append({"truncated": True})
    return output


def _is_dynamic_model(model: Any) -> bool | None:
    check = getattr(model, "is_dynamic", None)
    if not callable(check):
        return None
    try:
        return bool(check())
    except Exception as exc:
        _diagnostic_trace_failure("is_dynamic", exc)
        return None


def _model_lifecycle_snapshot(model: Any) -> dict[str, Any]:
    if model is None:
        return {"patcher": None}
    base_model = getattr(model, "model", None)
    backup = getattr(model, "backup", None)
    patches = getattr(model, "patches", None)
    patch_entry_count = None
    if isinstance(patches, dict):
        try:
            patch_entry_count = sum(len(value or []) for value in patches.values())
        except Exception as exc:
            _diagnostic_trace_failure("patch_entry_count", exc)
    result = {
        "patcher_id": id(model),
        "patcher_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "base_model_id": id(base_model) if base_model is not None else None,
        "base_model_class": (
            f"{type(base_model).__module__}.{type(base_model).__qualname__}"
            if base_model is not None
            else None
        ),
        "clone_base_uuid": _uuid_text(getattr(model, "clone_base_uuid", None)),
        "patches_uuid": _uuid_text(getattr(model, "patches_uuid", None)),
        "resident_patch_uuid": _uuid_text(
            getattr(base_model, "current_weight_patches_uuid", None)
        ),
        "parent_patcher_id": (
            id(getattr(model, "parent", None))
            if getattr(model, "parent", None) is not None
            else None
        ),
        "backup_id": id(backup) if isinstance(backup, dict) else None,
        "backup_count": len(backup) if isinstance(backup, dict) else None,
        "backup_keys_sha256": _mapping_key_digest(backup),
        "patches_id": id(patches) if isinstance(patches, dict) else None,
        "patch_key_count": len(patches) if isinstance(patches, dict) else None,
        "patch_entry_count": patch_entry_count,
        "patch_keys_sha256": _mapping_key_digest(patches),
        "loaded_weight_memory": int(
            getattr(base_model, "model_loaded_weight_memory", 0) or 0
        ),
        "offload_buffer_memory": int(
            getattr(base_model, "model_offload_buffer_memory", 0) or 0
        ),
        "model_lowvram": bool(getattr(base_model, "model_lowvram", False)),
        "lowvram_patch_counter": int(
            getattr(base_model, "lowvram_patch_counter", 0) or 0
        ),
        "model_device": str(getattr(base_model, "device", None)),
        "load_device": str(getattr(model, "load_device", None)),
        "offload_device": str(getattr(model, "offload_device", None)),
        "weight_inplace_update": bool(
            getattr(model, "weight_inplace_update", False)
        ),
        "is_clip": bool(getattr(model, "is_clip", False)),
        "is_dynamic": _is_dynamic_model(model),
        "python_refcount": sys.getrefcount(model),
    }
    return result


def _loaded_model_lifecycle_snapshot(loaded: Any) -> dict[str, Any]:
    if hasattr(loaded, "_model"):
        try:
            model = loaded.model
        except Exception as exc:
            _diagnostic_trace_failure("loaded_model.model", exc)
            model = None
    else:
        model = None
    real_ref = getattr(loaded, "real_model", None)
    try:
        real_model = real_ref() if callable(real_ref) else None
    except Exception as exc:
        _diagnostic_trace_failure("loaded_model.real_model", exc)
        real_model = None
    model_finalizer = getattr(loaded, "model_finalizer", None)
    patcher_finalizer = getattr(loaded, "_patcher_finalizer", None)
    return {
        "loaded_model_id": id(loaded),
        "device": str(getattr(loaded, "device", None)),
        "currently_used": bool(getattr(loaded, "currently_used", False)),
        "real_model_id": id(real_model) if real_model is not None else None,
        "model_finalizer_alive": (
            bool(getattr(model_finalizer, "alive", False))
            if model_finalizer is not None
            else None
        ),
        "patcher_finalizer_alive": (
            bool(getattr(patcher_finalizer, "alive", False))
            if patcher_finalizer is not None
            else None
        ),
        "patcher": _model_lifecycle_snapshot(model),
    }


def _loaded_registry_snapshot() -> list[dict[str, Any]]:
    try:
        import comfy.model_management as model_management

        return [
            _loaded_model_lifecycle_snapshot(value)
            for value in list(getattr(model_management, "current_loaded_models", []))
        ]
    except Exception as exc:
        _diagnostic_trace_failure("loaded_registry_snapshot", exc)
        return [{"error": f"{type(exc).__name__}: {exc}"}]


def _cuda_stream_snapshot() -> dict[str, Any]:
    try:
        import comfy.model_management as model_management

        result: dict[str, Any] = {
            "num_offload_streams": int(getattr(model_management, "NUM_STREAMS", 0)),
            "stream_counters": {
                str(key): int(value)
                for key, value in dict(
                    getattr(model_management, "stream_counters", {}) or {}
                ).items()
            },
            "offload_streams": {},
        }
        for device, streams in dict(
            getattr(model_management, "STREAMS", {}) or {}
        ).items():
            result["offload_streams"][str(device)] = [
                int(getattr(stream, "cuda_stream", id(stream))) for stream in streams
            ]
        if torch.cuda.is_available():
            current = torch.cuda.current_stream()
            default = torch.cuda.default_stream()
            result.update(
                {
                    "current_stream": int(
                        getattr(current, "cuda_stream", id(current))
                    ),
                    "default_stream": int(
                        getattr(default, "cuda_stream", id(default))
                    ),
                }
            )
        return result
    except Exception as exc:
        _diagnostic_trace_failure("cuda_stream_snapshot", exc)
        return {"error": f"{type(exc).__name__}: {exc}"}


def _tensor_metadata(value: torch.Tensor) -> dict[str, Any]:
    detached = value.detach()
    result = {
        "type": "tensor",
        "tensor_id": id(value),
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "numel": int(detached.numel()),
    }
    try:
        result["data_ptr"] = int(detached.data_ptr())
        result["storage_data_ptr"] = int(detached.untyped_storage().data_ptr())
        result["storage_bytes"] = int(detached.untyped_storage().nbytes())
    except Exception as exc:
        _diagnostic_trace_failure("tensor_storage_metadata", exc)
        result["storage_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _argument_metadata(value: Any, *, depth: int = 0) -> Any:
    if depth > 3:
        return {"type": type(value).__name__, "truncated": True}
    if isinstance(value, torch.Tensor):
        return _tensor_metadata(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return _json_scalar(value)
    if hasattr(value, "patches_uuid") and hasattr(value, "model"):
        return _model_lifecycle_snapshot(value)
    if isinstance(value, dict):
        if len(value) > 32:
            return {
                "type": "dict",
                "count": len(value),
                "keys_sha256": _mapping_key_digest(value),
            }
        return {
            str(key): _argument_metadata(item, depth=depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        if len(value) > 16:
            return {"type": type(value).__name__, "count": len(value)}
        return [_argument_metadata(item, depth=depth + 1) for item in value]
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "object_id": id(value),
    }


def _subject_lifecycle_snapshot(subject: Any) -> dict[str, Any]:
    if hasattr(subject, "patcher"):
        return {
            "object_id": id(subject),
            "object_class": f"{type(subject).__module__}.{type(subject).__qualname__}",
            "patcher": _model_lifecycle_snapshot(getattr(subject, "patcher", None)),
        }
    if type(subject).__name__ == "LoadedModel":
        return _loaded_model_lifecycle_snapshot(subject)
    return _model_lifecycle_snapshot(subject)


def _lifecycle_capture_enabled() -> bool:
    with _LIFECYCLE_LOCK:
        return _LIFECYCLE_ACTIVE or _LIFECYCLE_ARMED


def _record_lifecycle(payload: dict[str, Any]) -> None:
    global _LIFECYCLE_SEQUENCE
    event = {
        "event": "lifecycle_event",
        "schema_version": 1,
        "monotonic_ns": time.monotonic_ns(),
        **payload,
    }
    emit_now = False
    with _LIFECYCLE_LOCK:
        _LIFECYCLE_SEQUENCE += 1
        event["sequence"] = _LIFECYCLE_SEQUENCE
        if _LIFECYCLE_ACTIVE and _LIFECYCLE_RUN_KEY:
            event["run_key"] = _LIFECYCLE_RUN_KEY
            emit_now = True
        elif _LIFECYCLE_ARMED:
            _LIFECYCLE_PRELUDE.append(event)
    if emit_now:
        try:
            _emit(event)
        except Exception as exc:
            _diagnostic_trace_failure("emit_lifecycle_event", exc)


def _method_lifecycle_wrapper(owner: type, method_name: str) -> None:
    original = getattr(owner, method_name, None)
    if original is None or getattr(original, _LIFECYCLE_WRAPPER_MARKER, False):
        return

    @functools.wraps(original)
    def wrapped(subject, *args, **kwargs):
        if not _lifecycle_capture_enabled():
            return original(subject, *args, **kwargs)
        started = time.monotonic_ns()
        before = _subject_lifecycle_snapshot(subject)
        arguments = {
            "args": _argument_metadata(args),
            "kwargs": _argument_metadata(kwargs),
        }
        try:
            result = original(subject, *args, **kwargs)
        except Exception as exc:
            print(
                "[Soya:ImageDiagnosticProbe] traced method failed: "
                f"operation={owner.__name__}.{method_name}, "
                f"arguments={arguments!r}, before={before!r}, "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            _record_lifecycle(
                {
                    "operation": f"{owner.__name__}.{method_name}",
                    "status": "error",
                    "duration_ns": time.monotonic_ns() - started,
                    "before": before,
                    "after": _subject_lifecycle_snapshot(subject),
                    "arguments": arguments,
                    "error": f"{type(exc).__name__}: {exc}",
                    "registry": _loaded_registry_snapshot(),
                    "streams": _cuda_stream_snapshot(),
                }
            )
            raise
        _record_lifecycle(
            {
                "operation": f"{owner.__name__}.{method_name}",
                "status": "ok",
                "duration_ns": time.monotonic_ns() - started,
                "before": before,
                "after": _subject_lifecycle_snapshot(subject),
                "arguments": arguments,
                "result": _argument_metadata(result),
                "registry": _loaded_registry_snapshot(),
                "streams": _cuda_stream_snapshot(),
            }
        )
        return result

    setattr(wrapped, _LIFECYCLE_WRAPPER_MARKER, True)
    setattr(owner, method_name, wrapped)


def _management_lifecycle_wrapper(module: Any, function_name: str) -> None:
    original = getattr(module, function_name, None)
    if original is None or getattr(original, _LIFECYCLE_WRAPPER_MARKER, False):
        return

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        if not _lifecycle_capture_enabled():
            return original(*args, **kwargs)
        started = time.monotonic_ns()
        before_registry = _loaded_registry_snapshot()
        requested = []
        if function_name == "load_models_gpu":
            models = args[0] if args else kwargs.get("models", [])
            requested = [
                _model_lifecycle_snapshot(model) for model in list(models or [])
            ]
        loaded_ids = {
            (entry.get("patcher") or {}).get("patcher_id")
            for entry in before_registry
        }
        loaded_base_ids = {
            (entry.get("patcher") or {}).get("base_model_id")
            for entry in before_registry
        }
        identity_hits = [
            model.get("patcher_id")
            for model in requested
            if model.get("patcher_id") in loaded_ids
        ]
        clone_conflicts = [
            model.get("patcher_id")
            for model in requested
            if model.get("patcher_id") not in loaded_ids
            and model.get("base_model_id") in loaded_base_ids
        ]
        try:
            result = original(*args, **kwargs)
        except Exception as exc:
            print(
                "[Soya:ImageDiagnosticProbe] traced model-management call failed: "
                f"operation=model_management.{function_name}, "
                f"requested={requested!r}, registry_before={before_registry!r}, "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            _record_lifecycle(
                {
                    "operation": f"model_management.{function_name}",
                    "status": "error",
                    "duration_ns": time.monotonic_ns() - started,
                    "arguments": _argument_metadata({"args": args, "kwargs": kwargs}),
                    "requested": requested,
                    "identity_hits": identity_hits,
                    "clone_conflicts": clone_conflicts,
                    "registry_before": before_registry,
                    "registry_after": _loaded_registry_snapshot(),
                    "streams": _cuda_stream_snapshot(),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            raise
        _record_lifecycle(
            {
                "operation": f"model_management.{function_name}",
                "status": "ok",
                "duration_ns": time.monotonic_ns() - started,
                "arguments": _argument_metadata({"args": args, "kwargs": kwargs}),
                "requested": requested,
                "identity_hits": identity_hits,
                "clone_conflicts": clone_conflicts,
                "registry_before": before_registry,
                "registry_after": _loaded_registry_snapshot(),
                "streams": _cuda_stream_snapshot(),
                "result": _argument_metadata(result),
            }
        )
        return result

    setattr(wrapped, _LIFECYCLE_WRAPPER_MARKER, True)
    setattr(module, function_name, wrapped)


def _install_lifecycle_hooks() -> None:
    global _LIFECYCLE_HOOKS_INSTALLED
    with _LIFECYCLE_LOCK:
        if _LIFECYCLE_HOOKS_INSTALLED:
            return
        try:
            import comfy.model_management as model_management
            import comfy.model_patcher as model_patcher
            import comfy.sd as comfy_sd

            patcher_classes = [model_patcher.ModelPatcher]
            dynamic_class = getattr(model_patcher, "ModelPatcherDynamic", None)
            if dynamic_class is not None and dynamic_class not in patcher_classes:
                patcher_classes.append(dynamic_class)
            for patcher_class in patcher_classes:
                for method_name in (
                    "clone",
                    "add_patches",
                    "model_patches_to",
                    "load",
                    "patch_model",
                    "unpatch_model",
                    "partially_load",
                    "partially_unload",
                    "detach",
                ):
                    if method_name in patcher_class.__dict__:
                        _method_lifecycle_wrapper(patcher_class, method_name)
            for method_name in (
                "_set_model",
                "_switch_parent",
                "model_load",
                "model_unload",
            ):
                _method_lifecycle_wrapper(model_management.LoadedModel, method_name)
            for method_name in ("load_models_gpu", "free_memory"):
                _management_lifecycle_wrapper(model_management, method_name)
            for method_name in ("load_model", "encode_from_tokens"):
                _method_lifecycle_wrapper(comfy_sd.CLIP, method_name)
            _LIFECYCLE_HOOKS_INSTALLED = True
        except Exception as exc:
            _diagnostic_trace_failure("install_lifecycle_hooks", exc)
            raise


def _start_lifecycle_trace(run_key: str, model: Any, anchor: str) -> None:
    global _LIFECYCLE_ACTIVE, _LIFECYCLE_ARMED, _LIFECYCLE_RUN_KEY
    _install_lifecycle_hooks()
    normalized_run_key = str(run_key or "").strip()
    if not normalized_run_key:
        raise ValueError("Lifecycle trace run_key is empty")
    with _LIFECYCLE_LOCK:
        if _LIFECYCLE_ACTIVE and _LIFECYCLE_RUN_KEY == normalized_run_key:
            same_run = True
            prelude = []
        else:
            same_run = False
            previous_run_key = _LIFECYCLE_RUN_KEY if _LIFECYCLE_ACTIVE else None
            prelude = list(_LIFECYCLE_PRELUDE)
            _LIFECYCLE_PRELUDE.clear()
            _LIFECYCLE_RUN_KEY = normalized_run_key
            _LIFECYCLE_ACTIVE = True
            _LIFECYCLE_ARMED = True
    if same_run:
        _record_lifecycle(
            {
                "operation": "trace.anchor",
                "status": "ok",
                "anchor": anchor,
                "model": _model_lifecycle_snapshot(model),
            }
        )
        return
    if previous_run_key:
        abandoned = {
            "event": "lifecycle_event",
            "schema_version": 1,
            "operation": "trace.previous_run_abandoned",
            "status": "warning",
            "run_key": previous_run_key,
            "next_run_key": normalized_run_key,
            "monotonic_ns": time.monotonic_ns(),
        }
        try:
            _emit(abandoned)
        except Exception as exc:
            _diagnostic_trace_failure("emit_abandoned_trace", exc)
    for event in prelude:
        replay = dict(event)
        replay["run_key"] = normalized_run_key
        replay["prelude"] = True
        try:
            _emit(replay)
        except Exception as exc:
            _diagnostic_trace_failure("emit_lifecycle_prelude", exc)
    _record_lifecycle(
        {
            "operation": "trace.start",
            "status": "ok",
            "anchor": anchor,
            "prelude_event_count": len(prelude),
            "model": _model_lifecycle_snapshot(model),
            "parent_chain": _parent_chain(model),
            "registry": _loaded_registry_snapshot(),
            "streams": _cuda_stream_snapshot(),
        }
    )


def _stop_lifecycle_trace(run_key: str, stage: str) -> None:
    global _LIFECYCLE_ACTIVE
    normalized_run_key = str(run_key or "").strip()
    with _LIFECYCLE_LOCK:
        matches = _LIFECYCLE_ACTIVE and _LIFECYCLE_RUN_KEY == normalized_run_key
    if not matches:
        return
    _record_lifecycle(
        {
            "operation": "trace.stop",
            "status": "ok",
            "stage": stage,
            "registry": _loaded_registry_snapshot(),
            "streams": _cuda_stream_snapshot(),
        }
    )
    with _LIFECYCLE_LOCK:
        if _LIFECYCLE_RUN_KEY == normalized_run_key:
            _LIFECYCLE_ACTIVE = False


def _patch_payload_shape(value: Any, *, depth: int = 0) -> Any:
    """Describe patch semantics without serializing complete LoRA tensors."""
    if depth > 6:
        return {"type": type(value).__name__, "truncated": True}
    if isinstance(value, torch.Tensor):
        return {
            "type": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if value is None or isinstance(value, (bool, int, float, str)):
        return _json_scalar(value)
    if isinstance(value, (list, tuple)):
        return [
            _patch_payload_shape(item, depth=depth + 1)
            for item in value
        ]
    if isinstance(value, dict):
        return {
            str(key): _patch_payload_shape(item, depth=depth + 1)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if callable(value):
        return {
            "type": "callable",
            "name": f"{getattr(value, '__module__', '')}.{getattr(value, '__qualname__', type(value).__qualname__)}",
        }
    state = getattr(value, "__dict__", None)
    if isinstance(state, dict):
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "state": _patch_payload_shape(state, depth=depth + 1),
        }
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _patch_structure(model: Any) -> dict[str, Any]:
    patches = getattr(model, "patches", {}) or {}
    entries = []
    entry_count = 0
    for key in sorted(patches, key=str):
        patch_list = patches.get(key) or []
        entry_count += len(patch_list)
        entries.append(
            {
                "key": str(key),
                "patches": _patch_payload_shape(patch_list),
            }
        )
    encoded = json.dumps(
        entries,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "patch_key_count": len(patches),
        "patch_entry_count": entry_count,
        "patch_structure_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _tensor_sample(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach()
    storage = _tensor_metadata(value)
    result: dict[str, Any] = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "numel": int(value.numel()),
        "tensor_id": storage.get("tensor_id"),
        "data_ptr": storage.get("data_ptr"),
        "storage_data_ptr": storage.get("storage_data_ptr"),
        "storage_bytes": storage.get("storage_bytes"),
    }
    if storage.get("storage_error"):
        result["storage_error"] = storage["storage_error"]
    if value.numel() == 0:
        result["samples"] = []
        return result
    flat = value.reshape(-1)
    stride = max(1, int(flat.numel() // 4096))
    sampled = flat[::stride][:4096]
    finite = torch.isfinite(sampled)
    nan_count = int(torch.isnan(sampled).sum().item())
    inf_count = int(torch.isinf(sampled).sum().item())
    finite_count = int(finite.sum().item())
    result.update(
        {
            "sampled_numel": int(sampled.numel()),
            "sampled_stride": stride,
            "sampled_finite_count": finite_count,
            "sampled_nan_count": nan_count,
            "sampled_inf_count": inf_count,
        }
    )
    if finite_count:
        finite_values = sampled[finite].to(device="cpu", dtype=torch.float32)
        result.update(
            {
                "sampled_min": float(finite_values.min().item()),
                "sampled_max": float(finite_values.max().item()),
                "sampled_mean": float(finite_values.mean().item()),
            }
        )
    indices = sorted({0, int(flat.numel() // 2), int(flat.numel() - 1)})
    scalar_samples = flat[indices].to(device="cpu", dtype=torch.float64)
    result["samples"] = [
        _json_scalar(float(item)) for item in scalar_samples.tolist()
    ]
    return result


def _sample_model_weights(model: Any, *, limit: int = 12) -> dict[str, Any]:
    patches = getattr(model, "patches", {}) or {}
    keys = sorted(patches, key=str)
    if not keys:
        return {"sampled": 0, "weights": []}
    if len(keys) <= limit:
        selected = keys
    else:
        selected = []
        for index in range(limit):
            position = round(index * (len(keys) - 1) / (limit - 1))
            selected.append(keys[position])
        selected = list(dict.fromkeys(selected))
    base_model = getattr(model, "model", None)
    samples = []
    for key in selected:
        try:
            weight = comfy.utils.get_attr(base_model, key)
            if not isinstance(weight, torch.Tensor):
                samples.append({"key": str(key), "type": type(weight).__name__})
                continue
            samples.append({"key": str(key), **_tensor_sample(weight)})
        except Exception as exc:
            print(
                "[Soya:ImageDiagnosticProbe] model weight sample failed: "
                f"key={key}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            samples.append(
                {"key": str(key), "error": f"{type(exc).__name__}: {exc}"}
            )
    encoded = json.dumps(
        samples,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "sampled": len(samples),
        "weights_sha256": hashlib.sha256(encoded).hexdigest(),
        "weights": samples,
    }


def _cuda_snapshot() -> dict[str, Any]:
    result: dict[str, Any] = {
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if not torch.cuda.is_available():
        return result
    try:
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        result.update(
            {
                "device_index": int(index),
                "device_name": str(properties.name),
                "device_capability": list(torch.cuda.get_device_capability(index)),
                "bf16_supported": bool(torch.cuda.is_bf16_supported()),
                "memory_allocated": int(torch.cuda.memory_allocated(index)),
                "memory_reserved": int(torch.cuda.memory_reserved(index)),
                "max_memory_allocated": int(torch.cuda.max_memory_allocated(index)),
                "max_memory_reserved": int(torch.cuda.max_memory_reserved(index)),
                "matmul_precision": str(torch.get_float32_matmul_precision()),
                "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
                "cudnn_version": torch.backends.cudnn.version(),
            }
        )
    except Exception as exc:
        print(
            "[Soya:ImageDiagnosticProbe] CUDA snapshot failed: "
            f"error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _model_snapshot(model: Any, *, sample_weights: bool = True) -> dict[str, Any]:
    base_model = getattr(model, "model", None)
    backup = getattr(model, "backup", {}) or {}
    structure = _patch_structure(model)
    snapshot = {
        "patcher_id": id(model),
        "model_id": id(base_model),
        "clone_base_uuid": _uuid_text(getattr(model, "clone_base_uuid", None)),
        "patches_uuid": _uuid_text(getattr(model, "patches_uuid", None)),
        "current_weight_patches_uuid": _uuid_text(
            getattr(base_model, "current_weight_patches_uuid", None)
        ),
        "backup_count": len(backup),
        "backup_id": id(backup),
        "backup_keys_sha256": _mapping_key_digest(backup),
        "patches_id": id(getattr(model, "patches", None)),
        "patch_keys_sha256": _mapping_key_digest(getattr(model, "patches", None)),
        "parent_chain": _parent_chain(model),
        "object_patch_count": len(getattr(model, "object_patches", {}) or {}),
        "weight_wrapper_patch_count": len(
            getattr(model, "weight_wrapper_patches", {}) or {}
        ),
        "model_option_keys": sorted((getattr(model, "model_options", {}) or {}).keys()),
        "weight_inplace_update": bool(
            getattr(model, "weight_inplace_update", False)
        ),
        "model_lowvram": bool(getattr(base_model, "model_lowvram", False)),
        "lowvram_patch_counter": int(
            getattr(base_model, "lowvram_patch_counter", 0) or 0
        ),
        "model_loaded_weight_memory": int(
            getattr(base_model, "model_loaded_weight_memory", 0) or 0
        ),
        "model_device": str(getattr(base_model, "device", None)),
        **structure,
    }
    if sample_weights:
        snapshot["patched_weight_samples"] = _sample_model_weights(model)
    return snapshot


def _walk_tensors(value: Any, path: str, output: list[tuple[str, torch.Tensor]]) -> None:
    if isinstance(value, torch.Tensor):
        output.append((path, value))
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _walk_tensors(item, f"{path}.{key}", output)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _walk_tensors(item, f"{path}[{index}]", output)


def _tensor_stats(value: Any) -> dict[str, Any]:
    tensors: list[tuple[str, torch.Tensor]] = []
    _walk_tensors(value, "root", tensors)
    results = []
    total_values = 0
    total_nonfinite = 0
    for path, tensor in tensors:
        detached = tensor.detach()
        count = int(detached.numel())
        total_values += count
        entry: dict[str, Any] = {
            "path": path,
            "shape": list(detached.shape),
            "dtype": str(detached.dtype),
            "device": str(detached.device),
            "numel": count,
        }
        if count:
            finite = torch.isfinite(detached)
            finite_count = int(finite.sum().item())
            nan_count = int(torch.isnan(detached).sum().item())
            inf_count = int(torch.isinf(detached).sum().item())
            nonfinite = count - finite_count
            total_nonfinite += nonfinite
            entry.update(
                {
                    "finite_count": finite_count,
                    "nan_count": nan_count,
                    "inf_count": inf_count,
                    "nonfinite_count": nonfinite,
                }
            )
            if finite_count:
                finite_values = detached[finite].to(dtype=torch.float32)
                entry.update(
                    {
                        "finite_min": float(finite_values.min().item()),
                        "finite_max": float(finite_values.max().item()),
                        "finite_mean": float(finite_values.mean().item()),
                    }
                )
        results.append(entry)
    return {
        "tensor_count": len(results),
        "total_values": total_values,
        "total_nonfinite": total_nonfinite,
        "all_finite": total_nonfinite == 0,
        "tensors": results,
    }


class SoyaDiagnosticStableModelReuse_mdsoya:
    """Pin the first fully LoRA-patched ModelPatcher for one diagnostic case."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "session_key": ("STRING", {"default": ""}),
                "run_key": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "reuse"
    CATEGORY = "Soya/Diagnostic"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def reuse(self, model, session_key, run_key):
        try:
            incoming = _model_snapshot(model, sample_weights=False)
            with _STABLE_MODELS_LOCK:
                cached = _STABLE_MODELS.get(session_key)
                if cached is None:
                    cached = {
                        "model": model,
                        "structure": {
                            key: incoming[key]
                            for key in (
                                "clone_base_uuid",
                                "patch_key_count",
                                "patch_entry_count",
                                "patch_structure_sha256",
                            )
                        },
                    }
                    _STABLE_MODELS[session_key] = cached
                    cache_hit = False
                else:
                    cache_hit = True
                chosen = cached["model"]
            structure_matches = cached["structure"] == {
                key: incoming[key]
                for key in (
                    "clone_base_uuid",
                    "patch_key_count",
                    "patch_entry_count",
                    "patch_structure_sha256",
                )
            }
            event = {
                "event": "stable_model_reuse",
                "run_key": run_key,
                "session_key": session_key,
                "cache_hit": cache_hit,
                "structure_matches": structure_matches,
                "incoming": incoming,
                "chosen": _model_snapshot(chosen, sample_weights=False),
                "cuda": _cuda_snapshot(),
            }
            _emit(event)
            if not structure_matches:
                raise RuntimeError(
                    "Stable ModelPatcher diagnostic received a different patch "
                    f"structure: session={session_key}, run={run_key}, "
                    f"cached={cached['structure']}, incoming={incoming}"
                )
            return (chosen,)
        except Exception as exc:
            print(
                "[Soya:ImageDiagnosticProbe] stable model reuse failed: "
                f"session={session_key}, run={run_key}, "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise


class SoyaDiagnosticModelProbe_mdsoya:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "stage": ("STRING", {"default": "model_before_sampler"}),
                "run_key": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "probe"
    CATEGORY = "Soya/Diagnostic"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def probe(self, model, stage, run_key):
        try:
            _start_lifecycle_trace(run_key, model, stage)
            _emit(
                {
                    "event": "model_state",
                    "stage": stage,
                    "run_key": run_key,
                    "model": _model_snapshot(model),
                    "cuda": _cuda_snapshot(),
                }
            )
        except Exception as exc:
            print(
                "[Soya:ImageDiagnosticProbe] model probe failed; passing through: "
                f"stage={stage}, run={run_key}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            _emit(
                {
                    "event": "probe_error",
                    "probe": "model",
                    "stage": stage,
                    "run_key": run_key,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        return (model,)


class SoyaDiagnosticConditioningProbe_mdsoya:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "stage": ("STRING", {"default": "conditioning"}),
                "run_key": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "probe"
    CATEGORY = "Soya/Diagnostic"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def probe(self, conditioning, stage, run_key):
        try:
            _emit(
                {
                    "event": "tensor_state",
                    "kind": "conditioning",
                    "stage": stage,
                    "run_key": run_key,
                    "stats": _tensor_stats(conditioning),
                    "cuda": _cuda_snapshot(),
                }
            )
        except Exception as exc:
            print(
                "[Soya:ImageDiagnosticProbe] conditioning probe failed; passing through: "
                f"stage={stage}, run={run_key}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            _emit(
                {
                    "event": "probe_error",
                    "probe": "conditioning",
                    "stage": stage,
                    "run_key": run_key,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        return (conditioning,)


class SoyaDiagnosticLatentProbe_mdsoya:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "model": ("MODEL",),
                "stage": ("STRING", {"default": "latent"}),
                "run_key": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "probe"
    CATEGORY = "Soya/Diagnostic"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def probe(self, latent, model, stage, run_key):
        try:
            _emit(
                {
                    "event": "tensor_state",
                    "kind": "latent",
                    "stage": stage,
                    "run_key": run_key,
                    "stats": _tensor_stats(latent),
                    "model": _model_snapshot(model),
                    "cuda": _cuda_snapshot(),
                }
            )
        except Exception as exc:
            print(
                "[Soya:ImageDiagnosticProbe] latent probe failed; passing through: "
                f"stage={stage}, run={run_key}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            _emit(
                {
                    "event": "probe_error",
                    "probe": "latent",
                    "stage": stage,
                    "run_key": run_key,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        return (latent,)


class SoyaDiagnosticImageProbe_mdsoya:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "stage": ("STRING", {"default": "vae_output"}),
                "run_key": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "probe"
    CATEGORY = "Soya/Diagnostic"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def probe(self, image, stage, run_key):
        try:
            _emit(
                {
                    "event": "tensor_state",
                    "kind": "image",
                    "stage": stage,
                    "run_key": run_key,
                    "stats": _tensor_stats(image),
                    "cuda": _cuda_snapshot(),
                }
            )
        except Exception as exc:
            print(
                "[Soya:ImageDiagnosticProbe] image probe failed; passing through: "
                f"stage={stage}, run={run_key}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            _emit(
                {
                    "event": "probe_error",
                    "probe": "image",
                    "stage": stage,
                    "run_key": run_key,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        finally:
            _stop_lifecycle_trace(run_key, stage)
        return (image,)
