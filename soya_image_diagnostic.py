"""Low-interference probes used by the production image-corruption diagnostic.

The probe nodes only observe and pass through their inputs.  They intentionally
avoid wrapping the diffusion model or replacing the sampler implementation so
the diagnostic keeps the same execution path as normal asset generation.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import traceback
from typing import Any

import torch

import comfy.utils


_LOG_PREFIX = "[Soya:ImageDiagnosticProbe] "
_STABLE_MODELS: dict[str, dict[str, Any]] = {}
_STABLE_MODELS_LOCK = threading.Lock()


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
    result: dict[str, Any] = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "numel": int(value.numel()),
    }
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
        return (image,)
