"""Conditioning-scoped LoRA residuals without merged copies of model weights."""

import copy
import math
import os
import traceback
from functools import lru_cache, partial

import torch

import comfy.hooks
import comfy.ops
import comfy.patcher_extension
import comfy.sd
import comfy.utils
from comfy.weight_adapter.bypass import get_module_type_info
from comfy.weight_adapter.lora import LoRAAdapter


_INJECTION_KEY = "soya_character_lora"


@lru_cache(maxsize=8)
def _load_character_lora_file(path, modified_ns, size):
    # Only CPU tensors live here. Replacing a file invalidates its entry.
    return comfy.utils.load_torch_file(path, safe_load=True)


def load_character_lora_file(path):
    path = os.path.normcase(os.path.realpath(path))
    stat = os.stat(path)
    return _load_character_lora_file(path, stat.st_mtime_ns, stat.st_size)


class _CharacterLoRAHook(comfy.hooks.TransformerOptionsHook):
    def __init__(self, patches=None, always_apply=False):
        super().__init__({}, hook_scope=comfy.hooks.EnumHookScope.HookedOnly)
        self.patches = patches
        self.always_apply = always_apply

    def clone(self):
        cloned = super().clone()
        cloned.patches = self.patches
        cloned.always_apply = self.always_apply
        return cloned


def is_linear_projection(module):
    # ComfyUI mixed-precision Linear exposes the same feature dimensions and
    # forward hooks as nn.Linear, but inherits from nn.Module instead.
    return (isinstance(module, (torch.nn.Linear, comfy.ops.MixedPrecisionOp))
            and hasattr(module, "in_features") and hasattr(module, "out_features"))


def _merged_path_reason(model, patches):
    for key, entries in patches.items():
        if not key.endswith(".weight"):
            return f"{key}: residual path requires a linear or convolution weight"
        module = model.get_model_object(key[:-7])
        is_conv = isinstance(module, torch.nn.Conv2d)
        if not is_conv and not is_linear_projection(module):
            return f"{key}: {type(module).__name__} uses the native weight adapter"
        if is_conv and (module.groups != 1 or module.padding_mode != "zeros"):
            return f"{key}: grouped/nonzero-padding convolution needs native application"
        for strength, adapter, model_strength, offset, function in entries:
            if not isinstance(adapter, LoRAAdapter):
                return f"{key}: {type(adapter).__name__} uses the native weight adapter"
            up, down, alpha, mid, dora, reshape = adapter.weights
            if mid is not None or dora is not None or reshape is not None:
                return f"{key}: mid/DoRA/reshape needs the native weight adapter"
            if is_conv:
                # Ordinary SDXL LoRA: spatial down projection, then 1x1 up.
                # Flattened kernels are also accepted by Comfy's adapter.h().
                rank = down.shape[0]
                down_shape = (rank, module.in_channels, *module.kernel_size)
                up_shape = (module.out_channels, rank, 1, 1)
                flat_down_shape = (rank, module.in_channels * math.prod(module.kernel_size))
                if tuple(down.shape) not in (down_shape, flat_down_shape) or tuple(up.shape) not in (up_shape, up_shape[:2]):
                    return f"{key}: nonstandard convolution LoRA needs native application"
            elif up.ndim != 2 or down.ndim != 2:
                return f"{key}: non-matrix LoRA needs the native weight adapter"
            if model_strength != 1.0 or offset is not None or function is not None:
                return f"{key}: transformed weight patch needs native application"
    return None


def load_character_lora(model, lora, strength):
    """Keep native key mapping and use residuals for ordinary LoRAs."""
    patched, _, group = comfy.hooks.load_hook_lora_for_models(model, None, lora, strength, 0.0)
    weight_hook = group.hooks[0]
    patches = patched.hook_patches[weight_hook.hook_ref]
    reason = _merged_path_reason(patched, patches)
    if reason is not None or not patches:
        print(f"[character LoRA] Native weight path: {reason or 'no matching model weights'}, strength={strength}")
        return patched, group

    hook = _CharacterLoRAHook(patches)
    group = comfy.hooks.HookGroup()
    group.add(hook)
    del patched.hook_patches[weight_hook.hook_ref]
    patched.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, _INJECTION_KEY)
    patched.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, _INJECTION_KEY, _sample_with_character_loras
    )
    print(f"[character LoRA] Unmerged residuals: layers={len(patches)}, strength={strength}")
    return patched, group


def load_character_lora_unmerged(model, lora, strength):
    """Apply an ordinary global character adapter without replacing base weights.

    Use the installed native loader for key mapping (including Anima remapping).
    Unsupported weight transformations retain native application for correctness.
    """
    if strength == 0:
        return model
    mapped, _ = comfy.sd.load_lora_for_models(model, None, lora, strength, 0.0)
    patches = {
        key: entries[len(model.patches.get(key, ())):]
        for key, entries in mapped.patches.items()
        if len(entries) > len(model.patches.get(key, ()))
    }
    if not patches:
        return model
    reason = _merged_path_reason(mapped, patches)
    if reason is not None:
        print(f"[character LoRA] Native weight path: {reason}, strength={strength}")
        return mapped
    patched = model.clone()
    options = patched.model_options.setdefault("transformer_options", {})
    options[_INJECTION_KEY] = (*options.get(_INJECTION_KEY, ()),
                               _CharacterLoRAHook(patches, always_apply=True))
    patched.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, _INJECTION_KEY)
    patched.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, _INJECTION_KEY, _sample_with_character_loras
    )
    print(f"[character LoRA] Base weights retained: layers={len(patches)}, strength={strength}")
    return patched


def _add_character_residual(patcher, adapters, module, args, kwargs, output):
    active = patcher.current_hooks
    for hook, adapter, strength in adapters:
        if hook.always_apply or (active is not None and active.contains(hook)):
            adapter.multiplier = strength * hook.strength
            x = args[0] if args else kwargs["input"]
            output = output + adapter.h(x, output)
    return output


class _CharacterLoRARuntime:
    """Own device copies and layer hooks for exactly one sampling execution."""

    def __init__(self, hooks):
        self.hooks = hooks
        self.states = {}

    def inject(self, patcher):
        state = self.states.get(patcher)
        if state is None:
            state = {"layers": {}, "handles": []}
            self.states[patcher] = state
            dtype = patcher.model.get_dtype_inference()
            for hook in self.hooks:
                for key, patches in hook.patches.items():
                    module = patcher.get_model_object(key[:-7])
                    module_info = get_module_type_info(module)
                    adapters = state["layers"].setdefault(module, [])
                    for strength, source, _, _, _ in patches:
                        adapter = copy.copy(source)
                        for name in ("is_conv", "conv_dim", "kernel_size", "in_channels", "out_channels"):
                            setattr(adapter, name, module_info[name])
                        adapter.kw_dict = {
                            name: module_info[name] for name in ("stride", "padding", "dilation", "groups")
                        } if module_info["is_conv"] else {}
                        adapter.weights = tuple(
                            value.to(device=patcher.load_device, dtype=dtype)
                            if torch.is_tensor(value) else value
                            for value in source.weights
                        )
                        adapters.append((hook, adapter, strength))
        # Native weight hooks temporarily eject injections. Keep the small
        # device tensors through those switches; release them at sample exit.
        for module, adapters in state["layers"].items():
            state["handles"].append(module.register_forward_hook(
                partial(_add_character_residual, patcher, adapters), with_kwargs=True
            ))

    def eject(self, patcher):
        state = self.states.get(patcher)
        if state is not None:
            for handle in state["handles"]:
                handle.remove()
            state["handles"].clear()

    def close(self, primary):
        # Clones on other devices may have inherited this execution's injection.
        patchers = list(dict.fromkeys([primary, *self.states]))
        for patcher in self.states:
            self.eject(patcher)
        self.states.clear()
        for patcher in patchers:
            with patcher.use_ejected():
                patcher.remove_injections(_INJECTION_KEY)


def _sample_with_character_loras(executor, *args, **kwargs):
    guider = executor.class_obj
    options = guider.model_patcher.model_options.get("transformer_options", {})
    hooks = {hook.hook_ref: hook for hook in options.get(_INJECTION_KEY, ())}
    for conds in guider.conds.values():
        for cond in conds:
            group = cond.get("hooks")
            if group is not None:
                for hook in group.hooks:
                    if isinstance(hook, _CharacterLoRAHook):
                        hooks[hook.hook_ref] = hook
    if not hooks:
        print("[character LoRA] No residual hooks in active conditioning; using the incoming sampler")
        return executor(*args, **kwargs)

    patcher = guider.model_patcher
    runtime = _CharacterLoRARuntime(list(hooks.values()))
    try:
        with patcher.use_ejected():
            patcher.set_injections(_INJECTION_KEY, [comfy.patcher_extension.PatcherInjection(
                inject=runtime.inject, eject=runtime.eject
            )])
        return executor(*args, **kwargs)
    except Exception as exc:
        print(f"[character LoRA] Sampling failed: groups={len(hooks)}, error={exc}")
        traceback.print_exc()
        raise
    finally:
        try:
            runtime.close(patcher)
        except Exception as exc:
            print(f"[character LoRA] Cleanup failed: groups={len(hooks)}, error={exc}")
            traceback.print_exc()
            raise
