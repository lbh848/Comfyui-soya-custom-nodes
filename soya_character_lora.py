"""Conditioning-scoped LoRA residuals without merged copies of model weights."""

import copy
import traceback
from functools import partial

import torch

import comfy.hooks
import comfy.patcher_extension
from comfy.weight_adapter.lora import LoRAAdapter


_INJECTION_KEY = "soya_character_lora"


class _CharacterLoRAHook(comfy.hooks.TransformerOptionsHook):
    def __init__(self, patches=None):
        super().__init__({}, hook_scope=comfy.hooks.EnumHookScope.HookedOnly)
        self.patches = patches

    def clone(self):
        cloned = super().clone()
        cloned.patches = self.patches
        return cloned


def _merged_path_reason(model, patches):
    for key, entries in patches.items():
        if not key.endswith(".weight"):
            return f"{key}: residual path requires a linear weight"
        module = model.get_model_object(key[:-7])
        if not isinstance(module, torch.nn.Linear):
            return f"{key}: {type(module).__name__} uses the native weight adapter"
        for strength, adapter, model_strength, offset, function in entries:
            if not isinstance(adapter, LoRAAdapter):
                return f"{key}: {type(adapter).__name__} uses the native weight adapter"
            up, down, alpha, mid, dora, reshape = adapter.weights
            if mid is not None or dora is not None or reshape is not None:
                return f"{key}: mid/DoRA/reshape needs the native weight adapter"
            if up.ndim != 2 or down.ndim != 2:
                return f"{key}: non-matrix LoRA needs the native weight adapter"
            if model_strength != 1.0 or offset is not None or function is not None:
                return f"{key}: transformed weight patch needs native application"
    return None


def load_character_lora(model, lora, strength):
    """Keep native key mapping and use residuals for ordinary linear LoRAs."""
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


def _add_character_residual(patcher, adapters, module, args, kwargs, output):
    active = patcher.current_hooks
    if active is not None:
        for hook, adapter, strength in adapters:
            if active.contains(hook):
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
                    adapters = state["layers"].setdefault(module, [])
                    for strength, source, _, _, _ in patches:
                        adapter = copy.copy(source)
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
    hooks = {}
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
