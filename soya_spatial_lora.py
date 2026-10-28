"""Anima token-routed LoRA residuals, with one shared DiT stream.

Inspired by FreeFuse's spatial residual routing (arXiv:2510.23515), not a
reproduction: use the caller's existing RGB masks, not FreeFuseAttn extraction.
Keep global self attention for composition. Only direct adapter injection is
spatially restricted; this does not promise zero downstream identity leakage.
"""

import copy
import math
import traceback
from functools import partial

import torch
import torch.nn.functional as F

import comfy.sd
import comfy.patcher_extension

from .soya_character_lora import _merged_path_reason
from .anima_regional_conditioning import (
    AnimaConditioningRegionChain,
    AnimaRegionalConditioningPatch,
    _build_flux_cross_attention_bias,
    _masked_attn_op,
    _match_context_length,
    _normalize_context,
    _validate_anima_model,
)


_KEY = "soya_spatial_lora"


def _layer_roles(dit):
    """Classify by actual Anima module topology, never by prompt/character text."""
    roles = {}
    for block in dit.blocks:
        for attention, cross in ((block.self_attn, False), (block.cross_attn, True)):
            for name in ("q_proj", "k_proj", "v_proj", "output_proj"):
                roles[getattr(attention, name)] = (
                    "text" if cross and name in ("k_proj", "v_proj") else "image"
                )
        for module in block.mlp.modules():
            if isinstance(module, torch.nn.Linear):
                roles[module] = "image"
    return roles


def _token_weights(masks, latent_shape, patch_spatial, patch_temporal):
    """Resize at latent resolution, then pad/pool exactly like the token grid.

    Overlapping masks share a bounded residual budget, rather than doubling
    character strength. They remain ambiguous blends, not inferred ownership.
    """
    t, h, w = latent_shape
    ph = math.ceil(h / patch_spatial) * patch_spatial
    pw = math.ceil(w / patch_spatial) * patch_spatial
    temporal = math.ceil(t / patch_temporal)
    weights = []
    for mask in masks:
        m = F.interpolate(mask.unsqueeze(1), size=(h, w), mode="nearest-exact")
        m = F.pad(m, (0, pw - w, 0, ph - h), mode="replicate")
        m = F.avg_pool2d(m, patch_spatial, patch_spatial).flatten()
        weights.append(m.repeat(temporal))
    weights = torch.stack(weights)
    weights = weights / weights.sum(0, keepdim=True).clamp(min=1.0)
    background = (1.0 - weights.sum(0, keepdim=True)).clamp(min=0.0)
    return torch.cat((background, weights))


def _route_residual(x, output, adapter, role, slot, weights, text_lengths):
    if role == "text":
        start = sum(text_lengths[:slot])
        end = start + text_lengths[slot]
        # K/V input is text, not pixels. The negative prompt also has its own
        # copy per slot, so CFG uses the same local adapters on both sides.
        result = output.clone()
        result[:, start:end] += adapter.h(x[:, start:end], output[:, start:end])
        return result
    if math.prod(output.shape[1:-1]) != weights.shape[1]:
        raise RuntimeError(
            f"Spatial LoRA token grid mismatch: output={tuple(output.shape)}, "
            f"mask={tuple(weights.shape)}, slot={slot}"
        )
    gate = weights[slot].reshape(1, *output.shape[1:-1], 1)
    return output + adapter.h(x, output) * gate


class _SpatialRuntime:
    def __init__(self, conditioning, patches, latent_format=None):
        self.conditioning = conditioning
        self.patches = patches  # CPU adapters, keyed by native model path
        self.prepared = {}
        self.adapters = {}
        self.layouts = {}
        self.forward_count = 0
        self.latent_format = latent_format
        self.tracker = None

    def clear(self):
        self.prepared.clear()
        self.adapters.clear()
        self.layouts.clear()
        if self.tracker is not None:
            self.tracker.clear_device_cache()

    def begin(self):
        self.tracker = None
        if self.latent_format is not None:
            try:
                from .soya_face_tracking import FaceTracker
                self.tracker = FaceTracker(self.conditioning.region_masks, self.latent_format)
                print("[FACE_TRACK] Dynamic masks enabled; CPU preview, no additional sampling steps")
            except Exception as exc:
                print(f"[FACE_TRACK] Initialization failed; retaining source masks: {exc}")
                traceback.print_exc()

    def contexts(self, dit, device, dtype):
        key = (device, dtype)
        if key not in self.prepared:
            self.prepared[key] = [
                self.conditioning.prepare_background_cond(dit, device, dtype),
                *self.conditioning.prepare_region_conds(dit, device, dtype),
            ]
        return self.prepared[key]

    def layout(self, shape, dit, device, dtype, lengths):
        key = (tuple(shape), dit.patch_spatial, dit.patch_temporal, device, dtype, tuple(lengths))
        if key not in self.layouts:
            weights = _token_weights(
                self.conditioning.region_masks, shape, dit.patch_spatial, dit.patch_temporal,
            ).to(device=device, dtype=dtype)
            bias = _build_flux_cross_attention_bias(
                weights > 1e-6, lengths, "uncovered_only", device, dtype,
            )
            self.layouts[key] = weights, bias
        original = self.layouts[key]
        if self.tracker is not None:
            tracked = self.tracker.layout(
                original[0], lengths, shape, dit, device, dtype, _build_flux_cross_attention_bias,
            )
            if tracked is not None:
                return tracked
        return original

    def layer_hook(self, entries, weights, lengths, module, args, kwargs, output):
        x = args[0] if args else kwargs["input"]
        for slot, role, strength, source in entries:
            key = (id(source), strength, x.device, x.dtype)
            adapter = self.adapters.get(key)
            if adapter is None:
                adapter = copy.copy(source)
                adapter.weights = tuple(
                    v.to(device=x.device, dtype=x.dtype) if torch.is_tensor(v) else v
                    for v in source.weights
                )
                adapter.multiplier = strength
                self.adapters[key] = adapter
            output = _route_residual(x, output, adapter, role, slot, weights, lengths)
        return output


def _unified_context(context, contexts, flags):
    if context.shape[0] % len(flags):
        raise RuntimeError(f"Spatial LoRA CFG batch mismatch: context={tuple(context.shape)}, flags={flags}")
    batch = context.shape[0] // len(flags)
    # Do not truncate long negatives or character prompts to another slot's
    # length. Padding is only within that slot, never across ownership bounds.
    lengths = [max(c.shape[1], context.shape[1]) for c in contexts]
    chunks = []
    for chunk, flag in zip(context.chunk(len(flags)), flags):
        slots = []
        for c, length in zip(contexts, lengths):
            if flag == 1:
                source = chunk
            elif c.shape[0] in (1, batch):
                source = c.expand(batch, -1, -1)
            else:
                raise RuntimeError(f"Spatial LoRA text batch mismatch: region={tuple(c.shape)}, batch={batch}")
            slots.append(_match_context_length(source, length))
        chunks.append(torch.cat(slots, dim=1))
    return torch.cat(chunks), lengths


def _diffusion_wrapper(executor, *args, **kwargs):
    options = kwargs.get("transformer_options", {})
    runtime = options[_KEY]
    dit = executor.class_obj
    x = args[0] if args else kwargs["x"]
    raw_context = args[2] if len(args) > 2 else kwargs["context"]
    handles, attention_ops = [], []
    try:
        context, _ = _normalize_context(raw_context)
        contexts = runtime.contexts(dit, context.device, context.dtype)
        flags = options.get("cond_or_uncond", [0])
        unified, lengths = _unified_context(context, contexts, flags)
        weights, bias = runtime.layout(x.shape[-3:], dit, context.device, context.dtype, lengths)
        for path, entries in runtime.patches.items():
            # Resolve against this execution's diffusion model (also supports
            # model clones); never retain a module or device copy in the node.
            module = dit.get_submodule(path)
            handles.append(module.register_forward_hook(
                partial(runtime.layer_hook, entries, weights, lengths), with_kwargs=True,
            ))
        for block in dit.blocks:
            attention = block.cross_attn
            attention_ops.append((attention, attention.attn_op))
            attention.attn_op = partial(_masked_attn_op, attn_bias=bias)
        args = list(args)
        if len(args) > 2:
            args[2] = unified
        else:
            kwargs["context"] = unified
        runtime.forward_count += 1
        return executor(*args, **kwargs)
    except Exception as exc:
        print(f"[spatial LoRA] Forward failed: latent={tuple(x.shape)}, error={exc}")
        traceback.print_exc()
        raise
    finally:
        for handle in handles:
            handle.remove()
        for attention, original in attention_ops:
            attention.attn_op = original
        if runtime.latent_format is not None and x.is_cuda:
            # Spectrum retains live feature history. Return only unused CUDA
            # allocator blocks; this does not modify that history or tensors.
            with torch.cuda.device(x.device):
                torch.cuda.empty_cache()


def _sample_wrapper(executor, *args, **kwargs):
    runtime = executor.class_obj.model_patcher.model_options["transformer_options"][_KEY]
    runtime.clear()
    runtime.forward_count = 0
    runtime.begin()
    try:
        if runtime.tracker is not None:
            from .soya_face_tracking import observation_steps
            sigmas = args[3] if len(args) > 3 else kwargs["sigmas"]
            checkpoints = observation_steps(len(sigmas) - 1)
            callback = args[5] if len(args) > 5 else kwargs.get("callback")

            def observe(step, x0, x, total_steps):
                if callback is not None:
                    callback(step, x0, x, total_steps)
                if step in checkpoints:
                    runtime.tracker.observe(step, x0)

            if len(args) > 5:
                args = list(args)
                args[5] = observe
            else:
                kwargs["callback"] = observe
            if not checkpoints:
                print(f"[FACE_TRACK] Schedule too short to apply an observation: steps={len(sigmas) - 1}")
        return executor(*args, **kwargs)
    except Exception as exc:
        runtime.tracker = None
        print(f"[spatial LoRA] Sampling failed: forwards={runtime.forward_count}, error={exc}")
        traceback.print_exc()
        raise
    finally:
        print(f"[spatial LoRA] Shared diffusion calls={runtime.forward_count}; releasing execution caches")
        runtime.clear()


def apply_spatial_loras(model, loras, masks, conditionings, background, *, face_tracking=False):
    """Return a spatially routed clone; never merge or change incoming weights."""
    try:
        dit = _validate_anima_model(model)
        roles = _layer_roles(dit)
        patches = {}
        for index, lora, strength in loras:
            # Use the same loader as unmasked sampling, including installed
            # Anima 28-to-40 remapping. Only route the newly added adapters;
            # incoming style patches already belong to the shared model.
            mapped, _ = comfy.sd.load_lora_for_models(model, None, lora, strength, 0.0)
            native = {
                key: entries[len(model.patches.get(key, ())):]
                for key, entries in mapped.patches.items()
                if len(entries) > len(model.patches.get(key, ()))
            }
            reason = _merged_path_reason(mapped, native)
            if reason is not None or not native:
                raise RuntimeError(f"Spatial LoRA cannot execute this adapter: {reason or 'no matching layers'}")
            for key, entries in native.items():
                module = model.get_model_object(key[:-7])
                role = roles.get(module)
                if role is None or not key.startswith("diffusion_model."):
                    raise RuntimeError(f"Spatial LoRA has no spatial/text routing for layer {key}; this adapter needs additional routing support")
                path = key[len("diffusion_model."):-7]
                for scale, adapter, _model_scale, _offset, _function in entries:
                    patches.setdefault(path, []).append((index + 1, role, scale, adapter))
            print(f"[spatial LoRA] Registered region={index + 1}, layers={len(native)}, strength={strength}")
        if not loras:
            print("[spatial LoRA] No character adapters; executing regional text conditioning only")
        regions = [AnimaConditioningRegionChain(None, mask, cond, 1.0) for mask, cond in zip(masks, conditionings)]
        conditioning = AnimaRegionalConditioningPatch(
            regions, "uncovered_only", 1.0, float("inf"), 0.0, 1.0, 0.0, 0.0, 1, 1, background,
        )
        runtime = _SpatialRuntime(conditioning, patches, model.model.latent_format if face_tracking else None)
        patched = model.clone()
        patched.model_options.setdefault("transformer_options", {})[_KEY] = runtime
        for kind, wrapper in (
            (comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, _diffusion_wrapper),
            (comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, _sample_wrapper),
        ):
            patched.remove_wrappers_with_key(kind, _KEY)
            patched.add_wrapper_with_key(kind, _KEY, wrapper)
        return patched
    except Exception as exc:
        print(f"[spatial LoRA] Registration failed: regions={len(masks)}, loras={len(loras)}, error={exc}")
        traceback.print_exc()
        raise
