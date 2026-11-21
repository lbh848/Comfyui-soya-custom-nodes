"""BBox-routed MOSS adapters on one shared image/text transformer stream.

Only direct adapter residuals are local. Shared self attention can still carry
their influence elsewhere, as it does for the Anima spatial LoRA node.
"""
import copy
import json
import math
from functools import partial

import torch
import comfy.lora
import comfy.lora_convert
import comfy.model_prefetch
import comfy.patcher_extension
import comfy.utils
from comfy.ldm.moss_image.model import MOSSImage
from comfy.weight_adapter import LoRAAdapter, LoHaAdapter, LoKrAdapter

from .soya_batch_lora_loader import SoyaBatchLoraLoader_mdsoya
from .soya_character_lora import is_linear_projection

_KEY = "soya_moss_spatial_lora"
_BASE_KEY = "soya_moss_spatial_base"


def attach_spatial_lora(model, runtime):
    """Attach changing character regions without changing fixed weight patches."""
    routed = model.clone()
    routed.set_attachments(_BASE_KEY, model)
    routed.model_options.setdefault("transformer_options", {})[_KEY] = runtime
    for kind, wrapper in ((comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, _diffusion_wrapper),
                          (comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, _sample_wrapper)):
        routed.remove_wrappers_with_key(kind, _KEY)
        routed.add_wrapper_with_key(kind, _KEY, wrapper)
    return routed


def character_boxes(layout, workflow):
    """Normalized boxes keyed by exact character name; repeated panels unite."""
    items = [item for item in layout.get("items", [])
             if isinstance(item, dict) and item.get("label") == "character"]
    if not items:
        return {}
    if (layout.get("bbox_coord_space") != "abs_centralized"
            or layout.get("bbox_format") != "yxyx"):
        raise ValueError("MOSS BBox LoRA requires abs_centralized yxyx boxes")
    size = layout.get("image_size") or workflow
    width, height = int(size["width"]), int(size["height"])
    if width <= 0 or height <= 0:
        raise ValueError("MOSS BBox LoRA requires positive canvas dimensions")
    result = {}
    for item in items:
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        box = item.get("bbox")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError(f"MOSS BBox LoRA: missing bbox for {name}")
        y0, x0, y1, x1 = map(float, box)
        if not all(math.isfinite(v) for v in (y0, x0, y1, x1)):
            raise ValueError(f"MOSS BBox LoRA: non-finite bbox for {name}")
        x0, x1 = max(0., (x0 + width // 2) / width), min(1., (x1 + width // 2) / width)
        y0, y1 = max(0., (y0 + height // 2) / height), min(1., (y1 + height // 2) / height)
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"MOSS BBox LoRA: empty bbox for {name}")
        result.setdefault(name, []).append((y0, x0, y1, x1))
    return result


def token_weights(boxes, height, width, device, dtype):
    """Token centres inside each bbox; overlap shares a bounded residual budget."""
    yy = (torch.arange(height, device=device, dtype=torch.float32) + .5) / height
    xx = (torch.arange(width, device=device, dtype=torch.float32) + .5) / width
    masks = []
    for regions in boxes:
        mask = torch.zeros((height, width), device=device, dtype=torch.bool)
        for y0, x0, y1, x1 in regions:
            mask |= ((yy >= y0) & (yy < y1))[:, None] & ((xx >= x0) & (xx < x1))[None, :]
        masks.append(mask.flatten())
    weights = torch.stack(masks).to(dtype=dtype)
    return weights / weights.sum(0, keepdim=True).clamp(min=1.)


class _Runtime:
    def __init__(self, boxes, patches):
        self.boxes, self.patches = boxes, patches
        self.adapters = {}
        self.active = False

    def layer_hook(self, entries, weights, sequence_length, module, args, kwargs, output, *, regions=None):
        if not self.active:
            return output
        if output.ndim != 3 or output.shape[1] != sequence_length:
            raise RuntimeError(f"MOSS BBox LoRA token mismatch: {tuple(output.shape)}, expected {sequence_length}")
        x = args[0] if args else kwargs["input"]
        count = weights.shape[1]
        if regions is None:
            regions = [(indices, weight.index_select(0, indices))
                       for weight in weights for indices in [(weight > 0).nonzero().flatten()]]
        result = output.clone()
        for slot, rows, strength, source in entries:
            indices, gate = regions[slot]
            if not indices.numel():
                continue
            key = (id(source), strength, x.device, x.dtype)
            adapter = self.adapters.get(key)
            if adapter is None:
                adapter = copy.copy(source)
                # These tensors survive the block. Keep them out of Comfy's
                # temporary-allocation graph so it can reuse its block buffer.
                with comfy.model_prefetch.pause_malloc_graph():
                    adapter.weights = tuple(v.to(device=x.device, dtype=x.dtype)
                                            if torch.is_tensor(v) else v for v in source.weights)
                adapter.multiplier = strength
                self.adapters[key] = adapter
            # Never evaluate the adapter on text or reference tokens. Row slices
            # also cover split q/k/v adapters mapped onto a fused projection.
            # Linear adapters are token-local: gather only owned tokens rather
            # than computing full-canvas residuals that the mask discards.
            delta = adapter.h(x.index_select(1, indices),
                              output[:, :, rows].index_select(1, indices))
            result[:, :count, rows].index_add_(1, indices, delta * gate[None, :, None])
        return result


def _diffusion_wrapper(executor, *args, **kwargs):
    options = args[3] if len(args) > 3 else kwargs.get("transformer_options", {})
    runtime = options[_KEY]
    dit = executor.class_obj
    x = args[0] if args else kwargs["x"]
    context = args[2] if len(args) > 2 else kwargs["context"]
    height, width = x.shape[-2:]
    refs = kwargs.get("ref_latents", [])
    method = kwargs.get("ref_latents_method")
    ref_count = (sum(r.shape[-2] * r.shape[-1] for r in refs)
                 if method in ("index", "index_timestep_zero") else 0)
    sequence_length = height * width + context.shape[1] + ref_count
    weights = token_weights(runtime.boxes, height, width, x.device, x.dtype)
    regions = [(indices, weight.index_select(0, indices))
               for weight in weights for indices in [(weight > 0).nonzero().flatten()]]
    handles, force_forward = [], []
    previous_preprocess = dit.__dict__.get("preprocess_refs")
    preprocess = dit.preprocess_refs

    def preprocess_refs(*ref_args, **ref_kwargs):
        previous = runtime.active
        runtime.active = False
        try:
            return preprocess(*ref_args, **ref_kwargs)
        finally:
            runtime.active = previous

    try:
        dit.preprocess_refs = preprocess_refs
        runtime.active = True
        for path, entries in runtime.patches.items():
            module = dit.get_submodule(path)
            # INT8 linear_input_act otherwise bypasses module.forward entirely.
            # Preserve quantized base weights; evaluate situ before the linear.
            force_forward.append((module, module.__dict__.get("comfy_force_forward")))
            module.comfy_force_forward = True
            handles.append(module.register_forward_hook(
                partial(runtime.layer_hook, entries, weights, sequence_length, regions=regions), with_kwargs=True))
        return executor(*args, **kwargs)
    finally:
        runtime.active = False
        for handle in handles:
            handle.remove()
        for module, previous in force_forward:
            if previous is None:
                module.__dict__.pop("comfy_force_forward", None)
            else:
                module.comfy_force_forward = previous
        if previous_preprocess is None:
            dit.__dict__.pop("preprocess_refs", None)
        else:
            dit.preprocess_refs = previous_preprocess


def _sample_wrapper(executor, *args, **kwargs):
    runtime = executor.class_obj.model_patcher.model_options["transformer_options"][_KEY]
    try:
        return executor(*args, **kwargs)
    finally:
        runtime.adapters.clear()


def _spatial_patches(model, lora, slot, strength):
    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    # ai-toolkit can save the same native paths under transformer.*.
    for key, target in list(key_map.items()):
        if key.startswith("diffusion_model."):
            key_map.setdefault("transformer." + key[len("diffusion_model."):], target)
    mapped = comfy.lora.load_lora(comfy.lora_convert.convert_lora(lora), key_map)
    patches, skipped = {}, []
    for key, adapter in mapped.items():
        target, rows = key, slice(None)
        if isinstance(key, tuple):
            if len(key) != 2 or key[1][0] != 0:
                raise ValueError(f"MOSS BBox LoRA cannot route transformed weight: {key}")
            target, (_, start, length) = key
            rows = slice(start, start + length)
        path = target.removeprefix("diffusion_model.").removesuffix(".weight")
        # Text embedder and timestep modulation have no image positions. Do not
        # let those adapter components silently become global character LoRAs.
        if not (target.endswith(".weight") and path.startswith("layers.") and ".proj." in path):
            skipped.append(path)
            continue
        if not isinstance(adapter, (LoRAAdapter, LoHaAdapter, LoKrAdapter)):
            raise ValueError(f"MOSS BBox LoRA does not support {type(adapter).__name__}: {path}")
        dora_index = 4 if isinstance(adapter, LoRAAdapter) else (7 if isinstance(adapter, LoHaAdapter) else 8)
        if adapter.weights[dora_index] is not None:
            raise ValueError(f"MOSS BBox LoRA cannot apply DoRA without global weights: {path}")
        if isinstance(adapter, LoRAAdapter) and adapter.weights[5] is not None:
            raise ValueError(f"MOSS BBox LoRA cannot reshape a base weight: {path}")
        module = model.get_model_object(target[:-7])
        if not is_linear_projection(module):
            raise ValueError(f"MOSS BBox LoRA requires linear projections: {path} ({type(module).__qualname__})")
        patches.setdefault(path, []).append((slot, rows, strength, adapter))
    if not patches:
        raise ValueError("MOSS BBox LoRA: no supported image transformer projections in adapter")
    return patches, skipped


class SoyaMossSpatialLoraV2_mdsoya:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "clip": ("CLIP",),
                             "prompt_json": ("STRING", {"multiline": True, "forceInput": True})}}

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "info")
    FUNCTION = "load_loras"
    CATEGORY = "Soya/LoRA"
    DESCRIPTION = "Apply character LoRA residuals inside their MOSS bboxes. Common LoRAs stay global."
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def load_loras(self, model, clip, prompt_json):
        payload = json.loads(prompt_json) if isinstance(prompt_json, str) else prompt_json
        workflow, layout = payload["workflow"], payload["moss"]
        config = workflow.get("loras") or {}
        if not config.get("activate", False):
            self._global_lora_loader = None
            return model, clip, "MOSS BBox LoRA: disabled"
        if not isinstance(model.get_model_object("diffusion_model"), MOSSImage):
            raise ValueError("MOSS BBox LoRA requires ComfyUI's MOSSImage model")
        boxes = character_boxes(layout, workflow)
        has_characters = any(item.get("label") == "character" for item in layout.get("items", []) if isinstance(item, dict))
        global_loras, regional, names, info = [], [], [], []
        for entry in config.get("list", []):
            if entry.get("enabled", True) is False:
                continue
            if config.get("filter") and entry.get("BASE", "").strip() != config["filter"].strip():
                continue
            strength = float(entry.get("str", 1.))
            if not math.isfinite(strength):
                raise ValueError("MOSS BBox LoRA strength must be finite")
            if not strength:
                continue
            name = str(entry.get("CHAR") or "").strip()
            if not name or not has_characters:
                global_loras.append(entry)
                continue
            if name not in boxes:
                raise ValueError(f"MOSS BBox LoRA: no named bbox for character {name!r}")
            if name not in names:
                names.append(name)
            regional.append((names.index(name), entry, strength))
        if getattr(self, "_global_lora_loader", None) is None:
            self._global_lora_loader = SoyaBatchLoraLoader_mdsoya()
        loader = self._global_lora_loader
        model, clip, global_info = loader.load_batch_lora(
            model, clip, "true", "false", json.dumps({"list": global_loras}), "")
        info.append(global_info)
        patches = {}
        for slot, entry, strength in regional:
            path = loader._resolve_lora_path(entry["lora_path"])
            if path is None:
                raise FileNotFoundError(f"MOSS BBox LoRA: {entry['lora_path']}")
            lora = comfy.utils.load_torch_file(path, safe_load=True)
            mapped, skipped = _spatial_patches(model, lora, slot, strength)
            for path, entries in mapped.items():
                patches.setdefault(path, []).extend(entries)
            info.append(f"BBox: {names[slot]}, strength={strength}, layers={len(mapped)}, boxes={len(boxes[names[slot]])}")
            if skipped:
                info.append(f"Excluded non-spatial components ({len(skipped)}): {', '.join(skipped)}")
        if patches:
            runtime = _Runtime([boxes[name] for name in names], patches)
            model = attach_spatial_lora(model, runtime)
        return model, clip, "\n".join(info)
