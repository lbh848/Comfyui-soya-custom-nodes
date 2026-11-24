"""SnapRef reference encoding for the existing spatial first sampler.

Adapted from 0ppxnhximxr/snapref-v1a, revision b06b901f8b8130acb58b2c446b84970267cd0a8c.
See third_party/SnapRef-LICENSE.md and SnapRef-NOTICE.txt.
"""

import copy
import json
import math
import traceback
from functools import partial

import torch
from comfy.ldm.modules.attention import AttentionTensorContainer
from safetensors import safe_open
from safetensors.torch import load_file

import comfy.sd
import comfy.utils
import comfy.model_management
import comfy.model_prefetch
from .anima_regional_conditioning import _masked_attn_op


def _adapter_memory(entries, dtype):
    sources = {(id(source), scale): source for scale, source in entries}
    return sum(value.numel() * comfy.model_management.dtype_size(dtype)
               for source in sources.values() for value in source.weights if torch.is_tensor(value))


def _device_adapter(source, scale, device, dtype, cache):
    key = (id(source), scale, device, dtype)
    if key not in cache:
        adapter = copy.copy(source)
        with comfy.model_prefetch.pause_malloc_graph():
            adapter.weights = tuple(value.to(device=device, dtype=dtype) if torch.is_tensor(value) else value
                                    for value in source.weights)
        adapter.multiplier = scale
        cache[key] = adapter
    return cache[key]


BRANCH_PREFIX = "anima_ref_kv.branch."
DEFAULT_STYLE_MODEL = "snapref_v2s_pre1.safetensors"
FALLBACK_STYLE_MODEL = "snapref_v2s_pre1_fp8.safetensors"


def resolve_snapref_model_path(resolve_lora_path, char_model="snapref_v2a_pre2.safetensors",
                               style_model=DEFAULT_STYLE_MODEL, *, has_style=False):
    if has_style:
        candidates = [style_model or DEFAULT_STYLE_MODEL]
        if candidates[0] == DEFAULT_STYLE_MODEL:
            candidates.append(FALLBACK_STYLE_MODEL)
        for name in candidates:
            try:
                return resolve_lora_path(name)
            except FileNotFoundError:
                continue
        raise FileNotFoundError(
            f"STYLE_REF 모델({DEFAULT_STYLE_MODEL})을 찾을 수 없습니다. "
            "설정의 업데이트 관리에서 누락 모델을 설치하거나 models/loras 폴더에 추가하세요."
        )
    candidates = [char_model or "snapref_v2a_pre2.safetensors"]
    if candidates[0] == "snapref_v2a_pre2.safetensors":
        candidates.extend((DEFAULT_STYLE_MODEL, FALLBACK_STYLE_MODEL))
    for name in candidates:
        try:
            return resolve_lora_path(name)
        except FileNotFoundError:
            continue
    return resolve_lora_path(candidates[0])


class _Adapter:
    def __init__(self, path, model):
        tensors = load_file(path)
        with safe_open(path, "pt") as handle:
            metadata = handle.metadata() or {}
        self.meta = json.loads(metadata.get("snapref") or metadata.get("anima_ref_kv") or "{}")
        self.ref_t = int(self.meta.get("ref_t", 10))
        self.modes = list(self.meta.get("modes", ["cs"]))
        self.mode_embed = tensors.get("anima_ref_kv.mode_embed")
        streams = {"ref": {}, "tgt": {}}
        for key in tensors:
            if not key.endswith(".lora_down"):
                continue
            root = key[:-len("lora_down")]
            _, stream, name = root[:-1].split(".", 2)
            target = "diffusion_model." + name
            for source, suffix in (("lora_down", "lora_down.weight"), ("lora_up", "lora_up.weight")):
                streams[stream][target + "." + suffix] = self._dequant(tensors, root + source)
            streams[stream][target + ".alpha"] = tensors[root + "alpha"]
        self.patches = {}
        for stream, state in streams.items():
            # Both streams use the same installed 28->40 remapper as character
            # LoRAs. Never put the original 28 layer indices on a 40 layer DiT.
            self.patches[stream] = self._map_lora_state(model, state, f"{stream} ({path})")
        self.branches = self._load_branches(path, model, tensors)
        self.device_adapters = {}
        self.cache = {}
        self.style_cache = {}
        branch_summary = ", ".join(f"{name}:{len(branch['patches'])}" for name, branch in self.branches.items())
        print(f"[SnapRef] Loaded {path}: ref_layers={len(self.patches['ref'])}, target_layers={len(self.patches['tgt'])}"
              + (f", branches=[{branch_summary}]" if branch_summary else ""))

    @staticmethod
    def _map_lora_state(model, state, label):
        mapped, _ = comfy.sd.load_lora_for_models(model, None, state, 1.0, 0.0)
        patches = {
            key[len("diffusion_model."):-len(".weight")]: entries[len(model.patches.get(key, ())):]
            for key, entries in mapped.patches.items()
            if key.startswith("diffusion_model.") and key.endswith(".weight")
            and len(entries) > len(model.patches.get(key, ()))
        }
        if not patches:
            raise ValueError(f"SnapRef {label} weights cannot be applied")
        return patches

    def _load_branches(self, path, model, tensors):
        specs = self.meta.get("branches") or {}
        if not isinstance(specs, dict) or not specs:
            return {}
        found = {}
        for key in tensors:
            if not (key.startswith(BRANCH_PREFIX) and key.endswith(".down")):
                continue
            root = key[:-len("down")]
            branch, name = root[len(BRANCH_PREFIX):-1].split(".", 1)
            if branch not in specs:
                raise ValueError(f"{path}: branch {branch} has weights without metadata")
            target = "diffusion_model." + name
            state = found.setdefault(branch, {})
            for source, suffix in (("down", "lora_down.weight"), ("up", "lora_up.weight")):
                state[target + "." + suffix] = self._dequant(tensors, root + source)
            state[target + ".alpha"] = tensors[root + "alpha"]
        branches = {}
        for branch, spec in specs.items():
            stream, combine = spec.get("stream"), spec.get("combine", "add")
            if stream not in ("ref", "tgt") or combine != "add":
                raise ValueError(f"{path}: unsupported branch {branch} (stream={stream}, combine={combine})")
            if branch in found:
                branches[branch] = {
                    "stream": stream,
                    "modes": list(spec.get("modes") or []),
                    "patches": self._map_lora_state(model, found[branch], f"branch.{branch} ({path})"),
                }
        return branches

    def branches_for(self, mode, stream=None):
        return [
            branch for branch in self.branches.values()
            if mode in branch["modes"] and (stream is None or branch["stream"] == stream)
        ]

    def check_mode(self, mode, label):
        if mode != "style":
            return
        if mode not in self.modes:
            raise ValueError(
                f"SnapRef 모델({label})은 style 모드를 지원하지 않습니다. "
                f"{DEFAULT_STYLE_MODEL} 모델을 설치해 사용하세요."
            )
        if self.mode_embed is None or self.mode_embed.shape[0] <= self.modes.index(mode) or not self.branches_for(mode):
            raise ValueError(
                f"SnapRef 모델({label})에 style 브랜치 가중치가 없습니다. "
                f"{DEFAULT_STYLE_MODEL} 모델을 확인하세요."
            )

    def ref_entries_for_modes(self, modes):
        entries = [
            (scale, source)
            for patch_entries in self.patches["ref"].values()
            for scale, source, *_ in patch_entries
        ]
        seen = set()
        for mode in modes:
            for branch in self.branches_for(mode, stream="ref"):
                if id(branch) in seen:
                    continue
                seen.add(id(branch))
                entries.extend(
                    (scale, source)
                    for patch_entries in branch["patches"].values()
                    for scale, source, *_ in patch_entries
                )
        return entries

    @staticmethod
    def _dequant(tensors, key):
        value = tensors[key]
        if key + "_scale" in tensors:
            value = value.float() * tensors[key + "_scale"]
        return value.to(torch.bfloat16)

    def clear(self):
        self.cache.clear()
        self.style_cache.clear()
        self.device_adapters.clear()

    def hook(self, entries, module, args, output):
        x = args[0]
        for scale, source, _model_scale, _offset, _function in entries:
            adapter = _device_adapter(source, scale, x.device, x.dtype, self.device_adapters)
            output = output + adapter.h(x, output)
        return output


def _rope(pe, t, h, w):
    def rot(pos, freqs):
        angle = torch.outer(pos.float(), freqs)
        return torch.stack([angle.cos(), -angle.sin(), angle.sin(), angle.cos()], dim=-1)
    dev = t.device
    emb = torch.cat([
        rot(t, 1.0 / ((10000.0 * pe.t_ntk_factor) ** pe.dim_temporal_range.to(dev))),
        rot(h, 1.0 / ((10000.0 * pe.h_ntk_factor) ** pe.dim_spatial_range.to(dev))),
        rot(w, 1.0 / ((10000.0 * pe.w_ntk_factor) ** pe.dim_spatial_range.to(dev))),
    ], dim=-2)
    return emb.reshape(*emb.shape[:-1], 2, 2).float()


def _reference_attention(original, record, index, q, k, v, transformer_options=None, preferred_attention=None):
    if isinstance(q, AttentionTensorContainer):
        q, k, v = q.take(), k.take(), v.take()
    record[index][0].copy_(k)
    record[index][1].copy_(v)
    return original(q, k, v, transformer_options=transformer_options or {}, preferred_attention=preferred_attention)


def _encode_reference(adapter, dit, ref, context, target_hw, mode, record):
    handles, attention_ops = [], []
    # Mod Guidance belongs to the target CFG batch. Reference encoding uses
    # its own sigma-zero/mode embedding and must not inherit target deltas.
    guidance_state = [(owner, name, getattr(owner, name))
                      for owner, name in ((dit.t_embedding_norm, "_anima_mod_delta"),
                                          (dit, "_anima_mod_block_state"))
                      if hasattr(owner, name)]
    try:
        for owner, name, _ in guidance_state:
            setattr(owner, name, None)
        for path, entries in adapter.patches["ref"].items():
            handles.append(dit.get_submodule(path).register_forward_hook(partial(adapter.hook, entries)))
        for branch in adapter.branches_for(mode, stream="ref"):
            for path, entries in branch["patches"].items():
                handles.append(dit.get_submodule(path).register_forward_hook(partial(adapter.hook, entries)))
        for index, block in enumerate(dit.blocks):
            attention = block.self_attn
            attention_ops.append((attention, attention.attn_op))
            attention.attn_op = partial(_reference_attention, attention.attn_op, record, index)
        x_emb, _, _ = dit.prepare_embedded_sequence(ref, fps=None, padding_mask=None)
        _, _, hp, wp, _ = x_emb.shape
        h0 = round((target_hw[0] // dit.patch_spatial) / 2 - hp / 2)
        w0 = round((target_hw[1] // dit.patch_spatial) / 2 - wp / 2)
        hh, ww = torch.meshgrid(torch.arange(hp, device=ref.device) + h0, torch.arange(wp, device=ref.device) + w0, indexing="ij")
        rope = _rope(dit.pos_embedder, torch.full_like(hh, adapter.ref_t).flatten(), hh.flatten(), ww.flatten())
        t_emb, adaln = dit.t_embedder[1](dit.t_embedder[0](torch.zeros(ref.shape[0], 1, device=ref.device)).to(x_emb.dtype))
        t_emb = dit.t_embedding_norm(t_emb)
        if adapter.mode_embed is not None and mode in adapter.modes:
            t_emb = t_emb + adapter.mode_embed[adapter.modes.index(mode)].to(t_emb)[None, None, :]
        # Match Cosmos' forward path: FP16 attention/MLP computation needs an
        # FP32 residual stream to preserve the model's large activation values.
        x = x_emb.float() if x_emb.dtype == torch.float16 else x_emb
        for block in dit.blocks:
            x = block(x, t_emb, context, rope_emb_L_1_1_D=rope.unsqueeze(1).unsqueeze(0),
                      adaln_lora_B_T_3D=adaln, extra_per_block_pos_emb=None, transformer_options={})
    finally:
        for handle in handles:
            handle.remove()
        for attention, original in attention_ops:
            attention.attn_op = original
        for owner, name, value in guidance_state:
            setattr(owner, name, value)


def _masked_reference_attention(original, references, weights, flags, strength, q, k, v, transformer_options=None, preferred_attention=None, fidelity=1.0, style_reference=None, style_fidelity=1.0):
    """Each query reads only its region's reference K/V, plus optional global style K/V.

    Target self-attention stays global. Blend reference deltas with the same
    overlap budget as character LoRAs. Fidelity scales negative CFG references;
    background queries keep their original attention unless style_reference is set.
    """
    if isinstance(q, AttentionTensorContainer):
        q, k, v = q.take(), k.take(), v.take()
    options = transformer_options or {}
    original_out = original(q, k, v, transformer_options=options, preferred_attention=preferred_attention)
    result = original_out.clone()
    batch = q.shape[0] // len(flags)
    for chunk, flag in enumerate(flags):
        sl = slice(chunk * batch, (chunk + 1) * batch)
        use_style = style_reference is not None and (flag == 0 or style_fidelity < 1.0)
        use_char = bool(references) and (flag == 0 or fidelity < 1.0)
        if not use_style and not use_char:
            continue
        if use_style:
            style_k, style_v, style_biases = style_reference
            chunk_style_bias = style_biases + (math.log(max(1.0 - style_fidelity, 1e-6)) if flag != 0 else 0.0)
            sk = style_k.expand(batch, -1, -1, -1)
            sv = style_v.expand(batch, -1, -1, -1)
            style_keys = torch.cat((sk, k[sl]), dim=1)
            style_values = torch.cat((sv, v[sl]), dim=1)
            if bool((chunk_style_bias == 0).all()):
                base_out = original(q[sl], style_keys, style_values, transformer_options=options, preferred_attention=preferred_attention)
            else:
                style_bias = torch.zeros(1, 1, 1, style_keys.shape[1], device=q.device, dtype=q.dtype)
                style_bias[..., :style_k.shape[1]] = chunk_style_bias
                base_out = _masked_attn_op(q[sl], style_keys, style_values, options, style_bias)
            result[sl] = base_out
        else:
            base_out = original_out[sl]
        if not use_char:
            continue
        reference_bias = strength + (math.log(max(1.0 - fidelity, 1e-6)) if flag != 0 else 0.0)
        for slot, (ref_k, ref_v) in references.items():
            indices = torch.nonzero(weights[slot] > 0, as_tuple=False).flatten()
            if indices.numel() == 0:
                continue
            query = q[sl].index_select(1, indices)
            rk = ref_k.expand(batch, -1, -1, -1)
            rv = ref_v.expand(batch, -1, -1, -1)
            if use_style:
                keys = torch.cat((sk, rk, k[sl]), dim=1)
                values = torch.cat((sv, rv, v[sl]), dim=1)
                bias = torch.zeros(1, 1, 1, keys.shape[1], device=q.device, dtype=q.dtype)
                style_len = style_k.shape[1]
                ref_len = ref_k.shape[1]
                bias[..., :style_len] = chunk_style_bias
                bias[..., style_len:style_len + ref_len] = reference_bias
                if bool((bias == 0).all()):
                    attended = original(query, keys, values, transformer_options=options, preferred_attention=preferred_attention)
                else:
                    attended = _masked_attn_op(query, keys, values, options, bias)
            else:
                keys = torch.cat((rk, k[sl]), dim=1)
                values = torch.cat((rv, v[sl]), dim=1)
                if reference_bias == 0:
                    attended = original(query, keys, values, transformer_options=options, preferred_attention=preferred_attention)
                else:
                    bias = torch.zeros(1, 1, 1, keys.shape[1], device=q.device, dtype=q.dtype)
                    bias[..., :ref_k.shape[1]] = reference_bias
                    attended = _masked_attn_op(query, keys, values, options, bias)
            delta = attended - base_out.index_select(1, indices)
            result[sl].index_add_(1, indices, delta * weights[slot, indices].reshape(1, -1, 1))
    return result


class SnapRef:
    """Per-execution state attached to the existing spatial ModelPatcher."""

    def __init__(self, model, clip, vae, images, path, mode="char", megapixels=1.0, strength=1.0, fidelity=1.0, *,
                 style_images=None, style_megapixels=1.0, style_strength=1.0, style_fidelity=1.0):
        self.adapter = _Adapter(path, model)
        self.mode = mode
        self.strength = math.log(strength) if strength > 0 else 0.0
        self.fidelity = fidelity
        self.style_strength = float(style_strength)
        self.style_fidelity = float(style_fidelity)
        self.refs = {}
        self.style_refs = []
        self.buffers = {}
        self.style_buffers = {}
        if images and strength > 0:
            for slot, slot_images in images.items():
                self.refs[slot] = []
                for index, image in enumerate(slot_images if isinstance(slot_images, list) else [slot_images]):
                    latent, width, height = self._encode_image_latent(model, vae, image, megapixels)
                    self.refs[slot].append(latent)
                    print(f"[SnapRef] Reference region={slot}, image={index + 1}, size={width}x{height}, mode={mode}")
            print(f"[SnapRef] Strength={strength}, Fidelity={fidelity}")
        if style_images and style_strength > 0:
            self.adapter.check_mode("style", path)
            for index, entry in enumerate(style_images):
                if isinstance(entry, tuple) and len(entry) == 2:
                    image, weight = entry
                else:
                    image, weight = entry, 1.0
                weight = float(weight)
                if weight <= 0:
                    continue
                latent, width, height = self._encode_image_latent(model, vae, image, style_megapixels)
                self.style_refs.append((latent, weight))
                print(f"[SnapRef] Style reference image={index + 1}, size={width}x{height}, weight={weight}, mode=style")
            if self.style_refs:
                print(f"[SnapRef] Style strength={style_strength}, fidelity={style_fidelity}, count={len(self.style_refs)}")
        self.cond = clip.encode_from_tokens_scheduled(clip.tokenize(""))[0]

    @staticmethod
    def _encode_image_latent(model, vae, image, megapixels):
        pixels = image[:1].movedim(-1, 1)
        h, w = pixels.shape[-2:]
        scale = min(1.0, (megapixels * 1024 * 1024 / (w * h)) ** 0.5)
        width, height = max(16, int(w * scale) // 16 * 16), max(16, int(h * scale) // 16 * 16)
        pixels = comfy.utils.common_upscale(pixels, width, height, "lanczos", "disabled").movedim(1, -1)
        latent = vae.encode(pixels[..., :3])
        if latent.ndim == 4:
            latent = latent.unsqueeze(2)
        return model.get_model_object("process_latent_in")(latent).cpu(), width, height

    @property
    def has_style(self):
        return bool(self.style_refs)

    @property
    def active_modes(self):
        modes = []
        if self.refs:
            modes.append(self.mode)
        if self.style_refs:
            modes.append("style")
        return tuple(modes)

    def clear(self):
        self.adapter.clear()
        self.buffers.clear()
        self.style_buffers.clear()

    @staticmethod
    def _latent_tokens(dit, ref):
        return (math.ceil(ref.shape[2] / dit.patch_temporal)
                * math.ceil(ref.shape[3] / dit.patch_spatial)
                * math.ceil(ref.shape[4] / dit.patch_spatial))

    def token_counts(self, dit):
        return {slot: [self._latent_tokens(dit, ref) for ref in refs]
                for slot, refs in self.refs.items()}

    def style_token_counts(self, dit):
        return [self._latent_tokens(dit, ref) for ref, _weight in self.style_refs]

    def memory_required(self, dit, dtype):
        tokens = sum(sum(counts) for counts in self.token_counts(dit).values()) + sum(self.style_token_counts(dit))
        return tokens * dit.model_channels * 2 * len(dit.blocks) * comfy.model_management.dtype_size(dtype)

    def allocate(self, dit, device, dtype):
        # Allocate persistent K/V before loading the diffusion model, so both
        # static and dynamic VRAM loaders see its actual resident footprint.
        self.buffers[device, dtype] = {
            slot: [tuple(torch.empty((1, sum(counts), dit.num_heads, dit.model_channels // dit.num_heads),
                                     device=device, dtype=dtype) for _ in range(2))
                   for _ in dit.blocks]
            for slot, counts in self.token_counts(dit).items()
        }
        style_counts = self.style_token_counts(dit)
        if style_counts:
            self.style_buffers[device, dtype] = [
                tuple(torch.empty((1, sum(style_counts), dit.num_heads, dit.model_channels // dit.num_heads),
                                  device=device, dtype=dtype) for _ in range(2))
                for _ in dit.blocks
            ]
        for scale, source in self.adapter.ref_entries_for_modes(self.active_modes):
            _device_adapter(source, scale, device, dtype, self.adapter.device_adapters)

    def _prepare_context(self, dit, x):
        cond, metadata = self.cond
        ids, weights = metadata.get("t5xxl_ids"), metadata.get("t5xxl_weights")
        return dit.preprocess_text_embeds(
            cond.to(x.device, x.dtype),
            None if ids is None else ids.unsqueeze(0).to(x.device),
            None if weights is None else weights.unsqueeze(0).unsqueeze(-1).to(x.device, x.dtype),
        )

    def prepare(self, dit, x):
        if not self.refs:
            return {}
        key = (id(dit), tuple(x.shape[-2:]), x.device, x.dtype)
        if key not in self.adapter.cache:
            self.adapter.cache.clear()
            context = self._prepare_context(dit, x)
            try:
                records = self.buffers[x.device, x.dtype]
                token_counts = self.token_counts(dit)
                for slot, refs in self.refs.items():
                    offset = 0
                    for ref, count in zip(refs, token_counts[slot]):
                        record = [tuple(value[:, offset:offset + count] for value in block) for block in records[slot]]
                        _encode_reference(self.adapter, dit, ref.to(x.device, x.dtype), context, x.shape[-2:], self.mode, record)
                        offset += count
                self.adapter.cache[key] = records
            except Exception as exc:
                self.adapter.clear()
                print(f"[SnapRef] Reference encoding failed: regions={list(self.refs)}, error={exc}")
                traceback.print_exc()
                raise
            print(f"[SnapRef] Cached reference K/V: regions={list(self.refs)}, blocks={len(dit.blocks)}")
        return self.adapter.cache[key]

    def prepare_style(self, dit, x):
        if not self.style_refs:
            return None
        key = (id(dit), tuple(x.shape[-2:]), x.device, x.dtype)
        if key not in self.adapter.style_cache:
            self.adapter.style_cache.clear()
            context = self._prepare_context(dit, x)
            try:
                records = self.style_buffers[x.device, x.dtype]
                counts = self.style_token_counts(dit)
                biases = torch.empty((1, 1, 1, sum(counts)), device=x.device, dtype=x.dtype)
                offset = 0
                for (ref, weight), count in zip(self.style_refs, counts):
                    record = [tuple(value[:, offset:offset + count] for value in block) for block in records]
                    _encode_reference(self.adapter, dit, ref.to(x.device, x.dtype), context, x.shape[-2:], "style", record)
                    biases[..., offset:offset + count] = math.log(max(self.style_strength * weight, 1e-6))
                    offset += count
                self.adapter.style_cache[key] = [(k_b, v_b, biases) for k_b, v_b in records]
            except Exception as exc:
                self.adapter.clear()
                print(f"[SnapRef] Style reference encoding failed: count={len(self.style_refs)}, error={exc}")
                traceback.print_exc()
                raise
            print(f"[SnapRef] Cached style reference K/V: images={len(self.style_refs)}, blocks={len(dit.blocks)}")
        return self.adapter.style_cache[key]
