"""MOSS-Image i2i reference node, owned by SOYA.

Vendored from ``ComfyUI-MOSS-Toolkit`` (sorryhyun, Apache-2.0): the LoRA and
reference-recipe handling, target-size helpers and the ComfyUI core-MOSS
reference path. See ``LICENSE-ComfyUI-MOSS-Toolkit.txt``.

The difference is the ``ref_cache = "cpu"`` store. The toolkit builds the whole
per-block K/V tensor on the GPU (about 3 GB at 1536x1152) before moving it to
host memory, so low-VRAM machines OOM before they can use the CPU cache. This
node builds the cache block by block: one ``(B, 2, N, dim)`` scratch tensor is
alive on the GPU at a time, then copied to pinned host memory. Each denoising
forward pulls only the current block's keys and values back to the GPU.

``gpu`` keeps the toolkit's full GPU cache. ``off`` recomputes the reference
rows in every forward. Only ComfyUI's own ``MOSSImage`` (``UNETLoader``) is
supported; the separate ``moss-image`` loader's DiT is not.
"""

import logging
import math
import time

import numpy as np
import torch
from PIL import Image

import comfy.model_management
import comfy.model_prefetch
import comfy.patcher_extension
import comfy.utils
import folder_paths
from comfy.ldm.flux.layers import timestep_embedding

from . import soya_moss_lora_compat as lora_compat

logger = logging.getLogger(__name__)

STRIDE = 16
AREA_TOLERANCE = 1.10
WRAPPER_KEY = "moss_i2i_ref"
TASKS = ("auto", "ai-toolkit, kv_cache off")
LEGACY_TASKS = ("colorize", "custom")
REF_CACHE = ("gpu", "cpu", "off")
AITK_JOINT = "ai-toolkit, kv_cache off"
CORE_NO_REFS = (
    "ComfyUI's MOSS-Image model takes a reference latent from ComfyUI 0.43.0 on; update "
    "ComfyUI, or load the DiT with the moss-image node's loader"
)

RESOLUTION = (
    "INT",
    {
        "default": 1024,
        "min": 256,
        "max": 2048,
        "step": 16,
        "tooltip": (
            "Target area ≈ resolution², native aspect, sides multiples of 16 (the "
            "training cache's target_size). Keep it equal on the Text Encode and "
            "Reference nodes so the vision tower and the ref see the same page."
        ),
    },
)


def target_size(width, height, resolution):
    """``(W, H)``: native aspect, multiples of 16, area ≤ ~``resolution²``, never
    larger than the source on either side."""
    cap = resolution * resolution
    if width % STRIDE == 0 and height % STRIDE == 0 and width * height <= cap * AREA_TOLERANCE:
        return width, height
    area = min(cap, width * height)
    aspect = width / height
    out_w = round(math.sqrt(area * aspect) / STRIDE) * STRIDE
    out_h = round(math.sqrt(area / aspect) / STRIDE) * STRIDE
    out_w = max(STRIDE, min(out_w, width // STRIDE * STRIDE))
    out_h = max(STRIDE, min(out_h, height // STRIDE * STRIDE))
    return out_w, out_h


def fit(image, width, height):
    """Resize to cover ``(width, height)`` and centre-crop; identity at that size."""
    if image.size == (width, height):
        return image
    scale = max(width / image.width, height / image.height)
    rw = max(width, math.ceil(image.width * scale))
    rh = max(height, math.ceil(image.height * scale))
    image = image.resize((rw, rh), Image.LANCZOS)
    left, top = (rw - width) // 2, (rh - height) // 2
    return image.crop((left, top, left + width, top + height))


def to_pil(image):
    """First frame of a ComfyUI ``IMAGE`` ``(B, H, W, C)`` in ``[0, 1]`` → RGB PIL."""
    array = image[0, ..., :3].clamp(0, 1).mul(255).round().byte().cpu().numpy()
    return Image.fromarray(array, "RGB")


def to_image(pil):
    """RGB PIL → ComfyUI ``IMAGE`` ``(1, H, W, 3)`` in ``[0, 1]``."""
    return torch.from_numpy(np.asarray(pil, dtype=np.float32) / 255.0)[None]


def page_at_target(image, resolution):
    page = to_pil(image)
    return fit(page, *target_size(page.width, page.height, resolution))


def is_core_moss(dit):
    try:
        from comfy.ldm.moss_image.model import MOSSImage
    except ImportError:
        return False
    return isinstance(dit, MOSSImage)


def _ref_timestep(value):
    return None if value in (None, "") or float(value) < 0 else float(value)


def _is_moss(model):
    dit = model.get_model_object("diffusion_model")
    return hasattr(dit, "video_head_modulation") and hasattr(dit, "final_norm_video")


def is_ai_toolkit(meta):
    """Whether ``meta`` is an ai-toolkit MOSS-Image save."""
    return meta.get("ss_base_model_version") == "moss_image" or '"ai-toolkit"' in meta.get("software", "")


def load_lora(model, lora_name, strength, task="auto", bypass=False):
    """``(model, meta)``: a MOSS LoRA on ``model`` and the reference settings it
    trained with."""
    path = folder_paths.get_full_path_or_raise("loras", lora_name)
    lora, meta = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
    meta = meta or {}
    if meta.get("base_model", "MOSS-Image") != "MOSS-Image":
        logger.warning("MOSS i2i: %s was trained on %s", lora_name, meta["base_model"])
    if strength:
        model = lora_compat.load(model, lora, strength, lora_name, bypass)
    if is_ai_toolkit(meta):
        return model, {
            "reference_latents": 1,
            "ref_scale": "1",
            "ref_timestep": 0.0,
            "ref_isolate": task != AITK_JOINT,
            "label": "ai-toolkit edit",
        }
    return model, {
        "reference_latents": int(meta.get("reference_latents", 0) or 0),
        "ref_scale": meta.get("ref_scale"),
        "ref_timestep": _ref_timestep(meta.get("ref_timestep")),
        "ref_isolate": meta.get("ref_isolate") in ("1", "True"),
    }


def ref_recipe(meta):
    """``(ref_scale, label)`` of the recipe ``meta`` trained with."""
    isolated = meta is None or meta["ref_isolate"]
    scale = 1 if isolated else 2
    if meta is not None and meta["ref_scale"]:
        scale = int(meta["ref_scale"])
    if meta is not None and "label" in meta:
        joint = "" if isolated else ", kv_cache off (joint)"
        return scale, f"{meta['label']}{joint}: full-size ref at t = 0"
    if isolated:
        return scale, "isolated ref, text-only prompt (CLIP Text Encode, no page)"
    return scale, "joint ref, page through the vision tower (MOSS i2i Text Encode)"


def _bypass_input(default):
    return (
        "BOOLEAN",
        {
            "default": default,
            "advanced": True,
            "tooltip": (
                "Add the LoRA in the forward instead of patching the weights. A patch "
                "requantizes every patched linear of an INT8 DiT, which loses a GPTQ "
                "file's weights; bypass keeps them and takes longer per forward."
            ),
        },
    )


class _ChunkedRefKV:
    """One host row per DiT block. ``_forward`` asks for ``ref_kv[:, i]``; that
    block's keys and values are copied to the GPU then and released when the
    block returns."""

    def __init__(self, rows, device, batch):
        self.rows = rows
        self.device = device
        self.batch = int(batch)
        first = rows[0]
        self.shape = (1, len(rows), 2, first.shape[-2], first.shape[-1])

    def expand(self, *args, **kwargs):
        return self

    def __getitem__(self, index):
        block = index[1] if isinstance(index, tuple) else index
        row = self.rows[block]
        if row.shape[0] != self.batch:
            row = row.expand(self.batch, -1, -1, -1)
        return row.to(self.device, non_blocking=True)


def _build_chunked_rows(dit, ref, transformer_options):
    """``preprocess_refs`` without the whole-cache GPU allocation. A single GPU
    scratch is reused for every block and copied straight into a per-block host
    row. Rows are pinned one at a time: one 3+ GB pinned allocation fails on
    Windows WDDM even when the same total in smaller chunks succeeds."""
    started = time.perf_counter()
    batch = int(ref.shape[0])
    x = dit.video_embedder(ref.flatten(2).transpose(1, 2))
    pe = dit.rope(0, 0, 0, x.device, transformer_options, [tuple(ref.shape[-2:])])
    vec = dit.time_embedding(timestep_embedding(torch.zeros(1, device=x.device), 256).to(x.dtype))
    mod = dit.shared_block_modulation(vec).unflatten(-1, (6, -1)).unsqueeze(2).unbind(1)
    row_shape = (batch, 2, x.shape[1], x.shape[2])
    try:
        probe = torch.empty(row_shape, dtype=x.dtype, pin_memory=True)
        pinned = True
    except RuntimeError:
        logger.warning("MOSS i2i: pinned host memory unavailable; using pageable CPU cache")
        probe = None
        pinned = False
        comfy.model_management.discard_cuda_async_error()
    allocated = time.perf_counter()
    rows = []
    slot = torch.empty(row_shape, device=x.device, dtype=x.dtype)
    prefetch = comfy.model_prefetch.make_prefetch_queue(list(dit.layers), x.device, transformer_options)
    for i, block in enumerate(dit.layers):
        comfy.model_prefetch.prefetch_queue_pop(prefetch, x.device, block, x.dtype)
        x = block(x, mod, pe, transformer_options, kv_out=slot)
        if i == 0 and probe is not None:
            row = probe
        else:
            try:
                row = torch.empty(row_shape, dtype=x.dtype, pin_memory=pinned)
            except RuntimeError:
                logger.warning("MOSS i2i: pinned host memory exhausted; remaining rows are pageable")
                pinned = False
                comfy.model_management.discard_cuda_async_error()
                row = torch.empty(row_shape, dtype=x.dtype)
        row.copy_(slot)
        rows.append(row)
    comfy.model_prefetch.prefetch_queue_pop(prefetch, x.device, None)
    total = sum(row.numel() * row.element_size() for row in rows)
    logger.info("MOSS i2i: chunked ref K/V cached, %.2f GB on %s (prep+alloc %.2fs, blocks %.2fs, total %.2fs)",
                total / 2**30,
                "pinned host memory" if pinned else "host memory",
                allocated - started, time.perf_counter() - allocated, time.perf_counter() - started)
    return rows


# The JSON loader rebuilds the model chain for every prompt, so a patched
# model from the previous prompt (and its host-side ref cache) can stay
# referenced by Comfy's model cache. Drop older caches when a new reference
# node is patched.
_REF_CACHES = []


def _clear_older_caches():
    for cache in list(_REF_CACHES):
        cache.clear()
    _REF_CACHES.clear()


def _make_ref_wrapper(ref, ref_timestep, ref_isolate, store, cache):
    def wrapper(executor, *args, **kwargs):
        dit = executor.class_obj.diffusion_model
        context = _executing_context()
        model_key = (
            id(dit),
            getattr(executor.class_obj, "current_weight_patches_uuid", None),
            None if context is None else context.prompt_id,
        )
        x = args[0]
        if not is_core_moss(dit):
            raise RuntimeError("MOSS i2i Reference: ComfyUI's own MOSS-Image model is required")
        if not hasattr(dit, "preprocess_refs"):
            raise RuntimeError(CORE_NO_REFS)
        options = args[5] if len(args) > 5 else kwargs.get("transformer_options", {})
        ref_device = ref.to(x.device, executor.class_obj.get_dtype_inference())
        if ref_timestep is None:
            if ref_isolate:
                raise ValueError("MOSS i2i: ComfyUI's MOSS-Image model isolates refs at t = 0 only")
            return executor(*args, **{**kwargs, "ref_latents": [ref_device],
                                       "ref_latents_method": "index"})
        if float(ref_timestep) != 0:
            raise ValueError(
                f"MOSS i2i: ComfyUI's MOSS-Image model runs refs at t = 0 or the step's, "
                f"not ref_timestep {ref_timestep:g}")
        if not ref_isolate:
            return executor(*args, **{**kwargs, "ref_latents": [ref_device],
                                       "ref_latents_method": "index_timestep_zero"})
        off_conds = {"ref_latents": [ref_device], "ref_latents_method": "index_timestep_zero_isolated"}
        if store == "off" or cache.get("disabled"):
            return executor(*args, **{**kwargs, **off_conds})
        key = (model_key, x.device)
        if cache.get("key") != key:
            cache.clear()
            oom = False
            try:
                if store == "gpu":
                    kv = dit.preprocess_refs([ref_device], options)
                else:
                    kv = _ChunkedRefKV(_build_chunked_rows(dit, ref_device, options), x.device, x.shape[0])
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                oom = True
            if oom:
                # Leave the except block first: its traceback keeps the failed
                # build's tensors alive, and the per-forward path needs that VRAM.
                logger.warning("MOSS i2i: ref cache build ran out of memory; using per-forward refs")
                cache.clear()
                cache["disabled"] = True
                comfy.model_management.discard_cuda_async_error()
                return executor(*args, **{**kwargs, **off_conds})
            cache.update(key=key, kv=kv)
        return executor(*args, **{**kwargs, "ref_kv": cache["kv"]})

    return wrapper


try:
    from comfy_execution.utils import get_executing_context as _executing_context
except ImportError:
    def _executing_context():
        return None


class SoyaMossI2iReferenceV2_mdsoya:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "lora_name": (
                    ["None"] + folder_paths.get_filename_list("loras"),
                    {"tooltip": "A LoRA trained with references (reference_latents in its metadata, "
                     "or an ai-toolkit edit LoRA). Fused (proj.qkv) or separate q/k/v keys, on "
                     "either DiT layout."},
                ),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05},
                ),
                "resolution": RESOLUTION,
                "ref_scale": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 8,
                        "tooltip": (
                            "Read only by workflows saved with task = custom; auto "
                            "takes the ref size from the LoRA (isolated ref: 1, full "
                            "size; joint ref: 2, 1/4 area)."
                        ),
                    },
                ),
                "task": (
                    list(TASKS),
                    {
                        "default": "auto",
                        "tooltip": (
                            "auto: the recipe the LoRA trained with. Isolated ref "
                            "(ref_isolate in its metadata): full size, prompt through "
                            "a plain CLIP Text Encode. Joint ref: half size, prompt through "
                            "MOSS i2i Text Encode. An ai-toolkit edit LoRA (no ref "
                            "metadata): full size at t = 0, isolated (its kv_cache); "
                            "pick 'ai-toolkit, kv_cache off' for one trained without."
                        ),
                    },
                ),
                "ref_cache": (
                    list(REF_CACHE),
                    {
                        "default": "cpu",
                        "tooltip": (
                            "An isolated ref depends on the page alone. cpu (default): build "
                            "the per-block keys/values block by block into pinned host memory, "
                            "so the GPU peak is one block instead of the whole cache; each "
                            "forward copies only that block's K/V back. gpu: keep the whole "
                            "cache on the GPU. off: run the ref rows in every forward."
                        ),
                    },
                ),
            },
            "optional": {"bypass": _bypass_input(False)},
        }

    RETURN_TYPES = ("MODEL", "LATENT")
    RETURN_NAMES = ("MODEL", "LATENT")
    FUNCTION = "patch"
    CATEGORY = "Soya/Illustration V2"
    DESCRIPTION = "MOSS i2i reference with a low-VRAM block-wise ref K/V cache."

    @classmethod
    def VALIDATE_INPUTS(cls, task):
        if task in TASKS or task in LEGACY_TASKS:
            return True
        return f"task {task!r} is not one of {TASKS}"

    def patch(self, model, vae, image, lora_name, strength, resolution, ref_scale, task="auto",
              ref_cache="cpu", bypass=False):
        if not _is_moss(model):
            raise ValueError("MOSS i2i Reference needs a MOSS-Image MODEL.")
        dit = model.get_model_object("diffusion_model")
        if not is_core_moss(dit):
            raise ValueError(
                "Soya MOSS i2i Reference supports ComfyUI's own MOSS-Image model "
                "(Load Diffusion Model on the Comfy-Org repack or vendor shards).")
        if not hasattr(dit, "preprocess_refs"):
            raise ValueError(f"MOSS i2i Reference: {CORE_NO_REFS}.")
        if ref_cache not in REF_CACHE:
            raise ValueError(f"ref_cache {ref_cache!r} is not one of {REF_CACHE}")
        meta = None
        if lora_name == "None":
            logger.warning("MOSS i2i: no LoRA; the base never saw a reference")
        else:
            model, meta = load_lora(model, lora_name, strength, task, bypass)
            if not meta["reference_latents"]:
                logger.warning(
                    "MOSS i2i: %s was trained without references; the ref is off its distribution",
                    lora_name,
                )
        if task != "custom":
            ref_scale, recipe = ref_recipe(meta)
            logger.info("MOSS i2i: %s, ref_scale %d", recipe, ref_scale)
        elif meta is not None and meta["ref_scale"] and int(meta["ref_scale"]) != ref_scale:
            logger.warning(
                "MOSS i2i: %s was trained at ref_scale %s, running %d",
                lora_name, meta["ref_scale"], ref_scale,
            )
        page = page_at_target(image, resolution)
        w, h = page.size
        rw = max(STRIDE, round(w / ref_scale / STRIDE) * STRIDE)
        rh = max(STRIDE, round(h / ref_scale / STRIDE) * STRIDE)
        ref = vae.encode(to_image(fit(page, rw, rh))).float().cpu()
        ref_timestep = None if meta is None else meta["ref_timestep"]
        ref_isolate = meta is not None and meta["ref_isolate"]
        out = model.clone()
        out.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.APPLY_MODEL, WRAPPER_KEY)
        _clear_older_caches()
        cache = {}
        _REF_CACHES.append(cache)
        out.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.APPLY_MODEL,
            WRAPPER_KEY,
            _make_ref_wrapper(ref, ref_timestep, ref_isolate, ref_cache, cache),
        )
        latent = torch.zeros(1, ref.shape[1], h // STRIDE, w // STRIDE)
        return (out, {"samples": latent})


NODE_CLASS_MAPPINGS = {
    "SoyaMossI2iReferenceV2_mdsoya": SoyaMossI2iReferenceV2_mdsoya,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SoyaMossI2iReferenceV2_mdsoya": "MOSS i2i Reference V2 (Soya)",
}
