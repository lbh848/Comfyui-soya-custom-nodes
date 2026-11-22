"""JSON-consuming V2 nodes. Each node reads its own keys from the same text.

Existing node classes and IDs retain their behavior. Models, samplers and
detailers are reused; only the V2 input contract and in-memory asset I/O differ.
"""
import base64
import copy
import hashlib
import importlib
import io
import json
import logging
import math
import re
from pathlib import Path

from PIL import Image, ImageOps

_MISSING = object()
CLIP_DEVICES = ("default", "gpu", "cpu")

COMPOSITE_SAMPLERS = {
    "SoyaFirstSampler_mdsoya", "SoyaHiresfixToggle_mdsoya", "SoyaHiresfixSpectrumToggle_mdsoya",
    "SoyaCharLoraFaceDetailer_mdsoya", "SoyaCharLoraEyeDetailer_mdsoya", "SoyaHandDetailerToggle_mdsoya",
}
PHASE_NODES = COMPOSITE_SAMPLERS | {"CLIPTextEncode", "KSampler", "MossSpectrumKSampler", "VAEDecode"}


# These nodes prepare reusable models or conditioning; none renders an image.
# Keep reuse inside the V2 wrappers so the standard Comfy cache stays untouched.
REUSABLE_BASES = frozenset({
    "CheckpointLoaderKJ", "UNETLoader", "MOSSImageUNETLoader", "CLIPLoader", "VAELoader",
    "UltralyticsDetectorProvider", "SoyaBatchLoraLoader_mdsoya",
    "PathchSageAttentionKJ", "ModelPatchTorchSettings",
    "SkimmedCFG_LinInterp_CFG_PreCFG", "DCWModelPatch",
    "CLIPSetLastLayer", "CLIPTextEncode",
})


def _load_clip_on_gpu(implementation, *, clip_name, type="stable_diffusion"):
    import comfy.model_management as model_management
    import comfy.sd
    import folder_paths

    gpu_device = model_management.get_torch_device()
    if model_management.is_device_cpu(gpu_device):
        logging.warning(
            "[Soya:CLIPDevice] GPU load requested for %s, but no GPU device is active; using default device.",
            clip_name,
        )
        return implementation.load_clip(clip_name=clip_name, type=type, device="default")
    clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
    clip_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
    logging.info("[Soya:CLIPDevice] Loading text encoder %s on GPU (%s)", clip_name, gpu_device)
    clip = comfy.sd.load_clip(
        ckpt_paths=[clip_path],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=clip_type,
        model_options={"load_device": gpu_device},
    )
    return (clip,)


def resolved_input_signature(value):
    """Compare extracted values, retaining no reference to the JSON document."""
    import torch
    if value is None or isinstance(value, (str, bool, int, float)):
        return (type(value), value)
    if isinstance(value, dict):
        return (dict, tuple((key, resolved_input_signature(item)) for key, item in sorted(value.items())))
    if isinstance(value, (tuple, list)):
        return (type(value), tuple(resolved_input_signature(item) for item in value))
    # Comfy objects are passed by reference. Patcher revisions also distinguish
    # in-place patch changes; tensors distinguish in-place writes.
    return (type(value), id(value), getattr(value, "patches_uuid", None),
            value._version if isinstance(value, torch.Tensor) and not value.is_inference() else None)


def reuse_resolved_output(node, inputs, execute, label):
    """Keep one resolved result, independent of transport JSON invalidation."""
    signature = resolved_input_signature(inputs)
    cached = getattr(node, "_resolved_cache", None)
    if cached is not None and cached[0] == signature:
        logging.info("[Soya:JSONReuse] %s: reuse", label)
        return cached[2]
    # Drop the old LoRA/model references before preparing the replacement.
    node._resolved_cache = None
    del cached
    result = execute()
    node._resolved_cache = (signature, inputs, result)
    logging.info("[Soya:JSONReuse] %s: updated", label)
    return result


def read_payload(prompt_json):
    payload = json.loads(prompt_json)
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON prompt object")
    if payload.get("schema") == "soya.illustration.v2":
        if not isinstance(payload.get("fields"), dict) or not isinstance(payload.get("assets"), dict):
            raise ValueError("Illustration JSON requires fields and assets objects")
        return payload
    # MOSS transport: {"moss": layout, "workflow": controller}. No schema key.
    if isinstance(payload.get("moss"), dict) and isinstance(payload.get("workflow"), dict):
        return payload
    raise ValueError("Expected a soya.illustration.v2 or MOSS JSON prompt")


def read_field(payload, key, default=None):
    """Read a legacy ``fields.X`` key or a dotted ``workflow.X``/``moss`` key."""
    key = str(key or "").strip()
    if key.startswith("fields."):
        root, path = payload.get("fields", {}), key[len("fields."):]
    else:
        head = key.split(".", 1)[0]
        if head in {"moss", "workflow"} and head in payload:
            root, path = payload, key
        else:
            root, path = payload.get("fields", {}), key
    value = root
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def asset_bytes(payload, key):
    asset = payload["assets"][key]
    if asset.get("encoding") != "base64":
        raise ValueError(f"Unsupported asset encoding: {key}")
    data = base64.b64decode(asset["data"], validate=True)
    if hashlib.sha256(data).hexdigest() != asset["sha256"]:
        raise ValueError(f"Illustration asset checksum mismatch: {key}")
    return data


def image_tensor(payload, key):
    import numpy as np
    import torch
    with Image.open(io.BytesIO(asset_bytes(payload, key))) as source:
        rgb = np.array(ImageOps.exif_transpose(source).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(rgb).unsqueeze(0)


def mask_tensor(payload, key):
    import numpy as np
    import torch
    with Image.open(io.BytesIO(asset_bytes(payload, key))) as source:
        gray = np.array(ImageOps.exif_transpose(source).convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(gray).unsqueeze(0)


def asset_key(payload, field):
    key = payload["fields"].get(field)
    if not isinstance(key, str) or key not in payload["assets"]:
        raise ValueError(f"Inpainting JSON requires a base64 asset referenced by fields.{field}")
    return key


class SoyaInpaintJSONV2_mdsoya:
    CATEGORY = "Soya/Illustration V2"
    FUNCTION = "resolve"
    RETURN_TYPES = ("STRING", "STRING", "IMAGE", "MASK", "INT", "INT", "FLOAT", "FLOAT", "INT", "FLOAT")
    RETURN_NAMES = ("EDIT_PROMPT", "NEGATIVE_PROMPT", "SOURCE_IMAGE", "MASK", "SEED", "STEPS", "CFG", "DENOISE", "MASK_GROW", "MASK_BLUR")
    DESCRIPTION = "Decodes the source and mask assets from the shared JSON prompt for inpainting."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"prompt_json": ("STRING", {"forceInput": True})}}

    def resolve(self, prompt_json):
        payload = read_payload(prompt_json)
        source = image_tensor(payload, asset_key(payload, "SOURCE_ASSET"))
        mask = mask_tensor(payload, asset_key(payload, "MASK_ASSET"))
        if mask.shape[-2:] != source.shape[1:3]:
            raise ValueError("Inpainting source and mask sizes differ")
        return (
            field_value(payload, "EDIT_PROMPT", "STRING"),
            field_value(payload, "NEGATIVE_PROMPT", "STRING"),
            source,
            mask,
            field_value(payload, "SEED", "INT"),
            field_value(payload, "STEPS", "INT"),
            field_value(payload, "CFG", "FLOAT"),
            field_value(payload, "DENOISE", "FLOAT"),
            field_value(payload, "MASK_GROW", "INT"),
            field_value(payload, "MASK_BLUR", "FLOAT"),
        )


def regional_masks(payload):
    fields = payload["fields"]
    multi = fields.get("MULTI_CHAR", {})
    if not multi.get("enable"):
        return []
    count = int(multi["char_num"])
    if not 1 <= count <= 3:
        raise ValueError("Regional RGB masks support one to three characters")
    rgb = image_tensor(payload, fields["REGION_MASK"])
    masks = [rgb[:, :, :, index].contiguous() for index in range(count)]
    for index, channel in enumerate(masks):
        if float(channel.max()) <= 0:
            raise ValueError(f"Regional mask channel {'RGB'[index]} is empty")
    return masks


def reference_stage_delivery(settings, stage, ref=None):
    key = "detailer_reference" if stage in {"face", "eye"} else "primary_reference"
    mode = settings.get(key) or settings.get("secondary_reference", "none")
    if mode == "llm":
        return "face_only" if ref == "face" else "none"
    return ("face_only" if stage in {"face", "eye"} else "none") if mode == "stage" else mode


def stage_reference_assets(payload, stage):
    """Select reference asset IDs for both execution and flow previews."""
    settings = payload["fields"].get("SNAPREF", {})
    if (not settings.get("enabled")
            or settings.get("strength_scale") == "linear" and float(settings.get("strength", 1.0)) <= 0):
        return {}
    grouped = {}
    for item in settings.get("references", []):
        grouped.setdefault(item["CHAR"], []).append(item)
    images = {}
    for name, refs in grouped.items():
        general, faces = [], []
        for index, item in enumerate(refs):
            # Older saved JSON represents a secondary face as a plain asset.
            is_face = item.get("role") == "face" or (index == 1 and "role" not in item
                       and settings.get("secondary_reference") == "face")
            if item.get("face_asset"):
                faces.append(item["face_asset"])
            (faces if is_face else general).append(item["asset"])
        mode = reference_stage_delivery(settings, stage, refs[0].get("ref"))
        if mode == "face_only":
            keys = faces[:1] if faces else general[:1]
        elif mode == "face":
            keys = general[:1] + faces[:1]
        elif mode == "representative":
            keys = general[:2]
        else:
            keys = general[:1]
        if keys:
            images[name] = list(dict.fromkeys(keys))
    return images


def stage_reference_images(payload, stage):
    return {name: [image_tensor(payload, key) for key in keys]
            for name, keys in stage_reference_assets(payload, stage).items()}


def field_value(payload, key, dtype):
    value = read_field(payload, key)
    if value is None:
        raise ValueError(f"JSON prompt is missing {key}")
    if dtype == "STRING":
        if isinstance(value, bool):
            return "true" if value else "false"
        if key == "CHAR_LIST":
            return ",".join(value)
        if isinstance(value, (dict, list)):
            # The MOSS layout is encoded verbatim; keep the native node's
            # compact separators so the model sees the same string.
            if key == "moss":
                return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            return json.dumps(value, ensure_ascii=False)
        return str(value)
    if dtype == "INT":
        if isinstance(value, bool) or int(value) != value:
            raise ValueError(f"{key} must be an integer")
        return int(value)
    if dtype == "FLOAT":
        return float(value)
    return value


def sampling_values(payload, sampling_key, sampling, declarations):
    settings = read_field(payload, sampling_key)
    if not isinstance(settings, dict):
        raise ValueError(f"JSON prompt requires a settings object at {sampling_key}")
    result = {}
    for name in sampling["inputs"]:
        path = f"{sampling_key}.{name}"
        if name not in settings:
            raise ValueError(f"JSON prompt is missing {path}")
        declaration = declarations[name]
        dtype = declaration[0]
        options = declaration[1] if len(declaration) > 1 else {}
        value = settings[name]
        if isinstance(dtype, list):
            if value not in dtype:
                raise ValueError(f"{path} has an unsupported value: {value!r}")
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{path} must be a number")
            if not math.isfinite(value):
                raise ValueError(f"{path} must be finite")
            if dtype == "INT" and int(value) != value:
                raise ValueError(f"{path} must be an integer")
            value = field_value({"fields": settings}, name, dtype)
            if "min" in options and value < options["min"]:
                raise ValueError(f"{path} must be at least {options['min']}")
            if "max" in options and value > options["max"]:
                raise ValueError(f"{path} must be at most {options['max']}")
        result[name] = value
    return result


class SoyaModelPreload_mdsoya:
    CATEGORY = "Soya/Util"
    FUNCTION = "load"
    RETURN_TYPES = ()
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}, "optional": {"secondary_model": ("MODEL",)}}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def load(self, model, secondary_model=None):
        import comfy.model_management as model_management
        import comfy.ops

        models = [model] + ([secondary_model] if secondary_model is not None else [])
        model_management.load_models_gpu(models)
        loaded = []
        for patcher in models:
            device = patcher.load_device
            if model_management.is_device_cpu(device):
                raise RuntimeError("그림 모델의 로드 장치가 CPU입니다. GPU 실행 설정을 확인하세요.")
            if patcher.is_dynamic():
                # Dynamic load_models_gpu only stages weights in host memory.
                # Touch each layer through the same cast/cache path used by
                # inference, releasing its pin while retaining the VRAM cache.
                dtype = getattr(patcher.model, "manual_cast_dtype", None) or patcher.model_dtype()
                for module in patcher.model.modules():
                    if not hasattr(module, "_v"):
                        continue
                    model_management.throw_exception_if_processing_interrupted()
                    with comfy.ops.CastBiasWeightContext(module, dtype=dtype, device=device, offloadable=True):
                        pass
                model_management.synchronize()
            resident_bytes = patcher.loaded_size()
            if resident_bytes <= 0:
                raise RuntimeError("그림 모델의 GPU 메모리 로드를 확인하지 못했습니다.")
            loaded.append({"model": type(patcher.model).__name__, "resident_bytes": resident_bytes})
            logging.info("[SOYA Model Preload] %s GPU resident: %.1f MiB", loaded[-1]["model"], resident_bytes / (1024 ** 2))
        return {"ui": {"text": ["그림 모델 GPU 로드 완료"], "soya_model_preload": loaded}, "result": ()}


class SoyaRenderTargetSelect_mdsoya:
    """Switch the decode target without rebuilding the shared diffusion model."""

    CATEGORY = "Soya/Illustration V2"
    FUNCTION = "select"
    RETURN_TYPES = ("MODEL", "VAE", "STRING")
    RETURN_NAMES = ("model", "vae", "sampler_prompt_json")
    DESCRIPTION = ("Reads fields.RENDER_TARGET and returns the character or background "
                   "model, decode VAE and the sampler's JSON prompt.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "prompt_json": ("STRING", {"forceInput": True}),
            "model_character": ("MODEL",),
            "model_background": ("MODEL",),
            "vae_character": ("VAE",),
            "vae_background": ("VAE",),
        }}

    def select(self, prompt_json, model_character, model_background, vae_character, vae_background):
        payload = read_payload(prompt_json)
        target = str(payload["fields"].get("RENDER_TARGET", "character")).strip().lower()
        if target not in {"character", "background"}:
            raise ValueError(f"RENDER_TARGET must be character or background: {target!r}")
        # Both branches applied their own LoRA set before this node.
        payload["fields"]["LORA_ACTIVATE"] = False
        sampler_json = json.dumps(payload, ensure_ascii=False)
        if target == "background":
            return model_background, vae_background, sampler_json
        return model_character, vae_character, sampler_json


_MOSS_PICTURE_MARKER = re.compile(r"Same character and art style as Picture \d+\.?\s*", re.IGNORECASE)


def moss_reference(payload):
    """The active ``workflow.reference`` object, or an empty dict."""
    reference = read_field(payload, "workflow.reference", {})
    return reference if isinstance(reference, dict) and reference.get("activate") else {}


# CharRef reference latents are cached as attention K/V for the whole run.
# Two 1MP pictures double that cache (~2.8GB on MOSS) and push a 16GB GPU over
# the edge, while the model card notes only Picture 1 is consistently stable.
# Keep the total reference area near the single-picture budget instead.
MOSS_REFERENCE_TOTAL_MEGAPIXELS = 1.0


def _moss_reference_megapixels(count):
    """Per-picture area budget: 1MP for one, 0.5MP for two, 0.33MP for three."""
    return MOSS_REFERENCE_TOTAL_MEGAPIXELS / max(1, min(int(count or 0), 3))


def _moss_fit_mp(tensor, megapixels=1.0):
    """Scale an IMAGE tensor to roughly ``megapixels`` megapixels, /16 snapped."""
    import comfy.utils
    height, width = int(tensor.shape[1]), int(tensor.shape[2])
    target = max(0.01, float(megapixels or 1.0)) * 1024 * 1024
    scale = (target / max(1, height * width)) ** 0.5
    new_w = max(16, int(round(width * scale / 16)) * 16)
    new_h = max(16, int(round(height * scale / 16)) * 16)
    if (new_h, new_w) == (height, width):
        return tensor
    return comfy.utils.common_upscale(tensor.movedim(-1, 1), new_w, new_h, "area", "disabled").movedim(1, -1)


def _moss_edit_prompt(payload, reference):
    """Copy the pure MOSS layout and add CharRef Picture markers for encoding.

    The stored/sent ``moss`` document is never modified. Markers are assembled
    here, at execution time, from ``workflow.reference.pictures``.
    """
    layout = copy.deepcopy(read_field(payload, "moss", {}))
    if not isinstance(layout, dict):
        return field_value(payload, "moss", "STRING")
    pictures = [entry for entry in (reference.get("pictures") or []) if isinstance(entry, dict)]
    by_name = {}
    for entry in pictures:
        name = str(entry.get("name") or "").strip().casefold()
        if name and name not in by_name:
            try:
                by_name[name] = int(entry.get("picture") or len(by_name) + 1)
            except (TypeError, ValueError):
                by_name[name] = len(by_name) + 1
    prefix = str(reference.get("prompt_prefix") or "Same character and art style as Picture 1.").strip()
    character_items = [item for item in (layout.get("items") or [])
                       if isinstance(item, dict) and item.get("label") == "character"]
    marked = 0
    for item in character_items:
        name = str(item.get("name") or "").strip().casefold()
        picture = by_name.get(name)
        if not picture:
            continue
        caption = _MOSS_PICTURE_MARKER.sub("", str(item.get("caption") or "")).strip()
        item["caption"] = f"Same character and art style as Picture {picture}. {caption}".strip()
        marked += 1
    if not marked:
        summary = layout.setdefault("summary", {})
        if isinstance(summary, dict):
            caption = _MOSS_PICTURE_MARKER.sub("", str(summary.get("caption") or "")).strip()
            summary["caption"] = f"{prefix} {caption}".strip()
    return json.dumps(layout, ensure_ascii=False, separators=(",", ":"))


class SoyaMossReferenceImagesV2_mdsoya:
    """Decode ``workflow.reference.pictures`` into the edit encoder's IMAGE inputs."""

    CATEGORY = "Soya/Illustration V2"
    FUNCTION = "images"
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("image1", "image2", "image3")
    OUTPUT_IS_LIST = (False, False, False)
    DESCRIPTION = "Decodes up to three embedded CharRef pictures from the JSON prompt."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"prompt_json": ("STRING", {"forceInput": True})}}

    def images(self, prompt_json):
        import torch
        payload = read_payload(prompt_json)
        reference = moss_reference(payload)
        pictures = [entry for entry in (reference.get("pictures") or []) if isinstance(entry, dict)]
        placeholder = torch.zeros(1, 64, 64, 3)
        megapixels = _moss_reference_megapixels(len(pictures))
        # Keep one slot per Picture: a corrupt/missing asset must not shift the
        # remaining images onto the wrong ``Picture N`` marker.
        tensors = []
        for entry in pictures[:3]:
            asset = entry.get("asset")
            if not isinstance(asset, str) or asset not in (payload.get("assets") or {}):
                tensors.append(None)
                continue
            tensors.append(_moss_fit_mp(image_tensor(payload, asset), megapixels))
        if pictures:
            logging.info(
                "[MOSS:CHARREF] pictures=%d names=%s mp=%.2f shapes=%s",
                len(pictures), [entry.get("name") for entry in pictures[:3]],
                megapixels,
                [None if tensor is None else tuple(tensor.shape) for tensor in tensors],
            )
        while len(tensors) < 3:
            tensors.append(None)
        return tuple(placeholder if tensor is None else tensor for tensor in tensors)


class SoyaMossPromptEncodeV2_mdsoya:
    """Dual-mode MOSS text encode: pure layout, or CharRef edit conditioning."""

    CATEGORY = "Soya/Illustration V2"
    FUNCTION = "encode"
    RETURN_TYPES = ("CONDITIONING",)
    DESCRIPTION = "Encodes the MOSS layout, or the CharRef reference/edit conditioning."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt_json": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "vae": ("VAE",),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
            },
        }

    def encode(self, clip, prompt_json, vae=None, image1=None, image2=None, image3=None):
        payload = read_payload(prompt_json)
        sequential_offload = read_field(payload, "workflow.sequential_offload", False) is True
        reference = moss_reference(payload)
        if not reference:
            from .soya_model_memory import model_phase
            with model_phase(clip.patcher, sequential_offload, "text encoder"):
                return (clip.encode_from_tokens_scheduled(clip.tokenize(field_value(payload, "moss", "STRING"))),)
        import folder_paths
        lora_name = str(reference.get("lora") or "").strip()
        if lora_name and lora_name not in folder_paths.get_filename_list("loras"):
            raise ValueError(f"MOSS CharRef LoRA를 찾을 수 없습니다: {lora_name}")
        from .soya_moss_edit import encode_moss_edit, reference_latent_shapes
        pictures = [entry for entry in (reference.get("pictures") or []) if isinstance(entry, dict)]
        images = [image for image in (image1, image2, image3) if image is not None][:len(pictures)]
        prompt = _moss_edit_prompt(payload, reference)
        conditioning = encode_moss_edit(clip, prompt, vae, images, sequential_offload=sequential_offload)
        logging.info(
            "[MOSS:CHARREF] prompt_head=%r pictures=%d ref_latents=%s",
            prompt[:180], len(pictures), reference_latent_shapes(conditioning),
        )
        return (conditioning,)


class SoyaMossNegativeEncodeV2_mdsoya:
    """Dual-mode MOSS negative text encode."""

    CATEGORY = "Soya/Illustration V2"
    FUNCTION = "encode"
    RETURN_TYPES = ("CONDITIONING",)
    DESCRIPTION = "Encodes the selected MOSS negative prompt for both modes."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"clip": ("CLIP",), "prompt_json": ("STRING", {"forceInput": True})}}

    def encode(self, clip, prompt_json):
        payload = read_payload(prompt_json)
        sequential_offload = read_field(payload, "workflow.sequential_offload", False) is True
        negative = field_value(payload, "workflow.negative", "STRING")
        if moss_reference(payload):
            from .soya_moss_edit import encode_moss_edit
            return (encode_moss_edit(clip, negative, None, [], sequential_offload=sequential_offload),)
        from .soya_model_memory import model_phase
        with model_phase(clip.patcher, sequential_offload, "text encoder"):
            return (clip.encode_from_tokens_scheduled(clip.tokenize(negative)),)


class SoyaMossOptionalLoraV2_mdsoya:
    """Apply the 4STEP LoRA with the strength configured in the workflow."""

    CATEGORY = "Soya/Illustration V2"
    FUNCTION = "load"
    RETURN_TYPES = ("MODEL",)
    DESCRIPTION = "Applies a MOSS LoRA with the workflow's configured strength in every mode."

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {"required": {
            "model": ("MODEL",),
            "lora_name": (folder_paths.get_filename_list("loras"),),
            "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            "prompt_json": ("STRING", {"forceInput": True}),
        }}

    def load(self, model, lora_name, strength, prompt_json):
        import folder_paths
        from .soya_batch_lora_loader import SoyaBatchLoraLoader_mdsoya
        from .soya_moss_spatial_lora import _BASE_KEY, _KEY, attach_spatial_lora
        strength = float(strength)
        if not strength:
            self._lora_loader = None
            return (model,)
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        if getattr(self, "_lora_loader", None) is None:
            self._lora_loader = SoyaBatchLoraLoader_mdsoya()
        # The spatial node creates a new wrapper for each character/bbox plan.
        # Cache the fixed 4STEP weights against its stable global-style base,
        # then attach this request's regions to the cached weight patcher.
        base = model.get_attachment(_BASE_KEY)
        patched, _, _ = self._lora_loader.load_batch_lora(
            base if base is not None else model, None, "true", "false",
            json.dumps({"list": [{"lora_path": path, "str": strength}]}), "")
        if base is not None:
            patched = attach_spatial_lora(patched, model.model_options["transformer_options"][_KEY])
        return (patched,)


class SoyaMossSkipLayerGuidanceV2_mdsoya:
    """Apply Skip Layer Guidance with the values configured in the workflow."""

    CATEGORY = "Soya/Illustration V2"
    FUNCTION = "patch"
    RETURN_TYPES = ("MODEL",)
    DESCRIPTION = "Skip Layer Guidance for MOSS using the configured layer and range."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "double_layers": ("STRING", {"default": ""}),
            "single_layers": ("STRING", {"default": "22"}),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "end_percent": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "prompt_json": ("STRING", {"forceInput": True}),
        }}

    def patch(self, model, double_layers, single_layers, start_percent, end_percent, prompt_json):
        import nodes
        implementation = nodes.NODE_CLASS_MAPPINGS["SkipLayerGuidanceDiTSimple"]()
        return getattr(implementation, implementation.FUNCTION)(
            model=model, double_layers=double_layers, single_layers=single_layers,
            start_percent=start_percent, end_percent=end_percent)


class SoyaMossSwapTextEncodeV2_mdsoya:
    """Encode a swap_text layout JSON with the Hangul syllable split.

    The MOSS Toolkit's CaptionHelper hard-imports anime-tools, which requires
    Python 3.13 and torch >= 2.12. swap_text only needs the layout string and
    the tokenizer split, so this node does that directly with ComfyUI's CLIP.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt_json": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "caption_key": ("STRING", {"default": "workflow.detailer.layout"}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "Soya/Illustration V2"
    DESCRIPTION = "MOSS swap_text layout JSON -> CONDITIONING (Hangul syllable split)."

    @staticmethod
    def _read(payload, key):
        value = payload
        for part in str(key).strip().removeprefix("fields.").split("."):
            if not isinstance(value, dict) or part not in value:
                raise ValueError(f"JSON prompt is missing {key}")
            value = value[part]
        return value

    def encode(self, clip, prompt_json, caption_key="workflow.detailer.layout"):
        # Imported here so the module also loads as a standalone test fixture.
        from .soya_moss_token_split import has_hangul, split_clip

        payload = read_payload(prompt_json)
        text = self._read(payload, caption_key)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"JSON prompt requires a layout string at {caption_key}")
        text = text.strip()
        encoder = split_clip(clip) if has_hangul(text) else clip
        from .soya_model_memory import model_phase
        with model_phase(encoder.patcher, read_field(payload, "workflow.sequential_offload", False) is True,
                              "text encoder"):
            return (encoder.encode_from_tokens_scheduled(encoder.tokenize(text)),)


def register_illustration_v2(class_mappings, display_mappings):
    from .soya_moss_edit import SoyaMOSSEditModelPatch_mdsoya
    class_mappings["SoyaMOSSEditModelPatch_mdsoya"] = SoyaMOSSEditModelPatch_mdsoya
    display_mappings["SoyaMOSSEditModelPatch_mdsoya"] = "MOSS Edit Model Patch (Soya)"
    from .soya_moss_i2i_reference import SoyaMossI2iReferenceV2_mdsoya
    class_mappings["SoyaMossI2iReferenceV2_mdsoya"] = SoyaMossI2iReferenceV2_mdsoya
    display_mappings["SoyaMossI2iReferenceV2_mdsoya"] = "MOSS i2i Reference V2 (Soya)"
    from .soya_moss_dialogue_repair import SoyaMossDialogueRepairV2_mdsoya
    class_mappings["SoyaMossDialogueRepairV2_mdsoya"] = SoyaMossDialogueRepairV2_mdsoya
    display_mappings["SoyaMossDialogueRepairV2_mdsoya"] = "MOSS Dialogue Repair V2 (Soya)"
    class_mappings["SoyaMossReferenceImagesV2_mdsoya"] = SoyaMossReferenceImagesV2_mdsoya
    display_mappings["SoyaMossReferenceImagesV2_mdsoya"] = "MOSS Reference Images V2 (Soya JSON)"
    class_mappings["SoyaMossPromptEncodeV2_mdsoya"] = SoyaMossPromptEncodeV2_mdsoya
    display_mappings["SoyaMossPromptEncodeV2_mdsoya"] = "MOSS Prompt Encode V2 (Soya JSON)"
    class_mappings["SoyaMossSwapTextEncodeV2_mdsoya"] = SoyaMossSwapTextEncodeV2_mdsoya
    display_mappings["SoyaMossSwapTextEncodeV2_mdsoya"] = "MOSS Swap Text Encode V2 (Soya JSON)"
    class_mappings["SoyaMossNegativeEncodeV2_mdsoya"] = SoyaMossNegativeEncodeV2_mdsoya
    display_mappings["SoyaMossNegativeEncodeV2_mdsoya"] = "MOSS Negative Encode V2 (Soya JSON)"
    class_mappings["SoyaMossOptionalLoraV2_mdsoya"] = SoyaMossOptionalLoraV2_mdsoya
    display_mappings["SoyaMossOptionalLoraV2_mdsoya"] = "MOSS Optional LoRA V2 (Soya JSON)"
    class_mappings["SoyaMossSkipLayerGuidanceV2_mdsoya"] = SoyaMossSkipLayerGuidanceV2_mdsoya
    display_mappings["SoyaMossSkipLayerGuidanceV2_mdsoya"] = "MOSS Skip Layer Guidance V2 (Soya JSON)"
    class_mappings["SoyaRenderTargetSelect_mdsoya"] = SoyaRenderTargetSelect_mdsoya
    display_mappings["SoyaRenderTargetSelect_mdsoya"] = "Render Target Select (Soya JSON)"
    class_mappings["SoyaInpaintJSONV2_mdsoya"] = SoyaInpaintJSONV2_mdsoya
    display_mappings["SoyaInpaintJSONV2_mdsoya"] = "Inpainting Inputs V2 (Soya JSON)"
    class_mappings["SoyaModelPreload_mdsoya"] = SoyaModelPreload_mdsoya
    display_mappings["SoyaModelPreload_mdsoya"] = "Load Image Model (Soya)"
    specs = json.loads(Path(__file__).with_name("illustration_v2_specs.json").read_text(encoding="utf-8"))
    # Original SOYA classes are available here. Other packages finish loading
    # before INPUT_TYPES or execution resolves their registered classes.
    originals = dict(class_mappings)

    def make_node(spec):
        sampling = spec.get("sampling")
        model = spec.get("model")
        clip_device = spec.get("clip_device")
        reuse_outputs = spec["base"] in REUSABLE_BASES

        def base_class():
            if spec["base"] in originals:
                base = originals[spec["base"]]
            else:
                import nodes
                base = nodes.NODE_CLASS_MAPPINGS[spec["base"]]
            core = {"SoyaFirstSampler_mdsoya": "soya_first_sampler_v2_core",
                    "SoyaIPAPatchMaker_mdsoya": "soya_face_match_v2_core",
                    "SoyaCharLoraFaceDetailer_mdsoya": "soya_char_lora_face_detailer_v2_core",
                    "SoyaCharLoraEyeDetailer_mdsoya": "soya_char_lora_eye_detailer_v2_core",
                    "SoyaHandDetailerToggle_mdsoya": "soya_hand_detailer_toggle_v2_core"}.get(spec["base"])
            if core:
                module = importlib.import_module(base.__module__.rsplit(".", 1)[0] + "." + core)
                return getattr(module, spec["base"])
            return base

        class JsonNode:
            RETURN_TYPES = tuple(spec["output"])
            RETURN_NAMES = tuple(spec["output_name"])
            OUTPUT_IS_LIST = tuple(spec["output_is_list"])
            INPUT_IS_LIST = spec.get("input_is_list", False)
            OUTPUT_NODE = spec.get("output_node", False)
            FUNCTION = "execute"
            CATEGORY = "Soya/Illustration V2"
            DESCRIPTION = "Reads its own fields from the shared illustration JSON prompt."

            @classmethod
            def INPUT_TYPES(cls):
                try:
                    implementation = base_class()
                except (KeyError, AttributeError):
                    # Optional third-party bases (for example the OpenMOSS MOSS
                    # loaders) may not be installed. Keep this node harmless so
                    # ComfyUI's global object_info request still succeeds.
                    logging.warning(
                        "[Soya:JSONReuse] 기반 노드가 없어 %s를 비활성화합니다: %s",
                        spec["name"], spec["base"])
                    return {"required": {"prompt_json": ("STRING", {"forceInput": True})}}
                result = copy.deepcopy(implementation.INPUT_TYPES())
                for group in ("required", "optional"):
                    for name in list(result.get(group, {})):
                        if (name in spec["json_inputs"] or name in spec.get("fixed_inputs", {})
                                or sampling and name in sampling["inputs"]
                                or model and name == model["input"] and not model.get("fallback_widget")):
                            result[group].pop(name)
                result.setdefault("required", {})["prompt_json"] = ("STRING", {"forceInput": True})
                if spec["base"] == "CLIPTextEncode":
                    result.setdefault("optional", {})["json_key"] = ("STRING", {
                        "default": spec["json_inputs"]["text"]["key"],
                        "multiline": False,
                        "tooltip": "JSON field to encode, e.g. ANIMA_CONTENT, SDXL_CONTENT or NEGATIVE_ANIMA. "
                                   "Separate keys with commas to combine them in order, e.g. ANIMA_QUALITY, ANIMA_CONTENT. "
                                   "The fields. prefix is optional for each key.",
                    })
                if sampling:
                    result.setdefault("optional", {})["sampling_key"] = ("STRING", {
                        "default": sampling["key"], "multiline": False,
                        "tooltip": "JSON settings object under fields. Reads " + ", ".join(sampling["inputs"])
                                   + ". Choose the ANIMA or ILXL block for this stage.",
                    })
                if model and not model.get("fallback_widget"):
                    result.setdefault("optional", {})["model_key"] = ("STRING", {
                        "default": model["key"], "multiline": False,
                        "tooltip": "JSON field containing the installed model filename, e.g. " + model["key"]
                                   + ". The fields. prefix is optional.",
                    })
                if clip_device and clip_device.get("fallback_widget"):
                    result.setdefault("optional", {})[clip_device["input"]] = (list(CLIP_DEVICES), {"advanced": True})
                if spec["base"] == "SoyaFirstSampler_mdsoya":
                    # Keep existing saved widget positions, including sampling_key.
                    optional = result["optional"]
                    optional["snapref_fidelity"] = optional.pop("snapref_fidelity")
                if spec["base"] == "SoyaHiresfixToggle_mdsoya" and sampling["key"] == "ANIMA_HIRESFIX":
                    result.setdefault("optional", {})["clip"] = ("CLIP",)
                return result

            def prepare_inputs(self, prompt_json, kwargs):
                # Keep the large parsed payload in this short-lived frame only.
                if self.INPUT_IS_LIST:
                    prompt_json = prompt_json[0]
                json_keys = None
                if spec["base"] == "CLIPTextEncode":
                    json_key = kwargs.pop("json_key", spec["json_inputs"]["text"]["key"])
                    if self.INPUT_IS_LIST:
                        json_key = json_key[0]
                    json_keys = [key.strip().removeprefix("fields.") for key in json_key.split(",")]
                    if not all(json_keys):
                        raise ValueError("json_key must contain non-empty field names, e.g. ANIMA_QUALITY, ANIMA_CONTENT")
                payload = read_payload(prompt_json)
                resolved = dict(spec.get("fixed_inputs", {}))
                if model:
                    model_key = kwargs.pop("model_key", model["key"])
                    if self.INPUT_IS_LIST:
                        model_key = model_key[0]
                    model_key = model_key.strip().removeprefix("fields.")
                    if not model_key:
                        raise ValueError("model_key must name a JSON field containing a model filename")
                    fallback = kwargs.get(model["input"]) if model.get("fallback_widget") else None
                    filename = read_field(payload, model_key, fallback)
                    if not isinstance(filename, str) or not filename.strip():
                        raise ValueError(f"JSON prompt requires a model filename at {model_key}")
                    filename = filename.strip()
                    schema = base_class().INPUT_TYPES()
                    declarations = {**schema.get("required", {}), **schema.get("optional", {})}
                    if filename not in declarations[model["input"]][0]:
                        raise ValueError(f"Model {filename!r} from {model_key} is not available for {spec['base']}")
                    resolved[model["input"]] = filename
                if clip_device:
                    device_value = read_field(payload, clip_device["key"])
                    if device_value is not None:
                        if not isinstance(device_value, str):
                            raise ValueError(f"{clip_device['key']} must be default, gpu, or cpu: {device_value!r}")
                        normalized_device = device_value.strip().lower()
                        if normalized_device not in CLIP_DEVICES:
                            raise ValueError(f"{clip_device['key']} must be default, gpu, or cpu: {device_value!r}")
                        resolved[clip_device["input"]] = normalized_device
                for name, binding in spec["json_inputs"].items():
                    if name == "text" and json_keys is not None:
                        values = []
                        for key in json_keys:
                            if read_field(payload, key, _MISSING) is _MISSING:
                                raise ValueError(f"JSON prompt is missing {key}; check json_key on this CLIP Text Encode node")
                            values.append(field_value(payload, key, binding["type"]))
                        resolved[name] = ", ".join(value for value in values if value)
                    else:
                        resolved[name] = field_value(payload, binding["key"], binding["type"])
                if sampling:
                    sampling_key = kwargs.pop("sampling_key", sampling["key"])
                    if self.INPUT_IS_LIST:
                        sampling_key = sampling_key[0]
                    sampling_key = sampling_key.strip().removeprefix("fields.")
                    if not sampling_key:
                        raise ValueError("sampling_key must name a JSON settings object")
                    schema = base_class().INPUT_TYPES()
                    declarations = {**schema.get("required", {}), **schema.get("optional", {})}
                    resolved.update(sampling_values(payload, sampling_key, sampling, declarations))
                if self.INPUT_IS_LIST:
                    resolved = {key: [value] for key, value in resolved.items()}
                kwargs.update(resolved)
                fields = payload.get("fields", {})
                if spec["base"] == "SoyaFirstSampler_mdsoya":
                    multi = fields.get("MULTI_CHAR", {})
                    masks = regional_masks(payload)
                    refs = fields.get("SNAPREF", {}).get("references", [])
                    names = multi.get("char_name_list", []) if multi.get("enable") else list(dict.fromkeys(item["CHAR"] for item in refs))
                    if not multi.get("enable") and len(names) > 1:
                        raise ValueError("Multiple SnapRef references require regional RGB masks")
                    by_name = stage_reference_images(payload, "first")
                    images = {index + 1: by_name[name] for index, name in enumerate(names) if name in by_name}
                    kwargs.update(_embedded_region_masks=masks, _embedded_reference_images=images)
                elif spec["base"] in {"SoyaHiresfixToggle_mdsoya", "SoyaHiresfixSpectrumToggle_mdsoya"} and sampling["key"] == "ANIMA_HIRESFIX":
                    if str(kwargs["enable"]).strip().lower() in {"true", "1", "yes", "on"}:
                        from .soya_face_tracking import TRACKED_REGION_MASKS
                        multi = fields.get("MULTI_CHAR", {})
                        tracked_masks = kwargs["model"].get_attachment(TRACKED_REGION_MASKS) if multi.get("enable") else None
                        if tracked_masks:
                            print("[FACE_TRACK_MASK] ANIMA hires.fix using tracked regional masks")
                        kwargs["_hires_regions"] = {
                            "multi": multi, "masks": tracked_masks or regional_masks(payload),
                            "lora_activate": fields.get("LORA_ACTIVATE", False),
                            "lora_data": fields.get("LORA_DATA", {"list": []}),
                            "images": stage_reference_images(payload, "hires"),
                            "settings": {key: value for key, value in fields.get("SNAPREF", {}).items() if key != "references"},
                        }
                elif spec["base"] in {"SoyaCharLoraFaceDetailer_mdsoya", "SoyaCharLoraEyeDetailer_mdsoya"}:
                    if kwargs.get("base_model", "").strip() == "anima" and str(kwargs["enable"]).strip().lower() in {"true", "1", "yes"}:
                        stage = "face" if spec["base"] == "SoyaCharLoraFaceDetailer_mdsoya" else "eye"
                        kwargs["_reference_images"] = stage_reference_images(payload, stage)
                        kwargs["_reference_settings"] = {key: value for key, value in fields.get("SNAPREF", {}).items() if key != "references"}
                elif spec["base"] == "SoyaHandDetailerToggle_mdsoya" and sampling["key"] == "ANIMA_HAND_DETAILER":
                    if str(kwargs["enable"]).strip().lower() in {"true", "1", "yes", "on"}:
                        images = stage_reference_images(payload, "hand")
                        if images:
                            multi = fields.get("MULTI_CHAR", {})
                            kwargs["_hand_references"] = {
                                "masks": regional_masks(payload),
                                "names": multi["char_name_list"] if multi.get("enable") else list(images),
                                "images": images,
                                "settings": {key: value for key, value in fields.get("SNAPREF", {}).items() if key != "references"},
                            }
                elif spec["base"] == "SoyaIPAPatchMaker_mdsoya":
                    import torch
                    entries = fields.get("FACE_EMBEDDINGS", [])
                    kwargs["embed_cache_data"] = json.dumps({"list": [
                        {"CHAR": entry["CHAR"], "emb_path": entry["asset"]} for entry in entries]})
                    kwargs["_embedded_caches"] = {
                        entry["CHAR"]: torch.load(io.BytesIO(asset_bytes(payload, entry["asset"])),
                                                  map_location="cpu", weights_only=True)
                        for entry in entries
                    }
                if spec["base"] in PHASE_NODES:
                    kwargs["_sequential_offload"] = read_field(payload, "workflow.sequential_offload", False) is True
                del payload, fields
                return kwargs

            def execute(self, prompt_json, **kwargs):
                kwargs = self.prepare_inputs(prompt_json, kwargs)
                sequential_offload = kwargs.pop("_sequential_offload", False)
                if spec["base"] == "SoyaBatchLoraLoader_mdsoya":
                    # Style/background LoRAs stay merged and cached. Only the
                    # changing ANIMA character branch uses residual adapters.
                    kwargs["_character_loras"] = (
                        spec["json_inputs"]["lora_list"]["key"] == "LORA_DATA"
                        and kwargs.get("base_filter", "").strip() == "anima"
                    )
                # The graph owns the shared source; this node no longer needs it.
                del prompt_json
                if (sequential_offload and spec["base"] in COMPOSITE_SAMPLERS
                        and str(kwargs.get("enable", True)).strip().lower() in {"true", "1", "yes", "on"}):
                    from .soya_model_memory import PhaseIO, sequential_sampling_model
                    if kwargs.get("clip") is not None:
                        kwargs["clip"] = PhaseIO(kwargs["clip"], "text encoder", {"encode_from_tokens", "encode_from_tokens_scheduled"})
                    if kwargs.get("vae") is not None:
                        kwargs["vae"] = PhaseIO(kwargs["vae"], "VAE", {"encode", "encode_tiled", "decode", "decode_tiled"})
                    if spec["base"] == "SoyaFirstSampler_mdsoya":
                        # Preserve the actual input patcher as the character LoRA cache key.
                        kwargs["_sequential_offload"] = True
                    else:
                        kwargs["model"] = sequential_sampling_model(kwargs["model"])
                regions = kwargs.pop("_hires_regions", None)
                if regions is not None:
                    from .soya_illustration_stages import prepare_hires_model
                    kwargs["model"] = prepare_hires_model(kwargs["model"], kwargs.get("clip"), kwargs["vae"],
                                                          kwargs["positive"], kwargs["image"], **regions)
                if spec["base"] == "SoyaHiresfixToggle_mdsoya":
                    kwargs.pop("clip", None)
                hand = kwargs.pop("_hand_references", None)
                if hand is not None:
                    from functools import partial
                    from .soya_illustration_stages import prepare_hand_model
                    kwargs["_reference_model"] = partial(prepare_hand_model, clip=kwargs["clip"], **hand)
                if not hasattr(self, "_implementation"):
                    self._implementation = base_class()()
                def execute_implementation():
                    if spec["base"] == "CLIPLoader" and kwargs.get("device") == "gpu":
                        return _load_clip_on_gpu(
                            self._implementation,
                            clip_name=kwargs["clip_name"],
                            type=kwargs.get("type", "stable_diffusion"),
                        )
                    execute = getattr(self._implementation, self._implementation.FUNCTION)
                    if not sequential_offload or spec["base"] in COMPOSITE_SAMPLERS:
                        return execute(**kwargs)
                    from .soya_model_memory import model_phase
                    sampling = spec["base"] in {"KSampler", "MossSpectrumKSampler"}
                    if sampling:
                        patcher, phase = kwargs["model"], "DiT"
                    elif spec["base"] == "VAEDecode":
                        patcher, phase = kwargs["vae"].patcher, "VAE"
                    else:
                        patcher, phase = kwargs["clip"].patcher, "text encoder"
                    with model_phase(patcher, True, phase, release_after=sampling):
                        return execute(**kwargs)
                if reuse_outputs:
                    return reuse_resolved_output(
                        self, kwargs, execute_implementation, spec["name"])
                return execute_implementation()

        if model and model.get("fallback_widget"):
            # The graph widget may name a different file than the prompt. Check
            # the resolved filename against Comfy's inventory in prepare_inputs.
            JsonNode.VALIDATE_INPUTS = staticmethod(lambda clip_name: isinstance(clip_name, str))
        if spec["base"] == "SoyaFirstSampler_mdsoya":
            # Queueing the same fixed seed explicitly requests another sample.
            # Only this node and its descendants are invalidated, not loaders.
            JsonNode.IS_CHANGED = classmethod(lambda cls, **kwargs: float("nan"))
        JsonNode.__name__ = spec["name"]
        return JsonNode

    for name, spec in specs.items():
        class_mappings[name] = make_node(spec)
        display_mappings[name] = spec["display_name"]
    from .soya_spatial_face_context import NODE_NAME, SoyaSpatialFaceContextV2
    class_mappings[NODE_NAME] = SoyaSpatialFaceContextV2
    display_mappings[NODE_NAME] = "Spatial Face Context V2 (no identity cache)"
