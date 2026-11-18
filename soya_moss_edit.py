"""MOSS-Image reference-image (edit) conditioning, integrated into SOYA.

``encode_moss_edit`` encodes the prompt + reference images through the
MOSS-Image Qwen3.5 text encoder (edit-plus style ``Picture N:`` vision
placeholders inside MOSS's own chat template) and attaches the VAE reference
latents to the conditioning.

ComfyUI's own MOSS-Image model implements the reference mechanism itself
(``ref_latents`` / ``ref_latents_method``, ``preprocess_refs``), mirroring the
ai-toolkit ``moss_image`` edit training implementation: VL images are
downscaled (never upscaled) to fit 384x384 total pixels; reference latents are
encoded at the size the image is fed in (snapped to /16, the FLUX.2 VAE's
latent grid) -- scale the reference upstream to pick its token budget.

``SoyaMOSSEditModelPatch_mdsoya``'s ``kv_cache`` toggle (default ON, matching
ai-toolkit's edit models) selects ``index_timestep_zero_isolated``: reference
tokens attend only to each other, so their per-block K/V are computed once at
t=0 and injected as extra keys on every denoising step, and the refs never
ride along in the per-step sequence. A LoRA trained without ``kv_cache`` needs
``index_timestep_zero`` (toggle off), where the references ride along at t=0.
"""

import math

import comfy.utils
import node_helpers

# ---------------------------------------------------------------------------
# Text encode (VL images -> conditioning, refs -> reference_latents)
# ---------------------------------------------------------------------------

VLM_MAX_PIXELS = 384 * 384  # Qwen3.5 sees a coarse reference
REF_SNAP = 16  # FLUX.2 VAE: f8 conv + 2x2 packing -> 16x latent grid


def _fit_area(samples, max_pixels):
    """Downscale (never upscale) a (B, C, H, W) image to fit ``max_pixels``,
    keeping aspect."""
    h, w = samples.shape[2], samples.shape[3]
    scale = min(1.0, math.sqrt(max_pixels / (w * h)))
    nw, nh = max(round(w * scale), 1), max(round(h * scale), 1)
    if (nh, nw) == (h, w):
        return samples
    return comfy.utils.common_upscale(samples, nw, nh, "area", "disabled")


def _snap(samples, snap):
    """Round a (B, C, H, W) image's dims to the nearest multiple of ``snap``.
    The reference is otherwise used at the size it is fed in."""
    h, w = samples.shape[2], samples.shape[3]
    nw = max(round(w / snap) * snap, snap)
    nh = max(round(h / snap) * snap, snap)
    if (nh, nw) == (h, w):
        return samples
    return comfy.utils.common_upscale(samples, nw, nh, "area", "disabled")


def encode_moss_edit(clip, prompt, vae=None, images=None, *, sequential_offload=False):
    """Encode ``prompt`` plus reference IMAGE tensors for a MOSS edit LoRA.

    ``images`` is an ordered list (Picture 1, Picture 2, ...). The MOSS Qwen3.5
    text encoder receives the page through its vision tower; when ``vae`` is
    given, each image is also attached as a VAE reference latent under
    ``reference_latents`` for the model's own index_timestep_zero method.
    """
    images_vl = []
    ref_latents = []
    image_prompt = ""

    for image in (images or []):
        if image is None:
            continue
        samples = image.movedim(-1, 1)
        images_vl.append(_fit_area(samples, VLM_MAX_PIXELS).movedim(1, -1))
        if vae is not None:
            # no resize: the reference's resolution is the caller's call
            s = _snap(samples, REF_SNAP)
            from .soya_moss_memory import moss_model_phase
            with moss_model_phase(vae.patcher, sequential_offload, "reference VAE"):
                ref_latents.append(vae.encode(s.movedim(1, -1)[:, :, :, :3]))
        image_prompt += (
            "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(
                len(images_vl)
            )
        )

    # MOSS's own template (user turn + open think block), applied by the
    # moss_image tokenizer: the trained LoRA saw the Picture markers at the
    # start of the user content.
    tokens = clip.tokenize(image_prompt + prompt, images=images_vl)
    from .soya_moss_memory import moss_model_phase
    with moss_model_phase(clip.patcher, sequential_offload, "text encoder"):
        conditioning = clip.encode_from_tokens_scheduled(tokens)
    if len(ref_latents) > 0:
        conditioning = node_helpers.conditioning_set_values(
            conditioning, {"reference_latents": ref_latents}, append=True
        )
    return conditioning


def reference_latent_shapes(conditioning):
    """``[(B, C, H, W), ...]`` attached to a conditioning, for diagnostics."""
    shapes = []
    for item in conditioning or []:
        values = item[1] if isinstance(item, (list, tuple)) and len(item) > 1 else {}
        for latent in values.get("reference_latents") or []:
            try:
                shapes.append(tuple(latent.shape))
            except AttributeError:
                shapes.append(type(latent).__name__)
    return shapes


class TextEncodeMOSSImageOstrisEdit:
    """Compatibility node kept for saved workflows; SOYA uses encode_moss_edit."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            },
            "optional": {
                "vae": ("VAE",),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "Soya/Illustration V2"
    DESCRIPTION = "MOSS-Image edit text encode (SOYA integration)."

    def encode(self, clip, prompt, vae=None, image1=None, image2=None, image3=None):
        return (encode_moss_edit(clip, prompt, vae, [image1, image2, image3]),)


# ---------------------------------------------------------------------------
# Model patch (select the reference method for the conditioning)
# ---------------------------------------------------------------------------


class SoyaMOSSEditModelPatch_mdsoya:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "kv_cache": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Cache the reference tokens' attention K/V: they "
                            "are precomputed in one t=0 pass and reused every "
                            "denoising step, so the refs never ride along in "
                            "the per-step sequence. Matches LoRAs trained with "
                            "ai-toolkit's kv_cache model kwarg (the default "
                            "for its MOSS-Image edit models). Turn off for a "
                            "LoRA trained with kv_cache disabled."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "Soya/Illustration V2"
    DESCRIPTION = (
        "Enable reference latents on a MOSS-Image model (index_timestep_zero "
        "method, as trained by ai-toolkit). Chain conditioning from "
        "SoyaMossPromptEncodeV2_mdsoya. kv_cache (default on) is the cached "
        "one-pass reference mode ai-toolkit's edit models train with; turn it "
        "off for a LoRA trained with kv_cache disabled."
    )

    def patch(self, model, kv_cache=True):
        from .soya_illustration_json_v2 import reuse_resolved_output
        return reuse_resolved_output(
            self, (model, kv_cache), lambda: self._patch(model, kv_cache),
            "SoyaMOSSEditModelPatch_mdsoya")

    def _patch(self, model, kv_cache):
        # ComfyUI's MOSSImage implements the references itself: extra_conds
        # turns ``reference_latents`` into ``ref_latents`` and, for a method
        # other than index/index_timestep_zero, precomputes the K/V once
        # (``preprocess_refs``). The patch only has to pick the method.
        method = "index_timestep_zero_isolated" if kv_cache else "index_timestep_zero"
        m = model.clone()
        orig_extra_conds = m.model.extra_conds

        def extra_conds(**kwargs):
            if kwargs.get("reference_latents"):
                kwargs.setdefault("reference_latents_method", method)
            return orig_extra_conds(**kwargs)

        m.add_object_patch("extra_conds", extra_conds)
        return (m,)


NODE_CLASS_MAPPINGS = {
    "SoyaMOSSEditModelPatch_mdsoya": SoyaMOSSEditModelPatch_mdsoya,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SoyaMOSSEditModelPatch_mdsoya": "MOSS Edit Model Patch (Soya)",
}

__all__ = ("SoyaMOSSEditModelPatch_mdsoya", "encode_moss_edit", "reference_latent_shapes")
