# Comfyui-soya-custom-nodes

`SoyaFirstSampler_mdsoya` (`1st sampler`) uses shared-pass spatial LoRA routing
when `MULTI_CHAR=true`. The existing node ID, inputs, outputs, and sampler-mode
choices are unchanged: restart ComfyUI to load the updated code; no workflow
rewiring or configuration changes are needed. There is no separate experimental
sampler node or switch. Single-character sampling keeps its existing global-LoRA
behavior. Common style LoRAs remain on the incoming model; do not additionally
merge multi-character LoRAs into that model.

## Shared-pass spatial LoRA

The multi-character path shares the DiT forward across characters. Image-side attention
and MLP LoRA residuals are multiplied by RGB token masks. Cross-attention K/V
residuals apply only to the owning character's text slot; positive and negative
CFG branches use the same routing. Cross-attention cannot read foreign prompt
slots outside their masks. Self-attention remains global for scene composition.
Overlapping masks share a normalized residual budget and may still mix identities.
There is no guarantee of zero indirect leakage, improved eyes, or a particular
speedup. This changes the denoising trajectory; equal seeds need not yield the
same pose or layout as the Hook sampler.

With Spectrum Mod Guidance, forecasting follows the shared stream, and the
modulation projection uses the shared background conditioning rather than a
different projection for each character branch. Thus this is not mathematically
equivalent to masked prediction composition, even outside overlapping masks.
Image quality should be evaluated with the intended Spectrum options.

The implementation handles ordinary linear character LoRAs on Anima's
self/cross-attention projections and MLP layers. Adapters on other layers or with
DoRA/nonlinear weight transformations require additional routing support and
raise a logged error. They are never silently merged globally.
Runtime device caches are released at sample exit, and temporary forward hooks
and attention operations are restored even on failure. Multi-character sampling
returns the clean incoming model to downstream detailers.

This is an independent, RGB-mask-based implementation inspired by
[FreeFuse: Multi-Subject LoRA Fusion via Adaptive Token-Level Routing at Test Time](https://arxiv.org/abs/2510.23515),
not a reproduction of the paper or an official Anima port. It does not implement
FreeFuseAttn's automatic subject-mask extraction. No FreeFuse code is vendored.

CPU tests (run with the existing uv-managed Comfy Python):

```powershell
uv run --no-project --python <ComfyUI>/.venv/Scripts/python.exe python tests/test_spatial_lora.py
uv run --no-project --python <ComfyUI>/.venv/Scripts/python.exe python tests/test_soya_first_sampler.py
uv run --no-project --python <ComfyUI>/.venv/Scripts/python.exe python tests/test_character_lora_residuals.py
```

`tests/manual_spatial_lora_benchmark.py` reads a backend workflow backup and its
`_info.json`, and writes images into a new directory supplied explicitly by
the developer. It never changes the live workflow, configuration, or input masks.
Run it only when the target GPU is available. Checkpoint load and VAE decode are
outside the reported sampler timing; repetition 0 is a warmup. The runner prints
settings, per-image timing, and peak allocated VRAM for an auditable comparison.
By default it exercises the current `1st sampler`. For historical Hook A/B,
pass `--baseline-sampler <path/to/pre-spatial/soya_first_sampler.py>`; the old
implementation is loaded only by this manual test, not registered as a node.
