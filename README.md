# Comfyui-soya-custom-nodes

`SoyaFirstSampler_mdsoya` applies each multi-character LoRA through its own
ComfyUI conditioning hook. Each character's positive and negative conditioning
share the same RGB mask and hooks; uncovered background uses the incoming model
without character LoRAs. Common style LoRAs remain on that incoming model.
All branches see the full latent, and ComfyUI combines their masked predictions
at each sampling step. Single-character sampling keeps its existing behavior.

Multi-character sampling requires more model evaluations and hook-weight memory.
Use the matching Spectrum update when selecting a Spectrum sampler: forecast
histories and modulation guidance must follow each conditioning branch.
Spectrum defers block-only LoRA weight changes until a full model evaluation is
needed. Forecasted steps keep their per-character histories without copying
unused block weights. Hooks that also modify layers used by forecasted steps
retain normal weight switching.
