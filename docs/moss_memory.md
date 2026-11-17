# Optional sequential MOSS offloading

Set `workflow.sequential_offload` to `true` in the JSON supplied to the MOSS
V2 nodes to enable phase-based VRAM handoff. The default is `false`; existing
workflows retain their normal memory behavior. ComfyUI must have DynamicVRAM
enabled for this option to take effect.

```json
{
  "workflow": {
    "sequential_offload": true
  }
}
```

This is one field within the existing controller; keep the other workflow and
MOSS layout fields. No additional nodes or model files are required.

Before text encoding or sampling, inactive models on the same CUDA device
return their VRAM allocations to the allocator. After successful sampling,
the diffusion model returns its VRAM allocations before the downstream VAE
decode. Positive and negative encoding share the same encoder residency.
Reference-image encoding also uses phase boundaries between its VAE and text
encoder.

Model objects, conditioning results, LoRA definitions, and DynamicVRAM host
weight caches remain available in RAM. Uploads continue through ComfyUI's
managed loading path. The option does not call full unload, discard host
caches, disable DynamicVRAM, or change the selected GPU.

Use this option when overlapping model residency causes memory pressure, such
as on an 8 GB GPU. Uploading weights again at each phase can add overhead.
It does not guarantee that a single model's own working set fits in VRAM or
recover a CUDA operation that is already stalled. Logs prefixed
`[Soya:MossMemory]` report phase timing, VRAM residency, and retained host
weight sizes.
