# Sequential offloading for MOSS and ANIMA

Set `workflow.sequential_offload` to `true` in the JSON supplied to the MOSS or ANIMA
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

For ANIMA, `workflow.clip` selects the installed text encoder filename.
Older documents inherit the CLIP loader widget. `workflow.clip_device` accepts
`"default"`, `"gpu"`, or `"cpu"` for both MOSS and ANIMA CLIP loaders
(`"default"` follows ComfyUI's VRAM setting, `"gpu"` forces `load_device` to the
active GPU while keeping ComfyUI's standard offload device, and `"cpu"` keeps
the text encoder on CPU). The first sampler, high-resolution
pass and detailers give the VAE, text encoder and DiT their own phases, including
the encoding calls inside SnapRef and Spectrum.

During each controlled phase, AIMDO also keeps 3 GiB of headroom (at most half
the device capacity, and never less than an existing larger user reservation).
On an 8 GiB device this leaves about 5 GiB for AIMDO-tracked weights and Torch
allocations. As activations grow, AIMDO can return weight pages to RAM instead
of retaining the whole encoder. The previous reservation is restored after
the phase, including when an exception is raised. This conservative budget
still requires validation on the affected workload; it is not a hard cap on
all driver allocations or a guarantee against Windows paging.

Model objects, conditioning results, LoRA definitions, and DynamicVRAM host
weight caches remain available in RAM. Uploads continue through ComfyUI's
managed loading path. The option does not call full unload, discard host
caches, disable DynamicVRAM, or change the selected GPU.

Use this option when overlapping model residency causes memory pressure, such
as on an 8 GB GPU. Uploading weights again at each phase can add overhead.
It does not guarantee that a single model's own working set fits in VRAM or
recover a CUDA operation that is already stalled. Logs prefixed
`[Soya:ModelMemory]` report phase timing, VRAM residency, and retained host
weight sizes.
