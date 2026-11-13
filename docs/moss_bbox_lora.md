# MOSS BBox LoRA (Soya)

Replace `Batch LoRA Loader (MOSS JSON)` with `MOSS BBox LoRA (Soya)`.
The inputs are `model`, `clip`, and the same `prompt_json` used by the MOSS
workflow. Outputs are `model`, unchanged `clip`, and `info`.

`workflow.loras.list` entries with `CHAR` use all character items whose `name`
matches that value exactly. Entries without `CHAR` remain global, including
style, distillation and CharRef LoRAs. A summary-only single-subject prompt
has no character bboxes, so its LoRAs retain full-image behavior.

```json
{
  "moss": {
    "summary": {"caption": "Two people standing in a garden."},
    "bbox_coord_space": "abs_centralized",
    "bbox_format": "yxyx",
    "items": [
      {"label": "character", "name": "Alice", "caption": "A woman in blue.", "bbox": [-512, -512, 512, 0]},
      {"label": "character", "name": "Bob", "caption": "A man in red.", "bbox": [-512, 0, 512, 512]}
    ]
  },
  "workflow": {
    "width": 1024,
    "height": 1024,
    "loras": {
      "activate": true,
      "filter": "",
      "list": [
        {"CHAR": "Alice", "lora_path": "alice.safetensors", "str": 0.8},
        {"CHAR": "Bob", "lora_path": "bob.safetensors", "str": 0.7},
        {"lora_path": "style.safetensors", "str": 0.5}
      ]
    }
  }
}
```

The node injects LoRA/LoHa/LoKr residuals in image-token projections without
merging character adapters into the base weights. Token centres inside a bbox
receive the adapter; overlap between characters shares a bounded strength.
Text and reference tokens receive no direct character-adapter injection.
The shared attention stream can still propagate visual influence outside the
box; this does not guarantee pixel-identical surroundings or zero identity bleed.

Text-embedder, timestep, and other non-spatial adapter components are excluded
and listed in `info`. DoRA, reshaped weights and unsupported adapter types raise
an error instead of silently applying globally. The current native ComfyUI
`MOSSImage` architecture is required. Split Q/K/V adapters are supported on
fused checkpoints through ComfyUI's native row mapping.

The INT8 activation path uses `comfy_force_forward` while sampling so FFN
residuals see the activated input. Base quantized weights remain unchanged;
disabling the fused activation path can affect speed. Hooks are removed after
every forward, and device adapter copies are released when sampling exits.

CPU regression command, from the ComfyUI directory:

```powershell
.venv/Scripts/python.exe -m unittest discover -s custom_nodes/comfyui-soya-custom-nodes/tests -p test_moss_spatial_lora.py -v
```

These are smoke/contract checks, including a tiny native MOSS forward with
references. They are not a full-checkpoint INT8 image-quality experiment.
