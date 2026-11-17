# Anima Tagger (Soya)

`Anima Tagger (Soya)` analyzes each complete input image with
[`sorryhyun/anime_tools`](https://github.com/sorryhyun/anime_tools)'s
`AnimaTagger.predict_caption`. It preserves the upstream vocabulary, thresholds,
sidecar head, tag rules, category order, and image preprocessing. It performs no
subject segmentation. An image batch returns one caption per image.

The node has three output switches, all off by default:

- `등급 태그 포함` (`include_rating`): rating labels such as `safe` and `sensitive`.
- `캐릭터 태그 포함` (`include_character`): character names such as `hatsune miku`.
- `작품 태그 포함` (`include_copyright`): `original` and franchises such as `vocaloid`.

These switches use the checkpoint's rating labels and vocabulary categories,
including its tag aliases, rather than a hand-maintained list of character names.
Counts such as `1girl`, artist tags, and appearance tags remain. The node's
`제외할 태그` (`exclude_tags`) field additionally removes comma-separated complete
tag names, with underscores normalized to spaces; it is empty by default.
Enable all three switches and leave this field empty to retain every upstream
output tag. Filtering uses the upstream tag-rule engine after caption generation;
the upstream inference code and model files are unchanged.

The MIT-licensed inference source is included unchanged under
`third_party/anime_tools`. `UPSTREAM.json` records the exact commit and every
included file's SHA-256. This avoids installing the upstream training, OCR,
segmentation, and GUI dependencies. Install this package's `requirements.txt`
in ComfyUI's Python environment and restart ComfyUI after updating the nodes.

## Model downloads

The node reuses existing model files and downloads missing files on first use.
The first run can take longer while the approximately 2 GB download completes.
You can also download them in advance. In Comfy Workshop, select
`배포_공방_범용태그분석_ANIMA_V1_0.json` under `워크플로우 목록` and click `설치`.
Workshop's download URLs, revisions, sizes, and SHA-256 checksums are owned
by Workshop's `src/comfy_installer/resources/install_manifest.json`.

The Anima checkpoint belongs in `ComfyUI/models/captioners/anima-tagger-pixai/`:
`config.json`, `vocab.json`, `rules.yaml`, `groups.yaml`,
`thresholds.safetensors`, `sidecar.json`, and `sidecar.safetensors`.
The backbone files are stored in the adjacent `huggingface/hub/` cache at the
revision specified by the original Anima checkpoint. Workshop downloads both
parts. The backbone is about 1.95 GB; the Anima-specific files are under 1 MB.

`auto` uses ComfyUI's selected device; `cpu` selects the CPU. A separate worker
loads the models once for the input batch and releases its memory on exit.
It uses a Hugging Face cache inside `models/captioners/` and does not
upload input images. CPU execution can take longer.

## Licenses

- Anima Tagger source: MIT, copyright (c) 2026 Seunghyun Ji. The complete notice
  is in `third_party/anime_tools/LICENSE`.
- Anima Tagger checkpoint: MIT, published at
  <https://huggingface.co/sorryhyun/anima-tagger>.
- The checkpoint's default PixAI backbone and its original model code:
  Apache-2.0, published at <https://huggingface.co/pixai-labs/pixai-tagger-v1.0>.
  The license is included as `LICENSE-PixAI-Tagger.txt`. No backbone source or
  weights are modified.

The legacy gated dbv4 weights are not included in this integration.
