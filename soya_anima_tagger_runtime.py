"""Anima Tagger worker; download missing models, then tag the input batch."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-tags", default="")
    args = parser.parse_args()

    # Keep optional ML imports inside this worker, outside ComfyUI node discovery.
    from anime_tools.contract import DBV4_REQUIRED_FILES, DBV4_OPTIONAL_FILES
    from anime_tools.downloads import ensure
    from anime_tools.tagger.fetch import ensure_tagger_backbone
    from anime_tools.tagger import AnimaTagger
    from anime_tools.captions.tag_rules import from_dict, parse_caption
    from PIL import Image

    checkpoint = args.models_dir / "anima-tagger-pixai"
    missing = [checkpoint / name for name in (*DBV4_REQUIRED_FILES, *DBV4_OPTIONAL_FILES)
               if not (checkpoint / name).is_file()]
    if missing:
        print("Downloading missing Anima Tagger checkpoint files...", flush=True)
        ensure(
            "tagger", dest=checkpoint,
            files=tuple(path.name for path in missing), optional=(),
            revision="fbe45963eb3e59d131b98d20375509e2582b5c33", log=print,
        )
    print("Checking Anima Tagger backbone; missing files download automatically (about 2 GB).", flush=True)
    ensure_tagger_backbone(checkpoint)
    tagger = AnimaTagger(ckpt_dir=checkpoint, device=args.device)
    output_rules = from_dict({
        "remove": [tag.strip().lower().replace("_", " ")
                   for tag in args.exclude_tags.split(",") if tag.strip()],
    })
    captions = []
    for path in sorted(args.input_dir.glob("*.png")):
        with Image.open(path) as image:
            caption = tagger.predict_caption(image)
            captions.append(", ".join(parse_caption(caption, output_rules)))
    args.output.write_text(json.dumps(captions, ensure_ascii=False), encoding="utf-8")
    print(f"Tagged {len(captions)} images with Anima Tagger.")


if __name__ == "__main__":
    main()
