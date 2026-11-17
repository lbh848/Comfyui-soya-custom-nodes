"""Load Anima Tagger once and serve image requests from ComfyUI over stdin."""

import argparse
import json
from pathlib import Path
import sys


RESULT_PREFIX = "SOYA_ANIMA_TAGGER_RESULT "


def build_output_rules(tagger, exclude_tags, include_rating, include_character, include_copyright):
    """Resolve output exclusions from the checkpoint's categories and aliases."""
    from anime_tools.captions.tag_rules import apply_aliases, from_dict

    excluded_categories = set()
    if not include_character:
        excluded_categories.add("character")
    if not include_copyright:
        excluded_categories.add("copyright")
    excluded = apply_aliases(
        [entry.name for entry in tagger.tag_entries if entry.category in excluded_categories],
        tagger.rules.aliases,
    )
    if not include_rating:
        excluded.extend(tagger.ratings)
    excluded.extend(exclude_tags.split(","))
    return from_dict({
        "remove": [tag.strip().lower().replace("_", " ") for tag in excluded if tag.strip()],
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()

    # Keep optional ML imports inside this worker, outside ComfyUI node discovery.
    from anime_tools.contract import DBV4_REQUIRED_FILES, DBV4_OPTIONAL_FILES
    from anime_tools.downloads import ensure
    from anime_tools.tagger.fetch import ensure_tagger_backbone
    from anime_tools.tagger import AnimaTagger
    from anime_tools.captions.tag_rules import parse_caption
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
    for line in sys.stdin:
        request = json.loads(line)
        output_rules = build_output_rules(
            tagger, request["exclude_tags"], request["include_rating"],
            request["include_character"], request["include_copyright"],
        )
        captions = []
        for path in sorted(Path(request["input_dir"]).glob("*.png")):
            with Image.open(path) as image:
                caption = tagger.predict_caption(image)
                captions.append(", ".join(parse_caption(caption, output_rules)))
        print(f"Tagged {len(captions)} images with Anima Tagger.", flush=True)
        print(RESULT_PREFIX + json.dumps(captions), flush=True)


if __name__ == "__main__":
    main()
