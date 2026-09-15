"""Manual GPU runner. Reads a backup; never changes workflows or config.

Run with the uv-managed Comfy Python. Output directory must be new. Timing
includes sampler setup/encoding, excludes checkpoint loading and VAE decoding.
The first repetition of each method is a warmup, not a steady-state result.
Optionally supply a pre-spatial soya_first_sampler.py via --baseline-sampler
for a historical Hook A/B. No legacy sampler node is registered in production.
"""

import argparse
import importlib.util
import json
import math
import sys
import time
import traceback
import types
from contextlib import ExitStack
from pathlib import Path
from unittest import mock


def import_file(name, path, package=False):
    spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=[str(path.parent)] if package else None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--sampler-mode")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--baseline-sampler", type=Path)
    args = parser.parse_args()
    sys.argv = [sys.argv[0]]
    root = Path(__file__).resolve().parents[1]
    comfy_root = root.parents[1]
    sys.path.insert(0, str(comfy_root))
    import comfy.cli_args
    comfy.cli_args.args.highvram = True
    import torch
    import numpy as np
    from PIL import Image
    import comfy.model_management as mm
    import comfy.sd
    import comfy.utils
    import folder_paths
    import nodes
    from comfy_extras.nodes_string import RegexExtract

    package = types.ModuleType("_soya_bench")
    package.__path__ = [str(root)]
    sys.modules[package.__name__] = package
    sampler = import_file("_soya_bench.soya_first_sampler", root / "soya_first_sampler.py")
    baseline = None
    if args.baseline_sampler is not None:
        baseline = import_file("_soya_bench.legacy_first_sampler", args.baseline_sampler.resolve())
    prompt_module = import_file("_soya_bench.soya_prompt_parser", root / "soya_prompt_parser.py")
    spectrum = import_file("_spectrum_bench", root.parent / "comfyui-spectrum-ksampler/__init__.py", True)
    nodes.NODE_CLASS_MAPPINGS.update(spectrum.NODE_CLASS_MAPPINGS)

    workflow = json.loads(args.workflow.read_text(encoding="utf-8"))
    info = json.loads(args.workflow.with_name(args.workflow.stem + "_info.json").read_text(encoding="utf-8"))
    graph = {n["id"]: n for n in workflow["nodes"]}
    links = {link[0]: link for link in workflow["links"]}
    def of_type(kind):
        return next(n for n in graph.values() if n["type"] == kind and n.get("mode", 0) == 0)
    def linked(n, name):
        inp = next(i for i in n["inputs"] if i["name"] == name)
        link = links[inp["link"]]
        return resolve(graph[link[1]], link[2])
    def resolve(n, slot=0):
        kind = n["type"]
        if kind == "GetNode":
            setter = next(v for v in graph.values() if v["type"] == "SetNode" and v["widgets_values"] == n["widgets_values"])
            return linked(setter, setter["inputs"][0]["name"])
        if kind in ("SetNode", "Reroute"):
            return linked(n, n["inputs"][0]["name"])
        if kind == "CLIPTextEncode":
            return linked(n, "text")
        if kind == "SoyaPromptParser_mdsoya":
            return prompt_module.SoyaPromptParser_mdsoya().parse(linked(n, "text"))[slot]
        if kind == "RegexExtract":
            return RegexExtract.execute(linked(n, "string"), *n["widgets_values"][1:])[0]
        if kind in ("PrimitiveStringMultiline", "PrimitiveString", "PrimitiveNode"):
            return n["widgets_values"][slot]
        raise RuntimeError(f"Benchmark cannot resolve node {n['id']} ({kind}); no guessed input")

    original_sampler = of_type("SoyaFirstSampler_mdsoya")
    prompt_node = of_type("SoyaPromptParser_mdsoya")
    values = dict(zip(prompt_module.SoyaPromptParser_mdsoya.RETURN_NAMES,
                      prompt_module.SoyaPromptParser_mdsoya().parse(linked(prompt_node, "text"))))
    negative_text = linked(original_sampler, "negative")
    multi = json.loads(values["MULTI_CHAR"])
    layout = info["illustration_multi_char"]["layout"]
    masks = []
    for region in layout["regions"]:
        mask = torch.zeros(1, layout["mask_height"], layout["mask_width"])
        # Same inclusive floor/ceil rectangle as modes.multi_char_mask.render_region_mask.
        x0, y0 = math.floor(region["x"] * mask.shape[-1]), math.floor(region["y"] * mask.shape[-2])
        x1 = math.ceil((region["x"] + region["width"]) * mask.shape[-1])
        y1 = math.ceil((region["y"] + region["height"]) * mask.shape[-2])
        mask[:, y0:y1, x0:x1] = 1
        masks.append(mask)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    model = nodes.UNETLoader().load_unet(*of_type("UNETLoader")["widgets_values"])[0]
    clip = nodes.CLIPLoader().load_clip(*of_type("CLIPLoader")["widgets_values"])[0]
    vae = nodes.VAELoader().load_vae(*of_type("VAELoader")["widgets_values"])[0]
    style_rows = of_type("Power Lora Loader (rgthree)")["widgets_values"]
    for style in style_rows:
        if isinstance(style, dict) and style.get("on") and style.get("lora"):
            path = folder_paths.get_full_path_or_raise("loras", style["lora"])
            data = comfy.utils.load_torch_file(path, safe_load=True)
            strength = style["strength"]
            clip_strength = style.get("strengthTwo")
            model, clip = comfy.sd.load_lora_for_models(model, clip, data, strength, strength if clip_strength is None else clip_strength)
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(values["ANIMA_ALL"]))
    negative = clip.encode_from_tokens_scheduled(clip.tokenize(negative_text))
    settings = dict(zip(
        ("mask_location", "steps", "cfg", "sampler_name", "scheduler", "denoise", "split_mode", "spd_scale", "spd_sigma", "adaptive_smc_alpha", "sampler_mode"),
        original_sampler["widgets_values"],
    ))
    if args.steps is not None:
        settings["steps"] = args.steps
    if args.sampler_mode:
        settings["sampler_mode"] = args.sampler_mode
    options = sampler.SoyaSpectrumModGuidanceOptions_mdsoya()
    spectrum_node = of_type("SoyaSpectrumModGuidanceOptions_mdsoya")
    settings["spectrum_options"] = getattr(options, options.FUNCTION)(*spectrum_node["widgets_values"])[0]
    if settings["sampler_mode"] != "SpectrumKSamplerModGuidance":
        settings["spectrum_options"] = None
    width, height = info["image_width"], info["image_height"]
    latent = nodes.EmptyLatentImage().generate(width, height, 1)[0]
    seed = values["SEED"] if args.seed is None else args.seed
    instances = {}
    if baseline is not None:
        instances["hook"] = baseline.SoyaFirstSampler_mdsoya()
    else:
        print("[BENCH] No --baseline-sampler supplied; testing the current 1st sampler only", flush=True)
    instances["spatial"] = sampler.SoyaFirstSampler_mdsoya()
    print("[BENCH_SETTINGS] " + json.dumps({"workflow": str(args.workflow), "size": [width, height], "seed": seed,
          "characters": multi["char_name_list"], "baseline_sampler": str(args.baseline_sampler) if baseline else None,
          "settings": settings}, ensure_ascii=False), flush=True)
    with torch.inference_mode(), ExitStack() as patches:
        patches.enter_context(mock.patch.object(sampler, "_load_channel_masks", return_value=masks))
        if baseline is not None:
            patches.enter_context(mock.patch.object(baseline, "_load_channel_masks", return_value=masks))
        for repeat in range(args.repeats):
            # Alternate method order; repeat 0 warms both implementations.
            order = list(instances) if repeat % 2 == 0 else list(reversed(instances))
            for method in order:
                current_seed = seed if repeat == 0 else seed + repeat - 1
                mm.load_models_gpu([model])
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                output, clean_model = instances[method].sample(
                    model=model, positive=positive, negative=negative, clip=clip, multi_char=values["MULTI_CHAR"],
                    seed=current_seed, latent_image=latent, LORA_ACT=values["LORA_ACTIVATE"], LORA_DATA=values["LORA_DATA"], **settings,
                )
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                assert clean_model is model, "Multi-character sampler must return the clean input model"
                record = {"method": method, "repeat": repeat, "warmup": repeat == 0, "seed": current_seed,
                          "seconds": elapsed, "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20}
                print("[BENCH_RESULT] " + json.dumps(record), flush=True)
                image = nodes.VAEDecode().decode(vae, output)[0][0].detach().cpu().numpy()
                Image.fromarray((image.clip(0, 1) * 255).astype(np.uint8)).save(args.output_dir / f"{method}_{repeat}_{current_seed}.png")
                del image, output
    mm.unload_all_models()
    mm.soft_empty_cache()
    print("[BENCH_DONE] All test models unloaded; process exiting", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[BENCH_FAILURE] {exc}", flush=True)
        traceback.print_exc()
        raise
