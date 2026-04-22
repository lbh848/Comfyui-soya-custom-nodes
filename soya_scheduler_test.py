"""
SoyaSchedulerTest – 단일 이미지로 얼굴 분석 파이프라인을 테스트하는 노드.
입력: 이미지 (+ 선택적 프롬프트)
출력: 완성된 프롬프트, 처리 정보
"""

import time
import numpy as np

from .soya_scheduler.config_manager import load_config, load_characters, save_config
from .soya_scheduler.ray_worker import (
    analyze_faces_sync,
    _filter_characters_by_prompt,
)


class SoyaSchedulerTest_mdsoya:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("final_prompt", "process_result")
    FUNCTION = "test"
    CATEGORY = "Soya/Scheduler"
    OUTPUT_NODE = True

    def test(self, image, prompt=""):
        config = load_config()
        settings = config.get("settings", {})
        base_path = settings.get("base_path", "")

        # Load and filter characters
        all_chars = load_characters(base_path)
        if prompt:
            matched_chars = _filter_characters_by_prompt(all_chars, prompt)
        else:
            matched_chars = all_chars
        config["characters"] = matched_chars

        # Resolve model paths
        try:
            import folder_paths
            bbox_model = settings.get("bbox_model", "face_yolov8m.pt")
            resolved = folder_paths.get_full_path("ultralytics_bbox", bbox_model)
            if resolved:
                settings["_resolved_bbox_path"] = resolved

            clip_model = settings.get("clip_vision_model", "clip_vision_vit_h.safetensors")
            resolved = folder_paths.get_full_path("clip_vision", clip_model)
            if resolved:
                settings["_resolved_clip_path"] = resolved
        except Exception:
            pass
        config["settings"] = settings

        # Prepare image
        image_numpy = image.cpu().numpy()
        if image_numpy.ndim == 4:
            image_numpy = image_numpy[0]
        if image_numpy.dtype != np.uint8:
            image_numpy = (image_numpy * 255).clip(0, 255).astype(np.uint8)

        # Run analysis synchronously
        t0 = time.time()
        result = analyze_faces_sync("test", image_numpy, config)
        elapsed = time.time() - t0

        prompts = result.get("prompts", "")
        timing = result.get("timing", {})
        assignments = result.get("assignments", [])
        similarities = result.get("similarities", [])

        # Format process result
        lines = ["[Soya Scheduler Test]"]
        lines.append(f"총 소요시간: {elapsed:.2f}초")
        if timing:
            for step, dur in timing.items():
                if step != "total":
                    lines.append(f"  - {step}: {dur:.2f}초")
        lines.append(f"감지된 얼굴: {len(assignments)}개")
        if assignments:
            for i, (name, score) in enumerate(zip(assignments, similarities)):
                lines.append(f"  [{i+1}] {name} (similarity: {score:.3f})")
        lines.append(f"사용된 캐릭터: {', '.join(ch['name'] for ch in matched_chars) if matched_chars else 'none'}")
        info = "\n".join(lines)

        # Save to config for web UI (same format as SoyaProcessCollector)
        config = load_config()
        config["last_process_result"] = {
            "timing": timing,
            "assignments": assignments,
            "similarities": similarities,
            "face_crops": result.get("face_crops", []),
            "ref_crops_map": result.get("ref_crops_map", {}),
            "elapsed": elapsed,
        }
        config["last_final_prompts"] = prompts
        save_config(config)

        print(info)
        return (prompts, info)
