import os
import json
import folder_paths
import comfy.sd
import comfy.utils


class SoyaBatchLoraLoader_mdsoya:
    """
    JSON 형태의 LoRA 리스트를 입력받아 BASE 필터로 필터링 후 순차적으로 적용하는 노드.
    enable이 "true"가 아니거나 필터에 걸리는 LoRA가 없으면 model/clip을 그대로 통과시킵니다.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "enable": ("STRING", {"default": "true"}),
                "lora_list": ("STRING", {"multiline": True, "default": '{"list":[]}'}),
                "base_filter": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "info")
    FUNCTION = "load_batch_lora"
    CATEGORY = "Soya/LoRA"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def _resolve_lora_path(self, lora_path):
        """LoRA 경로를 해석합니다. 절대경로 우선, 없으면 folder_paths의 lora 폴더 기준."""
        # 절대경로로 바로 존재하는 경우
        if os.path.isfile(lora_path):
            return lora_path

        # folder_paths의 "loras" 카테고리에서 탐색
        try:
            resolved = folder_paths.get_full_path("loras", lora_path)
            if resolved and os.path.isfile(resolved):
                return resolved
        except Exception:
            pass

        # 모든 lora 등록 폴더에 대해 직접 join 시도
        try:
            for base in folder_paths.get_folder_paths("loras"):
                candidate = os.path.join(base, lora_path)
                if os.path.isfile(candidate):
                    return candidate
        except Exception:
            pass

        return None

    def _format_file_size(self, path):
        size = os.path.getsize(path)
        if size >= 1024 * 1024 * 1024:
            return f"{size / (1024 * 1024 * 1024):.1f} GB"
        if size >= 1024 * 1024:
            return f"{size / (1024 * 1024):.1f} MB"
        return f"{size / 1024:.1f} KB"

    def load_batch_lora(self, model, clip, enable, lora_list, base_filter):
        use = enable.strip().lower() in ("true", "1", "yes")
        info_lines = []
        info_lines.append(f"Enable: {enable.strip()} → {'ON' if use else 'OFF'}")

        if not use:
            info_lines.append("Result: Passing through (disabled)")
            return (model, clip, "\n".join(info_lines))

        # JSON 파싱
        try:
            data = json.loads(lora_list)
        except json.JSONDecodeError as e:
            info_lines.append(f"[ERROR] JSON parse failed: {e}")
            return (model, clip, "\n".join(info_lines))

        loras = data.get("list", [])
        info_lines.append(f"Total LoRAs in list: {len(loras)}")

        if not loras:
            info_lines.append("Result: Passing through (empty list)")
            return (model, clip, "\n".join(info_lines))

        # 사용 가능한 BASE 목록
        available_bases = sorted(set(l.get("BASE", "").strip() for l in loras if l.get("BASE", "").strip()))
        info_lines.append(f"Available BASEs: {', '.join(available_bases) or '(none)'}")

        # BASE 필터 적용
        base_filter = base_filter.strip()
        info_lines.append(f"Filter: '{base_filter or '(all)'}'")
        if base_filter:
            filtered = [l for l in loras if l.get("BASE", "").strip() == base_filter]
        else:
            filtered = loras
        info_lines.append(f"Matched by filter: {len(filtered)}")
        info_lines.append("─" * 50)

        if not filtered:
            info_lines.append("Result: Passing through (no matching LoRAs)")
            return (model, clip, "\n".join(info_lines))

        # 순차 적용
        current_model = model
        current_clip = clip
        applied_count = 0
        skipped_count = 0

        for i, lora_entry in enumerate(filtered):
            lora_path = lora_entry.get("lora_path", "")
            strength = float(lora_entry.get("str", 1.0))
            base = lora_entry.get("BASE", "")

            resolved = self._resolve_lora_path(lora_path)
            if resolved is None:
                skipped_count += 1
                info_lines.append(f"  {i+1}. [SKIP] File not found")
                info_lines.append(f"     Path: {lora_path}")
                continue

            filename = os.path.basename(resolved)
            file_size = self._format_file_size(resolved)

            if strength == 0:
                skipped_count += 1
                info_lines.append(f"  {i+1}. [SKIP] strength=0")
                info_lines.append(f"     File: {filename} ({file_size})")
                continue

            lora_data = comfy.utils.load_torch_file(resolved, safe_load=True)
            current_model, current_clip = comfy.sd.load_lora_for_models(
                current_model, current_clip, lora_data, strength, strength
            )
            applied_count += 1
            info_lines.append(f"  {i+1}. [APPLIED] {filename}")
            info_lines.append(f"     Strength: {strength} (model={strength}, clip={strength})")
            info_lines.append(f"     BASE: {base}")
            info_lines.append(f"     Size: {file_size}")

        # 요약
        info_lines.append("─" * 50)
        info_lines.append(f"Summary: {applied_count} applied, {skipped_count} skipped / {len(filtered)} matched")

        if applied_count == 0:
            info_lines.append("Result: Passing through (nothing applied)")
            return (model, clip, "\n".join(info_lines))

        return (current_model, current_clip, "\n".join(info_lines))
