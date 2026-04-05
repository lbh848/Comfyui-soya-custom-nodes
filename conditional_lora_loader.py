import os
import comfy.sd
import comfy.utils


class ConditionalLoraLoader_mdsoya:
    """
    프롬프트에서 캐릭터 이름을 추출하여, 해당 캐릭터의 LoRA를 자동으로 로드하는 노드.
    각 캐릭터의 LoRA 강도는 {model_name}_conditional_lora_info.txt 파일로 관리됩니다.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "lora_folder_path": ("STRING", {"multiline": False, "default": ""}),
                "model_name": ("STRING", {"multiline": False, "default": "default"}),
                "default_strength": ("FLOAT", {"default": 0.4, "min": -100.0, "max": 100.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "info")
    FUNCTION = "load_conditional_lora"
    CATEGORY = "Soya/LoRA"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def _get_info_file_path(self, lora_folder_path, model_name):
        return os.path.join(lora_folder_path, f"{model_name}_conditional_lora_info.txt")

    def _scan_character_folders(self, lora_folder_path):
        """폴더 내 캐릭터 하위폴더와 LoRA 파일을 스캔 (캐시 없이 항상 새로 읽음)"""
        characters = {}
        if not os.path.isdir(lora_folder_path):
            return characters
        for name in os.listdir(lora_folder_path):
            char_path = os.path.join(lora_folder_path, name)
            if os.path.isdir(char_path):
                lora_files = [
                    os.path.join(char_path, f)
                    for f in os.listdir(char_path)
                    if f.endswith(('.safetensors', '.ckpt', '.pt'))
                ]
                if lora_files:
                    characters[name] = lora_files
        return characters

    def _load_saved_strengths(self, info_file_path):
        """info 파일에서 저장된 LoRA 강도를 읽음 (항상 새로 읽음)"""
        strengths = {}
        if not os.path.isfile(info_file_path):
            return strengths
        with open(info_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if ':' in line:
                    char_name, strength_str = line.rsplit(':', 1)
                    char_name = char_name.strip()
                    try:
                        strengths[char_name] = float(strength_str.strip())
                    except ValueError:
                        pass
        return strengths

    def _save_strengths(self, info_file_path, strengths):
        """LoRA 강도를 info 파일에 저장"""
        os.makedirs(os.path.dirname(info_file_path), exist_ok=True)
        with open(info_file_path, 'w', encoding='utf-8') as f:
            f.write("# Conditional LoRA Info\n")
            f.write("# Format: character_name:strength\n")
            f.write("# 캐릭터별 LoRA 강도를 수정하려면 숫자를 변경하세요\n\n")
            for char_name in sorted(strengths.keys()):
                f.write(f"{char_name}:{strengths[char_name]}\n")

    def _extract_characters_from_prompt(self, prompt, character_names):
        """프롬프트에서 캐릭터 이름을 추출 (중복 제거, 순서 유지)"""
        def preprocess(text):
            return text.lower().replace("_", " ").replace("-", " ")

        prompt_processed = preprocess(prompt)
        found = []
        seen = set()
        # 긴 이름부터 매칭 (예: "haganai"보다 "haganai_yozora" 우선)
        sorted_names = sorted(character_names, key=len, reverse=True)
        for name in sorted_names:
            if name in seen:
                continue
            if preprocess(name) in prompt_processed:
                found.append(name)
                seen.add(name)
        return found

    def _sync_strengths_file(self, info_file_path, available_chars, default_strength):
        """
        폴더의 캐릭터 목록과 info 파일을 동기화.
        - 기존 강도 유지, 새 캐릭터는 default_strength, 삭제된 캐릭터는 제거
        """
        saved_strengths = self._load_saved_strengths(info_file_path)
        updated_strengths = {}

        for char_name in available_chars:
            if char_name in saved_strengths:
                updated_strengths[char_name] = saved_strengths[char_name]
            else:
                updated_strengths[char_name] = default_strength

        # 항상 파일에 기록 (변경 없어도 강제 저장하여 최신 상태 보장)
        self._save_strengths(info_file_path, updated_strengths)
        return updated_strengths

    def load_conditional_lora(self, model, prompt, lora_folder_path, model_name, default_strength):
        # 항상 강제로 폴더와 파일을 새로 읽음 (캐시 사용 안함)
        character_folders = self._scan_character_folders(lora_folder_path)
        info_file_path = self._get_info_file_path(lora_folder_path, model_name)

        # 모든 캐릭터 강도 동기화
        strengths = self._sync_strengths_file(info_file_path, character_folders, default_strength)

        # 프롬프트에서 캐릭터 추출
        detected_chars = self._extract_characters_from_prompt(prompt, list(character_folders.keys()))

        info_lines = []
        info_lines.append(f"[Available] {', '.join(sorted(character_folders.keys())) or 'None'}")
        info_lines.append(f"[Detected] {', '.join(detected_chars) if detected_chars else 'None'}")
        info_lines.append(f"[Info file] {info_file_path}")
        info_lines.append("")

        if not detected_chars:
            info_text = "\n".join(info_lines) + "No matching characters in prompt."
            return (model, info_text)

        current_model = model
        for char_name in detected_chars:
            lora_files = character_folders[char_name]
            strength = strengths.get(char_name, default_strength)

            for lora_file in lora_files:
                if strength == 0:
                    info_lines.append(f"[SKIP] {char_name}: {os.path.basename(lora_file)} (strength=0)")
                    continue

                lora_data = comfy.utils.load_torch_file(lora_file, safe_load=True)
                current_model, _ = comfy.sd.load_lora_for_models(
                    current_model, None, lora_data, strength, 0
                )
                info_lines.append(f"[APPLIED] {char_name}: {os.path.basename(lora_file)} @ {strength}")

        info_text = "\n".join(info_lines)
        return (current_model, info_text)
