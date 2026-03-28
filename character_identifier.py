import re
import math
import matplotlib.colors as mcolors

class IdentifyCharacters_mdsoya:
    """
    Identifies characters from cropped face tags based on the main prompt and character dictionary.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_size": ("INT", {"default": 1, "min": 1}),
                "prompt": ("STRING", {"multiline": True}),
                "character_features": ("STRING", {"multiline": True}),
            },
            "optional": {
                "text_batch": ("STRING", {"multiline": True, "default": ""}),
                "estimated_character_names": ("STRING", {"forceInput": True}),
                "estimated_character_scores": ("FLOAT", {"forceInput": True}),
                "text_weight": ("STRING", {"default": "1.0"}),
                "embedding_weight": ("STRING", {"default": "1.0"}),
                "male_enhance_prompt": ("STRING", {"multiline": True, "default": ""}),
                "female_enhance_prompt": ("STRING", {"multiline": True, "default": ""}),
                "character_enhance_prompt": ("STRING", {"multiline": True, "default": ""}),
                "common_prompt": ("STRING", {"multiline": True, "default": ""}),
                "asc_prefix": ("STRING", {"default": "[ASC]"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("combined",)
    OUTPUT_IS_LIST = (False,)
    INPUT_IS_LIST = True
    FUNCTION = "identify"
    CATEGORY = "Soya/Character"

    def get_hue(self, color_str):
        name = color_str.replace(" hair", "").strip()
        try:
            rgb = mcolors.to_rgb(name)
            hsv = mcolors.rgb_to_hsv(rgb)
            return hsv[0] * 360 # 0-360
        except:
            return -1

    def token_distance(self, tags1, tags2):
        set1 = set(tags1)
        set2 = set(tags2)
        if not set1 or not set2:
            return 999
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        return 1.0 - (len(intersection) / len(union)) # Jaccard distance

    def parse_dict(self, text):
        char_dict = {}
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        current_char = None
        for line in lines:
            if line.startswith('###'):
                current_char = line.replace('###', '').strip().replace(' ', '_').lower()
            elif current_char:
                tags = [t.strip().replace(' ', '_').lower() for t in line.split(',') if t.strip()]
                char_dict[current_char] = tags
        return char_dict

    def parse_character_enhance(self, text):
        """Parse character enhancement prompts in ## 캐릭터이름 format"""
        char_enhance = {}
        if not text or not text.strip():
            return char_enhance

        lines = text.split('\n')
        current_char = None
        current_tags = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith('##'):
                # Save previous character if exists
                if current_char and current_tags:
                    char_enhance[current_char] = ', '.join(current_tags)
                # Start new character
                current_char = stripped.replace('##', '').strip().replace(' ', '_').lower()
                current_tags = []
            elif current_char and stripped:
                # Add tags to current character
                tags = [t.strip() for t in stripped.split(',') if t.strip()]
                current_tags.extend(tags)

        # Save last character
        if current_char and current_tags:
            char_enhance[current_char] = ', '.join(current_tags)

        return char_enhance

    def identify(self, batch_size, prompt, character_features, text_batch=None,
                 estimated_character_names=None, estimated_character_scores=None,
                 text_weight=1.0, embedding_weight=1.0,
                 male_enhance_prompt=None, female_enhance_prompt=None,
                 character_enhance_prompt=None, common_prompt=None, asc_prefix=None):
        # INPUT_IS_LIST = True 이므로 모든 입력이 리스트로 들어옴
        # batch_size, prompt, character_features도 리스트 형태
        batch_size = batch_size[0] if isinstance(batch_size, list) else batch_size
        prompt = prompt[0] if isinstance(prompt, list) else prompt
        character_features = character_features[0] if isinstance(character_features, list) else character_features

        text_weight = text_weight[0] if isinstance(text_weight, list) else (text_weight if text_weight is not None else 1.0)
        embedding_weight = embedding_weight[0] if isinstance(embedding_weight, list) else (embedding_weight if embedding_weight is not None else 1.0)

        def safe_float(val, default):
            try:
                if val == "" or val is None: return default
                return float(val)
            except:
                return default

        w_t = safe_float(text_weight, 1.0)
        w_e = safe_float(embedding_weight, 1.0)
        if w_t == 0.0 and w_e == 0.0:
            w_t, w_e = 0.5, 0.5
        total_w = w_t + w_e
        w_t /= total_w
        w_e /= total_w

        # 새로운 optional 입력 처리
        male_enhance = male_enhance_prompt[0] if isinstance(male_enhance_prompt, list) else male_enhance_prompt
        male_enhance = male_enhance.strip() if male_enhance else ""

        female_enhance = female_enhance_prompt[0] if isinstance(female_enhance_prompt, list) else female_enhance_prompt
        female_enhance = female_enhance.strip() if female_enhance else ""

        char_enhance_text = character_enhance_prompt[0] if isinstance(character_enhance_prompt, list) else character_enhance_prompt
        char_enhance_text = char_enhance_text if char_enhance_text else ""

        common = common_prompt[0] if isinstance(common_prompt, list) else common_prompt
        common = common.strip() if common else ""

        # asc_prefix 처리
        asc = asc_prefix[0] if isinstance(asc_prefix, list) else asc_prefix
        asc = asc.strip() if asc else "[ASC]"
        if not asc:
            asc = "[ASC]"

        # 캐릭터 강화 프롬프트 파싱
        char_enhance_dict = self.parse_character_enhance(char_enhance_text)
        print(f"[IdentifyCharacters] char_enhance_dict keys: {list(char_enhance_dict.keys())}")

        # batch_size가 0 이하면 최소 1로 설정
        if batch_size < 1:
            print(f"[IdentifyCharacters] batch_size was {batch_size}, setting to 1")
            batch_size = 1

        # text_batch 파싱
        face_tags_list = []
        if w_t > 0:
            tb = text_batch
            if isinstance(tb, list) and len(tb) > 0:
                tb = tb[0] if len(tb) == 1 else tb

            if isinstance(tb, str):
                for line in tb.split('\n'):
                    if line.strip():
                        tags = [t.strip().lower() for t in line.split(',')]
                        face_tags_list.append(tags)
            elif getattr(tb, '__iter__', False) and not isinstance(tb, dict):
                for item in tb:
                    if isinstance(item, str) and item.strip():
                        tags = [t.strip().lower() for t in item.split(',')]
                        face_tags_list.append(tags)
                    elif isinstance(item, str):
                        face_tags_list.append([])

        # estimated_character_names 파싱
        est_names = []
        est_scores = []
        if w_e > 0:
            if isinstance(estimated_character_names, str):
                est_names = [n.strip() for n in estimated_character_names.replace('\n', ',').split(',') if n.strip()]
            elif getattr(estimated_character_names, '__iter__', False) and not isinstance(estimated_character_names, dict):
                if len(estimated_character_names) > 0 and isinstance(estimated_character_names[0], list):
                    est_names = estimated_character_names[0]
                else:
                    est_names = list(estimated_character_names)
            
            if isinstance(estimated_character_scores, (float, int)):
                est_scores = [float(estimated_character_scores)]
            elif getattr(estimated_character_scores, '__iter__', False) and not isinstance(estimated_character_scores, dict):
                if len(estimated_character_scores) > 0 and isinstance(estimated_character_scores[0], list):
                    est_scores = [float(x) for x in estimated_character_scores[0]]
                else:
                    est_scores = [float(x) for x in estimated_character_scores]

        has_text = len(face_tags_list) > 0
        has_embed = len(est_names) > 0

        if not has_text and not has_embed:
            print("[IdentifyCharacters] Both Text and Embed inputs are empty, returning common_prompt only")
            return (f"{asc}\n{common}",)

        prompt_norm = prompt.lower().replace('_', ' ')

        char_dict = self.parse_dict(character_features)

        expected_chars = []
        for c_name in char_dict.keys():
            c_name_space = c_name.replace('_', ' ')
            if c_name in prompt_norm or c_name_space in prompt_norm:
                expected_chars.append(c_name)

        if not expected_chars:
            expected_chars = list(char_dict.keys())

        preferences = []
        face_gender_list = []
        
        # Ensure we have enough assignments for batch size
        assigned_chars = set()  # 이미 할당된 캐릭터 추적
        
        for i in range(batch_size):
            tags = face_tags_list[i] if i < len(face_tags_list) else []
            face_gender = 'unknown'
            if '1girl' in tags or 'girl' in tags: face_gender = 'female'
            elif '1boy' in tags or 'boy' in tags: face_gender = 'male'
            face_gender_list.append(face_gender)

            text_based_probs = {}
            if w_t > 0 and tags:
                face_hair = next((t for t in tags if 'hair' in t), '')
                face_eye = next((t for t in tags if 'eyes' in t), '')

                for c_name in expected_chars:
                    c_tags = char_dict.get(c_name, [])
                    c_gender = 'unknown'
                    if '1girl' in c_tags or 'girl' in c_tags: c_gender = 'female'
                    elif '1boy' in c_tags or 'boy' in c_tags or 'mature_male' in c_tags or '(mature_male:0.9)' in c_tags: c_gender = 'male'

                    if face_gender != 'unknown' and c_gender != 'unknown' and face_gender != c_gender:
                        text_based_probs[c_name] = 0.0
                        continue

                    c_hair = next((t for t in c_tags if 'hair' in t), '')
                    dist = self.token_distance(tags, c_tags)
                    score = dist

                    if face_hair and c_hair:
                        fhue = self.get_hue(face_hair.replace('_', ' '))
                        chue = self.get_hue(c_hair.replace('_', ' '))
                        if fhue >= 0 and chue >= 0:
                            diff = abs(fhue - chue)
                            diff = min(diff, 360 - diff)
                            score += diff / 360.0

                    text_based_probs[c_name] = max(0.0, 1.0 - score)
            else:
                for c_name in expected_chars:
                    text_based_probs[c_name] = 0.0

            embedding_based_probs = {}
            if w_e > 0 and i < len(est_names) and i < len(est_scores):
                pred_c_name = str(est_names[i]).strip().replace(' ', '_').lower()
                
                # Check for "Unknown_X" case from AssignCharactersCLIP
                if pred_c_name.startswith('unknown_'):
                    pass # treat as 0 prob for all
                else:    
                    for c_name in expected_chars:
                        embedding_based_probs[c_name] = 0.0
                    if pred_c_name in expected_chars:
                        embedding_based_probs[pred_c_name] = est_scores[i]
                    else:
                        embedding_based_probs[pred_c_name] = est_scores[i]
            else:
                for c_name in expected_chars:
                    embedding_based_probs[c_name] = 0.0

            char_scores = []
            all_considered_chars = set(expected_chars).union(set(embedding_based_probs.keys()))
            for c_name in all_considered_chars:
                prob_t = text_based_probs.get(c_name, 0.0)
                prob_e = embedding_based_probs.get(c_name, 0.0)
                final_prob = (prob_t * w_t) + (prob_e * w_e)
                cost = 1.0 - final_prob
                char_scores.append((cost, c_name))

            char_scores.sort(key=lambda x: x[0])
            if not char_scores:
                best_char = expected_chars[0] if expected_chars else 'unknown'
                char_scores.append((999, best_char))
            preferences.append(char_scores)

        # 충돌 해결 (Conflict Resolution)
        assigned_indices = [0] * batch_size
        assigned_chars_dict = {}
        for i in range(batch_size):
            if preferences[i]:
                assigned_chars_dict[i] = preferences[i][0][1]
            else:
                assigned_chars_dict[i] = 'unknown'

        changed = True
        while changed:
            changed = False
            char_to_i = {}
            for i, c in assigned_chars_dict.items():
                if c == 'unknown': continue
                if c not in char_to_i:
                    char_to_i[c] = []
                char_to_i[c].append(i)

            for c, indices in char_to_i.items():
                if len(indices) > 1:
                    # 점수가 높은(score가 낮은) 순으로 정렬
                    indices.sort(key=lambda idx: preferences[idx][assigned_indices[idx]][0])
                    # 첫 번째 요소가 해당 캐릭터를 차지하고 나머지는 다음 순위로
                    for idx in indices[1:]:
                        assigned_indices[idx] += 1
                        if assigned_indices[idx] < len(preferences[idx]):
                            assigned_chars_dict[idx] = preferences[idx][assigned_indices[idx]][1]
                        else:
                            assigned_chars_dict[idx] = 'unknown'
                        changed = True

        # === 사용되지 않은 캐릭터 강제 할당 ===
        assigned_set = set(assigned_chars_dict.values())
        unused_chars = [c for c in expected_chars if c not in assigned_set]
        
        # 'unknown' 인 슬롯에 우선적으로 미사용 캐릭터 할당 (점수와 무관하게 무조건 추출된 캐릭터 우선 사용)
        for i in range(batch_size):
            if assigned_chars_dict[i] == 'unknown' and unused_chars:
                assigned_chars_dict[i] = unused_chars.pop(0)

        genders = []
        eye_colors = []
        char_names = []
        enhance_prompts = []
        for i in range(batch_size):
            best_char = assigned_chars_dict.get(i, 'unknown')

            if best_char == 'unknown':
                # 남은 캐릭터도 없고 face tag 정보도 없어 매칭할 대상이 없는 경우
                print(f"[IdentifyCharacters] No char assigned, using dummy value")
                genders.append("male")
                eye_colors.append("black eyes")
                char_names.append("unknown")
                enhance_prompts.append(male_enhance)
                continue

            char_name_normalized = best_char.replace('_', ' ')
            char_names.append(char_name_normalized)

            c_tags = char_dict.get(best_char, [])
            c_gender = 'female'
            if '1boy' in c_tags or 'boy' in c_tags or 'mature_male' in c_tags or '(mature_male:0.9)' in c_tags:
                c_gender = 'male'

            face_gender = face_gender_list[i]
            # 얼굴 성별 정보가 없거나 불명확한 경우 캐릭터 사전의 성별을 따름
            final_gender = face_gender if face_gender != 'unknown' else c_gender
            genders.append(final_gender)

            c_eye = next((t for t in c_tags if 'eyes' in t), 'black eyes').replace('_', ' ')
            eye_colors.append(c_eye)

            print(f"[IdentifyCharacters] best_char: '{best_char}', in dict: {best_char in char_enhance_dict}")
            if best_char in char_enhance_dict:
                enhance_prompts.append(char_enhance_dict[best_char])
                print(f"[IdentifyCharacters] Using char enhance: {char_enhance_dict[best_char][:50]}...")
            else:
                if final_gender == 'male':
                    enhance_prompts.append(male_enhance)
                else:
                    enhance_prompts.append(female_enhance)

        # {asc}\n[1] eye_color, gender, character_name, enhance_prompt, common_prompt\n[2] ... 형식으로 출력
        entries = []
        for i in range(len(char_names)):
            gender_display = "boy" if genders[i] == "male" else "girl"
            entry = f"{eye_colors[i]}, {gender_display}, {char_names[i]}, {enhance_prompts[i]}, {common}"
            entries.append(entry)

        result = f"{asc}\n" + "\n".join([f"[{i+1}] {e}" for i, e in enumerate(entries)])

        # 결과가 batch_size보다 부족하면 더미값으로 채움
        while len(entries) < batch_size:
            print(f"[IdentifyCharacters] Padding result with dummy value")
            dummy_entry = f"black eyes, girl, unknown, {female_enhance}, {common}"
            entries.append(dummy_entry)
            result = f"{asc}\n" + "\n".join([f"[{i+1}] {e}" for i, e in enumerate(entries)])

        print(f"[IdentifyCharacters] Returning result (batch_size was {batch_size})")
        return (result,)
