import re

with open('character_identifier.py', 'r', encoding='utf-8') as f:
    content = f.read()

# find def identify
start_idx = content.find('    def identify')

target_start = content.find('        # text_batch 처리 (리스트 또는 None)', start_idx)
target_end = content.find('        # 충돌 해결 (Conflict Resolution)', target_start)

# We will inject the new logic in between.

new_logic = """        # text_batch 파싱
        face_tags_list = []
        if w_t > 0:
            tb = text_batch
            if isinstance(tb, list) and len(tb) > 0:
                tb = tb[0] if len(tb) == 1 else tb

            if isinstance(tb, str):
                for line in tb.split('\\n'):
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
                est_names = [n.strip() for n in estimated_character_names.replace('\\n', ',').split(',') if n.strip()]
            elif getattr(estimated_character_names, '__iter__', False) and not isinstance(estimated_character_names, dict):
                if len(estimated_character_names) == 1 and isinstance(estimated_character_names[0], list):
                    est_names = estimated_character_names[0]
                else:
                    est_names = list(estimated_character_names)
            
            if isinstance(estimated_character_scores, (float, int)):
                est_scores = [float(estimated_character_scores)]
            elif getattr(estimated_character_scores, '__iter__', False) and not isinstance(estimated_character_scores, dict):
                if len(estimated_character_scores) == 1 and isinstance(estimated_character_scores[0], list):
                    est_scores = [float(x) for x in estimated_character_scores[0]]
                else:
                    est_scores = [float(x) for x in estimated_character_scores]

        has_text = len(face_tags_list) > 0
        has_embed = len(est_names) > 0

        if not has_text and not has_embed:
            print("[IdentifyCharacters] Both Text and Embed inputs are empty, returning common_prompt only")
            return (f"{asc}\\n{common}",)

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

"""

new_content = content[:target_start] + new_logic + content[target_end:]

with open('character_identifier.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("done")
