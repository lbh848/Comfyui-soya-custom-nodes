import re


class FilterClosedEyes_mdsoya:
    """
    Identify Characters (Soya) 출력에서 closed eyes 캐릭터를 필터링하는 노드

    - UPSCALE PROMPT를 캐릭터 정렬 텍스트(assigned_names) 순서로 정렬
    - 정렬된 UPSCALE PROMPT에서 closed eyes가 감지된 캐릭터의 [N] 블록을 제거
    - 출력은 Identify Characters (Soya)와 동일한 형식
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "identify_output": ("STRING", {"multiline": True, "forceInput": True}),
                "assigned_names": ("STRING", {"forceInput": True}),
                "upscale_prompt": ("STRING", {"multiline": True, "forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filtered_output",)
    OUTPUT_IS_LIST = (False,)
    INPUT_IS_LIST = True
    FUNCTION = "filter_closed_eyes"
    CATEGORY = "Soya/Character"

    def _normalize(self, text):
        """소문자 + 언더스코어→공백 정규화"""
        return text.lower().replace('_', ' ').strip()

    def _has_closed_eyes(self, text):
        """closed eyes 감지 (대소문자 무시, _/공백 무시)"""
        normalized = self._normalize(text)
        return 'closed eyes' in normalized

    def _has_wink(self, text):
        """wink 감지 (대소문자 무시, _/공백 무시)"""
        normalized = self._normalize(text)
        return 'wink' in normalized

    def _is_eyes_tag(self, tag):
        """태그가 eyes 관련인지 판별"""
        return 'eyes' in self._normalize(tag)

    def _replace_eyes_with_closed(self, content):
        """content에서 eyes 관련 태그를 모두 제거하고 'closed eyes'를 첫 태그로 삽입"""
        parts = [p.strip() for p in content.split(',')]
        filtered = [p for p in parts if p and not self._is_eyes_tag(p)]
        filtered.insert(0, 'closed eyes')
        return ', '.join(filtered)

    def _add_wink(self, content):
        """content 첫 태그 앞에 'wink' 추가"""
        parts = [p.strip() for p in content.split(',')]
        parts = [p for p in parts if p]
        parts.insert(0, 'wink')
        return ', '.join(parts)

    def _parse_identify_output(self, text):
        """Identify Characters 출력 파싱 → (header, blocks)"""
        lines = text.strip().split('\n')
        header = lines[0].strip() if lines else ""
        blocks = []
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            match = re.match(r'\[(\d+)\]\s*(.*)', line)
            if match:
                idx = int(match.group(1))
                content = match.group(2)
                # Format: "eye_color, gender, char_name, ..."
                parts = [p.strip() for p in content.split(',')]
                char_name = parts[2] if len(parts) > 2 else ""
                blocks.append({
                    'idx': idx,
                    'content': content,
                    'char_name': self._normalize(char_name)
                })
        return header, blocks

    def _parse_assigned_names(self, raw):
        """assigned_names 파싱 → 정렬된 캐릭터 이름 리스트"""
        if isinstance(raw, list):
            names = []
            for item in raw:
                if isinstance(item, str):
                    text = item.strip().strip('[]')
                    names.extend([self._normalize(n) for n in text.split(',') if n.strip()])
                else:
                    names.append(self._normalize(str(item)))
            return names

        text = str(raw).strip().strip('[]')
        return [self._normalize(n) for n in text.split(',') if n.strip()]

    def _parse_upscale_segments(self, text):
        """UPSCALE PROMPT를 | 로 분리"""
        return [s.strip() for s in text.split('|') if s.strip()]

    def _match_char_in_segment(self, segment, ordered_names):
        """세그먼트에서 캐릭터 이름 매칭 (assigned_names 순서 우선)"""
        seg_norm = self._normalize(segment)
        for name in ordered_names:
            if name and name in seg_norm:
                return name
        return None

    def filter_closed_eyes(self, identify_output, assigned_names, upscale_prompt):
        # INPUT_IS_LIST = True 이므로 모든 입력이 리스트로 들어옴
        # 빈 리스트 방어
        identify_output = identify_output[0] if identify_output else ""
        upscale_prompt = upscale_prompt[0] if upscale_prompt else ""

        if not identify_output or not identify_output.strip():
            print("[FilterClosedEyes] identify_output is empty, returning empty string")
            return ("",)

        # assigned_names: 리스트 그대로 사용 (빈 리스트면 매칭 없이 통과)
        # parsed later in _parse_assigned_names

        # 입력 파싱
        header, blocks = self._parse_identify_output(identify_output)
        ordered_names = self._parse_assigned_names(assigned_names)
        segments = self._parse_upscale_segments(upscale_prompt)

        print(f"[FilterClosedEyes] header={repr(header)}, blocks={len(blocks)}, "
              f"ordered_names={ordered_names}, segments={len(segments)}")

        # 각 upscale 세그먼트를 캐릭터 이름에 매칭
        char_to_segment = {}
        for seg in segments:
            matched = self._match_char_in_segment(seg, ordered_names)
            if matched:
                char_to_segment[matched] = seg
            else:
                print(f"[FilterClosedEyes] No character match for segment: {seg[:80]}...")

        # closed eyes / wink 감지된 캐릭터 처리
        for block in blocks:
            seg = char_to_segment.get(block['char_name'])
            if not seg:
                continue
            if self._has_closed_eyes(seg):
                block['content'] = self._replace_eyes_with_closed(block['content'])
                print(f"[FilterClosedEyes] '{block['char_name']}' - replaced eyes tags with 'closed eyes'")
            elif self._has_wink(seg):
                block['content'] = self._add_wink(block['content'])
                print(f"[FilterClosedEyes] '{block['char_name']}' - added 'wink' tag")

        result = header + '\n' + '\n'.join(
            f"[{b['idx']}] {b['content']}" for b in blocks
        )

        return (result,)
