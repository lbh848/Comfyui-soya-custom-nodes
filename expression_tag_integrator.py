import re


class ExpressionTagIntegrator_mdsoya:
    """
    Expression Tag Extractor의 표정 태그를 캐릭터별 프롬프트에 통합하는 노드.
    배치 순서대로 [1], [2], ... 캐릭터에 표정 태그를 추가한다.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "character_prompts": ("STRING", {"multiline": True}),
                "expression_tags": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("combined",)
    OUTPUT_IS_LIST = (False,)
    INPUT_IS_LIST = True
    FUNCTION = "integrate"
    CATEGORY = "Soya"

    def integrate(self, character_prompts, expression_tags):
        # INPUT_IS_LIST = True: 모든 입력이 리스트로 들어옴
        # character_prompts: [single_string] from IdentifyCharacters (OUTPUT_IS_LIST=False)
        # expression_tags: [tag_str_1, tag_str_2, ...] from ExpressionTagExtractor (OUTPUT_IS_LIST=True)
        char_text = character_prompts[0] if isinstance(character_prompts, list) and len(character_prompts) == 1 else character_prompts

        lines = char_text.strip().split('\n')
        result_lines = []
        char_idx = 0

        for line in lines:
            stripped = line.strip()
            if re.match(r'^\[\d+\]', stripped):
                expr_tags = expression_tags[char_idx] if char_idx < len(expression_tags) else ""
                if expr_tags:
                    result_lines.append(f"{stripped}, {expr_tags}")
                else:
                    result_lines.append(stripped)
                char_idx += 1
            else:
                result_lines.append(stripped)

        result = "\n".join(result_lines)
        print(f"[Soya: ExpressionTagIntegrator] Integrated {char_idx} characters with expression tags")
        return (result,)
