import traceback


class SoyaQwenEditPromptParser_mdsoya:
    """Parse the explicit Qwen Edit payload used by the hooking server."""

    _FIELDS = {
        "EDIT_PROMPT": "STRING",
        "NEGATIVE_PROMPT": "STRING",
        "IMAGE_PATH": "STRING",
        "MASK_PATH": "STRING",
        "SEED": "INT",
        "STEPS": "INT",
        "CFG": "FLOAT",
        "DENOISE": "FLOAT",
        "MASK_GROW": "INT",
        "MASK_BLUR": "FLOAT",
        "FILENAME_PREFIX": "STRING",
        "WIDTH": "INT",
        "HEIGHT": "INT",
    }

    _DEFAULTS = {
        "EDIT_PROMPT": "",
        "NEGATIVE_PROMPT": "",
        "IMAGE_PATH": "",
        "MASK_PATH": "",
        "SEED": 0,
        "STEPS": 6,
        "CFG": 1.0,
        "DENOISE": 1.0,
        "MASK_GROW": 8,
        "MASK_BLUR": 4.0,
        "FILENAME_PREFIX": "qwen_edit/output",
        "WIDTH": 1024,
        "HEIGHT": 1024,
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": False,
                    },
                ),
            },
        }

    RETURN_TYPES = tuple(field_type for field_type in _FIELDS.values())
    RETURN_NAMES = tuple(_FIELDS.keys())
    FUNCTION = "parse"
    CATEGORY = "Soya/Util"

    @classmethod
    def _parse_sections(cls, text):
        if not isinstance(text, str):
            print(
                "[QWEN_EDIT_PARSER] 입력 형식 오류: "
                f"type={type(text).__name__}, value={text!r}"
            )
            raise TypeError("Qwen Edit 파서 입력은 문자열이어야 합니다")
        if not text.strip():
            print("[QWEN_EDIT_PARSER] 입력값이 비어 있어 파싱할 수 없습니다")
            raise ValueError("Qwen Edit 파서 입력값이 비어 있습니다")

        parsed = {}
        current_key = None
        current_lines = []

        def commit_current():
            if current_key is None:
                return
            if current_key in parsed:
                print(
                    "[QWEN_EDIT_PARSER] 중복 섹션 감지: "
                    f"field={current_key}, input_preview={text[:300]!r}"
                )
                raise ValueError(f"중복된 Qwen Edit 필드: {current_key}")
            parsed[current_key] = "\n".join(current_lines).strip()

        for line_number, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.rstrip("\r")
            stripped = line.strip()
            is_section = stripped.startswith("[") and stripped.endswith("]")
            if is_section:
                commit_current()
                candidate = stripped[1:-1].strip()
                if candidate not in cls._FIELDS:
                    print(
                        "[QWEN_EDIT_PARSER] 알 수 없는 섹션: "
                        f"line={line_number}, field={candidate!r}"
                    )
                    raise ValueError(f"알 수 없는 Qwen Edit 필드: {candidate}")
                current_key = candidate
                current_lines = []
            elif current_key is not None:
                current_lines.append(line)
            elif stripped:
                print(
                    "[QWEN_EDIT_PARSER] 섹션 밖 텍스트 감지: "
                    f"line={line_number}, value={line!r}"
                )
                raise ValueError(
                    f"Qwen Edit 필드 헤더 이전에 값이 있습니다 (line {line_number})"
                )

        commit_current()
        return parsed

    @classmethod
    def _convert_value(cls, key, field_type, raw):
        if raw == "":
            default = cls._DEFAULTS[key]
            if key in ("EDIT_PROMPT", "IMAGE_PATH", "MASK_PATH"):
                print(
                    "[QWEN_EDIT_PARSER] 필수 필드가 비어 있음: "
                    f"field={key}, parsed_value={raw!r}"
                )
                raise ValueError(f"Qwen Edit 필수 필드가 비어 있습니다: {key}")
            print(
                "[QWEN_EDIT_PARSER] 선택 필드가 비어 있어 기본값 사용: "
                f"field={key}, default={default!r}"
            )
            return default

        try:
            if field_type == "INT":
                value = int(raw)
            elif field_type == "FLOAT":
                value = float(raw)
            else:
                value = raw
        except (TypeError, ValueError, OverflowError) as exc:
            print(
                "[QWEN_EDIT_PARSER] 필드 변환 실패: "
                f"field={key}, type={field_type}, value={raw!r}, error={exc}"
            )
            traceback.print_exc()
            raise ValueError(
                f"Qwen Edit 필드 변환 실패: {key}={raw!r}"
            ) from exc

        if key == "STEPS" and not 1 <= value <= 100:
            print(f"[QWEN_EDIT_PARSER] STEPS 범위 오류: value={value}")
            raise ValueError("STEPS는 1~100이어야 합니다")
        if key == "CFG" and not 0.0 <= value <= 100.0:
            print(f"[QWEN_EDIT_PARSER] CFG 범위 오류: value={value}")
            raise ValueError("CFG는 0~100이어야 합니다")
        if key == "DENOISE" and not 0.0 <= value <= 1.0:
            print(f"[QWEN_EDIT_PARSER] DENOISE 범위 오류: value={value}")
            raise ValueError("DENOISE는 0~1이어야 합니다")
        if key == "MASK_BLUR" and not 0.0 <= value <= 100.0:
            print(f"[QWEN_EDIT_PARSER] MASK_BLUR 범위 오류: value={value}")
            raise ValueError("MASK_BLUR는 0~100이어야 합니다")
        if key in ("WIDTH", "HEIGHT"):
            if not 16 <= value <= 16384:
                print(
                    "[QWEN_EDIT_PARSER] 이미지 크기 범위 오류: "
                    f"field={key}, value={value}"
                )
                raise ValueError(f"{key}는 16~16384여야 합니다")
            if value % 16 != 0:
                print(
                    "[QWEN_EDIT_PARSER] 이미지 크기 배수 오류: "
                    f"field={key}, value={value}"
                )
                raise ValueError(f"{key}는 16의 배수여야 합니다")
        return value

    def parse(self, text):
        try:
            parsed = self._parse_sections(text)
            missing = [
                key
                for key in ("EDIT_PROMPT", "IMAGE_PATH", "MASK_PATH")
                if key not in parsed
            ]
            if missing:
                print(
                    "[QWEN_EDIT_PARSER] 필수 섹션 누락: "
                    f"missing={missing}, input_preview={text[:300]!r}"
                )
                raise ValueError(
                    "Qwen Edit 필수 필드가 누락되었습니다: " + ", ".join(missing)
                )

            results = tuple(
                self._convert_value(
                    key,
                    field_type,
                    parsed.get(key, ""),
                )
                for key, field_type in self._FIELDS.items()
            )
            print(
                "[QWEN_EDIT_PARSER] 파싱 완료: "
                f"image={parsed.get('IMAGE_PATH')!r}, "
                f"mask={parsed.get('MASK_PATH')!r}, "
                f"steps={results[5]}, cfg={results[6]}, denoise={results[7]}"
            )
            return results
        except Exception as exc:
            print(
                "[QWEN_EDIT_PARSER] 파싱 예외: "
                f"type={type(exc).__name__}, error={exc}, "
                f"input_preview={str(text)[:300]!r}"
            )
            traceback.print_exc()
            raise
