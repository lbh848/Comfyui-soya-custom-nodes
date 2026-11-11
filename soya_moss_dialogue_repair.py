"""Repair corrupted Korean speech-balloon dialogue in a MOSS document.

The node sits between the document PrimitiveString and the MOSS V2 wrappers.
It reads the already-loaded Qwen3.5-9B CLIP (the MOSS text encoder), corrects
only ``moss.items`` entries whose ``label`` is ``"text"``, patches the
``workflow.detailer.layout`` copy, and returns the same document JSON.

The control block is authoritative::

    "workflow": {
      "dialogue_repair": {
        "enabled": true,
        "anchor": "C017",
        "source": "[C001]\\n...\\n\\n[C017]\\n..."
      }
    }

``enabled`` false (or a missing source) returns the input untouched.  The
model is asked for plain ``index. text`` lines, never JSON, so a malformed
reply cannot break the document.
"""
from __future__ import annotations

import json
import math
import re

_SECTION_RE = re.compile(r"^\[(C\d+)\]\s*$", re.MULTILINE)
_LINE_RE = re.compile(r"^\s*(\d+)\s*[.):]\s*(.*)$")
_HANGUL_RE = re.compile(r"[가-힣]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_CJK_RE = re.compile(r"[一-鿿぀-ヿ]")
_BAD_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f�]")

# Roughly 0.63 tokens per Korean character with the Qwen3.5 tokenizer.  Only
# used to choose the source window; the final prompt is tokenized exactly.
_TOKENS_PER_CHAR = 0.63
_BLOCK_OVERHEAD_TOKENS = 8

SYSTEM_PROMPT = """You repair corrupted Korean dialogue for manga speech balloons.
Use the supplied source passages as ground truth.
The input lists broken lines as "index. text", one per line.
Return exactly one corrected line per input line, in the same "index. text" format.
Do not output JSON, explanations, headings, blank lines, or any other text.
Never add, remove, merge, reorder or invent lines.
If a line is already correct, return it unchanged.
Preserve the source wording; shorten only when the input line is shorter than the source line.
Keep any literal \\n escape sequences."""


# ---------------------------------------------------------------------------
# Pure helpers (kept import-free so they can be unit tested without ComfyUI)
# ---------------------------------------------------------------------------

def text_items(moss: dict) -> list[tuple[int, dict]]:
    """``label:"text"`` items in document order, skipping empty text.

    ``build_swap_text_layout`` uses the same filter, so the list lines up with
    the ``workflow.detailer.layout`` text items.
    """
    out = []
    for index, item in enumerate((moss or {}).get("items") or []):
        if not isinstance(item, dict):
            continue
        if str(item.get("label") or "").strip().lower() != "text":
            continue
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        out.append((index, item))
    return out


def source_blocks(source: str) -> list[tuple[str, str]]:
    """Split a ``[Cxxx]`` block string into ``(id, text)`` pairs."""
    text = str(source or "")
    matches = list(_SECTION_RE.finditer(text))
    if not matches:
        body = text.strip()
        return [("", body)] if body else []
    blocks = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        blocks.append((match.group(1), text[match.end():end].strip()))
    return blocks


def estimate_tokens(text: str) -> int:
    return int(math.ceil(len(text) * _TOKENS_PER_CHAR))


def render_user_text(blocks: list[tuple[str, str]], items: list[tuple[int, dict]]) -> str:
    lines = ["# Source"]
    for block_id, body in blocks:
        lines.append(f"[{block_id}]" if block_id else "[Source]")
        lines.append(body)
        lines.append("")
    lines.append("# Broken lines (index. text, one per line)")
    for index, (_, item) in enumerate(items):
        body = str(item.get("text") or "").replace("\n", "\\n")
        lines.append(f"{index}. {body}")
    return "\n".join(lines)


def order_blocks(blocks: list[tuple[str, str]], anchor: str) -> list[int]:
    """Anchor first, then alternately outward: -1, +1, -2, +2, ..."""
    if not blocks:
        return []
    anchor_index = next((i for i, (block_id, _) in enumerate(blocks)
                         if block_id == anchor), None)
    if anchor_index is None:
        anchor_index = 0
    order = [anchor_index]
    for distance in range(1, len(blocks)):
        before = anchor_index - distance
        after = anchor_index + distance
        if before >= 0:
            order.append(before)
        if after < len(blocks):
            order.append(after)
    return order


def select_blocks(blocks: list[tuple[str, str]], anchor: str, items: list[tuple[int, dict]],
                  budget: int, count_tokens=None) -> list[tuple[str, str]]:
    """Grow the source window from the anchor until the token budget is hit.

    ``count_tokens`` receives the full user text and returns an exact count.
    It is called once for the final candidate, so an over-budget window is
    trimmed by dropping the farthest block.
    """
    if not blocks:
        return []
    order = order_blocks(blocks, anchor)
    picked: list[int] = []
    used = estimate_tokens(render_user_text([], items))
    for index in order:
        block_id, body = blocks[index]
        cost = estimate_tokens(f"[{block_id}]\n{body}\n\n") + _BLOCK_OVERHEAD_TOKENS
        if picked and used + cost > budget:
            break
        if not picked and used + cost > budget:
            return []  # even the anchor alone does not fit
        picked.append(index)
        used += cost
    while picked:
        chosen = [blocks[i] for i in sorted(picked)]
        if count_tokens is None or count_tokens(render_user_text(chosen, items)) <= budget:
            return chosen
        picked.pop()
    return []


def count_token_structure(tokens) -> int:
    """Count tokens in the ``{clip_name: [[(token, weight), ...]]}`` structure."""
    total = 0
    groups = tokens.values() if isinstance(tokens, dict) else [tokens]
    for chunks in groups:
        for chunk in chunks or []:
            total += len(chunk)
    return total


def parse_output(raw: str, expected: int) -> dict[int, str]:
    """Parse plain ``index. text`` lines. Non-matching lines are ignored."""
    text = str(raw or "")
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    out: dict[int, str] = {}
    for line in text.splitlines():
        match = _LINE_RE.match(line)
        if not match:
            continue
        index = int(match.group(1))
        if not 0 <= index < expected or index in out:
            continue
        out[index] = match.group(2).strip().replace("\\n", "\n")
    return out


def is_valid_korean(candidate: str, original: str) -> bool:
    """Reject replies that are not plain, source-shaped Korean."""
    value = str(candidate or "").strip()
    if not value:
        return False
    if _BAD_CHAR_RE.search(value):
        return False
    if _LATIN_RE.search(value) or _CJK_RE.search(value):
        return False
    hangul = len(_HANGUL_RE.findall(value))
    if hangul == 0:
        return False
    visible = sum(1 for char in value if not char.isspace())
    if hangul / max(visible, 1) < 0.4:
        return False
    original_length = len(str(original or "").strip())
    if original_length >= 8:
        ratio = len(value) / original_length
        if ratio < 0.5 or ratio > 2.0:
            return False
    return True


def bbox_key(item: dict):
    bbox = item.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        return tuple(int(round(float(value))) for value in bbox)
    except (TypeError, ValueError):
        return None


def patch_detailer_layout(document: dict, items: list[tuple[int, dict]]) -> None:
    """Mirror corrected ``moss.items`` text into ``workflow.detailer.layout``."""
    workflow = document.get("workflow")
    if not isinstance(workflow, dict):
        return
    detailer = workflow.get("detailer")
    if not isinstance(detailer, dict) or not isinstance(detailer.get("layout"), str):
        return
    try:
        layout = json.loads(detailer["layout"])
    except (TypeError, ValueError):
        return
    by_bbox = {}
    for _, item in items:
        key = bbox_key(item)
        if key is not None:
            by_bbox[key] = str(item.get("text") or "")
    for entry in layout.get("items") or []:
        if not isinstance(entry, dict) or str(entry.get("label") or "").lower() != "text":
            continue
        key = bbox_key(entry)
        if key is not None and key in by_bbox:
            entry["text"] = by_bbox[key]
    detailer["layout"] = json.dumps(layout, ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class SoyaMossDialogueRepairV2_mdsoya:
    """Correct only ``label:"text"`` dialogue with the loaded MOSS text encoder."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_json": ("STRING", {"forceInput": True}),
                "clip": ("CLIP",),
            },
            "optional": {
                "max_prompt_tokens": ("INT", {"default": 24576, "min": 512,
                                              "max": 32000, "step": 256}),
                "max_new_tokens": ("INT", {"default": 2048, "min": 64,
                                           "max": 4096, "step": 64}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt_json",)
    FUNCTION = "repair"
    CATEGORY = "Soya/MOSS"

    def repair(self, prompt_json, clip, max_prompt_tokens=24576, max_new_tokens=2048):
        try:
            document = json.loads(prompt_json)
        except Exception as exc:
            print(f"[Soya:MossDialogueRepair] document JSON parse failed: {exc}")
            return (prompt_json,)
        try:
            config = ((document.get("workflow") or {}).get("dialogue_repair")
                      if isinstance(document, dict) else None)
            if not isinstance(config, dict) or not config.get("enabled"):
                return (prompt_json,)
            source = str(config.get("source") or "")
            if not source.strip():
                return (prompt_json,)
            items = text_items(document.get("moss") or {})
            if not items:
                return (prompt_json,)

            def count_tokens(text: str) -> int:
                tokens = clip.tokenize(text, thinking=False,
                                       system_prompt=SYSTEM_PROMPT, min_length=1)
                return count_token_structure(tokens)

            blocks = select_blocks(source_blocks(source), str(config.get("anchor") or ""),
                                   items, int(max_prompt_tokens), count_tokens)
            if not blocks:
                print("[Soya:MossDialogueRepair] source window does not fit the budget")
                return (prompt_json,)
            user_text = render_user_text(blocks, items)
            prompt_tokens = count_tokens(user_text)
            tokens = clip.tokenize(user_text, thinking=False,
                                   system_prompt=SYSTEM_PROMPT, min_length=1)
            ids = clip.generate(tokens, do_sample=False, max_length=int(max_new_tokens),
                                temperature=1.0, top_k=50, top_p=0.95, min_p=0.0,
                                repetition_penalty=1.0, seed=None,
                                presence_penalty=0.0, mtp=True)
            raw = clip.decode(ids)
            corrected = parse_output(raw, len(items))
            accepted = 0
            for index, (_, item) in enumerate(items):
                candidate = corrected.get(index)
                original = str(item.get("text") or "")
                if candidate is not None and is_valid_korean(candidate, original):
                    item["text"] = candidate
                    accepted += 1
            if not accepted:
                print("[Soya:MossDialogueRepair] no valid correction; keeping input: "
                      f"items={len(items)}, prompt_tokens={prompt_tokens}")
                return (prompt_json,)
            patch_detailer_layout(document, items)
            print("[Soya:MossDialogueRepair] corrected "
                  f"{accepted}/{len(items)} lines, prompt_tokens={prompt_tokens}, "
                  f"blocks={[block_id for block_id, _ in blocks]}")
            return (json.dumps(document, ensure_ascii=False,
                               separators=(",", ":")),)
        except Exception as exc:
            print(f"[Soya:MossDialogueRepair] failed: {type(exc).__name__}: {exc}")
            return (prompt_json,)


NODE_CLASS_MAPPINGS = {"SoyaMossDialogueRepairV2_mdsoya": SoyaMossDialogueRepairV2_mdsoya}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SoyaMossDialogueRepairV2_mdsoya": "MOSS Dialogue Repair V2 (Soya)",
}
