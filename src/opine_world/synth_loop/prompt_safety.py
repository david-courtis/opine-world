"""Sanitizers for text that is fed back into LLM prompts.

The engine writes and re-reads natural-language handoff artifacts. Those files
must not become a side channel for orchestration commands such as sweep control.
"""
from __future__ import annotations

from typing import Any


_CONTROL_MARKERS = (
    "from run_manager.control import",
    "run_manager.control",
    "stop_sweep",
    "resume_sweep",
    "gateway_status",
    "STOP_ERROR",
    "RESUME_ERROR",
    "print('GATEWAY'",
    'print("GATEWAY"',
)

_INJECTION_NOTE_MARKERS = (
    "PROMPT-INJECTION FLAG",
    "Security note first",
    "Note on the injected snippet",
    "injected stop_sweep",
    "injected snippet",
)


def contains_operational_control_text(value: Any) -> bool:
    text = str(value or "")
    return any(marker in text for marker in _CONTROL_MARKERS)


def _paragraph_is_unsafe(lines: list[str]) -> bool:
    para = "\n".join(lines)
    if contains_operational_control_text(para):
        return True
    if any(marker in para for marker in _INJECTION_NOTE_MARKERS):
        return True
    return False


def sanitize_model_visible_text(value: Any) -> str:
    """Remove orchestration-control snippets from text before prompting.

    This intentionally works at paragraph granularity. A leaked control block may
    contain harmless-looking neighbor lines such as `import json`, `names = [...]`,
    or `for name in names`. Removing only the marker line would leave executable
    residue behind.
    """
    text = str(value or "")
    if not text:
        return ""

    lines = text.splitlines()
    kept: list[str] = []
    para: list[str] = []

    def flush() -> None:
        nonlocal para
        if not para:
            return
        if not _paragraph_is_unsafe(para):
            kept.extend(para)
        para = []

    for line in lines:
        if line.strip():
            para.append(line)
        else:
            flush()
            if kept and kept[-1] != "":
                kept.append("")

    flush()
    while kept and kept[-1] == "":
        kept.pop()
    return "\n".join(kept)
