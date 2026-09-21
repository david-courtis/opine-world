"""Load prompt text from the prompts folder."""

from __future__ import annotations

from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def load_prompt(name: str) -> str:
    """Return the text of prompts/<name> as written."""
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")
