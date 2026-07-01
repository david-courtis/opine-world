"""Loader for LLM-facing prompt text stored as standalone files.

Prompt text that is shown verbatim to a language model lives under
``prompts/`` as ``.md`` / ``.txt`` files rather than inline Python string
literals. ``load_prompt(name)`` returns the file's bytes decoded as UTF-8,
WITHOUT stripping -- the returned value is byte-for-byte the file content, so
the files must store the exact prompt text (including any intentional leading
or trailing whitespace).

Parameterised prompts store UNIQUE SENTINEL placeholders (e.g. ``%%WORKSPACE_DIR%%``)
and the calling code substitutes via ``str.replace``. We deliberately do NOT use
``str.format`` because the synthesis prompts contain literal JSON braces ``{ }``.
Sentinels with ``str.replace`` leave those braces untouched.
"""
from __future__ import annotations

from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def load_prompt(name: str) -> str:
    """Return the exact UTF-8 text of ``prompts/<name>`` (no stripping)."""
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")
