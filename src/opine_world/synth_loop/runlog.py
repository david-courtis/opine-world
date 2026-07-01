"""Append-only, grep-friendly run log for the agentic consumer.

Each step is a fixed-format block delimited by [STEP N] ... [/STEP N] markers.
Synthesis events, resume markers, and notes share the same file with their own markers.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class RunLog:
    """Writes each step to both run_log.txt (master) and levels/level_N.log (per-level).

    Per-level files prevent cross-level state pollution when the analyzer filters transitions.
    Cross-level events (synthesis, level_advance, resume, notes) land only in run_log.txt.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("")
        self.levels_dir = self.path.parent / "levels"
        self.levels_dir.mkdir(parents=True, exist_ok=True)

    def _level_path(self, level: int) -> Path:
        return self.levels_dir / f"level_{int(level)}.log"

    def write_header(self, header: dict[str, Any]) -> None:
        with open(self.path, "a") as f:
            f.write("[RUN_HEADER]\n")
            f.write(json.dumps(header, indent=2, default=str))
            f.write("\n[/RUN_HEADER]\n\n")

    def append_step(
        self,
        step: int,
        action_id: int,
        action_name: str,
        level: int,
        reward: float,
        done: bool,
        diff_text: str,
        state_desc: str,
        after_state: list[dict],
        ascii_frame: str | None = None,
        action_sequence_diffs: list[str] | None = None,
    ) -> None:
        buf: list[str] = []
        buf.append(f"[STEP {step}]\n")
        buf.append(
            f"[ACTION] action_id={action_id} action_name={action_name}\n"
        )
        buf.append(f"[LEVEL] {level}\n")
        buf.append(f"[REWARD] {reward} done={done}\n")
        buf.append(f"[DIFF] {diff_text}\n")
        buf.append(f"[STATE_DESC] {state_desc}\n")
        buf.append("[STATE_JSON] ")
        buf.append(json.dumps(after_state, default=_json_default))
        buf.append("\n")
        if ascii_frame:
            buf.append("[ASCII_FRAME]\n")
            buf.append(ascii_frame.rstrip())
            buf.append("\n[/ASCII_FRAME]\n")
        if action_sequence_diffs:
            n = len(action_sequence_diffs)
            buf.append(f"[ACTION_SEQUENCE n={n}]\n")
            for i, d in enumerate(action_sequence_diffs):
                buf.append(f"  tick {i+1}/{n}: {d}\n")
            buf.append("[/ACTION_SEQUENCE]\n")
        buf.append(f"[/STEP {step}]\n\n")
        block = "".join(buf)
        with open(self.path, "a") as f:
            f.write(block)
        with open(self._level_path(level), "a") as f:
            f.write(block)

    def append_note(self, step: int, source: str, text: str) -> None:
        flat = " ".join(text.splitlines()).strip()
        with open(self.path, "a") as f:
            f.write(f"[NOTE step={step} source={source}] {flat}\n")

    def append_synthesis(
        self,
        step: int,
        run_idx: int,
        accuracy: str,
        reward_src: str | None,
        duration_s: float | None = None,
        escalated: bool = False,
    ) -> None:
        with open(self.path, "a") as f:
            tag = " [ESCALATED]" if escalated else ""
            f.write(f"[SYNTHESIS step={step} run={run_idx}{tag}]\n")
            f.write(f"  accuracy: {accuracy}\n")
            if duration_s is not None:
                f.write(f"  duration_s: {duration_s}\n")
            if reward_src:
                f.write("  reward_function:\n")
                f.write("  ```python\n")
                for line in reward_src.splitlines():
                    f.write(f"  {line}\n")
                f.write("  ```\n")
            f.write(f"[/SYNTHESIS run={run_idx}]\n\n")

    def append_level_advance(
        self, step: int, from_level: int, to_level: int,
    ) -> None:
        marker = (
            f"[LEVEL_ADVANCE step={step} "
            f"from={from_level} to={to_level}]\n\n"
        )
        with open(self.path, "a") as f:
            f.write(marker)
        with open(self._level_path(from_level), "a") as f:
            f.write(f"[LEVEL_END from={from_level} at_step={step}]\n\n")
        with open(self._level_path(to_level), "a") as f:
            f.write(f"[LEVEL_START to={to_level} at_step={step}]\n\n")

    def append_resume_marker(self, step: int) -> None:
        with open(self.path, "a") as f:
            f.write(f"[RESUME at_step={step}]\n\n")


def _json_default(obj: Any) -> Any:
    try:
        import numpy as np
    except Exception:
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
