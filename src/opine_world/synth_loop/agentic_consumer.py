"""Agentic consumer: a Claude Code subprocess that picks actions via Bash + Read + Grep over the engine's workspace artifacts.

Output protocol: the agent writes next_actions.json containing {"plan": [<action_id>, ...], "reasoning": "..."}. The engine reads it and feeds it into PLAN-mode machinery.
"""
from __future__ import annotations

import base64
import importlib.util as _importlib_util
import json
import os
import pickle
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


def _load_prompt_loader():
    """Import the sibling ``prompts.py`` loader by file path.

    A plain ``from .prompts import load_prompt`` is unsafe here: this module
    is loaded by some entrypoints (e.g. play.py) via
    importlib under stub parent packages that have no ``__path__``, so a
    relative import of an un-preloaded submodule fails. Loading by path works
    in every scheme.
    """
    path = Path(__file__).resolve().parent / "prompts.py"
    spec = _importlib_util.spec_from_file_location(
        "_synth_loop_prompts_loader", path
    )
    mod = _importlib_util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_prompt


load_prompt = _load_prompt_loader()


def _load_prompt_safety():
    path = Path(__file__).resolve().parent / "prompt_safety.py"
    spec = _importlib_util.spec_from_file_location(
        "_synth_loop_prompt_safety", path
    )
    mod = _importlib_util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.sanitize_model_visible_text


sanitize_model_visible_text = _load_prompt_safety()


_VIEW_SPRITE_SCRIPT = '''#!/usr/bin/env python3
"""ASCII slice of any sprite or rectangular region in the latest frame.

Source of truth: the most recent ``[ASCII_FRAME]`` block in run_log.txt.
The 64x64 ASCII grid is the canonical post-action board state -- slicing
it at a sprite's (x, y, w, h) gives you that sprite's exact pixel content.

Usage:
    # by sprite name (from current_state.json)
    python tools/view_sprite.py --name sfqyzhzkij

    # by tag (matches any sprite carrying this tag)
    python tools/view_sprite.py --tag rhsxkxzdjz

    # arbitrary box: x=col, y=row, w=width, h=height (image convention)
    python tools/view_sprite.py --xywh 12 38 9 9

    # widen the slice with --pad to see surrounding context
    python tools/view_sprite.py --name wgmbtyhvbc --pad 2

Output: header line + one line per row of the slice, in ASCII codepoints
(digits 0-9 then A-F for palette values 10-15). Same encoding as
[ASCII_FRAME] blocks in run_log.txt.
"""
import argparse, json, os, re, sys


def latest_ascii_frame(path):
    """Pull the most recent [ASCII_FRAME] block."""
    if not os.path.exists(path):
        return None
    data = open(path).read()
    matches = re.findall(
        r"\\[ASCII_FRAME\\]\\s*(.*?)\\s*\\[/ASCII_FRAME\\]",
        data, re.DOTALL,
    )
    if not matches:
        return None
    return [ln for ln in matches[-1].splitlines() if ln.strip()]


def find_sprite(name=None, tag=None, state_path="current_state.json"):
    """Return (x, y, w, h) for the first matching sprite, or None."""
    if not os.path.exists(state_path):
        return None
    state = json.load(open(state_path)).get("state", [])
    for o in state:
        if name and o.get("name") == name:
            return o["x"], o["y"], o["w"], o["h"]
        if tag and tag in (o.get("tags") or []):
            return o["x"], o["y"], o["w"], o["h"]
    return None


def slice_grid(grid, x, y, w, h, pad=0):
    """Return rows of the (x,y,w,h) box, optionally padded."""
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(len(grid[0]) if grid else 0, x + w + pad)
    y1 = min(len(grid), y + h + pad)
    return [row[x0:x1] for row in grid[y0:y1]], (x0, y0, x1, y1)


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--name", help="match sprite by exact name")
    g.add_argument("--tag", help="match sprite by tag")
    g.add_argument("--xywh", nargs=4, type=int,
                   metavar=("X", "Y", "W", "H"),
                   help="explicit box: x=col, y=row, w=width, h=height")
    p.add_argument("--pad", type=int, default=0,
                   help="extend the slice by N cells on each side")
    p.add_argument("--frame", default="run_log.txt",
                   help="ascii source (default: run_log.txt)")
    p.add_argument("--state", default="current_state.json",
                   help="state json source")
    args = p.parse_args()

    grid = latest_ascii_frame(args.frame)
    if grid is None:
        print(f"ERROR: no [ASCII_FRAME] block in {args.frame}", file=sys.stderr)
        sys.exit(2)

    if args.xywh:
        x, y, w, h = args.xywh
    else:
        bbox = find_sprite(name=args.name, tag=args.tag, state_path=args.state)
        if bbox is None:
            who = args.name or args.tag
            print(f"ERROR: no sprite matching {who!r} in {args.state}",
                  file=sys.stderr)
            sys.exit(3)
        x, y, w, h = bbox

    sliced, (x0, y0, x1, y1) = slice_grid(grid, x, y, w, h, pad=args.pad)
    label = (args.name or args.tag
             or f"region x={x},y={y},w={w},h={h}")
    print(f"# {label}: actual slice cols {x0}-{x1-1} rows {y0}-{y1-1} "
          f"(pad={args.pad})")
    for row in sliced:
        print(row)


if __name__ == "__main__":
    main()
'''


_ESCAPE_TOOL_SCRIPT = '''#!/usr/bin/env python3
"""L\\u00e9vy-flight + self-avoiding random walk for escape from stuck states.

Madras-Slade self-avoiding walk (no immediate reverse) + geometric(p=0.7)
clipped run lengths in [1,10] (heavy-tailed dashes for basin escape) +
tabular novelty bias (least-used direction first). The consumer invokes
this when it diagnoses itself as stuck (e.g., the last K transitions
all show "Nothing changed").

Usage:
    python tools/escape_sequence.py --actions 1 2 3 4 --length 50

Output (stdout, JSON):
    {"plan": [1, 1, 1, 3, 3, 2, ...]}
"""
import argparse, json, random

REVERSE = {1: 2, 2: 1, 3: 4, 4: 3}


def build(actions, length=50, seed=None):
    rng = random.Random(seed)
    available = [a for a in actions if a in REVERSE]
    if not available:
        return []
    seq, last = [], None
    while len(seq) < length:
        forbidden = REVERSE.get(last)
        candidates = [a for a in available if a != forbidden] or available
        usage = {a: seq.count(a) for a in candidates}
        candidates.sort(key=lambda a: (usage[a], rng.random()))
        direction = candidates[0]
        run = 1
        while rng.random() < 0.7 and run < 10:
            run += 1
        for _ in range(run):
            if len(seq) >= length:
                break
            seq.append(direction)
        last = direction
    return seq[:length]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--actions", type=int, nargs="+", required=True)
    p.add_argument("--length", type=int, default=50)
    p.add_argument("--seed", type=int, default=None)
    a = p.parse_args()
    print(json.dumps({"plan": build(a.actions, a.length, a.seed)}))
'''


_PLANNER_TOOL_SCRIPT = r'''#!/usr/bin/env python3
"""Plan a route to the goal using the synthesised world model (game_engine.py).

This is YOUR tool. It searches transition_function + reward_function for an
action sequence the MODEL predicts reaches reward, starting from the CURRENT
board. The model can be WRONG in states it has not seen, so treat the route as a
suggestion to sanity-check and try, NOT as ground truth. If the model looks
stale or mispredicts, FORCE A CEGIS REPAIR FIRST -- write synth_control.json
{"force_now": true, "focus": "<concrete mismatch>"}, let it re-synthesise, then
re-run this tool against the fresh model.

Usage:
    python tools/plan.py                                  # plan from current board
    python tools/plan.py --max-depth 12 --timeout 30 --max-nodes 6000
Output (stdout JSON):
    {"ok": bool, "plan": [...], "source": "...", "reason": "...", "nodes": N}
"""
import argparse, copy, json, os, time, importlib.util
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)


def load_engine():
    path = os.path.join(WS, "game_engine.py")
    if not os.path.exists(path):
        print(json.dumps({"ok": False, "reason": "no game_engine.py yet (synthesis has not run)"}))
        raise SystemExit(0)
    spec = importlib.util.spec_from_file_location("_plan_game_engine", path)
    mod = importlib.util.module_from_spec(spec)
    mod.copy = copy
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        print(json.dumps({"ok": False, "reason": "load error: %s: %s" % (type(e).__name__, e)}))
        raise SystemExit(0)
    return mod


def cur_state():
    d = json.load(open(os.path.join(WS, "current_state.json")))
    actions = [int(a) for a in d.get("available_actions", [])]
    if d.get("frame") is not None:
        return d["frame"], actions, True            # frames-only: start is the grid
    return d.get("state", []), actions, False        # sprite mode: start is the object list


def reward_tuple(v):
    try:
        if isinstance(v, (list, tuple)):
            return (float(v[0]) if v else 0.0), (bool(v[1]) if len(v) > 1 else (len(v) > 0 and float(v[0]) > 0))
        if isinstance(v, bool):
            return (1.0 if v else 0.0), v
        return float(v), float(v) > 0
    except Exception:
        return 0.0, False


def candidates(actions, frames, state, eng, max_click):
    out = [a for a in sorted(set(actions)) if a not in (0, 6, 7)]
    if 6 in actions:
        objs = []
        if frames and hasattr(eng, "extract_objects"):
            try: objs = eng.extract_objects(copy.deepcopy(state)) or []
            except Exception: objs = []
        elif not frames and isinstance(state, list):
            objs = [o for o in state if isinstance(o, dict)]
        seen = set()
        for o in objs:
            try:
                x = int(o.get("display_x", o.get("x", 0))); y = int(o.get("display_y", o.get("y", 0)))
                w = int(o.get("display_w", o.get("w", 1))); h = int(o.get("display_h", o.get("h", 1)))
            except Exception:
                continue
            cx = max(0, min(63, x + w // 2)); cy = max(0, min(63, y + h // 2))
            if (cx, cy) in seen: continue
            seen.add((cx, cy)); out.append({"action_id": 6, "x": cx, "y": cy})
            if max_click and len(seen) >= max_click: break
    return out


def skey(s):
    try: return json.dumps(s, sort_keys=True, separators=(",", ":"), default=str)
    except Exception: return repr(s)


def bfs(eng, start, actions, frames, max_depth, max_nodes, timeout, max_click):
    q = deque([(start, [])]); seen = {skey(start)}; nodes = 0; t0 = time.time()
    while q:
        if timeout and time.time() - t0 > timeout:
            return None, nodes, "timeout"
        st, plan = q.popleft()
        if max_depth and len(plan) >= max_depth:
            continue
        for a in candidates(actions, frames, st, eng, max_click):
            if max_nodes and nodes >= max_nodes:
                return None, nodes, "max_nodes exhausted"
            nodes += 1
            try:
                nxt = eng.transition_function(copy.deepcopy(st), copy.deepcopy(a))
                r, done = reward_tuple(eng.reward_function(copy.deepcopy(st), copy.deepcopy(a), copy.deepcopy(nxt)))
            except Exception:
                continue
            np = plan + [a]
            if r > 0 or done:
                return np, nodes, "reaches reward under the model"
            k = skey(nxt)
            if k in seen:
                continue
            seen.add(k); q.append((nxt, np))
    return None, nodes, "no reward reachable within budget"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-depth", type=int, default=10)
    p.add_argument("--max-nodes", type=int, default=4000)
    p.add_argument("--timeout", type=float, default=25)
    p.add_argument("--max-click-targets", type=int, default=24)
    a = p.parse_args()
    eng = load_engine()
    start, actions, frames = cur_state()
    if hasattr(eng, "planner"):
        try:
            pl = eng.planner(copy.deepcopy(start), available_actions=actions, max_depth=a.max_depth)
            if isinstance(pl, (list, tuple)) and pl:
                print(json.dumps({"ok": True, "plan": list(pl), "source": "synth_planner", "reason": "synth-provided planner hook"}))
                return
        except Exception:
            pass
    plan, nodes, reason = bfs(eng, start, actions, frames, a.max_depth, a.max_nodes, a.timeout, a.max_click_targets)
    print(json.dumps({"ok": bool(plan), "plan": plan or [], "source": "bfs", "reason": reason, "nodes": nodes}))


if __name__ == "__main__":
    main()
'''


SYSTEM_PROMPT = load_prompt("analyzer/object_centric/system.md")


SYSTEM_PROMPT_FRAMES_ONLY = load_prompt("analyzer/frames/system.md")


_INITIAL_USER_TASK = load_prompt("analyzer/shared/initial_user_task.txt")


_RESUME_USER_TASK = load_prompt("analyzer/shared/resume_user_task.txt")


_CONTEXT_OVERFLOW_MARKERS = (
    "context length exceeded",
    "context_length_exceeded",
    "prompt is too long",
    "input is too long",
    "context overflow",
    "exceeds maximum",
    "too many tokens",
)


def _is_context_overflow(out_txt: str, err_txt: str) -> bool:
    blob = (out_txt + "\n" + err_txt).lower()
    return any(m in blob for m in _CONTEXT_OVERFLOW_MARKERS)


_SUBPROCESS_CRASH_MARKERS = (
    "memoryexhaustion",
    "memory allocation of",
    "bun has crashed",
    "out of memory",
    "cannot allocate memory",
    "std::bad_alloc",
    "javascriptcore",
    "allocationfailuremode",
)


def _is_subprocess_crash(out_txt: str, err_txt: str) -> bool:
    blob = (out_txt + "\n" + err_txt).lower()
    return any(m in blob for m in _SUBPROCESS_CRASH_MARKERS)


def _build_tools_readme(
    available_actions: list[int],
    has_world_model: bool,
    project_root: str,
) -> str:
    actions_str = ", ".join(str(a) for a in sorted(available_actions))
    wm_section = load_prompt(
        "analyzer/object_centric/tools_readme_wm.txt" if has_world_model
        else "analyzer/object_centric/tools_readme_nowm.txt"
    )
    escape_section = load_prompt("analyzer/object_centric/tools_readme_escape.txt").replace(
        "%%ACTIONS_STR_SPACES%%", actions_str.replace(", ", " ")
    )
    return (
        load_prompt("analyzer/object_centric/tools_readme.md")
        .replace("%%WM_SECTION%%", wm_section)
        .replace("%%ESCAPE_SECTION%%", escape_section)
        .replace("%%ACTIONS_STR%%", actions_str)
        .replace("%%PROJECT_ROOT%%", project_root)
    )


def _build_tools_readme_frames(
    available_actions: list[int],
    has_world_model: bool,
    project_root: str,
) -> str:
    """TOOLS.md for the frames-only regime (no type aliases, no view_sprite)."""
    actions_str = ", ".join(str(a) for a in sorted(available_actions))
    wm_section = load_prompt(
        "analyzer/frames/tools_readme_wm.txt" if has_world_model
        else "analyzer/frames/tools_readme_nowm.txt"
    )
    escape_section = load_prompt(
        "analyzer/frames/tools_readme_escape.txt"
    ).replace("%%ACTIONS_STR_SPACES%%", actions_str.replace(", ", " "))
    return (
        load_prompt("analyzer/frames/tools_readme.md")
        .replace("%%WM_SECTION%%", wm_section)
        .replace("%%ESCAPE_SECTION%%", escape_section)
        .replace("%%ACTIONS_STR%%", actions_str)
        .replace("%%PROJECT_ROOT%%", project_root)
    )


def _safe_serialise_state(state: list[dict]) -> list[dict]:
    """Convert numpy scalars and non-JSON-serialisable values to plain Python types."""
    out = []
    for o in state:
        d = {}
        for k, v in o.items():
            try:
                json.dumps(v)
                d[k] = v
            except TypeError:
                if hasattr(v, "tolist"):
                    d[k] = v.tolist()
                else:
                    d[k] = str(v)
        out.append(d)
    return out


def _clip_prompt_text(value: Any, limit: int = 1200) -> str:
    txt = sanitize_model_visible_text(value).strip()
    if not txt:
        return ""
    if len(txt) > limit:
        return txt[:limit] + "\n...[truncated]"
    return txt


def _synth_handoff_from_status(synth_status_src: Path) -> str:
    """Build an explicit prompt block from synth_status.json handoff fields."""
    try:
        status = json.loads(synth_status_src.read_text())
    except Exception:
        return ""
    lines: list[str] = []
    goal = _clip_prompt_text(status.get("goal_in_english"), 500)
    learnings = _clip_prompt_text(status.get("synth_learnings"), 1400)
    critique = _clip_prompt_text(status.get("critique_findings"), 1200)
    critique_response = _clip_prompt_text(status.get("critique_response"), 1200)
    animation = _clip_prompt_text(status.get("animation_findings"), 1200)
    shared_updates = _clip_prompt_text(
        status.get("shared_model_updates"), 1200
    )
    planner = status.get("planner") or {}
    gate = status.get("synthesis_gate") or {}

    if goal:
        lines.append("## Synth goal hypothesis\n" + goal)
    if learnings:
        lines.append("## Synth learnings for exploration\n" + learnings)
    if critique:
        lines.append("## Independent critique of synth model\n" + critique)
    if critique_response:
        lines.append("## Synth response to critique\n" + critique_response)
    if animation:
        lines.append("## Animation-review findings\n" + animation)
    if shared_updates:
        lines.append(
            "## Shared world-model file updates from synth\n"
            + shared_updates
        )
    if gate:
        gate_lines = [
            f"active={bool(gate.get('active'))}",
            f"ready={bool(gate.get('ready'))}",
            f"error_count={gate.get('error_count', 0)}",
            f"action_plan_count={gate.get('action_plan_count', 0)}",
            f"moves_since_first_error={gate.get('moves_since_first_error', 0)}",
            f"moves_left={gate.get('moves_until_auto_synthesis', 0)}",
            f"errors_left={gate.get('errors_until_auto_synthesis', 0)}",
            f"action_plans_left="
            f"{gate.get('action_plans_until_auto_synthesis', 0)}",
            "ready_reasons="
            + ", ".join(str(r) for r in (gate.get("ready_reasons") or [])),
        ]
        lines.append(
            "## Delayed CEGIS repair gate\n"
            + "\n".join(gate_lines)
        )
    if planner:
        lines.append(
            "## Engine C3 planner status\n"
            + json.dumps(planner, indent=2, default=str)[:1000]
        )
    if not lines:
        return ""
    return (
        "\n\nSYNTHESIZER HANDOFF (read and use explicitly)\n"
        "These are hypotheses and review notes from the synthesis side. Use them "
        "as priors for exploration, but the real environment remains ground "
        "truth. If an item is contradicted by a newly executed transition or "
        "reward observation, write `synth_control.json` with a focused "
        "correction; `force_now` is honored only after a concrete model "
        "mismatch, not after speculation.\n\n"
        + "\n\n".join(lines)
        + "\n"
    )


def _setup_workspace(
    workspace_dir: Path,
    *,
    run_log_src: Path,
    epistemic_matrix_src: Path,
    synth_status_src: Path,
    replay_buffer: list[dict],
    current_state: list[dict],
    state_desc: str,
    available_actions: list[int],
    moves_remaining: int | None,
    step: int,
    level: int,
    synthesis_dir: Path | None,
    notes_persistent: Path,
    project_root: str,
    frames_only: bool = False,
    current_frame: list[list[int]] | None = None,
    game_over: bool = False,
) -> bool:
    """Stage artifacts into workspace_dir and return has_wm.

    Under frames_only, current_state.json carries a raw frame field. If the
    synth has exported a spriteless object abstraction, epistemic/ontology
    diagnostics are still staged for the analyzer.
    """
    workspace_dir.mkdir(parents=True, exist_ok=True)

    def _relative_symlink(src: Path, dst: Path) -> None:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if src.exists():
            target = os.path.relpath(src.resolve(), dst.parent.resolve())
            dst.symlink_to(target)

    def _writable_copy(src: Path, dst: Path) -> None:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if src.exists():
            shutil.copy2(src, dst)

    rl = workspace_dir / "run_log.txt"
    _relative_symlink(run_log_src, rl)

    em = workspace_dir / "epistemic_matrix.json"
    _relative_symlink(epistemic_matrix_src, em)

    for diag_name in ("ontology_error.json", "spriteless_object_abstraction.json"):
        src = epistemic_matrix_src.parent / diag_name
        dst = workspace_dir / diag_name
        _relative_symlink(src, dst)

    ss = workspace_dir / "synth_status.json"
    _relative_symlink(synth_status_src, ss)

    output_dir = synth_status_src.parent
    for name in (
        "synth_learnings.md",
        "last_critique.md",
        "critique_response.md",
        "animation_analysis.md",
    ):
        _relative_symlink(output_dir / name, workspace_dir / name)
    for name in ("shared_model_updates.md", "world_model.md"):
        _writable_copy(output_dir / name, workspace_dir / name)
    for pattern in ("level_*_reasoning_log.md", "level_*_report.md"):
        for src in sorted(output_dir.glob(pattern)):
            _writable_copy(src, workspace_dir / src.name)
    _relative_symlink(
        output_dir / "frames" / "animation_events.jsonl",
        workspace_dir / "animation_events.jsonl",
    )

    with open(workspace_dir / "replay_buffer.pkl", "wb") as f:
        pickle.dump(replay_buffer, f)

    if frames_only:
        cur = {
            "frame": current_frame if current_frame is not None else None,
            "available_actions": list(available_actions),
            "moves_remaining": moves_remaining,
            "game_over": bool(game_over),
            "step": step,
            "level": level,
        }
    else:
        cur = {
            "state": _safe_serialise_state(current_state),
            "describe": state_desc,
            "available_actions": list(available_actions),
            "moves_remaining": moves_remaining,
            "game_over": bool(game_over),
            "step": step,
            "level": level,
        }
    with open(workspace_dir / "current_state.json", "w") as f:
        json.dump(cur, f, indent=2, default=str)

    has_wm = False
    if (
        synthesis_dir is not None
        and synthesis_dir.exists()
    ):
        ge = synthesis_dir / "game_engine.py"
        if ge.exists():
            shutil.copy2(ge, workspace_dir / "game_engine.py")
            has_wm = True

    nm = workspace_dir / "notes.md"
    if notes_persistent.exists():
        nm.write_text(sanitize_model_visible_text(notes_persistent.read_text()))
    else:
        nm.write_text("# Consumer scratchpad\n\n")

    if frames_only:
        (workspace_dir / "TOOLS.md").write_text(_build_tools_readme_frames(
            available_actions=available_actions,
            has_world_model=has_wm,
            project_root=project_root,
        ))
    else:
        (workspace_dir / "TOOLS.md").write_text(_build_tools_readme(
            available_actions=available_actions,
            has_world_model=has_wm,
            project_root=project_root,
        ))

    tools_dir = workspace_dir / "tools"
    tools_dir.mkdir(exist_ok=True)
    (tools_dir / "escape_sequence.py").write_text(_ESCAPE_TOOL_SCRIPT)
    if not frames_only:
        (tools_dir / "view_sprite.py").write_text(_VIEW_SPRITE_SCRIPT)
    if has_wm:
        (tools_dir / "plan.py").write_text(_PLANNER_TOOL_SCRIPT)

    na = workspace_dir / "next_actions.json"
    if na.exists():
        na.unlink()

    return has_wm


_RATE_LIMIT_MARKERS = (
    "you've hit your limit",
    "you have hit your limit",
    "rate limit",
    "rate-limited",
    "ratelimit",
    "rate_limit",
    "429",
    "usage limit reached",
    "quota exceeded",
    "authentication_failed",
    "please login again",
    "does not have access to claude",
    "do not have access to claude",
)


def _json_event_has_rate_limit_error(line: str) -> bool:
    try:
        ev = json.loads(line)
    except Exception:
        return False
    if not isinstance(ev, dict):
        return False
    event_blob = json.dumps(ev, sort_keys=True).lower()
    if not any(m in event_blob for m in _RATE_LIMIT_MARKERS):
        return False
    if ev.get("type") == "error":
        return True
    if ev.get("error"):
        return True
    if ev.get("is_error") is True:
        return True
    if ev.get("type") == "result" and ev.get("is_error") is True:
        return True
    turn = ev.get("turn")
    if isinstance(turn, dict) and turn.get("is_error") is True:
        return True
    return False


def _is_rate_limited(out_txt: str, err_txt: str, rc: int,
                      duration_s: float) -> bool:
    blob = (out_txt + "\n" + err_txt).lower()
    if rc == 0:
        for line in (out_txt + "\n" + err_txt).splitlines():
            if line.strip().startswith("{") and _json_event_has_rate_limit_error(line):
                return True
        return False
    if any(m in blob for m in _RATE_LIMIT_MARKERS):
        return True
    if duration_s < 30 and rc != 0 and ("limit" in blob or "reset" in blob):
        return True
    return False


def _read_plan(workspace_dir: Path) -> tuple[list, str]:
    """Read next_actions.json and return (plan, reasoning_or_error).

    Normalises plan items: a bare int or ACTION-string becomes an int, and ACTION6 dicts become {"action_id":6,"x":int,"y":int}. Returns ([], error_str) on any failure.
    """
    p = workspace_dir / "next_actions.json"
    if not p.exists():
        return [], "no next_actions.json written"
    try:
        data = json.loads(p.read_text())
    except Exception as e:
        return [], f"bad JSON: {type(e).__name__}: {e}"
    plan_raw = data.get("plan")
    if not isinstance(plan_raw, list):
        return [], f"missing/invalid 'plan' field: {plan_raw!r}"

    out: list = []
    for v in plan_raw:
        if isinstance(v, int):
            out.append(v)
            continue
        if isinstance(v, str):
            s = v.strip()
            if s.upper() == "RESET":
                out.append(0)
                continue
            if s.upper().startswith("ACTION"):
                rest = s[6:]
                import re as _re
                m = _re.match(r"^6\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*$", rest)
                if m:
                    out.append({
                        "action_id": 6,
                        "x": int(m.group(1)),
                        "y": int(m.group(2)),
                    })
                    continue
                try:
                    out.append(int(rest))
                    continue
                except ValueError:
                    return [], f"unparseable action string: {v!r}"
            try:
                out.append(int(s))
                continue
            except ValueError:
                return [], f"unparseable action string: {v!r}"
        if isinstance(v, dict):
            if "action_id" in v:
                aid = int(v["action_id"])
            elif "action" in v:
                a = str(v["action"]).strip().upper()
                if a == "RESET":
                    aid = 0
                elif a.startswith("ACTION"):
                    try:
                        aid = int(a[6:])
                    except ValueError:
                        return [], f"unparseable action name: {v!r}"
                else:
                    return [], f"unparseable action name: {v!r}"
            else:
                return [], f"plan dict missing 'action'/'action_id': {v!r}"
            if aid == 6:
                if "x" not in v or "y" not in v:
                    return [], f"ACTION6 missing x/y: {v!r}"
                out.append({
                    "action_id": 6,
                    "x": int(v["x"]),
                    "y": int(v["y"]),
                })
            else:
                out.append(aid)
            continue
        return [], f"unrecognised plan item form: {v!r}"

    if not out:
        return [], "empty plan"
    return out, str(data.get("reasoning", ""))[:200]


def _validate_available_plan(
    plan: list,
    available_actions: list[int],
) -> tuple[list, str | None]:
    valid = {int(a) for a in available_actions}

    def _id(a):
        if isinstance(a, dict):
            return a.get("action_id")
        return a

    invalid = [a for a in plan if _id(a) not in valid]
    if invalid:
        return [], (
            f"plan used unavailable action(s) {invalid!r}; "
            f"available_actions={sorted(valid)}"
        )
    return plan, None


class AgenticConsumer:
    """Claude Code subprocess action selector. Writes next_actions.json and reads it back as a plan."""

    def __init__(
        self,
        model: str,
        max_turns: int = 30,
        timeout_s: int = 0,
        log_dir: Path | None = None,
        rate_limit_retry_s: int = 60,
        rate_limit_max_wait_s: int = 12 * 3600,
        sandbox: bool = True,
        engine_output_dir: Path | None = None,
        effort: str = "max",
        backend: str = "claude",
        codex_home: str | None = None,
        codex_image: str = "codex-agent",
        codex_network: str = "codex-filtered",
        codex_gateway: str | None = None,
        claude_isolation: str = "bwrap",
        claude_image: str = "claude-agent",
        claude_network: str = "claude-filtered",
        claude_gateway: str | None = None,
        claude_docker_memory: str = "12g",
        claude_docker_cpus: str = "2.0",
        claude_docker_pids_limit: str = "512",
    ):
        self.cumulative_usage: dict[str, int | float] = {
            "input_tokens": 0, "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "estimated_cost_usd": 0.0, "calls": 0,
        }
        self._needs_fresh_session: bool = False
        self._continue_chain_len: int = 0
        self.backend = backend
        self.codex_home = codex_home
        self.codex_image = codex_image
        self.codex_network = codex_network
        self.codex_gateway = codex_gateway
        self._claude_session_id: str | None = None
        self._codex_session_id: str | None = None
        self.model = model
        self.max_turns = max_turns
        self.timeout_s = timeout_s
        self.effort = effort
        self.log_dir = log_dir
        self.rate_limit_retry_s = rate_limit_retry_s
        self.rate_limit_max_wait_s = rate_limit_max_wait_s
        self.sandbox = sandbox
        self.claude_isolation = claude_isolation
        self.claude_image = claude_image
        self.claude_network = claude_network
        self.claude_gateway = claude_gateway
        self.claude_docker_memory = claude_docker_memory
        self.claude_docker_cpus = claude_docker_cpus
        self.claude_docker_pids_limit = claude_docker_pids_limit
        self._docker_container_name: str | None = None
        self.engine_output_dir = engine_output_dir
        self.call_count = 0

    def _wrap_claude_cmd(
        self, cmd: list[str], workspace_dir: Path
    ) -> tuple[list[str], dict[str, Any]]:
        """Apply the configured claude isolation to a `claude ...` argv.

        Returns (possibly-wrapped cmd, Popen kwargs). For "docker" it sets
        self._docker_container_name so a timeout can `docker rm -f` the run.
        Resource bounding for docker is the container's (--memory/--cpus/--pids).
        The host rlimit preexec is only meaningful for the bwrap and none paths.
        """
        from .sandbox import claude_popen_kwargs
        self._docker_container_name = None
        if not self.sandbox:
            return cmd, claude_popen_kwargs()
        engine_out = self.engine_output_dir or workspace_dir.parent
        if self.claude_isolation == "docker":
            from .sandbox import wrap_for_docker
            name = (f"arc-claude-{os.getpid()}-{self.call_count:04d}-"
                    f"{int(time.time())}")
            self._docker_container_name = name
            cmd = wrap_for_docker(
                cmd, workspace_dir=workspace_dir, engine_output_dir=engine_out,
                image=self.claude_image, network=self.claude_network,
                gateway=self.claude_gateway, container_name=name,
                memory=self.claude_docker_memory, cpus=self.claude_docker_cpus,
                pids_limit=self.claude_docker_pids_limit,
            )
            return cmd, {"start_new_session": True}
        from .sandbox import wrap_for_sandbox
        cmd = wrap_for_sandbox(
            cmd, workspace_dir=workspace_dir, engine_output_dir=engine_out,
        )
        return cmd, claude_popen_kwargs()

    def _cleanup_docker_container(self) -> None:
        """Best-effort force-remove the current call's docker container."""
        name = self._docker_container_name
        if not name:
            return
        try:
            subprocess.run(
                ["docker", "rm", "-f", name],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=15, check=False,
            )
        except Exception:
            pass

    def _run_compact_session(self, workspace_dir: Path) -> None:
        """Summarise the analyzer's --continue session in place via /compact.

        Keeps continuity and a coherent prompt cache while shrinking context,
        unlike a fresh session (which discards the conversation). Best-effort:
        a failure just leaves the session unchanged. Claude path only.
        """
        if self.backend == "codex":
            return
        claude = shutil.which("claude")
        if not claude:
            return
        cmd = [claude, "-p", "/compact", "--continue",
               "--model", self.model, "--output-format", "json"]
        cmd, popen_kwargs = self._wrap_claude_cmd(cmd, workspace_dir)
        from .sandbox import (
            terminate_process_group,
            wait_with_resource_monitor,
        )
        t0 = time.time()
        try:
            with open(os.devnull, "w") as devnull:
                proc = subprocess.Popen(
                    cmd, stdout=devnull, stderr=devnull,
                    cwd=str(workspace_dir), **popen_kwargs,
                )
                try:
                    wait_with_resource_monitor(
                        proc,
                        timeout_s=(self.timeout_s if self.timeout_s and self.timeout_s > 0 else None),
                    )
                except subprocess.TimeoutExpired:
                    terminate_process_group(proc)
                    self._cleanup_docker_container()
            print(f"  [agentic-consumer] /compact done in {time.time() - t0:.1f}s",
                  flush=True)
        except Exception as exc:
            print(f"  [agentic-consumer] /compact failed: "
                  f"{type(exc).__name__}: {exc}", flush=True)

    def choose_actions(
        self,
        *,
        workspace_dir: Path,
        run_log_src: Path,
        epistemic_matrix_src: Path,
        synth_status_src: Path,
        replay_buffer: list[dict],
        current_state: list[dict],
        state_desc: str,
        available_actions: list[int],
        synthesis_dir: Path | None,
        notes_persistent: Path,
        project_root: str,
        moves_remaining: int | None = None,
        step: int = 0,
        level: int = 0,
        ascii_grid: str = "",
        score: int = 0,
        recent_history: list[dict] | None = None,
        last_plan_hint: str | None = "",
        extra_user_prompt: str = "",
        frames_only: bool = False,
        current_frame: list[list[int]] | None = None,
        game_over: bool = False,
        divergence_images: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Run one consumer call and return a result dict with plan, reasoning, duration_s, returncode, and reason."""
        self.call_count += 1
        _setup_workspace(
            workspace_dir=workspace_dir,
            run_log_src=run_log_src,
            epistemic_matrix_src=epistemic_matrix_src,
            synth_status_src=synth_status_src,
            replay_buffer=replay_buffer,
            current_state=current_state,
            state_desc=state_desc,
            available_actions=available_actions,
            moves_remaining=moves_remaining,
            step=step,
            level=level,
            synthesis_dir=synthesis_dir,
            notes_persistent=notes_persistent,
            project_root=project_root,
            frames_only=frames_only,
            current_frame=current_frame,
            game_over=game_over,
        )

        claude = shutil.which("claude")
        if self.backend != "codex" and not claude:
            return {"plan": [], "duration_s": 0.0, "returncode": -1,
                    "reasoning": "", "reason": "claude CLI not found"}

        actions_csv = ", ".join(str(a) for a in sorted(available_actions))
        user_task = (
            _INITIAL_USER_TASK if self.call_count == 1
            else _RESUME_USER_TASK
        )

        log_path = (workspace_dir.resolve().parent / "run_log.txt")
        system_prompt = (
            SYSTEM_PROMPT_FRAMES_ONLY if frames_only else SYSTEM_PROMPT
        )
        synth_handoff = _synth_handoff_from_status(synth_status_src)
        user_prompt = f"""{system_prompt}

WORKSPACE: {workspace_dir.resolve()}
RUN LOG: {log_path}

CALL #{self.call_count}. STEP: {step}  LEVEL: {level + 1}  SCORE: {score}  AVAILABLE ACTIONS: {actions_csv}
LEGAL ACTION CONTRACT: you may output and hypothesize only action ids listed in AVAILABLE ACTIONS. Any absent id does not exist for this game/state; do not propose, probe, or mention it as an available interaction.

{user_task}
{synth_handoff}

{extra_user_prompt}"""
        user_prompt = sanitize_model_visible_text(user_prompt)

        if self.backend == "codex":
            return self._choose_actions_codex(
                workspace_dir=workspace_dir,
                user_prompt=user_prompt,
                divergence_images=divergence_images,
                available_actions=available_actions,
                step=step,
                level=level,
            )

        allowed_tools = "Bash(python3:*),Bash(python:*),Read,Grep,Task"

        inject = [
            im for im in (divergence_images or [])
            if im.get("path") and Path(im["path"]).exists()
        ][:8]
        stdin_msg = None
        if inject:
            labels = "; ".join(
                f"{im['role']} of step {im['step']}" for im in inject
            )
            user_prompt = user_prompt + load_prompt(
                "analyzer/shared/divergence_preamble.txt"
            ).replace("%%LABELS%%", labels)
            user_prompt = sanitize_model_visible_text(user_prompt)
            content: list[dict] = [{"type": "text", "text": user_prompt}]
            for im in inject:
                try:
                    b64 = base64.b64encode(
                        Path(im["path"]).read_bytes()
                    ).decode("ascii")
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64,
                        },
                    })
                except Exception:
                    pass
            stdin_msg = json.dumps(
                {"type": "user",
                 "message": {"role": "user", "content": content}}
            )

        base_flags = [
            "--model", self.model,
            "--effort", self.effort,
            "--max-turns", str(self.max_turns),
            "--output-format", "stream-json", "--verbose",
            "--allowedTools", allowed_tools,
            "--permission-mode", "bypassPermissions",
        ]
        if stdin_msg is not None:
            cmd = [claude, "-p", "--input-format", "stream-json", *base_flags]
        else:
            cmd = [claude, "-p", user_prompt, *base_flags]
        max_chain = int(os.environ.get("ARC3_ANALYZER_MAX_CONTINUE", "0") or 0)
        if max_chain > 0 and self._continue_chain_len >= max_chain:
            self._needs_fresh_session = True
            print(f"  [agentic-consumer] --continue chain hit "
                  f"{self._continue_chain_len}; starting a fresh session "
                  f"(rebuilds from notes.md/world_model.md)", flush=True)
        use_continue = self.call_count > 1 and not self._needs_fresh_session
        if use_continue and self._claude_session_id:
            cmd.extend(["--resume", self._claude_session_id])
            self._continue_chain_len += 1
        elif use_continue:
            cmd.append("--continue")
            self._continue_chain_len += 1
        else:
            self._continue_chain_len = 0
        if self._needs_fresh_session:
            self._needs_fresh_session = False

        from .sandbox import (
            terminate_process_group,
            wait_with_resource_monitor,
        )
        cmd, popen_kwargs = self._wrap_claude_cmd(cmd, workspace_dir)

        stdout_path = workspace_dir / "consumer_stdout.txt"
        stderr_path = workspace_dir / "consumer_stderr.txt"
        prompt_path = workspace_dir / "consumer_prompt.txt"
        try:
            prompt_path.write_text(sanitize_model_visible_text(user_prompt))
        except Exception:
            pass

        timeout_val = self.timeout_s if self.timeout_s and self.timeout_s > 0 else None
        t_overall = time.time()
        waited_s = 0.0
        attempts = 0
        timed_out = False
        rc = 0
        rate_limited_giveup = False
        while True:
            attempts += 1
            t0 = time.time()
            try:
                with open(stdout_path, "w") as out_f, open(stderr_path, "w") as err_f:
                    proc = subprocess.Popen(
                        cmd,
                        stdin=(subprocess.PIPE if stdin_msg is not None else None),
                        stdout=out_f, stderr=err_f, text=True,
                        cwd=str(workspace_dir),
                        **popen_kwargs,
                    )
                    if stdin_msg is not None:
                        try:
                            proc.stdin.write(stdin_msg + "\n")
                            proc.stdin.close()
                        except Exception:
                            pass
                    try:
                        rc = wait_with_resource_monitor(
                            proc,
                            timeout_s=timeout_val,
                        )
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        terminate_process_group(proc)
                        self._cleanup_docker_container()
                        rc = -1
            except Exception as e:
                rc = -1
                self._cleanup_docker_container()
                return {"plan": [], "duration_s": round(time.time() - t_overall, 1),
                        "returncode": rc, "reasoning": "",
                        "reason": f"subprocess: {type(e).__name__}: {e}"}

            duration_s = time.time() - t0

            out_txt = ""
            err_txt = ""
            try:
                out_txt = stdout_path.read_text()
            except Exception:
                pass
            try:
                err_txt = stderr_path.read_text()
            except Exception:
                pass
            if _is_rate_limited(out_txt, err_txt, rc, duration_s):
                if waited_s >= self.rate_limit_max_wait_s:
                    rate_limited_giveup = True
                    break
                print(f"  [agentic-consumer rate-limited] attempt={attempts} "
                      f"rc={rc} dur={duration_s:.1f}s; retrying in "
                      f"{self.rate_limit_retry_s}s "
                      f"(cumulative wait={int(waited_s)}s)",
                      flush=True)
                time.sleep(self.rate_limit_retry_s)
                waited_s += self.rate_limit_retry_s
                continue
            break

        duration = time.time() - t_overall

        nm = workspace_dir / "notes.md"
        if nm.exists():
            try:
                notes_persistent.write_text(
                    sanitize_model_visible_text(nm.read_text())
                )
            except Exception:
                pass

        if self.log_dir is not None:
            try:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                tag = f"call_{self.call_count:04d}"
                if (workspace_dir / "consumer_prompt.txt").exists():
                    shutil.copy2(
                        workspace_dir / "consumer_prompt.txt",
                        self.log_dir / f"{tag}.prompt.txt",
                    )
                if stdout_path.exists():
                    shutil.copy2(stdout_path,
                                 self.log_dir / f"{tag}.chat.jsonl")
                if stderr_path.exists():
                    shutil.copy2(stderr_path,
                                 self.log_dir / f"{tag}.stderr.txt")
                if (workspace_dir / "next_actions.json").exists():
                    shutil.copy2(
                        workspace_dir / "next_actions.json",
                        self.log_dir / f"{tag}.next_actions.json",
                    )
            except Exception:
                pass

        try:
            out_blob = stdout_path.read_text() if stdout_path.exists() else ""
            err_blob = stderr_path.read_text() if stderr_path.exists() else ""
        except Exception:
            out_blob = err_blob = ""
        crashed = _is_subprocess_crash(out_blob, err_blob)
        if _is_context_overflow(out_blob, err_blob) or crashed:
            self._needs_fresh_session = True
        if crashed:
            print(
                f"  [agentic-consumer subprocess crash] attempt={attempts} "
                f"rc={rc} dur={duration_s:.1f}s; dropping --continue session "
                f"so the retry starts fresh",
                flush=True,
            )

        max_ctx_tokens = 0
        try:
            for line in (out_blob.splitlines()):
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                if ev.get("session_id"):
                    self._claude_session_id = str(ev["session_id"])
                if ev.get("type") == "assistant":
                    mu = (ev.get("message") or {}).get("usage")
                    if isinstance(mu, dict):
                        ctx = (int(mu.get("input_tokens", 0) or 0)
                               + int(mu.get("cache_read_input_tokens", 0) or 0)
                               + int(mu.get("cache_creation_input_tokens", 0) or 0))
                        if ctx > max_ctx_tokens:
                            max_ctx_tokens = ctx
                if (ev.get("type") == "result"
                        and isinstance(ev.get("usage"), dict)):
                    u = ev["usage"]
                    for k in ("input_tokens", "output_tokens",
                              "cache_creation_input_tokens",
                              "cache_read_input_tokens"):
                        v = u.get(k) or 0
                        try:
                            self.cumulative_usage[k] += int(v)
                        except Exception:
                            pass
                    cost = ev.get("total_cost_usd") or 0
                    try:
                        self.cumulative_usage["estimated_cost_usd"] += float(cost)
                    except Exception:
                        pass
                    self.cumulative_usage["calls"] += 1
                    break
        except Exception:
            pass

        self._last_context_tokens = max_ctx_tokens
        max_ctx = int(os.environ.get("ARC3_ANALYZER_MAX_CONTEXT_TOKENS", "500000") or 0)
        if (max_ctx > 0 and max_ctx_tokens >= max_ctx
                and not self._needs_fresh_session):
            print(f"  [agentic-consumer] analyzer context {max_ctx_tokens:,} >= "
                  f"{max_ctx:,} tokens; running /compact", flush=True)
            self._run_compact_session(workspace_dir)
        if self.log_dir is not None:
            try:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                (self.log_dir / "token_usage.json").write_text(
                    json.dumps(self.cumulative_usage, indent=2)
                )
            except Exception:
                pass

        plan, reasoning = _read_plan(workspace_dir)
        plan, invalid_reason = _validate_available_plan(plan, available_actions)
        if invalid_reason:
            reasoning = invalid_reason
        return {
            "plan": plan,
            "reasoning": reasoning,
            "duration_s": round(duration, 1),
            "returncode": rc,
            "timed_out": timed_out,
            "rate_limited": rate_limited_giveup,
            "rate_limit_wait_s": int(waited_s),
            "attempts": attempts,
            "reason": (
                "subprocess_crash" if crashed and not plan
                else "rate_limited_giveup" if rate_limited_giveup
                else "timed_out" if timed_out
                else "ok" if plan
                else (reasoning or "no plan")
            ),
        }

    def _choose_actions_codex(
        self,
        *,
        workspace_dir: Path,
        user_prompt: str,
        divergence_images: list[dict] | None,
        available_actions: list[int],
        step: int,
        level: int,
    ) -> dict[str, Any]:
        """Codex backend of choose_actions: run one turn in the locked-down
        codex-agent container. Same file contract -- the agent writes
        next_actions.json, which _read_plan parses back."""
        from . import codex_backend as cx

        run_dir = workspace_dir.resolve().parent
        inject = [
            im for im in (divergence_images or [])
            if im.get("path") and Path(im["path"]).exists()
        ][:8]
        container_images: list[str] = []
        if inject:
            labels = "; ".join(
                f"{im['role']} of step {im['step']}" for im in inject
            )
            user_prompt = user_prompt + load_prompt(
                "analyzer/shared/divergence_preamble.txt"
            ).replace("%%LABELS%%", labels)
            user_prompt = sanitize_model_visible_text(user_prompt)
            for im in inject:
                try:
                    rel = Path(im["path"]).resolve().relative_to(run_dir)
                    container_images.append(f"/run/{rel}")
                except Exception:
                    pass

        try:
            (workspace_dir / "consumer_prompt.txt").write_text(
                sanitize_model_visible_text(user_prompt)
            )
        except Exception:
            pass

        use_resume = (
            self.call_count > 1
            and not self._needs_fresh_session
            and self._codex_session_id is not None
        )
        if self._needs_fresh_session:
            self._needs_fresh_session = False

        ws_rel = workspace_dir.resolve().relative_to(run_dir)
        timeout_val = self.timeout_s if self.timeout_s and self.timeout_s > 0 else None
        stdout_path = workspace_dir / "consumer_stdout.txt"
        stderr_path = workspace_dir / "consumer_stderr.txt"
        for stale_retry_log in (
            workspace_dir / "consumer_codex_retry_tracker.jsonl",
            workspace_dir / "consumer_quota_wait.txt",
        ):
            try:
                stale_retry_log.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass
        codex_home = cx.codex_home_path(self.codex_home)
        attempts = 0
        quota_wait_s = 0
        while True:
            attempts += 1
            session_for_attempt = (
                self._codex_session_id if use_resume else None
            )
            attempt_used_resume = session_for_attempt is not None
            res = cx.run_codex_turn(
                prompt=user_prompt,
                workspace_dir=workspace_dir,
                run_dir=run_dir,
                container_cd=f"/run/{ws_rel}",
                model=self.model,
                effort=self.effort,
                codex_home=codex_home,
                images=container_images or None,
                session_id=session_for_attempt,
                timeout_s=timeout_val,
                image_name=self.codex_image,
                network=self.codex_network,
                gateway=self.codex_gateway,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            )
            retryable_infra = bool(res.get("retryable_infra_failure"))
            quota_limited = bool(res.get("quota_limited"))
            if retryable_infra and (workspace_dir / "next_actions.json").exists():
                dirty_plan, _dirty_reason = _read_plan(workspace_dir)
                dirty_plan, _invalid_reason = _validate_available_plan(
                    dirty_plan, available_actions,
                )
                if dirty_plan:
                    res["retryable_infra_failure"] = False
                    res["reason"] = "ok_dirty_stream_with_next_actions"
                    retryable_infra = False
            if not (quota_limited or retryable_infra):
                break
            if retryable_infra:
                try:
                    (workspace_dir / "next_actions.json").unlink()
                except FileNotFoundError:
                    pass
                except Exception:
                    pass
            wait_s = max(1, int(self.rate_limit_retry_s or 60))
            quota_wait_s += wait_s
            event = {
                "kind": "quota_exhausted" if quota_limited else "codex_infra_retry",
                "call": self.call_count,
                "step": step,
                "level": level + 1,
                "attempt": attempts,
                "reason": res.get("reason"),
                "returncode": res.get("returncode"),
                "duration_s": res.get("duration_s"),
                "quota_limited": quota_limited,
                "retryable_infra_failure": retryable_infra,
                "remote_compact_failed": bool(res.get("remote_compact_failed")),
                "using_resume": attempt_used_resume,
                "wait_s": wait_s,
                "total_wait_s": quota_wait_s,
                "ts": time.time(),
            }
            try:
                with open(
                    workspace_dir / "consumer_codex_retry_tracker.jsonl", "a"
                ) as f:
                    f.write(json.dumps(event, sort_keys=True) + "\n")
            except Exception:
                pass
            if self.log_dir is not None:
                try:
                    self.log_dir.mkdir(parents=True, exist_ok=True)
                    with open(self.log_dir / "codex_retry_tracker.jsonl", "a") as f:
                        f.write(json.dumps(event, sort_keys=True) + "\n")
                except Exception:
                    pass
                if quota_limited:
                    try:
                        with open(self.log_dir / "quota_exhausted.jsonl", "a") as f:
                            f.write(json.dumps(event, sort_keys=True) + "\n")
                    except Exception:
                        pass
            try:
                with open(workspace_dir / "consumer_quota_wait.txt", "a") as f:
                    f.write(json.dumps(event, sort_keys=True) + "\n")
            except Exception:
                pass
            time.sleep(wait_s)
        if res.get("session_id"):
            self._codex_session_id = res["session_id"]
        u = res.get("usage") or {}
        for k in ("input_tokens", "output_tokens"):
            try:
                self.cumulative_usage[k] += int(u.get(k, 0) or 0)
            except Exception:
                pass
        self.cumulative_usage["calls"] += 1

        nm = workspace_dir / "notes.md"
        notes_persistent = run_dir / "consumer_notes.md"
        if nm.exists():
            try:
                notes_persistent.write_text(
                    sanitize_model_visible_text(nm.read_text())
                )
            except Exception:
                pass
        if self.log_dir is not None:
            try:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                tag = f"call_{self.call_count:04d}"
                for src, dst in (
                    (workspace_dir / "consumer_prompt.txt", f"{tag}.prompt.txt"),
                    (stdout_path, f"{tag}.chat.jsonl"),
                    (stderr_path, f"{tag}.stderr.txt"),
                    (workspace_dir / "next_actions.json", f"{tag}.next_actions.json"),
                ):
                    if src.exists():
                        shutil.copy2(src, self.log_dir / dst)
                (self.log_dir / "token_usage.json").write_text(
                    json.dumps(self.cumulative_usage, indent=2)
                )
            except Exception:
                pass

        plan, reasoning = _read_plan(workspace_dir)
        plan, invalid_reason = _validate_available_plan(plan, available_actions)
        if invalid_reason:
            reasoning = invalid_reason
        timed_out = res.get("reason") == "timed_out"
        quota_limited = bool(res.get("quota_limited"))
        return {
            "plan": plan,
            "reasoning": reasoning,
            "duration_s": res.get("duration_s", 0.0),
            "returncode": res.get("returncode", -1),
            "timed_out": timed_out,
            "rate_limited": quota_limited,
            "quota_limited": quota_limited,
            "rate_limit_wait_s": quota_wait_s,
            "attempts": attempts,
            "reason": (
                "timed_out" if timed_out
                else "quota_limited" if quota_limited
                else "ok" if plan
                else (reasoning or res.get("reason") or "no plan")
            ),
        }
