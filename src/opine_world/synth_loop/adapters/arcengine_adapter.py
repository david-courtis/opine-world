"""ARC-AGI-3 domain adapter for arcengine games. Extracts typed objects from sprite state."""
from __future__ import annotations

import importlib.util as _importlib_util
import json
import pickle
from pathlib import Path
from typing import Any

from ..domain_adapter import DomainAdapter


def _load_prompt_loader():
    """Import the ``prompts.py`` loader (in the parent synth_loop package) by
    file path. A relative import is unsafe because some entrypoints load this
    module via importlib under stub parent packages without ``__path__``.
    """
    path = Path(__file__).resolve().parent.parent / "prompts.py"
    spec = _importlib_util.spec_from_file_location(
        "_synth_loop_prompts_loader", path
    )
    mod = _importlib_util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_prompt


load_prompt = _load_prompt_loader()


class ArcEngineEnv:
    """Wraps an arcengine ARCBaseGame to provide the EnvironmentInterface."""

    def __init__(self, game, action_names: dict[int, str] | None = None):
        self.game = game
        self.action_names = action_names or {
            1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT", 5: "ACTION5",
        }
        self._prev_level = game.level_index
        self._last_before_frame = None
        self._last_during_frames: list = []
        self._last_intermediate_states: list[list[dict]] = []
        self._last_after_frame = None

    def reset(self) -> list[dict]:
        self.game.full_reset()
        self._prev_level = self.game.level_index
        return self.extract_state()

    def step(self, action) -> tuple[list[dict], float, bool]:
        """Execute one action. Accepts int, dict with action_id/x/y, or "RESET"."""
        import numpy as np
        from arcengine import ActionInput, GameAction

        action_map = {
            1: GameAction.ACTION1, 2: GameAction.ACTION2,
            3: GameAction.ACTION3, 4: GameAction.ACTION4,
            5: GameAction.ACTION5, 6: GameAction.ACTION6,
            7: GameAction.ACTION7,
        }

        if isinstance(action, str) and action.upper() == "RESET":
            action_id = 0
            click_x = None
            click_y = None
        elif isinstance(action, dict):
            action_id = int(action["action_id"])
            click_x = int(action["x"]) if "x" in action else None
            click_y = int(action["y"]) if "y" in action else None
        else:
            action_id = int(action)
            click_x = None
            click_y = None

        if action_id == 0:
            try:
                self._last_before_frame = None
                self.game.handle_reset()
                self._last_after_frame = None
                self._last_during_frames = []
                return self.extract_state(), 0.0, False
            except Exception:
                return self.extract_state(), 0.0, False

        if str(getattr(self.game, "_state", "")) in (
            "GameState.GAME_OVER", "GameState.WIN"
        ):
            self._last_before_frame = None
            self._last_after_frame = None
            self._last_during_frames = []
            self._last_intermediate_states = []
            return self.extract_state(), 0.0, False

        level_before = self.game.level_index

        try:
            self._last_before_frame = self._render_canonical_frame()
        except Exception:
            self._last_before_frame = None

        if action_id == 6 and click_x is not None and click_y is not None:
            a = ActionInput(
                id=action_map[action_id],
                data={"x": int(click_x), "y": int(click_y)},
            )
        else:
            a = ActionInput(id=action_map[action_id])
        ticks_frames: list = []
        ticks_states: list[list[dict]] = []
        try:
            from arcengine import base_game as _base_game
            max_frames = int(getattr(_base_game, "MAX_FRAME_PER_ACTION", 1000))
            self.game._set_action(a)
            safety = 0
            while not self.game.is_action_complete():
                if safety > max_frames:
                    break
                safety += 1
                if getattr(self.game, "_next_level", False):
                    self.game._really_set_next_level()
                else:
                    self.game.step()
                frame = self.game.camera.render(
                    self.game.current_level.get_sprites()
                )
                ticks_frames.append(np.asarray(frame).copy())
                ticks_states.append(self.extract_state())
        except Exception:
            try:
                self.game.perform_action(a)
            except Exception:
                pass

        if ticks_frames:
            self._last_after_frame = ticks_frames[-1].copy()
            self._last_during_frames = ticks_frames[:-1]
            self._last_intermediate_states = ticks_states[:-1]
        else:
            try:
                self._last_after_frame = self._render_canonical_frame()
            except Exception:
                self._last_after_frame = None
            self._last_during_frames = []
            self._last_intermediate_states = []

        level_after = self.game.level_index
        game_won = self.is_game_won()
        reward = 1.0 if (level_after > level_before or game_won) else 0.0
        done = level_after > level_before or game_won

        state = self.extract_state()
        return state, reward, done

    def get_available_actions(self) -> list[int]:
        """Return the game's declared actions plus ACTION0 (RESET) and ACTION7 (UNDO)."""
        declared = list(self.game._available_actions)
        out = [0] + declared
        if 7 not in declared:
            out.append(7)
        return out

    def get_level_index(self) -> int:
        return self.game.level_index

    def get_mission(self) -> str | None:
        return None

    def is_game_won(self) -> bool:
        return hasattr(self.game, '_state') and str(self.game._state) == 'GameState.WIN'

    def is_game_over(self) -> bool:
        """True iff the current level has ended in failure (step budget exhausted).
        The board is frozen until a RESET (action 0 -> level_reset)."""
        return (hasattr(self.game, '_state')
                and str(self.game._state) == 'GameState.GAME_OVER')

    def get_move_budget_info(self) -> dict | None:
        """Return per-attempt move budget and lives, or None if the game has no budget.
        Accessors are private on the arcengine side. Failures return None.
        """
        try:
            ui = getattr(self.game, "_step_counter_ui", None)
            if ui is None or getattr(ui, "osgviligwp", 0) == 0:
                return None
            return {
                "remaining": int(getattr(ui, "current_steps", 0)),
                "max": int(ui.osgviligwp),
                "decrement_per_step": int(getattr(ui, "efipnixsvl", 1)),
                "lives_remaining": int(getattr(self.game, "aqygnziho", 0)),
                "max_lives": 3,
            }
        except Exception:
            return None

    def extract_state(self) -> list[dict]:
        """Extract typed objects from arcengine sprite state including rotation and pixels.
        Full-screen backgrounds (w >= 64) are skipped to avoid bloating state records.
        """
        import numpy as np
        try:
            cam = self.game.camera
            cam_w = int(cam.width) if cam.width > 0 else 64
            cam_h = int(cam.height) if cam.height > 0 else 64
            scale = max(1, min(64 // cam_w, 64 // cam_h))
        except Exception:
            scale = 1
        objects = []
        for s in self.game.current_level._sprites:
            if s.width >= 64 and s.height >= 64:
                continue
            entry = {
                "name": s.name,
                "tags": s.tags if s.tags else [],
                "x": s.x,
                "y": s.y,
                "w": s.width,
                "h": s.height,
                "display_x": int(s.x) * scale,
                "display_y": int(s.y) * scale,
                "display_w": int(s.width) * scale,
                "display_h": int(s.height) * scale,
                "visible": s.is_visible,
                "collidable": s.is_collidable,
                "layer": s.layer,
                "rotation": int(getattr(s, "rotation", 0)),
            }
            try:
                rendered = s.render() if hasattr(s, "render") else None
                if rendered is not None:
                    entry["pixels"] = np.asarray(rendered).astype(int).tolist()
            except Exception:
                pass
            objects.append(entry)
        return objects

    def get_frame(self):
        """Return the canonical 64x64 display frame via camera.render(sprites)."""
        try:
            return self._render_canonical_frame()
        except Exception:
            return None

    def _render_canonical_frame(self):
        import numpy as np
        frame = self.game.camera.render(self.game.current_level.get_sprites())
        return np.asarray(frame).copy()

    def describe_state(self, state: list[dict]) -> str:
        """Compact description skipping wall tiles."""
        parts = []
        n_walls = 0
        for o in state:
            is_wall = any(t in o.get("tags", []) for t in ["ihdgageizm"])
            if is_wall:
                n_walls += 1
                continue

            tags = ",".join(o.get("tags", []))
            tag_str = f"[{tags}]" if tags else ""
            vis = " (hidden)" if not o.get("visible", True) else ""
            parts.append(f"{o['name']}{tag_str}({o['x']},{o['y']}){vis}")

        if n_walls:
            parts.append(f"[{n_walls} wall tiles]")
        return ", ".join(parts)

    def compute_diff(self, before: list[dict], after: list[dict]) -> str:
        before_by_name: dict[str, list[dict]] = {}
        after_by_name: dict[str, list[dict]] = {}
        for o in before:
            before_by_name.setdefault(o["name"], []).append(o)
        for o in after:
            after_by_name.setdefault(o["name"], []).append(o)

        diffs = []
        all_names = set(list(before_by_name.keys()) + list(after_by_name.keys()))

        for name in sorted(all_names):
            b_list = before_by_name.get(name, [])
            a_list = after_by_name.get(name, [])

            b_pos = sorted((o["x"], o["y"]) for o in b_list)
            a_pos = sorted((o["x"], o["y"]) for o in a_list)

            if b_pos != a_pos:
                b_set, a_set = set(b_pos), set(a_pos)
                removed, added = b_set - a_set, a_set - b_set

                if len(removed) == 1 and len(added) == 1:
                    r, a = list(removed)[0], list(added)[0]
                    tags = b_list[0].get("tags", [])
                    tag_str = f" [{','.join(tags)}]" if tags else ""
                    diffs.append(f"{name}{tag_str} moved ({r[0]},{r[1]})->({a[0]},{a[1]})")
                else:
                    if removed:
                        diffs.append(f"{name} disappeared from {removed}")
                    if added:
                        diffs.append(f"{name} appeared at {added}")

            for bo, ao in zip(b_list, a_list):
                if bo.get("visible") != ao.get("visible"):
                    v = "visible" if ao.get("visible") else "hidden"
                    diffs.append(f"{name} at ({ao['x']},{ao['y']}) became {v}")
                if bo.get("rotation") != ao.get("rotation"):
                    diffs.append(
                        f"{name} at ({ao['x']},{ao['y']}) rotation "
                        f"{bo.get('rotation', 0)}->{ao.get('rotation', 0)}"
                    )
                bp = bo.get("pixels")
                ap = ao.get("pixels")
                if bp is not None and ap is not None and bp != ap:
                    n = sum(
                        1
                        for br, ar in zip(bp, ap)
                        for bv, av in zip(br, ar)
                        if bv != av
                    )
                    bh = len(bp)
                    ah = len(ap)
                    bw = len(bp[0]) if bp else 0
                    aw = len(ap[0]) if ap else 0
                    if (bh, bw) != (ah, aw):
                        n += abs(bh * bw - ah * aw)
                    if n > 0:
                        diffs.append(
                            f"{name} at ({ao['x']},{ao['y']}) internal "
                            f"pattern changed ({n} cells)"
                        )

        return "; ".join(diffs) if diffs else "Nothing changed"


class ArcEngineAdapter(DomainAdapter):
    """Domain adapter for ARC-AGI-3 games loaded from arcengine."""

    def __init__(self, action_names: dict[int, str] | None = None):
        self._action_names = action_names or {
            1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT",
        }

    @property
    def name(self) -> str:
        return "arcengine"

    def write_replay_buffer(self, transitions: list[dict], workspace_dir: Path) -> None:
        with open(workspace_dir / "replay_buffer.pkl", "wb") as f:
            pickle.dump(transitions, f)

    def write_replay_buffer_frames(
        self, transitions: list[dict], workspace_dir: Path,
    ) -> None:
        """Frames-only replay buffer with before_frame/after_frame instead of before_state/after_state."""
        with open(workspace_dir / "replay_buffer.pkl", "wb") as f:
            pickle.dump(transitions, f)

    def write_initial_data(self, initial_state: Any, workspace_dir: Path) -> None:
        with open(workspace_dir / "initial_state.json", "w") as f:
            json.dump(initial_state, f, indent=2)

    def write_test_runner(self, workspace_dir: Path) -> None:
        script = _TEST_RUNNER_SCRIPT
        (workspace_dir / "test_runner.py").write_text(script)

    def write_test_runner_frames(self, workspace_dir: Path) -> None:
        (workspace_dir / "test_runner.py").write_text(
            _TEST_RUNNER_SCRIPT_FRAMES
        )

    def format_code_stub(self, structure: str = "free") -> str:
        """Return the initial game_engine.py template for "free", "oop", or "monolithic" structure."""
        if structure == "free":
            return _CODE_STUB_FREE
        if structure == "monolithic":
            return _CODE_STUB_MONO
        if structure == "oop":
            return _CODE_STUB
        raise ValueError(
            f"structure must be 'free', 'oop', or 'monolithic', "
            f"got {structure!r}"
        )

    def format_code_stub_frames(self, structure: str = "free") -> str:
        """Return the frames-only game_engine.py template."""
        if structure not in ("free", "oop", "monolithic"):
            raise ValueError(
                f"structure must be 'free', 'oop', or 'monolithic', "
                f"got {structure!r}"
            )
        if structure == "oop":
            return _CODE_STUB_FRAMES_OOP
        if structure == "free":
            return _CODE_STUB_FRAMES_FREE
        return _CODE_STUB_FRAMES_MONO

    def format_synthesis_prompt(
        self,
        workspace_dir,
        test_runner_path,
        project_root,
        structure: str = "free",
    ) -> str:
        if structure not in ("free", "oop", "monolithic"):
            raise ValueError(
                f"structure must be 'free', 'oop', or 'monolithic', "
                f"got {structure!r}"
            )
        rules_block = load_prompt(f"synthesizer/object_centric/rules_{structure}.txt")
        objects_clause = load_prompt("synthesizer/object_centric/objects_clause.txt")
        closing = load_prompt("synthesizer/object_centric/closing.txt")
        return self._format_synthesis_prompt_with_blocks(
            workspace_dir=workspace_dir,
            test_runner_path=test_runner_path,
            project_root=project_root,
            rules_block=rules_block,
            objects_clause=objects_clause,
            closing=closing,
        )

    def write_test_runner_crystallised(
        self,
        workspace_dir: Path,
        scope_tags: list[str],
    ) -> None:
        """Write a scoped test runner that only checks sprites in scope_tags."""
        scope_repr = repr(tuple(sorted(set(str(t) for t in scope_tags))))
        script = _TEST_RUNNER_SCRIPT_CRYSTALLISED.replace(
            "__SCOPE_TAGS__", scope_repr,
        )
        (workspace_dir / "test_runner.py").write_text(script)

    def format_synthesis_prompt_crystallised(
        self,
        workspace_dir,
        test_runner_path,
        project_root,
        *,
        partition: dict,
        scope_extra_tags: list[str],
        structure: str = "oop",
    ) -> str:
        """Synthesis prompt for post-crystallisation scoped world model synthesis."""
        if structure not in ("free", "oop", "monolithic"):
            raise ValueError(
                f"structure must be 'free', 'oop', or 'monolithic', "
                f"got {structure!r}"
            )
        classes_line = load_prompt(
            f"synthesizer/crystallised/classes_{structure}.txt"
        )
        if partition:
            labelled_lines = "\n".join(
                f"  - {tag}  ->  role={alias!r}"
                for tag, alias in sorted(partition.items())
            )
        else:
            labelled_lines = "  (no analyzer-committed tags this round)"
        if scope_extra_tags:
            extra_lines = "\n".join(
                f"  - {tag}  (no analyzer role; observed changing)"
                for tag in sorted(scope_extra_tags)
            )
        else:
            extra_lines = "  (none)"
        all_scope = sorted(
            set(partition.keys())
            | set(str(t) for t in scope_extra_tags)
        )
        scope_count = len(all_scope)
        return (
            load_prompt("synthesizer/crystallised/main.md")
            .replace("%%WORKSPACE_DIR%%", str(workspace_dir))
            .replace("%%TEST_RUNNER_PATH%%", str(test_runner_path))
            .replace("%%LABELLED_LINES%%", labelled_lines)
            .replace("%%EXTRA_LINES%%", extra_lines)
            .replace("%%SCOPE_COUNT%%", str(scope_count))
            .replace("%%CLASSES_LINE%%", classes_line)
        )

    def format_synthesis_prompt_frames(
        self,
        workspace_dir,
        test_runner_path,
        project_root,
        structure: str = "oop",
    ) -> str:
        """Synthesis prompt for frames-only (raw 64x64 palette grid) world model."""
        if structure not in ("free", "oop", "monolithic"):
            raise ValueError(
                f"structure must be 'free', 'oop', or 'monolithic', "
                f"got {structure!r}"
            )
        classes_line = load_prompt(f"synthesizer/frames/classes_{structure}.txt")
        return (
            load_prompt("synthesizer/frames/main.md")
            .replace("%%WORKSPACE_DIR%%", str(workspace_dir))
            .replace("%%TEST_RUNNER_PATH%%", str(test_runner_path))
            .replace("%%CLASSES_LINE%%", classes_line)
        )

    def _format_synthesis_prompt_with_blocks(
        self, *, workspace_dir, test_runner_path, project_root,
        rules_block: str, objects_clause: str, closing: str,
    ) -> str:
        return (
            load_prompt("synthesizer/object_centric/main.md")
            .replace("%%WORKSPACE_DIR%%", str(workspace_dir))
            .replace("%%TEST_RUNNER_PATH%%", str(test_runner_path))
            .replace("%%RULES_BLOCK%%", rules_block)
            .replace("%%OBJECTS_CLAUSE%%", objects_clause)
            .replace("%%CLOSING%%", closing)
        )

    def format_goal_description(self, mission=None, **kwargs):
        return load_prompt("synthesizer/shared/goal_description.txt")

    def format_transitions_for_context(self, transitions, max_examples=40, mission=None):
        sections = []

        sections.append("# Observed Transitions\n")

        action_set = set()
        for t in transitions:
            action_set.add((t["action_id"], t["action_name"]))
        sections.append("Actions: " + ", ".join(
            f"{aid}={aname}" for aid, aname in sorted(action_set)
        ))

        for t in transitions[:max_examples]:
            line = f'Step {t["timestep"]} {t["action_name"]} (L{t["level"]+1}): {t["diff_text"]}'
            if t["reward"] > 0:
                line += f'  *** REWARD: {t["reward"]} ***'
            sections.append(line)

        return "\n".join(sections)


_TEST_RUNNER_SCRIPT = r'''#!/usr/bin/env python3
"""Test runner: structured state + reward verification for ARC-AGI-3."""
import sys, os, re, pickle, json, importlib.util, copy

# Patterns that indicate the canonical lookup-table cheat: the model
# reads replay_buffer.pkl (the verifier's input data) and uses it as a
# (state, action) -> after_state lookup, passing the verifier by
# memorisation rather than rule synthesis. We scan for any string
# literal naming the file. Other file I/O is allowed; a model may
# legitimately load per-level initial-state caches to handle the
# uncomputable hand-designed level-advance transitions.
_FORBIDDEN_PATTERNS = (
    (
        r"\breplay_buffer\b",
        "references replay_buffer (the verifier input), canonical "
        "lookup-table cheat. The next state must be COMPUTED from "
        "(state, action), not looked up from the buffer this file is "
        "verified against. Per-level initial-state caches "
        "(e.g. l2_initial.pkl) are fine; those carry only "
        "level-entry layouts, not the full transition table.",
    ),
)

def _check_static_no_file_io(code_path):
    """Reject any game_engine.py that performs file I/O or memorisation
    in transition_function / reward_function. Returns None if clean,
    a (line_no, message) tuple if a violation is found.

    Conservative scan: strip line comments and multi-line string blocks
    before pattern-matching, so a forbidden pattern mentioned in a
    docstring or comment is not a false positive.
    """
    try:
        with open(code_path) as f:
            src = f.read()
    except Exception as e:
        return (0, "could not read " + str(code_path) + ": "
                + type(e).__name__ + ": " + str(e))
    # chr(39) is single quote; chr(34) is double quote. Three of each
    # constructs the Python triple-quote markers without putting them
    # as literal characters anywhere in this file (which would break
    # the embedding of this whole function inside an r-string).
    tq_single = chr(39) * 3
    tq_double = chr(34) * 3
    # Drop triple-quoted blocks (docstrings, multi-line strings).
    # Non-greedy DOTALL match between matching triple quotes.
    cleaned = re.sub(
        tq_single + r".*?" + tq_single, "", src, flags=re.DOTALL,
    )
    cleaned = re.sub(
        tq_double + r".*?" + tq_double, "", cleaned, flags=re.DOTALL,
    )
    # Drop line comments. Naive but adequate.
    cleaned = re.sub(r"#[^\n]*", "", cleaned)
    for pattern, message in _FORBIDDEN_PATTERNS:
        m = re.search(pattern, cleaned)
        if m is not None:
            line_no = cleaned[:m.start()].count("\n") + 1
            return (line_no, message)
    return None

def load_engine(code_path):
    # Static anti-cheat gate: reject before importing.
    violation = _check_static_no_file_io(code_path)
    if violation is not None:
        line_no, message = violation
        return None, (
            f"STATIC REJECTION (file I/O / memorisation): "
            f"line {line_no}: {message}. game_engine.py must "
            f"compute transitions from (state, action) only; "
            f"no open(), no pickle/json.load, no __file__-relative "
            f"reads. See the synth prompt rule on NO FILE I/O AND "
            f"NO MEMORISATION."
        )
    spec = importlib.util.spec_from_file_location("game_engine", code_path)
    module = importlib.util.module_from_spec(spec)
    module.copy = __import__("copy")
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        return None, f"Load error: {type(e).__name__}: {e}"
    missing = []
    if not hasattr(module, "transition_function"): missing.append("transition_function")
    if not hasattr(module, "reward_function"): missing.append("reward_function")
    if missing:
        return None, f"Missing: {', '.join(missing)}"
    return module, None

def compare_states(predicted, actual, wall_tags=("ihdgageizm",)):
    """Compare non-wall objects by name, position, visibility, rotation,
    AND internal pixel content.

    Pixel-content comparison is required for phi_1; the synthesised
    transition_function must predict EVERY observable change per step.
    Sprite-internal transformations (HUD rotates, cells toggle, animation
    frame advances) leave (x, y, w, h, visible) unchanged but produce
    pixel-level differences. Without including pixels here, those
    transformations are invisible to CEGIS.
    """
    def _pixel_hash(p):
        if p is None:
            return None
        # Tuple-of-tuples is hashable and equality-comparable.
        return tuple(tuple(row) for row in p)

    def _key(o):
        # Use position too so multi-instance same-name sprites don't
        # collide (the engine has its own pairing fix; here we just
        # need a unique key per before/after object).
        return (o["name"], int(o.get("x", 0)), int(o.get("y", 0)))

    def extract(state):
        objs = {}
        for o in state:
            tags = o.get("tags", [])
            if any(t in tags for t in wall_tags):
                continue
            if o.get("w", 0) >= 64:
                continue
            objs[_key(o)] = (
                o["x"], o["y"],
                o.get("visible", True),
                int(o.get("rotation", 0)),
                _pixel_hash(o.get("pixels")),
            )
        return objs

    p, a = extract(predicted), extract(actual)
    if p == a:
        return True, ""

    diffs = []
    for key in set(list(p.keys()) + list(a.keys())):
        pv, av = p.get(key), a.get(key)
        if pv != av:
            # Render a compact summary: which field(s) differ?
            if pv is None or av is None:
                diffs.append(f"  {key}: predicted={pv is not None} "
                             f"actual={av is not None}")
                continue
            field_names = ("x", "y", "visible", "rotation", "pixels")
            for i, fname in enumerate(field_names):
                if pv[i] != av[i]:
                    if fname == "pixels":
                        # Don't dump the full grid; just say it differs
                        diffs.append(f"  {key}: pixel pattern differs")
                    else:
                        diffs.append(
                            f"  {key}: {fname} predicted={pv[i]} actual={av[i]}"
                        )
    return False, "\n".join(diffs)

def main():
    workspace = os.path.dirname(os.path.abspath(__file__))
    engine, err = load_engine(os.path.join(workspace, "game_engine.py"))
    if err:
        print(f"LOAD_ERROR: {err}")
        sys.exit(1)

    buffer_path = os.environ.get("OOP_EVAL_BUFFER")
    if not buffer_path:
        buffer_path = os.path.join(workspace, "replay_buffer.pkl")
    with open(buffer_path, "rb") as f:
        replay_buffer = pickle.load(f)

    total = len(replay_buffer)
    trans_passed = 0
    reward_passed = 0
    failures = []

    for i, trans in enumerate(replay_buffer):
        # For click actions (ACTION6) the synthesised model needs the
        # click coords too. We pass the action as a dict if click_x/y
        # are present in the transition, else as a bare int. The
        # synthesised transition_function should accept either form.
        aid = trans["action_id"]
        if "click_x" in trans and "click_y" in trans and aid == 6:
            action_arg = {"action_id": 6, "x": trans["click_x"], "y": trans["click_y"]}
        else:
            action_arg = aid

        # Transition check
        try:
            before = copy.deepcopy(trans["before_state"])
            predicted = engine.transition_function(before, action_arg)

            if predicted is None:
                failures.append({"i": i, "t": trans["timestep"], "a": trans["action_name"],
                                 "type": "transition", "error": "returned None"})
                continue

            ok, diff = compare_states(predicted, trans["after_state"])
            if ok:
                trans_passed += 1
            else:
                failures.append({"i": i, "t": trans["timestep"], "a": trans["action_name"],
                                 "type": "transition", "diff": diff})
        except Exception as e:
            failures.append({"i": i, "t": trans["timestep"], "a": trans["action_name"],
                             "type": "transition", "error": str(e)})
            continue

        # Reward check
        try:
            pred_r, pred_d = engine.reward_function(
                copy.deepcopy(trans["before_state"]), action_arg, predicted)
            actual_r = trans.get("reward", 0.0)
            actual_d = trans.get("done", False)
            if abs(pred_r - actual_r) < 1e-6 and pred_d == actual_d:
                reward_passed += 1
            else:
                failures.append({"i": i, "t": trans["timestep"], "a": trans["action_name"],
                                 "type": "reward",
                                 "diff": f"predicted=({pred_r},{pred_d}) actual=({actual_r},{actual_d})"})
        except Exception as e:
            failures.append({"i": i, "t": trans["timestep"], "a": trans["action_name"],
                             "type": "reward", "error": str(e)})

    both = sum(1 for idx in range(total) if not any(f["i"] == idx for f in failures))
    print(f"TRANSITION: {trans_passed}/{total} passed ({trans_passed/total*100:.0f}%)")
    print(f"REWARD: {reward_passed}/{total} passed ({reward_passed/total*100:.0f}%)")
    print(f"RESULT: {both}/{total} passed ({both/total*100:.0f}%)")

    if failures:
        print(f"\nFAILURES ({len(failures)}):")
        for f in failures[:20]:
            print(f"  Step {f['i']} (T{f['t']} {f['a']}) [{f['type']}]:")
            if "error" in f:
                print(f"    ERROR: {f['error']}")
            else:
                print(f"    {f['diff']}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")
    else:
        print("ALL TESTS PASSED")

    # Phi_2 check: reward function must not be trivially zero
    all_zero_reward = True
    for trans in replay_buffer:
        try:
            aid_arg = trans["action_id"]
            if "click_x" in trans and "click_y" in trans and aid_arg == 6:
                aid_arg = {"action_id": 6, "x": trans["click_x"], "y": trans["click_y"]}
            r, d = engine.reward_function(
                copy.deepcopy(trans["before_state"]),
                aid_arg,
                copy.deepcopy(trans["after_state"]))
            if r > 0 or d:
                all_zero_reward = False
                break
        except:
            pass
    if all_zero_reward:
        print("PHI2_WARNING: reward_function returns (0,False) for ALL transitions. "
              "It must hypothesize a goal condition even if no reward was observed.")

if __name__ == "__main__":
    main()
'''


_CODE_STUB = '''# ARC-AGI-3 -- Object-oriented world model
# Read context.txt for observed transitions and goal.
# Organize the transition logic with `class` statements (one or more).
# Helper functions at module scope are allowed.

import copy


def transition_function(state, action_id):
    # state: list[dict] of object records with keys: name, tags, x, y, w, h,
    #   display_x, display_y, display_w, display_h, visible, collidable,
    #   layer, rotation, pixels.
    # action_id: int (or dict {"action_id": 6, "x": int, "y": int} for clicks).
    # Return: new state (list[dict]) with the same schema as the input.
    # TODO: implement the action -> state delta from context.txt using class(es).
    return [copy.deepcopy(o) for o in state]


def reward_function(state, action_id, new_state):
    # IMPORTANT: This function must NOT always return (0.0, False).
    # Even if no reward was observed in training, hypothesize the reward
    # condition. The condition usually requires preconditions to be satisfied
    # and a joint configuration of objects, not a single-object check.
    # Return (1.0, True) when the hypothesized condition is met.
    # TODO: implement reward detection based on context.txt
    return (0.0, False)


def planner(state, available_actions=None, max_depth=None):
    # Optional C3 hook. Search through transition_function + reward_function
    # and return a reward-reaching action list, or None if no plan is found.
    return None
'''


_TEST_RUNNER_SCRIPT_CRYSTALLISED = r'''#!/usr/bin/env python3
"""Scoped sprite-level test runner for crystallised world models.

Only sprites whose tag intersects SCOPE_TAGS are compared between
predicted and observed states. Sprites whose tag is NOT in scope are
ignored entirely: the synth's transition_function need not predict
them. This matches the post-crystallisation scoped verifier (paper
§6 + C2 refinement): the partition is frozen and only goal-relevant
tags plus observed-changing tags must be modelled."""
import sys, os, re, pickle, json, importlib.util, copy

SCOPE_TAGS = __SCOPE_TAGS__

_FORBIDDEN_PATTERNS = (
    (
        r"\breplay_buffer\b",
        "references replay_buffer (the verifier input), canonical "
        "lookup-table cheat. The next state must be COMPUTED from "
        "(state, action), not looked up from the buffer this file is "
        "verified against.",
    ),
)


def _check_static_no_file_io(code_path):
    try:
        with open(code_path) as f:
            src = f.read()
    except Exception as e:
        return (0, "could not read " + str(code_path) + ": "
                + type(e).__name__ + ": " + str(e))
    tq_single = chr(39) * 3
    tq_double = chr(34) * 3
    cleaned = re.sub(tq_single + r".*?" + tq_single, "", src, flags=re.DOTALL)
    cleaned = re.sub(tq_double + r".*?" + tq_double, "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"#[^\n]*", "", cleaned)
    for pattern, message in _FORBIDDEN_PATTERNS:
        m = re.search(pattern, cleaned)
        if m is not None:
            line_no = cleaned[:m.start()].count("\n") + 1
            return (line_no, message)
    rf_match = re.search(
        r"def\s+reward_function\s*\([^\n]*\n(.*?)(?=\n(?:def |class |\Z))",
        cleaned, flags=re.DOTALL,
    )
    if rf_match:
        rf_body = rf_match.group(1)
        cheat = (
            (r"_L\d+_INITIAL",
             "reward_function references a level-initial cache "
             "(_L<N>_INITIAL). Goal predicate must be derivable from "
             "observable state, not from cached future frames."),
            (r"l\d+_initial\.pkl",
             "reward_function loads a level-initial pkl."),
            (r"_load_level_initial",
             "reward_function calls _load_level_initial."),
        )
        for pat, msg in cheat:
            m = re.search(pat, rf_body)
            if m is not None:
                line_no = cleaned[:rf_match.start() + m.start()].count("\n") + 1
                return (line_no, msg)
    return None


def load_engine(code_path):
    violation = _check_static_no_file_io(code_path)
    if violation is not None:
        line_no, message = violation
        return None, (
            f"STATIC REJECTION (file I/O / memorisation): "
            f"line {line_no}: {message}. game_engine.py must "
            f"compute transitions from (state, action) only."
        )
    spec = importlib.util.spec_from_file_location("game_engine", code_path)
    module = importlib.util.module_from_spec(spec)
    module.copy = __import__("copy")
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        return None, f"Load error: {type(e).__name__}: {e}"
    missing = []
    if not hasattr(module, "transition_function"): missing.append("transition_function")
    if not hasattr(module, "reward_function"): missing.append("reward_function")
    if missing:
        return None, f"Missing: {', '.join(missing)}"
    return module, None


def _in_scope(obj, scope=SCOPE_TAGS):
    if not scope:
        return True
    tags = obj.get("tags") or []
    for t in tags:
        if t in scope:
            return True
    return False


def compare_states_scoped(predicted, actual):
    """Compare ONLY in-scope sprites by name, position, visibility,
    rotation, and internal pixel content. Out-of-scope sprites are
    ignored entirely (pass-through semantics)."""
    def _pixel_hash(p):
        if p is None:
            return None
        return tuple(tuple(row) for row in p)

    def _key(o):
        return (o["name"], int(o.get("x", 0)), int(o.get("y", 0)))

    def extract_in_scope(state):
        objs = {}
        for o in state:
            if not _in_scope(o):
                continue
            if o.get("w", 0) >= 64:
                continue
            objs[_key(o)] = (
                o["x"], o["y"],
                o.get("visible", True),
                int(o.get("rotation", 0)),
                _pixel_hash(o.get("pixels")),
            )
        return objs

    p, a = extract_in_scope(predicted), extract_in_scope(actual)
    if p == a:
        return True, ""
    diffs = []
    for key in set(list(p.keys()) + list(a.keys())):
        pv, av = p.get(key), a.get(key)
        if pv != av:
            if pv is None or av is None:
                diffs.append(f"  {key}: predicted={pv is not None} "
                             f"actual={av is not None}")
                continue
            field_names = ("x", "y", "visible", "rotation", "pixels")
            for i, fname in enumerate(field_names):
                if pv[i] != av[i]:
                    if fname == "pixels":
                        diffs.append(f"  {key}: pixel pattern differs")
                    else:
                        diffs.append(
                            f"  {key}: {fname} predicted={pv[i]} actual={av[i]}"
                        )
    return False, "\n".join(diffs)


def main():
    workspace = os.path.dirname(os.path.abspath(__file__))
    engine, err = load_engine(os.path.join(workspace, "game_engine.py"))
    if err:
        print(f"LOAD_ERROR: {err}")
        sys.exit(1)

    buffer_path = os.environ.get("OOP_EVAL_BUFFER")
    if not buffer_path:
        buffer_path = os.path.join(workspace, "replay_buffer.pkl")
    with open(buffer_path, "rb") as f:
        replay_buffer = pickle.load(f)

    total = len(replay_buffer)
    trans_passed = 0
    reward_passed = 0
    failures = []

    for i, trans in enumerate(replay_buffer):
        aid = trans["action_id"]
        if "click_x" in trans and "click_y" in trans and aid == 6:
            action_arg = {"action_id": 6, "x": trans["click_x"], "y": trans["click_y"]}
        else:
            action_arg = aid

        try:
            before = copy.deepcopy(trans["before_state"])
            predicted = engine.transition_function(before, action_arg)
            if predicted is None:
                failures.append({"i": i, "t": trans["timestep"], "a": trans["action_name"],
                                 "type": "transition", "error": "returned None"})
                continue
            ok, diff = compare_states_scoped(predicted, trans["after_state"])
            if ok:
                trans_passed += 1
            else:
                failures.append({"i": i, "t": trans["timestep"], "a": trans["action_name"],
                                 "type": "transition", "diff": diff})
        except Exception as e:
            failures.append({"i": i, "t": trans["timestep"], "a": trans["action_name"],
                             "type": "transition", "error": str(e)})
            continue

        try:
            pred_r, pred_d = engine.reward_function(
                copy.deepcopy(trans["before_state"]), action_arg, predicted)
            actual_r = trans.get("reward", 0.0)
            actual_d = trans.get("done", False)
            if abs(pred_r - actual_r) < 1e-6 and pred_d == actual_d:
                reward_passed += 1
            else:
                failures.append({"i": i, "t": trans["timestep"], "a": trans["action_name"],
                                 "type": "reward",
                                 "diff": f"predicted=({pred_r},{pred_d}) actual=({actual_r},{actual_d})"})
        except Exception as e:
            failures.append({"i": i, "t": trans["timestep"], "a": trans["action_name"],
                             "type": "reward", "error": str(e)})

    both = sum(1 for idx in range(total) if not any(f["i"] == idx for f in failures))
    n_scope = len(SCOPE_TAGS)
    print(f"TRANSITION: {trans_passed}/{total} passed ({trans_passed/total*100:.0f}%)")
    print(f"REWARD: {reward_passed}/{total} passed ({reward_passed/total*100:.0f}%)")
    print(f"RESULT: {both}/{total} passed ({both/total*100:.0f}%)")
    print(f"SCOPE: {n_scope} in-scope tag(s) checked; out-of-scope sprites pass-through.")
    if failures:
        print(f"\nFAILURES ({len(failures)}):")
        for f in failures[:20]:
            print(f"  Step {f['i']} (T{f['t']} {f['a']}) [{f['type']}]:")
            if "error" in f:
                print(f"    ERROR: {f['error']}")
            else:
                print(f"    {f['diff']}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")
    else:
        print("ALL TESTS PASSED")

    all_zero_reward = True
    for trans in replay_buffer:
        try:
            aid_arg = trans["action_id"]
            if "click_x" in trans and "click_y" in trans and aid_arg == 6:
                aid_arg = {"action_id": 6, "x": trans["click_x"], "y": trans["click_y"]}
            r, d = engine.reward_function(
                copy.deepcopy(trans["before_state"]),
                aid_arg,
                copy.deepcopy(trans["after_state"]))
            if r > 0 or d:
                all_zero_reward = False
                break
        except:
            pass
    if all_zero_reward:
        print("PHI2_WARNING: reward_function returns (0,False) for ALL transitions.")


if __name__ == "__main__":
    main()
'''


_TEST_RUNNER_SCRIPT_FRAMES = r'''#!/usr/bin/env python3
"""Pixel-equality test runner for frames-only ARC-AGI-3 synthesis."""
import sys, os, re, pickle, importlib.util, copy

_FORBIDDEN_PATTERNS = (
    (
        r"\breplay_buffer\b",
        "references replay_buffer (the verifier input), canonical "
        "lookup-table cheat. The next frame must be COMPUTED from "
        "(frame, action), not looked up from the buffer this file is "
        "verified against. Per-level initial-frame caches "
        "(e.g. l2_initial.pkl) are fine; those carry only "
        "level-entry frames, not the full transition table.",
    ),
)


def _check_static_no_file_io(code_path):
    try:
        with open(code_path) as f:
            src = f.read()
    except Exception as e:
        return (0, "could not read " + str(code_path) + ": "
                + type(e).__name__ + ": " + str(e))
    tq_single = chr(39) * 3
    tq_double = chr(34) * 3
    cleaned = re.sub(
        tq_single + r".*?" + tq_single, "", src, flags=re.DOTALL,
    )
    cleaned = re.sub(
        tq_double + r".*?" + tq_double, "", cleaned, flags=re.DOTALL,
    )
    cleaned = re.sub(r"#[^\n]*", "", cleaned)
    for pattern, message in _FORBIDDEN_PATTERNS:
        m = re.search(pattern, cleaned)
        if m is not None:
            line_no = cleaned[:m.start()].count("\n") + 1
            return (line_no, message)
    # Anti-cheat: reward_function must not reference any l<N>_initial cache
    # or load any pkl file. The cache is for modelling transition_function's
    # un-computable after_frames (RESET, level-advance into already-cleared
    # levels), NOT for defining the goal predicate. A reward_function that
    # checks "next_frame == _L<N>_INITIAL" is treating the env's
    # level-advance signal as the goal predicate, which defeats the entire
    # point of hypothesising the goal.
    rf_match = re.search(
        r"def\s+reward_function\s*\([^\n]*\n(.*?)(?=\n(?:def |class |\Z))",
        cleaned, flags=re.DOTALL,
    )
    if rf_match:
        rf_body = rf_match.group(1)
        cheat_patterns = (
            (
                r"_L\d+_INITIAL",
                "reward_function references a level-initial cache "
                "(_L<N>_INITIAL). That treats the env's level-advance "
                "signal as the goal predicate instead of synthesising "
                "one; it would always return reward=0 on any state the "
                "env hasn't advanced from. Goal predicates must be "
                "derivable from observable state, not from cached future "
                "frames. Remove the cache reference from reward_function.",
            ),
            (
                r"l\d+_initial\.pkl",
                "reward_function loads a level-initial pkl. Same issue "
                "as above: the goal predicate must be derivable from "
                "the current state, not from the next-level's cached "
                "entry frame.",
            ),
            (
                r"_load_level_initial",
                "reward_function calls _load_level_initial. Goal "
                "predicates must not depend on cached future-state "
                "frames. Keep _load_level_initial calls in "
                "transition_function only.",
            ),
        )
        for pattern, message in cheat_patterns:
            m = re.search(pattern, rf_body)
            if m is not None:
                line_no = cleaned[:rf_match.start() + m.start()].count("\n") + 1
                return (line_no, message)
    return None


def load_engine(code_path):
    violation = _check_static_no_file_io(code_path)
    if violation is not None:
        line_no, message = violation
        return None, (
            f"STATIC REJECTION (file I/O / memorisation): "
            f"line {line_no}: {message}."
        )
    spec = importlib.util.spec_from_file_location("game_engine", code_path)
    module = importlib.util.module_from_spec(spec)
    module.copy = __import__("copy")
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        return None, f"Load error: {type(e).__name__}: {e}"
    missing = []
    if not hasattr(module, "transition_function"): missing.append("transition_function")
    if not hasattr(module, "reward_function"): missing.append("reward_function")
    if missing:
        return None, f"Missing: {', '.join(missing)}"
    return module, None


def counter_mask(engine):
    """Read+validate game_engine.move_counter_mask() (optional). Returns a set of
    (row, col) cells to EXCLUDE from transition verification, or empty set.

    Fail closed: only ONE continuous line at most 2px wide is honored (the
    move-counter HUD strip). A model may NOT mask an arbitrary region to dodge
    verification of real mechanics."""
    fn = getattr(engine, "move_counter_mask", None)
    if not callable(fn):
        return set()
    try:
        pts = {(int(r), int(c)) for (r, c) in (fn() or [])}
    except Exception:
        return set()
    if not pts:
        return set()
    rows = {r for r, _ in pts}; cols = {c for _, c in pts}
    rspan = max(rows) - min(rows) + 1; cspan = max(cols) - min(cols) + 1
    if min(rspan, cspan) > 2:
        return set()
    long_span = max(rspan, cspan)
    if len(pts) > 2 * long_span:
        return set()
    long_idx = sorted(cols if cspan >= rspan else rows)
    if long_idx[-1] - long_idx[0] + 1 != len(long_idx):
        return set()
    return pts


def frames_equal(a, b, mask=None):
    """Pixel-equality between two 2D palette grids. Tolerant of list / numpy
    inputs and minor shape mismatches (returns False if shapes differ). Cells in
    ``mask`` (the validated move-counter region) are ignored."""
    if a is None or b is None:
        return False
    try:
        if hasattr(a, "tolist"):
            a = a.tolist()
        if hasattr(b, "tolist"):
            b = b.tolist()
    except Exception:
        return False
    if len(a) != len(b):
        return False
    for r in range(len(a)):
        ra, rb = a[r], b[r]
        if hasattr(ra, "tolist"):
            ra = ra.tolist()
        if hasattr(rb, "tolist"):
            rb = rb.tolist()
        if len(ra) != len(rb):
            return False
        for c in range(len(ra)):
            if mask and (r, c) in mask:
                continue
            if int(ra[c]) != int(rb[c]):
                return False
    return True


def frame_diff_summary(predicted, actual, max_cells=12):
    """Return a compact string listing the first few cells where predicted
    and actual differ. Used in failure output."""
    if predicted is None or actual is None:
        return "predicted_is_None" if predicted is None else "actual_is_None"
    try:
        if hasattr(predicted, "tolist"):
            predicted = predicted.tolist()
        if hasattr(actual, "tolist"):
            actual = actual.tolist()
    except Exception:
        return "shape error"
    diffs = []
    h = min(len(predicted), len(actual))
    for r in range(h):
        rp, ra = predicted[r], actual[r]
        if hasattr(rp, "tolist"): rp = rp.tolist()
        if hasattr(ra, "tolist"): ra = ra.tolist()
        w = min(len(rp), len(ra))
        for c in range(w):
            pv, av = int(rp[c]), int(ra[c])
            if pv != av:
                diffs.append(f"({r},{c}) predicted={pv} actual={av}")
                if len(diffs) >= max_cells:
                    return "; ".join(diffs) + f" ... ({_total_diff(predicted, actual)} cells differ)"
    if len(predicted) != len(actual) or any(
        len(predicted[r]) != len(actual[r]) for r in range(min(len(predicted), len(actual)))
    ):
        diffs.append("shape mismatch")
    if not diffs:
        return "equal"
    return "; ".join(diffs)


def _total_diff(a, b):
    n = 0
    for r in range(min(len(a), len(b))):
        ra, rb = a[r], b[r]
        if hasattr(ra, "tolist"): ra = ra.tolist()
        if hasattr(rb, "tolist"): rb = rb.tolist()
        for c in range(min(len(ra), len(rb))):
            if int(ra[c]) != int(rb[c]):
                n += 1
    return n


def _object_has_location(obj):
    if isinstance(obj, dict):
        if obj.get("bbox") is not None:
            return True
        return (
            ("x" in obj or "col" in obj)
            and ("y" in obj or "row" in obj)
        )
    if getattr(obj, "bbox", None) is not None:
        return True
    return (
        (hasattr(obj, "x") or hasattr(obj, "col"))
        and (hasattr(obj, "y") or hasattr(obj, "row"))
    )


def validate_extract_objects(engine, replay_buffer, max_frames=40):
    """Sanity-check the synth's frames-to-object abstraction.

    Spriteless ETA diagnostics are built from this hook. Transition/reward
    checks still report separately, but a dead extractor must not be hidden by
    an otherwise passing pixel transition model.
    """
    extractor = getattr(engine, "extract_objects", None)
    if not callable(extractor):
        return 0, 0, ["game_engine.py must define extract_objects(frame)"]

    checked = 0
    usable = 0
    failures = []
    for trans in replay_buffer:
        if checked >= max_frames:
            break
        for frame_key in ("before_frame", "after_frame"):
            if checked >= max_frames:
                break
            frame = trans.get(frame_key)
            if frame is None:
                continue
            checked += 1
            try:
                objects = extractor(copy.deepcopy(frame))
            except Exception as e:
                failures.append(
                    f"T{trans.get('timestep')} {frame_key}: "
                    f"{type(e).__name__}: {e}"
                )
                continue
            if objects is None:
                failures.append(
                    f"T{trans.get('timestep')} {frame_key}: returned None"
                )
                continue
            if not isinstance(objects, (list, tuple)):
                failures.append(
                    f"T{trans.get('timestep')} {frame_key}: returned "
                    f"{type(objects).__name__}, expected list"
                )
                continue
            if objects:
                usable += 1
                for j, obj in enumerate(list(objects)[:10]):
                    if not _object_has_location(obj):
                        failures.append(
                            f"T{trans.get('timestep')} {frame_key}: object {j} "
                            "is missing bbox or x/y (row/col accepted)"
                        )
                        break

    if checked and usable == 0:
        failures.append("extract_objects returned no objects for any checked frame")
    return checked, usable, failures


def main():
    workspace = os.path.dirname(os.path.abspath(__file__))
    engine, err = load_engine(os.path.join(workspace, "game_engine.py"))
    if err:
        print(f"LOAD_ERROR: {err}")
        sys.exit(1)

    # Optional move-counter mask: a thin (<=2px) line of cells the model is
    # allowed not to predict (the per-level-quantized HUD counter). Validated;
    # excluded from the transition frame check so counter ticks do not force
    # endless resynthesis.
    MASK = counter_mask(engine)
    if MASK:
        print(f"MASK: move-counter mask active ({len(MASK)} cells excluded)")

    buffer_path = os.environ.get("OOP_EVAL_BUFFER")
    if not buffer_path:
        buffer_path = os.path.join(workspace, "replay_buffer.pkl")
    with open(buffer_path, "rb") as f:
        replay_buffer = pickle.load(f)

    total = len(replay_buffer)
    trans_passed = 0
    reward_passed = 0
    failures = []
    object_checked, object_usable, object_failures = validate_extract_objects(
        engine, replay_buffer
    )

    for i, trans in enumerate(replay_buffer):
        aid = trans["action_id"]
        if "click_x" in trans and "click_y" in trans and aid == 6:
            action_arg = {"action_id": 6, "x": trans["click_x"], "y": trans["click_y"]}
        else:
            action_arg = aid

        # Transition check.
        try:
            before = copy.deepcopy(trans["before_frame"])
            predicted = engine.transition_function(before, action_arg)
            if predicted is None:
                failures.append({"i": i, "t": trans["timestep"], "a": trans["action_name"],
                                 "type": "transition", "error": "returned None"})
                continue
            if frames_equal(predicted, trans["after_frame"], MASK):
                trans_passed += 1
            else:
                failures.append({"i": i, "t": trans["timestep"], "a": trans["action_name"],
                                 "type": "transition",
                                 "diff": frame_diff_summary(predicted, trans["after_frame"])})
        except Exception as e:
            failures.append({"i": i, "t": trans["timestep"], "a": trans["action_name"],
                             "type": "transition", "error": str(e)})
            continue

        # Reward check.
        try:
            pred_r, pred_d = engine.reward_function(
                copy.deepcopy(trans["before_frame"]), action_arg, predicted)
            actual_r = trans.get("reward", 0.0)
            actual_d = trans.get("done", False)
            if abs(pred_r - actual_r) < 1e-6 and pred_d == actual_d:
                reward_passed += 1
            else:
                failures.append({"i": i, "t": trans["timestep"], "a": trans["action_name"],
                                 "type": "reward",
                                 "diff": f"predicted=({pred_r},{pred_d}) actual=({actual_r},{actual_d})"})
        except Exception as e:
            failures.append({"i": i, "t": trans["timestep"], "a": trans["action_name"],
                             "type": "reward", "error": str(e)})

    both = sum(1 for idx in range(total) if not any(f["i"] == idx for f in failures))
    print(f"TRANSITION: {trans_passed}/{total} passed ({trans_passed/total*100:.0f}%)")
    print(f"REWARD: {reward_passed}/{total} passed ({reward_passed/total*100:.0f}%)")
    print(f"RESULT: {both}/{total} passed ({both/total*100:.0f}%)")
    print(f"OBJECTS: {object_usable}/{object_checked} frames produced objects")

    if failures:
        print(f"\nFAILURES ({len(failures)}):")
        for f in failures[:20]:
            print(f"  Step {f['i']} (T{f['t']} {f['a']}) [{f['type']}]:")
            if "error" in f:
                print(f"    ERROR: {f['error']}")
            else:
                print(f"    {f['diff']}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")
    if object_failures:
        print(f"\nOBJECT EXTRACTION FAILURES ({len(object_failures)}):")
        for msg in object_failures[:20]:
            print(f"  {msg}")
        if len(object_failures) > 20:
            print(f"  ... and {len(object_failures) - 20} more")
    if not failures and not object_failures:
        print("ALL TESTS PASSED")

    # phi_2 check.
    all_zero_reward = True
    for trans in replay_buffer:
        try:
            aid_arg = trans["action_id"]
            if "click_x" in trans and "click_y" in trans and aid_arg == 6:
                aid_arg = {"action_id": 6, "x": trans["click_x"], "y": trans["click_y"]}
            r, d = engine.reward_function(
                copy.deepcopy(trans["before_frame"]),
                aid_arg,
                copy.deepcopy(trans["after_frame"]))
            if r > 0 or d:
                all_zero_reward = False
                break
        except:
            pass
    if all_zero_reward:
        print("PHI2_WARNING: reward_function returns (0,False) for ALL transitions. "
              "It must hypothesize a goal condition even if no reward was observed.")


if __name__ == "__main__":
    main()
'''


_FRAMES_RENDERER_BLOCK = '''# --- Mini renderer (mirrors arcengine.Camera / Sprite, minimal) -------------
# Use these helpers if your hypothesis is easier to express as a list of
# Sprites + a per-step composite render than as direct cell edits. Drop them
# entirely if you don't need them. They are not required by the verifier.
#
# Pixel convention: palette indices 0..15. Inside a Sprite, -1 means
# transparent (does not overwrite the canvas under the sprite). The env
# always emits non-negative pixels in the observed frame.

class Sprite:
    """A 2D sprite. ``pixels`` is a list[list[int]] of palette indices;
    -1 marks transparent cells. ``layer`` controls draw order (lower
    first, higher on top). ``visible`` toggles rendering without removing
    the sprite from your hypothesis."""

    def __init__(self, pixels, x=0, y=0, layer=0, name="", visible=True):
        self.pixels = pixels
        self.x = int(x)
        self.y = int(y)
        self.layer = int(layer)
        self.name = name
        self.visible = bool(visible)


def render_sprites(sprites, w=64, h=64, background=5):
    """Composite a list of Sprites onto a (h, w) canvas. Lower layers
    render first; higher layers overwrite. Transparent cells (-1) do
    not overwrite. Returns a list[list[int]].
    """
    canvas = [[int(background)] * int(w) for _ in range(int(h))]
    for s in sorted([sp for sp in sprites if sp.visible], key=lambda s: s.layer):
        for dy, row in enumerate(s.pixels):
            ry = s.y + dy
            if ry < 0 or ry >= h:
                continue
            for dx, v in enumerate(row):
                rx = s.x + dx
                if rx < 0 or rx >= w:
                    continue
                v = int(v)
                if v < 0:
                    continue
                canvas[ry][rx] = v
    return canvas


def load_level_frame(fname):
    """Load a per-level entry frame from the workspace. Returns the
    2D palette grid the engine recorded on first entry to that level,
    or None if the cache is absent. Use for the level-advance and
    RESET transitions whose after_frame cannot be derived from rules."""
    import os, pickle
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        with open(os.path.join(here, fname), 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


def extract_objects(frame):
    """Best-effort visual-object abstraction for spriteless ETA diagnostics.

    Replace or extend this with the same object parser your transition model
    uses. The engine calls it after synthesis to build epistemic/ontology
    diagnostics, and the frames-only verifier sanity-checks that it runs and
    returns located objects.
    """
    grid = [[int(v) for v in row] for row in frame]
    if not grid or not grid[0]:
        return []
    h, w = len(grid), len(grid[0])
    counts = {}
    for row in grid:
        for v in row:
            counts[v] = counts.get(v, 0) + 1
    background = max(counts, key=counts.get)
    seen = [[False] * w for _ in range(h)]
    objects = []

    for y0 in range(h):
        for x0 in range(w):
            if seen[y0][x0] or grid[y0][x0] == background:
                continue
            color = grid[y0][x0]
            stack = [(x0, y0)]
            seen[y0][x0] = True
            cells = []
            while stack:
                x, y = stack.pop()
                cells.append((x, y))
                for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if nx < 0 or nx >= w or ny < 0 or ny >= h:
                        continue
                    if seen[ny][nx] or grid[ny][nx] != color:
                        continue
                    seen[ny][nx] = True
                    stack.append((nx, ny))
            xs = [x for x, _ in cells]
            ys = [y for _, y in cells]
            x, y = min(xs), min(ys)
            bw, bh = max(xs) - x + 1, max(ys) - y + 1
            pixels = [[-1] * bw for _ in range(bh)]
            for cx, cy in cells:
                pixels[cy - y][cx - x] = color
            name = f"cc_{color}_{len(objects):03d}"
            objects.append({
                "name": name,
                "type": f"color_{color}_component",
                "tags": [f"color_{color}"],
                "x": x,
                "y": y,
                "w": bw,
                "h": bh,
                "pixels": pixels,
                "layer": 0,
                "visible": True,
            })
    return objects
'''


_CODE_STUB_FRAMES_OOP = (
    '''# ARC-AGI-3 -- Frames-only world model (object-organised)
# Read context.txt for observed transitions and the goal hypothesis.
# Organise the transition logic with `class` statements (one or more).
# Helper functions at module scope are allowed.

import copy

'''
    + _FRAMES_RENDERER_BLOCK
    + '''

def transition_function(frame, action_id):
    """Predict the next frame from the current frame and an action.

    frame: 2D list[list[int]] of palette indices 0..15.
    action_id: int (or dict {"action_id": 6, "x": int, "y": int} for clicks).
    Return: 2D list[list[int]] with the SAME SHAPE as the input.
    """
    # TODO: implement. If your hypothesis groups pixels into objects,
    # maintain those as Sprite instances and re-render each step with
    # render_sprites(...).
    return [list(row) for row in frame]


def reward_function(frame, action_id, next_frame):
    """Return (reward: float, done: bool). MUST hypothesise a goal
    condition; must not always return (0.0, False)."""
    # TODO: implement.
    return (0.0, False)


def planner(frame, available_actions=None, max_depth=None):
    """Optional C3 hook: return a reward-reaching action list under this
    model, or None if no plan is found."""
    return None
'''
)


_CODE_STUB_FRAMES_MONO = (
    '''# ARC-AGI-3 -- Frames-only world model (monolithic)
# Read context.txt for observed transitions and the goal hypothesis.
# A SINGLE transition_function implements the full state transition.
# NO classes. Helper functions at module scope are allowed.

import copy

'''
    + _FRAMES_RENDERER_BLOCK
    + '''

def transition_function(frame, action_id):
    """Predict the next frame from the current frame and an action.

    frame: 2D list[list[int]] of palette indices 0..15.
    action_id: int (or dict {"action_id": 6, "x": int, "y": int} for clicks).
    Return: 2D list[list[int]] with the SAME SHAPE as the input.
    """
    # TODO: implement.
    return [list(row) for row in frame]


def reward_function(frame, action_id, next_frame):
    """Return (reward: float, done: bool). MUST hypothesise a goal
    condition; must not always return (0.0, False)."""
    # TODO: implement.
    return (0.0, False)


def planner(frame, available_actions=None, max_depth=None):
    """Optional C3 hook: return a reward-reaching action list under this
    model, or None if no plan is found."""
    return None
'''
)


_CODE_STUB_FRAMES_FREE = (
    '''# ARC-AGI-3 -- Frames-only world model (free)
# Read context.txt for observed transitions and the goal hypothesis.
# Free structure: classes, helper functions, lookup tables, object parsers,
# sprite renderers, or direct pixel logic are all allowed when they fit.
# Keep transition_function/reward_function/planner/extract_objects as the
# public contract expected by the verifier and engine.

import copy

'''
    + _FRAMES_RENDERER_BLOCK
    + '''

def transition_function(frame, action_id):
    """Predict the next frame from the current frame and an action.

    frame: 2D list[list[int]] of palette indices 0..15.
    action_id: int (or dict {"action_id": 6, "x": int, "y": int} for clicks).
    Return: 2D list[list[int]] with the SAME SHAPE as the input.
    """
    # TODO: implement using the representation that best explains context.txt.
    return [list(row) for row in frame]


def reward_function(frame, action_id, next_frame):
    """Return (reward: float, done: bool). MUST hypothesise a goal
    condition; must not always return (0.0, False)."""
    # TODO: implement.
    return (0.0, False)


def planner(frame, available_actions=None, max_depth=None):
    """Optional C3 hook: return a reward-reaching action list under this
    model, or None if no plan is found."""
    return None
'''
)


_CODE_STUB_MONO = '''# ARC-AGI-3 -- Monolithic world model
# Read context.txt for observed transitions and goal.
# A SINGLE transition_function implements the full state transition.
# NO classes. Helper functions at module scope are allowed.

import copy


def transition_function(state, action_id):
    # state: list[dict] of object records with keys: name, tags, x, y, w, h,
    #   display_x, display_y, display_w, display_h, visible, collidable,
    #   layer, rotation, pixels.
    # action_id: int (or dict {"action_id": 6, "x": int, "y": int} for clicks).
    # Return: new state (list[dict]) with the same schema.
    # TODO: implement the action -> state delta from context.txt.
    return [copy.deepcopy(o) for o in state]


def reward_function(state, action_id, new_state):
    # IMPORTANT: This function must NOT always return (0.0, False).
    # Even if no reward was observed in training, hypothesize the reward
    # condition. The condition usually requires preconditions to be satisfied
    # and a joint configuration of objects, not a single-object check.
    # Return (1.0, True) when the hypothesized condition is met.
    # TODO: implement reward detection based on context.txt
    return (0.0, False)


def planner(state, available_actions=None, max_depth=None):
    # Optional C3 hook. Search through transition_function + reward_function
    # and return a reward-reaching action list, or None if no plan is found.
    return None
'''


_CODE_STUB_FREE = '''# ARC-AGI-3 -- Free-form object world model
# Read context.txt for observed transitions and goal.
# Free structure: classes, helper functions, lookup tables, or direct object
# transforms are all allowed when they fit the mechanic.
# Keep transition_function/reward_function/planner as the public contract
# expected by the verifier and engine.

import copy


def transition_function(state, action_id):
    # state: list[dict] of object records with keys: name, tags, x, y, w, h,
    #   display_x, display_y, display_w, display_h, visible, collidable,
    #   layer, rotation, pixels.
    # action_id: int (or dict {"action_id": 6, "x": int, "y": int} for clicks).
    # Return: new state (list[dict]) with the same schema.
    # TODO: implement using the representation that best explains context.txt.
    return [copy.deepcopy(o) for o in state]


def reward_function(state, action_id, new_state):
    # IMPORTANT: This function must NOT always return (0.0, False).
    # Even if no reward was observed in training, hypothesize the reward
    # condition. The condition usually requires preconditions to be satisfied
    # and a joint configuration of objects, not a single-object check.
    # Return (1.0, True) when the hypothesized condition is met.
    # TODO: implement reward detection based on context.txt
    return (0.0, False)


def planner(state, available_actions=None, max_depth=None):
    # Optional C3 hook. Search through transition_function + reward_function
    # and return a reward-reaching action list, or None if no plan is found.
    return None
'''
