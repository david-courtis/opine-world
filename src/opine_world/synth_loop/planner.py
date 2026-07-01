"""Bounded planning over a synthesized ARC-AGI-3 world model.

The engine owns policy switching and real-environment execution. This module is
only the model-side search/verification piece: load ``game_engine.py``, prefer a
synth-authored planner hook when available, and otherwise run a bounded BFS over
``transition_function`` + ``reward_function``.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import signal
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class PlannerModel:
    """Imported synthesized model plus optional planning helpers."""

    module: Any
    transition_function: Callable
    reward_function: Callable
    planner: Callable | None = None
    action_candidates: Callable | None = None
    extract_objects: Callable | None = None
    move_counter_mask: frozenset[tuple[int, int]] = field(
        default_factory=frozenset
    )


@dataclass
class PlanStep:
    """One predicted model step in a plan trace."""

    action: Any
    next_state: Any
    reward: float
    done: bool

    def to_dict(self) -> dict:
        return {
            "action": copy.deepcopy(self.action),
            "next_state": copy.deepcopy(self.next_state),
            "reward": float(self.reward),
            "done": bool(self.done),
        }


@dataclass
class PlanResult:
    """Planner outcome."""

    ok: bool
    source: str
    plan: list[Any] = field(default_factory=list)
    predicted_steps: list[PlanStep] = field(default_factory=list)
    nodes: int = 0
    depth: int = 0
    reason: str = ""

    def summary(self) -> dict:
        return {
            "ok": bool(self.ok),
            "source": self.source,
            "plan_len": len(self.plan),
            "nodes": int(self.nodes),
            "depth": int(self.depth),
            "reason": self.reason,
        }


class PlannerTimeout(TimeoutError):
    """Raised when a planning attempt exceeds its wall-clock budget."""


class _Deadline:
    """Small SIGALRM-based timeout guard.

    The ARC engine runs planning on the main thread on Unix-like systems, where
    SIGALRM is available. If signal installation fails, elapsed-time checks in
    the BFS loop still cap the search.
    """

    def __init__(self, seconds: float | int | None):
        self.seconds = float(seconds or 0)
        self._old = None
        self._enabled = False

    def __enter__(self):
        if self.seconds <= 0:
            return self
        try:
            self._old = signal.signal(signal.SIGALRM, self._raise)
            signal.setitimer(signal.ITIMER_REAL, self.seconds)
            self._enabled = True
        except Exception:
            self._enabled = False
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._enabled:
            try:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, self._old)
            except Exception:
                pass
        return False

    @staticmethod
    def _raise(*_args):
        raise PlannerTimeout("planner timed out")


def validate_counter_mask(cells: Any) -> frozenset[tuple[int, int]]:
    """Validate the optional move-counter mask contract."""

    try:
        pts = {(int(r), int(c)) for (r, c) in (cells or [])}
    except Exception:
        return frozenset()
    if not pts:
        return frozenset()
    rows = {r for r, _ in pts}
    cols = {c for _, c in pts}
    rspan = max(rows) - min(rows) + 1
    cspan = max(cols) - min(cols) + 1
    if min(rspan, cspan) > 2:
        return frozenset()
    long_span = max(rspan, cspan)
    if len(pts) > 2 * long_span:
        return frozenset()
    long_idx = sorted(cols if cspan >= rspan else rows)
    if long_idx[-1] - long_idx[0] + 1 != len(long_idx):
        return frozenset()
    return frozenset(pts)


def load_model(code_path: str | Path, *, module_name: str = "_arc3_planner_model") -> PlannerModel:
    """Import a synthesized ``game_engine.py`` file."""

    code_path = Path(code_path)
    spec = importlib.util.spec_from_file_location(module_name, str(code_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"could not create import spec for {code_path}")
    module = importlib.util.module_from_spec(spec)
    module.copy = copy
    spec.loader.exec_module(module)

    transition = getattr(module, "transition_function", None)
    reward = getattr(module, "reward_function", None)
    if not callable(transition):
        raise AttributeError("game_engine.py is missing transition_function")
    if not callable(reward):
        raise AttributeError("game_engine.py is missing reward_function")

    mask = frozenset()
    mask_fn = getattr(module, "move_counter_mask", None)
    if callable(mask_fn):
        try:
            mask = validate_counter_mask(mask_fn())
        except Exception:
            mask = frozenset()

    return PlannerModel(
        module=module,
        transition_function=transition,
        reward_function=reward,
        planner=getattr(module, "planner", None),
        action_candidates=getattr(module, "action_candidates", None),
        extract_objects=getattr(module, "extract_objects", None),
        move_counter_mask=mask,
    )


def normalize_action(action: Any, available_actions: list[int]) -> Any | None:
    """Normalize a planner action and reject actions unavailable in the env."""

    available = {int(a) for a in available_actions}
    try:
        if isinstance(action, dict):
            aid = int(action.get("action_id", action.get("id")))
            if aid not in available:
                return None
            if aid == 6:
                if "x" not in action or "y" not in action:
                    return None
                return {
                    "action_id": 6,
                    "x": max(0, min(63, int(action["x"]))),
                    "y": max(0, min(63, int(action["y"]))),
                }
            return aid
        if isinstance(action, str):
            upper = action.upper()
            if upper == "RESET":
                aid = 0
            elif upper.startswith("ACTION"):
                aid = int(upper[len("ACTION"):])
            else:
                return None
        else:
            aid = int(action)
    except Exception:
        return None
    if aid not in available:
        return None
    if aid == 6:
        return None
    return aid


def _dedupe_actions(actions: list[Any]) -> list[Any]:
    out: list[Any] = []
    seen: set[str] = set()
    for action in actions:
        key = json.dumps(action, sort_keys=True, separators=(",", ":"))
        if key in seen:
            continue
        seen.add(key)
        out.append(action)
    return out


def _object_click_targets(objects: list[dict], max_targets: int) -> list[dict]:
    out: list[dict] = []
    seen: set[tuple[int, int]] = set()
    limit = int(max_targets or 0)
    for obj in objects or []:
        if not obj.get("visible", True):
            continue
        try:
            x = int(obj.get("display_x", obj.get("x", 0)))
            y = int(obj.get("display_y", obj.get("y", 0)))
            w = int(obj.get("display_w", obj.get("w", 1)))
            h = int(obj.get("display_h", obj.get("h", 1)))
        except (TypeError, ValueError):
            continue
        cx = max(0, min(63, x + max(0, w // 2)))
        cy = max(0, min(63, y + max(0, h // 2)))
        if (cx, cy) in seen:
            continue
        seen.add((cx, cy))
        out.append({"action_id": 6, "x": cx, "y": cy})
        if limit > 0 and len(out) >= limit:
            break
    return out


def _frame_click_targets(
    model: PlannerModel, frame: Any, max_targets: int,
) -> list[dict]:
    if not callable(model.extract_objects):
        return []
    try:
        objects = model.extract_objects(copy.deepcopy(frame))
    except Exception:
        return []
    if not isinstance(objects, list):
        return []
    return _object_click_targets(
        [o for o in objects if isinstance(o, dict)], max_targets
    )


def default_action_candidates(
    model: PlannerModel,
    state: Any,
    available_actions: list[int],
    *,
    frames_only: bool = False,
    max_click_targets: int = 0,
) -> list[Any]:
    """Build a conservative primitive action set for model search."""

    candidates: list[Any] = []
    for aid in sorted({int(a) for a in available_actions}):
        if aid in (0, 7):
            continue
        if aid == 6:
            continue
        candidates.append(aid)

    if 6 in {int(a) for a in available_actions}:
        if frames_only:
            candidates.extend(_frame_click_targets(model, state, max_click_targets))
        elif isinstance(state, list):
            objects = [o for o in state if isinstance(o, dict)]
            candidates.extend(_object_click_targets(objects, max_click_targets))

    return _dedupe_actions(candidates)


def get_action_candidates(
    model: PlannerModel,
    state: Any,
    available_actions: list[int],
    *,
    frames_only: bool = False,
    max_click_targets: int = 0,
) -> list[Any]:
    """Use a model-provided action candidate hook, falling back to defaults."""

    raw: Any = None
    if callable(model.action_candidates):
        for call in (
            lambda: model.action_candidates(
                copy.deepcopy(state), available_actions=available_actions
            ),
            lambda: model.action_candidates(copy.deepcopy(state), available_actions),
            lambda: model.action_candidates(copy.deepcopy(state)),
        ):
            try:
                raw = call()
                break
            except TypeError:
                continue
            except Exception:
                raw = None
                break
    if raw is None:
        raw = default_action_candidates(
            model,
            state,
            available_actions,
            frames_only=frames_only,
            max_click_targets=max_click_targets,
        )
    if not isinstance(raw, (list, tuple)):
        return []
    out: list[Any] = []
    for action in raw:
        norm = normalize_action(action, available_actions)
        if norm is None or norm == 0 or norm == 7:
            continue
        out.append(norm)
    return _dedupe_actions(out)


def _reward_tuple(value: Any) -> tuple[float, bool]:
    try:
        if isinstance(value, tuple) or isinstance(value, list):
            reward = float(value[0]) if len(value) > 0 else 0.0
            done = bool(value[1]) if len(value) > 1 else reward > 0
            return reward, done
        if isinstance(value, bool):
            return (1.0 if value else 0.0), bool(value)
        reward = float(value)
        return reward, reward > 0
    except Exception:
        return 0.0, False


def simulate_step(
    model: PlannerModel, state: Any, action: Any,
) -> tuple[Any, float, bool]:
    before = copy.deepcopy(state)
    next_state = model.transition_function(before, copy.deepcopy(action))
    reward, done = _reward_tuple(
        model.reward_function(
            copy.deepcopy(state), copy.deepcopy(action), copy.deepcopy(next_state)
        )
    )
    return next_state, reward, done


def validate_plan(
    model: PlannerModel,
    start_state: Any,
    plan: list[Any],
    available_actions: list[int],
    *,
    source: str,
    max_depth: int,
    nodes: int = 0,
) -> PlanResult:
    """Check that a proposed action sequence reaches reward under the model."""

    if not plan:
        return PlanResult(False, source, reason="empty plan")
    if max_depth > 0 and len(plan) > max_depth:
        return PlanResult(False, source, reason="plan exceeds max_depth")

    state = copy.deepcopy(start_state)
    norm_plan: list[Any] = []
    steps: list[PlanStep] = []
    for action in plan:
        norm = normalize_action(action, available_actions)
        if norm is None or norm == 0 or norm == 7:
            return PlanResult(False, source, reason="invalid action in plan")
        try:
            next_state, reward, done = simulate_step(model, state, norm)
        except Exception as exc:
            return PlanResult(
                False,
                source,
                plan=norm_plan,
                predicted_steps=steps,
                nodes=nodes,
                depth=len(norm_plan),
                reason=f"simulation error: {type(exc).__name__}: {exc}",
            )
        norm_plan.append(copy.deepcopy(norm))
        steps.append(PlanStep(copy.deepcopy(norm), next_state, reward, done))
        if reward > 0 or done:
            return PlanResult(
                True,
                source,
                plan=norm_plan,
                predicted_steps=steps,
                nodes=nodes,
                depth=len(norm_plan),
                reason="reaches reward",
            )
        state = next_state

    return PlanResult(
        False,
        source,
        plan=norm_plan,
        predicted_steps=steps,
        nodes=nodes,
        depth=len(norm_plan),
        reason="plan did not reach reward",
    )


def _call_synth_planner(
    model: PlannerModel,
    start_state: Any,
    available_actions: list[int],
    max_depth: int,
) -> list[Any] | None:
    if not callable(model.planner):
        return None
    max_depth_arg = max_depth if max_depth > 0 else None
    for call in (
        lambda: model.planner(
            copy.deepcopy(start_state),
            available_actions=available_actions,
            max_depth=max_depth_arg,
        ),
        lambda: model.planner(copy.deepcopy(start_state), available_actions),
        lambda: model.planner(copy.deepcopy(start_state)),
    ):
        try:
            plan = call()
            if plan is None:
                return None
            if isinstance(plan, tuple):
                plan = list(plan)
            return plan if isinstance(plan, list) else None
        except TypeError:
            continue
    return None


def _state_key(state: Any) -> str:
    try:
        return json.dumps(
            state, sort_keys=True, separators=(",", ":"), default=str
        )
    except Exception:
        return repr(state)


def _generic_bfs(
    model: PlannerModel,
    start_state: Any,
    available_actions: list[int],
    *,
    frames_only: bool,
    max_depth: int,
    max_nodes: int,
    max_click_targets: int,
    started_at: float,
    timeout_s: float,
) -> PlanResult:
    queue = deque([(copy.deepcopy(start_state), [], [])])
    seen = {_state_key(start_state)}
    nodes = 0

    node_cap = int(max_nodes or 0)
    depth_cap = int(max_depth or 0)

    while queue and (node_cap <= 0 or nodes < node_cap):
        if timeout_s and time.time() - started_at > timeout_s:
            raise PlannerTimeout("planner timed out")
        state, plan, trace = queue.popleft()
        if depth_cap > 0 and len(plan) >= depth_cap:
            continue
        candidates = get_action_candidates(
            model,
            state,
            available_actions,
            frames_only=frames_only,
            max_click_targets=max_click_targets,
        )
        for action in candidates:
            nodes += 1
            try:
                next_state, reward, done = simulate_step(model, state, action)
            except Exception:
                continue
            step = PlanStep(copy.deepcopy(action), next_state, reward, done)
            next_plan = plan + [copy.deepcopy(action)]
            next_trace = trace + [step]
            if reward > 0 or done:
                return PlanResult(
                    True,
                    "generic_bfs",
                    plan=next_plan,
                    predicted_steps=next_trace,
                    nodes=nodes,
                    depth=len(next_plan),
                    reason="reaches reward",
                )
            key = _state_key(next_state)
            if key in seen:
                continue
            seen.add(key)
            queue.append((next_state, next_plan, next_trace))

    if node_cap > 0 and nodes >= node_cap:
        reason = "max_nodes exhausted"
    elif depth_cap > 0:
        reason = "no reward reachable within depth"
    else:
        reason = "no reward reachable before search exhausted"
    return PlanResult(
        False,
        "generic_bfs",
        nodes=nodes,
        depth=depth_cap,
        reason=reason,
    )


def plan_from_model(
    model: PlannerModel,
    start_state: Any,
    available_actions: list[int],
    *,
    frames_only: bool = False,
    max_depth: int = 0,
    max_nodes: int = 0,
    timeout_s: float | int = 30,
    max_click_targets: int = 0,
) -> PlanResult:
    """Return a predicted reward-reaching plan from the synthesized planner.

    The synthesizer authors ``planner`` explicitly. There is no engine-side
    search fallback (matching baseline1): an unauthored planner, or one whose
    plan does not reach reward under the model, yields a not-ok result and the
    engine bails to the analyzer rather than running an uninformed search whose
    branching factor does not scale to ARC-3.
    """

    try:
        with _Deadline(timeout_s):
            synth_plan = _call_synth_planner(
                model, start_state, available_actions, max_depth
            )
            if not synth_plan:
                return PlanResult(
                    False, "synth_planner",
                    reason="synth planner returned no plan",
                )
            return validate_plan(
                model,
                start_state,
                synth_plan,
                available_actions,
                source="synth_planner",
                max_depth=max_depth,
            )
    except PlannerTimeout as exc:
        return PlanResult(False, "timeout", reason=str(exc))
    except Exception as exc:
        return PlanResult(False, "model_error",
                          reason=f"{type(exc).__name__}: {exc}")


def _pixel_hash(pixels: Any) -> Any:
    if pixels is None:
        return None
    try:
        return tuple(tuple(int(v) for v in row) for row in pixels)
    except Exception:
        return repr(pixels)


def object_state_signature(
    state: list[dict],
    *,
    scope_tags: set[str] | None = None,
    wall_tags: tuple[str, ...] = ("ihdgageizm",),
) -> dict:
    """Verifier-compatible object-state signature."""

    sig = {}
    for obj in state or []:
        if not isinstance(obj, dict):
            continue
        tags = [str(t) for t in (obj.get("tags") or [])]
        if scope_tags is not None and scope_tags:
            if not any(t in scope_tags for t in tags):
                continue
        elif any(t in tags for t in wall_tags):
            continue
        try:
            if int(obj.get("w", 0)) >= 64:
                continue
            x = int(obj.get("x", 0))
            y = int(obj.get("y", 0))
        except Exception:
            continue
        key = (str(obj.get("name", "?")), x, y)
        sig[key] = (
            x,
            y,
            bool(obj.get("visible", True)),
            int(obj.get("rotation", 0)),
            _pixel_hash(obj.get("pixels")),
        )
    return sig


def object_states_equal(
    predicted: list[dict],
    actual: list[dict],
    *,
    scope_tags: set[str] | None = None,
) -> bool:
    """Return True when object states match under the verifier signature."""

    return object_state_signature(
        predicted, scope_tags=scope_tags
    ) == object_state_signature(actual, scope_tags=scope_tags)
