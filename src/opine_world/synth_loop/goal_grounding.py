"""Goal-requirement grounding: localize coverage gaps named by the goal hypothesis.

The LLM proposes each lifted requirement of the current natural-language goal
hypothesis as a small executable Boolean predicate over object states. It
never checks constraints itself. Two mechanical buffer scans dispose each
predicate:

1. Necessity: the predicate must hold at, or immediately before, every
   historical reward step across all levels. Violators are rejected. This is
   also the mechanical enforcement of ARC-3 goal continuity, since a revised
   hypothesis must still explain every previous level's rewards.
2. Satisfiability: a surviving predicate satisfied by no observed state names
   a localized coverage hole and yields a goal-stratum mode class on the
   object it concerns.

Until reward fires on the current level, a nonzero coverage gap provably
exists because the rewarding state is outside the buffer by definition.
Grounded requirements localize that gap onto named objects.
"""
from __future__ import annotations

import json
import signal
from typing import Any, Callable

MAX_REQUIREMENTS = 6

# Wall-clock budget for compiling and scanning ONE predicate over the whole
# buffer. A predicate that cannot finish is invalid, not slow.
REQUIREMENT_TIMEOUT_S = 5

STATUS_ACCEPTED = "accepted"
STATUS_REJECTED_NECESSITY = "rejected_necessity"
STATUS_INVALID = "invalid"


class _Timeout(Exception):
    pass


def _raise_timeout(*_args):
    raise _Timeout()


def parse_raw_requirements(raw: Any) -> tuple[list[dict], list[str]]:
    """Validate the LLM-written payload. Returns (requirements, errors)."""
    errors: list[str] = []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception as exc:
            return [], [f"payload is not JSON: {exc}"]
    if not isinstance(raw, dict):
        return [], [f"payload must be a dict, got {type(raw).__name__}"]
    entries = raw.get("requirements")
    if not isinstance(entries, list):
        return [], ["payload has no 'requirements' list"]

    out: list[dict] = []
    seen_ids: set[str] = set()
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            errors.append(f"requirement {i} is not a dict")
            continue
        rid = str(entry.get("id", "") or "").strip()
        obj = str(entry.get("object", "") or "").strip()
        mode = str(entry.get("mode", "") or "").strip()
        src = entry.get("predicate_src")
        if not rid or rid in seen_ids:
            rid = next(
                f"p{k}" for k in range(1, len(entries) + len(out) + 2)
                if f"p{k}" not in seen_ids
            )
            errors.append(
                f"requirement {i}: missing or duplicate id, assigned {rid}"
            )
        if not obj or not mode:
            errors.append(f"requirement {rid}: missing object or mode")
            continue
        if not isinstance(src, str) or "def predicate" not in src:
            errors.append(
                f"requirement {rid}: predicate_src must define predicate(state)"
            )
            continue
        seen_ids.add(rid)
        out.append({
            "id": rid, "object": obj, "mode": mode, "predicate_src": src,
        })
        if len(out) >= MAX_REQUIREMENTS:
            if len(entries) > MAX_REQUIREMENTS:
                errors.append(
                    f"truncated to the first {MAX_REQUIREMENTS} requirements"
                )
            break
    return out, errors


def _compile_predicate(src: str) -> tuple[Callable | None, str | None]:
    namespace: dict[str, Any] = {}
    try:
        exec(src, namespace)
    except Exception as exc:
        return None, f"compile failed: {type(exc).__name__}: {exc}"
    fn = namespace.get("predicate")
    if not callable(fn):
        return None, "predicate_src did not define a callable predicate"
    return fn, None


def _states_of(transition: dict) -> list[tuple[list[dict], int, bool]]:
    """States of a transition as (state, level, at_reward_moment). A
    reward-firing transition's after-state belongs to the NEXT level
    (ARC-3 reward means level advance), which is what makes per-level
    satisfaction well defined at level boundaries.

    Reward transitions also carry mid-animation tick states, attributed to
    the completing level. The completing move and the level sweep are one
    transition, so the goal-completed configuration exists only in those
    ticks. The necessity scan accepts only reward-moment states (before
    and ticks): the after-state of a reward transition is the next level's
    entry, and counting it lets predicates about next-level objects pass
    as goal requirements."""
    level = int(transition.get("level", 0) or 0)
    advanced = float(transition.get("reward", 0.0) or 0.0) > 0.0
    out = []
    before = transition.get("before_state")
    if before:
        out.append((before, level, True))
    for tick_state in transition.get("intermediate_states") or []:
        if tick_state:
            out.append((tick_state, level, True))
    after = transition.get("after_state")
    if after:
        out.append((after, level + 1 if advanced else level, not advanced))
    return out


def _scan_requirement(
    fn: Callable, transitions: list[dict], current_level: int | None,
) -> dict[str, Any]:
    necessity_ok = True
    necessity_reason = ""
    satisfied_step = None
    satisfied_in_current = False

    for t in transitions:
        step = t.get("timestep")
        holds_at_reward_moment = False
        for state, level, at_reward_moment in _states_of(t):
            if bool(fn(state)):
                if at_reward_moment:
                    holds_at_reward_moment = True
                if satisfied_step is None:
                    satisfied_step = step
                if current_level is not None and level == current_level:
                    satisfied_in_current = True
        if (
            float(t.get("reward", 0.0) or 0.0) > 0.0
            and not holds_at_reward_moment
        ):
            necessity_ok = False
            necessity_reason = (
                f"does not hold at reward step {step}"
            )
            break

    return {
        "necessity_ok": necessity_ok,
        "necessity_reason": necessity_reason,
        "satisfied_in_buffer": satisfied_step is not None,
        "first_satisfied_step": satisfied_step,
        "satisfied_in_current_level": satisfied_in_current,
    }


def ground_requirements(
    requirements: list[dict],
    transitions: list[dict],
    *,
    current_level: int | None = None,
) -> dict[str, Any]:
    """Dispose each requirement against the buffer. Purely mechanical.

    Necessity is checked cross-level (goal continuity). Satisfaction is
    tracked both globally and for ``current_level``: a level-agnostic
    completion condition is satisfied somewhere as soon as any level has
    been solved, so the coverage hole is its non-satisfaction on the level
    being played now.
    """
    results: list[dict] = []
    for req in requirements:
        fn, err = _compile_predicate(req["predicate_src"])
        record = {
            "id": req["id"],
            "object": req["object"],
            "mode": req["mode"],
            "predicate_src": req["predicate_src"],
        }
        if fn is None:
            record.update({
                "status": STATUS_INVALID,
                "reason": err,
                "satisfied_in_buffer": False,
                "first_satisfied_step": None,
                "satisfied_in_current_level": False,
            })
            results.append(record)
            continue

        old_handler = None
        alarm_set = False
        try:
            old_handler = signal.signal(signal.SIGALRM, _raise_timeout)
            signal.alarm(REQUIREMENT_TIMEOUT_S)
            alarm_set = True
        except Exception:
            alarm_set = False
        try:
            scan = _scan_requirement(fn, transitions, current_level)
        except _Timeout:
            record.update({
                "status": STATUS_INVALID,
                "reason": f"timed out after {REQUIREMENT_TIMEOUT_S}s",
                "satisfied_in_buffer": False,
                "first_satisfied_step": None,
                "satisfied_in_current_level": False,
            })
            results.append(record)
            continue
        except Exception as exc:
            record.update({
                "status": STATUS_INVALID,
                "reason": f"raised {type(exc).__name__}: {exc}",
                "satisfied_in_buffer": False,
                "first_satisfied_step": None,
                "satisfied_in_current_level": False,
            })
            results.append(record)
            continue
        finally:
            if alarm_set:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

        if scan["necessity_ok"]:
            record["status"] = STATUS_ACCEPTED
            record["reason"] = ""
        else:
            record["status"] = STATUS_REJECTED_NECESSITY
            record["reason"] = scan["necessity_reason"]
        record["satisfied_in_buffer"] = scan["satisfied_in_buffer"]
        record["first_satisfied_step"] = scan["first_satisfied_step"]
        record["satisfied_in_current_level"] = scan[
            "satisfied_in_current_level"
        ]
        results.append(record)

    n_reward_steps = sum(
        1 for t in transitions
        if float(t.get("reward", 0.0) or 0.0) > 0.0
    )
    return {
        "requirements": results,
        "current_level": current_level,
        "n_transitions": len(transitions),
        "n_reward_steps": n_reward_steps,
        "n_accepted": sum(
            1 for r in results if r["status"] == STATUS_ACCEPTED
        ),
        "n_rejected": sum(
            1 for r in results if r["status"] == STATUS_REJECTED_NECESSITY
        ),
        "n_invalid": sum(
            1 for r in results if r["status"] == STATUS_INVALID
        ),
    }


def goal_mode_injections(grounded: dict) -> list[dict]:
    """Accepted requirements not yet satisfied on the current level are the
    localized coverage holes. Falls back to global non-satisfaction when no
    current level was given."""
    per_level = grounded.get("current_level") is not None
    out = []
    for r in grounded.get("requirements", []):
        if r["status"] != STATUS_ACCEPTED:
            continue
        satisfied = (
            r["satisfied_in_current_level"] if per_level
            else r["satisfied_in_buffer"]
        )
        if not satisfied:
            out.append({
                "object": r["object"],
                "mode": r["mode"],
                "predicate_id": r["id"],
            })
    return out
