"""Model-declared fluent harvest and disposal.

The synthesized model already computes the derived predicates and relations
its CPFs condition on. The synthesizer registers them where they are
defined, one decorator line, no second copy:

    FLUENTS = {}
    def fluent(f):
        FLUENTS[f.__name__] = f
        return f

    @fluent
    def is_colorable(o, state):
        return "gOi" in (o.get("tags") or [])

A fluent is a lifted, subject-first function over (object_record, state) or
(object_record,) returning a hashable value. Lifted means its grounding is
unknown until it runs: returning None or raising on an object marks the
fluent NOT APPLICABLE to that object, never invalid, and the discovered
applies-to set is part of the report. Evaluation goes through the imported
module (the same module object the engine loads each step), so intra-module
helpers, caches, and imports work unchanged; module-level mutable state is
already excluded by the verifier's double-run check.

Harvest reads the registry off the module; disposal evaluates every fluent
over the buffer and admits it only where it de-conflates: some mixed action
cell splits into near-deterministic bins (the propose step is synthesis
itself, the dispose step is this check, and the entropy-reduction gate of
the legacy ledger plays no part). Admitted fluents refine the Σ class table
with provenance "model"; bins the fluent produces on observed states but
that carry no engaged evidence are the coverage holes. Everything is
recomputable from (buffer, current code), so each synthesis round
re-harvests from scratch and nothing migrates through code wipes.
"""
from __future__ import annotations

import inspect
import signal
from collections import Counter, defaultdict
from typing import Any, Callable

from .epistemic import _pixels_hash_bucket
from .sigma import (
    _action_cell_key,
    _chebyshev,
    _click_hits,
    _click_xy,
    _effect_signature,
    _pair_before_after,
    _primary_tag,
)

MAX_FLUENTS = 16
MAX_FLUENT_CODOMAIN = 12
FLUENT_EVAL_TIMEOUT_S = 10
MIN_BIN_SUPPORT = 2
ADMIT_MODAL_FRAC = 0.9
# A cell counts as mixed (a de-conflation target) below this concentration.
MIXED_MODAL_FRAC = 0.9
MIN_MIXED_CELL_N = 3


def _value_outcome(bo: dict, ao: dict | None, sig: str) -> str:
    """Disposal outcome symbol: the effect signature refined by value
    DELTAS. The display alphabet certifies WHICH fields changed; disposal
    must also see what they became (a mechanic that always recolors reads
    uniformly "pixels" while the game's logic lives in which color
    results). Deltas rather than absolutes so accumulators stay uniform (a
    +90-per-step rotator is one outcome, not a fresh one each step);
    pixels use the bucket transition, the one field whose exact target
    value carries game logic."""
    if ao is None:
        return sig
    dx = int(ao.get("x", 0)) - int(bo.get("x", 0))
    dy = int(ao.get("y", 0)) - int(bo.get("y", 0))
    drot = (
        int(ao.get("rotation", 0) or 0) - int(bo.get("rotation", 0) or 0)
    )
    dvis = (
        int(bool(ao.get("visible", True)))
        - int(bool(bo.get("visible", True)))
    )
    pb = _pixels_hash_bucket(bo.get("pixels"))
    pa = _pixels_hash_bucket(ao.get("pixels"))
    dpx = f"{pb}>{pa}" if pb != pa else "="
    return f"{sig}|d{dx},{dy}|r{drot}|v{dvis}|px{dpx}"


class _Timeout(Exception):
    pass


def _raise_timeout(_signum, _frame):
    raise _Timeout()


def harvest_fluents(module: Any) -> tuple[dict[str, Callable], list[str]]:
    """Read the FLUENTS registry off an imported model module.

    Absence is not an error (the contract is optional per round); malformed
    entries are, loudly.
    """
    errors: list[str] = []
    reg = getattr(module, "FLUENTS", None)
    if reg is None:
        return {}, errors
    if not isinstance(reg, dict):
        return {}, [f"FLUENTS must be a dict, got {type(reg).__name__}"]
    out: dict[str, Callable] = {}
    for name in sorted(str(k) for k in reg):
        fn = reg[name]
        if not callable(fn):
            errors.append(f"fluent {name}: not callable")
            continue
        try:
            params = [
                p for p in inspect.signature(fn).parameters.values()
                if p.kind in (
                    p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD,
                )
            ]
        except (TypeError, ValueError):
            params = []
        if not params:
            errors.append(
                f"fluent {name}: signature must be (object_record) or "
                "(object_record, state)"
            )
            continue
        if len(out) >= MAX_FLUENTS:
            errors.append(
                f"fluent registry truncated to the first {MAX_FLUENTS}"
            )
            break
        out[name] = _adapt_arity(fn, len(params))
    return out, errors


def _adapt_arity(fn: Callable, n_positional: int) -> Callable:
    """Uniform (object, state) calling convention over arity-1 predicates
    like ``is_hud(o)`` and arity-2 relations like ``governors(o, state)``."""
    if n_positional >= 2:
        return fn

    def unary(obj, _state, _fn=fn):
        return _fn(obj)

    return unary


def _engaged_samples(
    transitions: list[dict],
    committed_features: list[dict] | None,
) -> list[tuple[int, dict, str, str, str, str]]:
    """Engaged (t_idx, before_obj, name, tag, cell, outcome) samples,
    mirroring SigmaState.observe's engagement and per-transition dedup.
    After-only births are skipped: a fluent needs a before-record subject.
    """
    samples: list[tuple[int, dict, str, str, str, str]] = []
    for t_idx, t in enumerate(transitions):
        before = t.get("before_state") or []
        after = t.get("after_state") or []
        action_id = int(t.get("action_id", -1))
        xy = _click_xy(t) if action_id == 6 else None
        pairs = _pair_before_after(before, after)
        sig_of = [(bo, ao, _effect_signature(bo, ao)) for bo, ao in pairs]
        changed = [
            bo if bo is not None else ao
            for bo, ao, sig in sig_of if sig != "no_change"
        ]
        seen: set[tuple[str, str, str]] = set()
        for bo, ao, sig in sig_of:
            if bo is None:
                continue
            name = bo.get("name")
            if not name:
                continue
            name = str(name)
            responded = sig != "no_change"
            targeted = xy is not None and _click_hits(bo, *xy)
            proximal = False
            if not responded and not targeted:
                for ch in changed:
                    if ch is not bo and _chebyshev(bo, ch) <= 1:
                        proximal = True
                        break
            if not (responded or targeted or proximal):
                continue
            cell = _action_cell_key(action_id, t, bo, committed_features)
            outcome = _value_outcome(bo, ao, sig)
            key = (name, cell, outcome)
            if key in seen:
                continue
            seen.add(key)
            samples.append(
                (t_idx, bo, name, _primary_tag(bo), cell, outcome)
            )
    return samples


def dispose_fluents(
    fluents: dict[str, Callable],
    transitions: list[dict],
    *,
    committed_features: list[dict] | None = None,
    alpha_0: float = 1.0,
) -> dict[str, Any]:
    """Evaluate each harvested fluent over the buffer and admit it where it
    de-conflates a mixed cell into near-deterministic bins.

    Admission is per (fluent, tag, cell). Bins the fluent produces on
    observed states of that tag but that have no engaged evidence in the
    cell are reported as holes.
    """
    report: dict[str, Any] = {"fluents": {}, "n_admitted": 0}
    if not fluents or not transitions:
        return report

    samples = _engaged_samples(transitions, committed_features)

    cell_outcomes: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for _t, _bo, _name, tag, cell, outcome in samples:
        cell_outcomes[(tag, cell)][outcome] += 1
    def _is_mixed(ctr: Counter) -> bool:
        n = sum(ctr.values())
        if n < MIN_MIXED_CELL_N or len(ctr) < 2:
            return False
        return ctr.most_common(1)[0][1] / n < MIXED_MODAL_FRAC

    mixed_cells = {tc for tc, ctr in cell_outcomes.items() if _is_mixed(ctr)}

    for fname, fn in fluents.items():
        entry: dict[str, Any] = {
            "status": "rejected", "reason": "", "codomain": {},
            "admitted": [],
        }
        report["fluents"][fname] = entry

        old_handler = None
        alarm_set = False
        try:
            old_handler = signal.signal(signal.SIGALRM, _raise_timeout)
            signal.alarm(FLUENT_EVAL_TIMEOUT_S)
            alarm_set = True
        except Exception:
            alarm_set = False
        # A lifted fluent's grounding is discovered by running it: None or a
        # per-object exception means not-applicable to that object, and only
        # a timeout or an empty applies-to set invalidates the fluent.
        def _apply(obj, state):
            try:
                v = fn(obj, state)
                if v is None:
                    return None, False
                hash(v)
                return v, True
            except _Timeout:
                raise
            except Exception:
                return None, False

        try:
            presence: dict[str, set] = defaultdict(set)
            applies_to: dict[str, int] = defaultdict(int)
            for t in transitions:
                before = t.get("before_state") or []
                for obj in before:
                    if not obj.get("name"):
                        continue
                    v, ok = _apply(obj, before)
                    if ok:
                        presence[_primary_tag(obj)].add(v)
                        applies_to[_primary_tag(obj)] += 1

            bins: dict[tuple[str, str], dict[Any, Counter]] = defaultdict(
                lambda: defaultdict(Counter)
            )
            for t_idx, bo, _name, tag, cell, outcome in samples:
                before = transitions[t_idx].get("before_state") or []
                v, ok = _apply(bo, before)
                if ok:
                    bins[(tag, cell)][v][outcome] += 1
        except _Timeout:
            entry["status"] = "invalid"
            entry["reason"] = f"timed out after {FLUENT_EVAL_TIMEOUT_S}s"
            continue
        finally:
            if alarm_set:
                signal.alarm(0)
                if old_handler is not None:
                    signal.signal(signal.SIGALRM, old_handler)

        if not presence:
            entry["status"] = "invalid"
            entry["reason"] = "applies to no observed object"
            continue
        entry["applies_to"] = dict(sorted(applies_to.items()))
        entry["codomain"] = {
            tag: sorted(map(repr, vals)) for tag, vals in presence.items()
        }
        # Values the fluent produces on observed states that were never
        # exercised under any action: the model's own untested branches,
        # reported for EVERY fluent (admitted or not) as probe hints.
        engaged_vals: dict[str, set] = defaultdict(set)
        for (tag, _cell), by_value in bins.items():
            engaged_vals[tag].update(by_value.keys())
        untested = {
            tag: sorted(
                repr(v) for v in vals - engaged_vals.get(tag, set())
            )
            for tag, vals in presence.items()
            if vals - engaged_vals.get(tag, set())
        }
        if untested:
            entry["untested"] = untested
        oversize = [
            tag for tag, vals in presence.items()
            if len(vals) > MAX_FLUENT_CODOMAIN
        ]
        if oversize:
            entry["reason"] = (
                f"codomain exceeds {MAX_FLUENT_CODOMAIN} on "
                f"{', '.join(sorted(oversize))}"
            )
            continue

        for (tag, cell) in sorted(mixed_cells):
            by_value = bins.get((tag, cell))
            if not by_value or len(by_value) < 2:
                continue
            supported = {
                v: ctr for v, ctr in by_value.items()
                if sum(ctr.values()) >= MIN_BIN_SUPPORT
            }
            if not supported:
                continue
            deterministic = all(
                ctr.most_common(1)[0][1] / sum(ctr.values())
                >= ADMIT_MODAL_FRAC
                for ctr in supported.values()
            )
            if not deterministic:
                continue
            bin_rows = []
            for v, ctr in sorted(by_value.items(), key=lambda kv: repr(kv[0])):
                n = sum(ctr.values())
                bin_rows.append({
                    "value": repr(v),
                    "n": n,
                    "modal_outcome": ctr.most_common(1)[0][0],
                    "modal_frac": round(ctr.most_common(1)[0][1] / n, 4),
                    # Laplace-style bin uncertainty: the global outcome
                    # alphabet is huge, so Dirichlet entropy would keep a
                    # deterministic 3-sample bin pinned near the prior.
                    "U": round(1.0 - ctr.most_common(1)[0][1] / (n + 1), 6),
                })
            holes = sorted(
                repr(v) for v in presence.get(tag, set())
                if v not in by_value
            )
            entry["admitted"].append({
                "tag": tag,
                "cell": cell,
                "bins": bin_rows,
                "holes": holes,
            })
        if entry["admitted"]:
            entry["status"] = "admitted"
            report["n_admitted"] += 1
        elif not entry["reason"]:
            entry["reason"] = "de-conflates no mixed cell"
    return report


def harvest_and_dispose(
    module: Any,
    transitions: list[dict],
    *,
    committed_features: list[dict] | None = None,
    alpha_0: float = 1.0,
) -> dict[str, Any]:
    """One-call entry: registry off the module, disposal against the buffer."""
    fluents, errors = harvest_fluents(module)
    report = dispose_fluents(
        fluents, transitions,
        committed_features=committed_features, alpha_0=alpha_0,
    )
    report["harvest_errors"] = errors
    report["n_registered"] = len(fluents)
    return report
