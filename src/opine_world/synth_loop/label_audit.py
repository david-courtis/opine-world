"""Mechanical audit of the sprite role labels against the replay buffer.

A semantic label can never be verified mechanically, but every label carries
behavioral commitments and commitments are checkable. The audit grades every
flag by evidence strength, and only the deductive grade triggers automatic
disposal:

  refuted             a decorative label whose inertness commitment is
                      falsified by a cited transition (deductive, auto-retired
                      by the engine)
  contradictions      two same-label tags behave differently in a shared
                      (action, context) stratum, both near-deterministic at
                      exact-delta granularity. Proves the disjunction
                      "mislabel OR missing context feature" and names both
                      disposal routes without picking one
  anomalies           correlational flags: role-posterior disagreement,
                      controllability mismatch. Never auto-acted-on
  unlabeled_relevant  tags with reward-adjacent or action-influence evidence
                      and no committed role. A statement of absence
  merge_compatible    differently-labeled pairs indistinguishable so far,
                      with sample support attached. The weakest grade

Grade A and B flags cite witness timesteps so every claim is independently
re-derivable from the buffer. Labels feed only rel(o) and prompt annotations,
never eta, C, coverage, or the verifier, so a wrong label misdirects
exploration priority at worst.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .aliases import best_alias, is_decorative
from .epistemic import (
    _pair_before_after,
    _pixels_hash_bucket,
    compute_role_posterior,
)
from .sigma import _action_cell_key, _outcome_hash, _primary_tag

_FULL_INERT_HEADS = frozenset({
    "wall", "scenery", "decoration", "decorative", "border", "tile",
    "floor", "background",
})
_POSITION_INERT_HEADS = frozenset({"hud"})
_CONTROLLABLE_HEADS = frozenset({
    "player", "cursor", "avatar", "agent", "controller",
})

DEFAULT_N_MIN = 3
DEFAULT_MODAL_FRAC_MIN = 0.9
DEFAULT_POSTERIOR_MARGIN = 0.5
DEFAULT_CAI_RELEVANT = 0.3
DEFAULT_CAI_CONTROLLABLE = 0.1


def _head(alias: str) -> str:
    return alias.strip().lower().split("_", 1)[0]


def _committed_labels(aliases: dict | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for tag in aliases or {}:
        top = best_alias(aliases, tag)
        if top:
            out[str(tag)] = top
    return out


def _record_delta(bo: dict, ao: dict) -> dict[str, int]:
    return {
        "dx": int(ao.get("x", 0)) - int(bo.get("x", 0)),
        "dy": int(ao.get("y", 0)) - int(bo.get("y", 0)),
        "dvis": int(
            bool(ao.get("visible", True)) != bool(bo.get("visible", True))
        ),
        "drot": (
            int(ao.get("rotation", 0) or 0) - int(bo.get("rotation", 0) or 0)
        ),
        "dpx": int(
            _pixels_hash_bucket(ao.get("pixels"))
            != _pixels_hash_bucket(bo.get("pixels"))
        ),
    }


def _refuted_commitments(
    transitions: list[dict], committed: dict[str, str],
) -> list[dict]:
    """Grade A. gone and born pairings are excluded as falsifiers because the
    pairing heuristic can fabricate them."""
    commitments: dict[str, tuple[str, str]] = {}
    for tag, alias in committed.items():
        head = _head(alias)
        if head in _FULL_INERT_HEADS:
            commitments[tag] = (alias, "inert")
        elif head in _POSITION_INERT_HEADS:
            commitments[tag] = (alias, "position_inert")
    if not commitments:
        return []

    refuted: dict[str, dict] = {}
    for t in transitions:
        before = t.get("before_state") or []
        after = t.get("after_state") or []
        for bo, ao in _pair_before_after(before, after)[: len(before)]:
            if bo is None or ao is None:
                continue
            tag = _primary_tag(bo)
            if tag not in commitments or tag in refuted:
                continue
            alias, kind = commitments[tag]
            d = _record_delta(bo, ao)
            if kind == "inert":
                violated = any(v != 0 for v in d.values())
            else:
                violated = d["dx"] != 0 or d["dy"] != 0
            if violated:
                refuted[tag] = {
                    "tag": tag,
                    "alias": alias,
                    "commitment": kind,
                    "witness_timestep": t.get("timestep"),
                    "delta": d,
                }
    return [refuted[tag] for tag in sorted(refuted)]


def _delta_strata(
    transitions: list[dict], committed_features: list[dict] | None,
) -> dict[str, dict[str, dict]]:
    """tag -> cell key -> {counter over exact-delta outcomes, first witness
    timestep per outcome}. Deduped per (tag, cell, delta) per transition."""
    strata: dict[str, dict[str, dict]] = defaultdict(dict)
    for t in transitions:
        before = t.get("before_state") or []
        after = t.get("after_state") or []
        seen: set[tuple[str, str, str]] = set()
        for bo, ao in _pair_before_after(before, after)[: len(before)]:
            if bo is None:
                continue
            name = bo.get("name")
            if not name:
                continue
            tag = _primary_tag(bo)
            cell = _action_cell_key(
                int(t.get("action_id", -1)), t, bo, committed_features,
            )
            delta = _outcome_hash(bo, ao)
            key = (tag, cell, delta)
            if key in seen:
                continue
            seen.add(key)
            rec = strata[tag].setdefault(
                cell, {"counter": Counter(), "first_ts": {}},
            )
            rec["counter"][delta] += 1
            rec["first_ts"].setdefault(delta, t.get("timestep"))
    return strata


def _deterministic_modal(
    rec: dict, n_min: int, modal_frac_min: float,
) -> tuple[str, int] | None:
    n = sum(rec["counter"].values())
    if n < n_min:
        return None
    modal, count = rec["counter"].most_common(1)[0]
    if count / n < modal_frac_min:
        return None
    return modal, n


def _pair_contradictions(
    tag_a: str, tag_b: str, strata: dict,
    n_min: int, modal_frac_min: float,
) -> tuple[list[dict], int]:
    """Shared strata where both tags are near-deterministic. Returns the
    contradicting cells and the count of cells where the modal deltas AGREE
    (the shared-support figure Grade E reports)."""
    contradictions: list[dict] = []
    agreements = 0
    cells = set(strata.get(tag_a, {})) & set(strata.get(tag_b, {}))
    for cell in sorted(cells):
        ra, rb = strata[tag_a][cell], strata[tag_b][cell]
        da = _deterministic_modal(ra, n_min, modal_frac_min)
        db = _deterministic_modal(rb, n_min, modal_frac_min)
        if da is None or db is None:
            continue
        (modal_a, n_a), (modal_b, n_b) = da, db
        if modal_a == modal_b:
            agreements += 1
            continue
        contradictions.append({
            "cell": cell,
            "modal_a": modal_a,
            "modal_b": modal_b,
            "n_a": n_a,
            "n_b": n_b,
            "witness_a": ra["first_ts"].get(modal_a),
            "witness_b": rb["first_ts"].get(modal_b),
        })
    return contradictions, agreements


def _label_contradictions(
    committed: dict[str, str], strata: dict,
    n_min: int, modal_frac_min: float,
) -> list[dict]:
    """Grade B: same-label tag pairs that are not behaviorally exchangeable
    under the current context vocabulary."""
    by_label: dict[str, list[str]] = defaultdict(list)
    for tag, alias in committed.items():
        by_label[alias].append(tag)

    out: list[dict] = []
    for label, tags in sorted(by_label.items()):
        if len(tags) < 2:
            continue
        tags = sorted(tags)
        for i, tag_a in enumerate(tags):
            for tag_b in tags[i + 1:]:
                cells, _ = _pair_contradictions(
                    tag_a, tag_b, strata, n_min, modal_frac_min,
                )
                if cells:
                    out.append({
                        "label": label,
                        "tag_a": tag_a,
                        "tag_b": tag_b,
                        "cells": cells,
                        "disposal_options": [
                            "relabel one tag",
                            "propose a distinguishing context feature",
                        ],
                    })
    return out


def _anomalies(
    transitions: list[dict], aliases: dict | None,
    committed: dict[str, str], cai: dict[str, float] | None,
    tag_of_name: dict[str, str],
    posterior_margin: float, cai_controllable: float,
) -> list[dict]:
    """Grade C. Correlational flags only, never auto-acted-on."""
    out: list[dict] = []

    role = compute_role_posterior(transitions, aliases)
    for tag, info in sorted(role.get("per_tag", {}).items()):
        voted = committed.get(tag)
        if not voted:
            continue
        post = info.get("posterior") or {}
        map_role = info.get("map_role")
        if not map_role or map_role == voted:
            continue
        margin = float(info.get("map_score", 0.0)) - float(
            post.get(voted, 0.0)
        )
        if margin >= posterior_margin:
            out.append({
                "kind": "role_posterior",
                "tag": tag,
                "voted": voted,
                "map_role": map_role,
                "margin": round(margin, 4),
            })

    if cai:
        cai_by_tag: dict[str, float] = {}
        for name, value in cai.items():
            tag = tag_of_name.get(name)
            if tag:
                cai_by_tag[tag] = max(cai_by_tag.get(tag, 0.0), float(value))
        for tag, alias in sorted(committed.items()):
            if _head(alias) not in _CONTROLLABLE_HEADS:
                continue
            if tag in cai_by_tag and cai_by_tag[tag] < cai_controllable:
                out.append({
                    "kind": "controllability",
                    "tag": tag,
                    "alias": alias,
                    "cai": round(cai_by_tag[tag], 4),
                })
    return out


def _delta_changed_tags(transition: dict) -> set[str]:
    tags: set[str] = set()
    before = transition.get("before_state") or []
    after = transition.get("after_state") or []
    for bo, ao in _pair_before_after(before, after)[: len(before)]:
        if bo is None or ao is None or not bo.get("name"):
            continue
        if any(v != 0 for v in _record_delta(bo, ao).values()):
            tags.add(_primary_tag(bo))
    return tags


def _reward_changed_tags(transitions: list[dict]) -> dict[str, int]:
    """Attribute-delta changes only, never gone or born. The ARC-3 reward
    transition is also the level sweep where every object reads gone or
    born, so the completing move is looked up in the previous same-level
    transition."""
    out: dict[str, int] = {}
    for i, t in enumerate(transitions):
        if float(t.get("reward", 0.0) or 0.0) <= 0.0:
            continue
        ts = t.get("timestep")
        tags = _delta_changed_tags(t)
        if i > 0 and transitions[i - 1].get("level") == t.get("level"):
            tags |= _delta_changed_tags(transitions[i - 1])
        for tag in tags:
            out.setdefault(tag, ts)
    return out


def _unlabeled_relevant(
    transitions: list[dict], committed: dict[str, str],
    cai: dict[str, float] | None, tag_of_name: dict[str, str],
    cai_relevant: float,
) -> list[dict]:
    """Grade D. A statement of absence over the label table."""
    out: dict[str, dict] = {}
    for tag, ts in _reward_changed_tags(transitions).items():
        if tag not in committed:
            out[tag] = {
                "tag": tag,
                "evidence": f"changed at reward step {ts}",
            }
    for name, value in (cai or {}).items():
        tag = tag_of_name.get(name)
        if not tag or tag in committed or tag in out:
            continue
        if float(value) >= cai_relevant:
            out[tag] = {
                "tag": tag,
                "evidence": f"action influence {round(float(value), 3)}",
            }
    return [out[tag] for tag in sorted(out)]


def _parse_merge_candidates(
    ontology_latest: dict | None,
) -> list[tuple[str, str, float]]:
    if not ontology_latest:
        return []
    flat = float(ontology_latest.get("flat_eta", 0.0) or 0.0)
    out: list[tuple[str, str, float]] = []
    for cand in ontology_latest.get("candidates") or []:
        label = str(cand.get("candidate", ""))
        if not (label.startswith("merge[") and label.endswith("]")):
            continue
        body = label[len("merge["):-1].rsplit("|", 1)[0]
        parts = body.split("+")
        if len(parts) != 2:
            continue
        reduction = flat - float(cand.get("eta", flat))
        out.append((parts[0], parts[1], round(reduction, 6)))
    return out


def _merge_compatible(
    committed: dict[str, str], strata: dict,
    ontology_latest: dict | None, n_min: int, modal_frac_min: float,
) -> list[dict]:
    """Grade E. Indistinguishability is never provable from finite data, so
    entries carry their support and are worded as compatible-so-far."""
    out: list[dict] = []
    for tag_a, tag_b, reduction in _parse_merge_candidates(ontology_latest):
        if reduction <= 0:
            continue
        label_a, label_b = committed.get(tag_a), committed.get(tag_b)
        if not label_a or not label_b or label_a == label_b:
            continue
        contradictions, agreements = _pair_contradictions(
            tag_a, tag_b, strata, n_min, modal_frac_min,
        )
        if contradictions or agreements == 0:
            continue
        out.append({
            "tag_a": tag_a,
            "tag_b": tag_b,
            "label_a": label_a,
            "label_b": label_b,
            "eta_reduction": reduction,
            "agreeing_deterministic_strata": agreements,
        })
    return out


def label_audit(
    transitions: list[dict],
    *,
    aliases: dict | None,
    committed_features: list[dict] | None = None,
    cai: dict[str, float] | None = None,
    ontology_latest: dict | None = None,
    n_min: int = DEFAULT_N_MIN,
    modal_frac_min: float = DEFAULT_MODAL_FRAC_MIN,
    posterior_margin: float = DEFAULT_POSTERIOR_MARGIN,
    cai_relevant: float = DEFAULT_CAI_RELEVANT,
    cai_controllable: float = DEFAULT_CAI_CONTROLLABLE,
) -> dict[str, Any]:
    committed = _committed_labels(aliases)
    committed_nondecorative = {
        tag: alias for tag, alias in committed.items()
        if not is_decorative(alias)
    }
    strata = _delta_strata(transitions, committed_features)

    tag_of_name: dict[str, str] = {}
    for t in transitions:
        for o in (t.get("before_state") or []) + (t.get("after_state") or []):
            name = o.get("name")
            if name and str(name) not in tag_of_name:
                tag_of_name[str(name)] = _primary_tag(o)

    return {
        "refuted": _refuted_commitments(transitions, committed),
        "contradictions": _label_contradictions(
            committed_nondecorative, strata, n_min, modal_frac_min,
        ),
        "anomalies": _anomalies(
            transitions, aliases, committed_nondecorative, cai,
            tag_of_name, posterior_margin, cai_controllable,
        ),
        "unlabeled_relevant": _unlabeled_relevant(
            transitions, committed, cai, tag_of_name, cai_relevant,
        ),
        "merge_compatible": _merge_compatible(
            committed_nondecorative, strata, ontology_latest,
            n_min, modal_frac_min,
        ),
        "n_transitions": len(transitions),
        "n_committed_labels": len(committed),
        "params": {
            "n_min": n_min,
            "modal_frac_min": modal_frac_min,
            "posterior_margin": posterior_margin,
            "cai_relevant": cai_relevant,
            "cai_controllable": cai_controllable,
        },
    }
