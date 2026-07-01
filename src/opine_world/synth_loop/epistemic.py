"""Per-(type, action) epistemic matrix computed from the replay buffer.

Implements the heuristic priority and the Bayesian Beta posterior with UCB
and Thompson sampling. It is recomputed from the replay buffer at dump time,
with no separate accumulation required.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


_EFFECT_SKIP_FIELDS = frozenset({
    "w", "h",
    "display_x", "display_y", "display_w", "display_h",
    "name", "tags",
})

_VALID_SORT_KEYS = ("thompson", "ucb", "heuristic")

MATRIX_DEPTH_BINARY = "binary"
MATRIX_DEPTH_BETA = "beta"
MATRIX_DEPTH_DIRICHLET = "dirichlet"
VALID_MATRIX_DEPTHS = (
    MATRIX_DEPTH_BINARY, MATRIX_DEPTH_BETA, MATRIX_DEPTH_DIRICHLET,
)
_BINARY_DET_THRESHOLD_N = 3
_BINARY_DET_C_THRESHOLD = 0.95


def _project_to_depth(cell: dict, depth: str) -> dict:
    """Project a full Dirichlet-depth cell to the requested depth, keeping
    only the fields visible at that depth."""
    if depth == MATRIX_DEPTH_BINARY:
        n = cell["n"]
        c = cell["c"]
        if n < _BINARY_DET_THRESHOLD_N:
            ternary = "UK"
        elif c >= _BINARY_DET_C_THRESHOLD:
            ternary = "KT"
        else:
            ternary = "KF"
        return {
            "type": cell["type"],
            "action_id": cell["action_id"],
            "n": n,
            "ternary": ternary,
        }
    if depth == MATRIX_DEPTH_BETA:
        return {
            "type": cell["type"],
            "action_id": cell["action_id"],
            "n": cell["n"],
            "mu": cell["mu"],
            "sigma": cell["sigma"],
            "priority_ucb": cell["priority_ucb"],
            "priority_thompson": cell["priority_thompson"],
            "majority_effect": cell["majority_effect"],
        }
    if depth == MATRIX_DEPTH_DIRICHLET:
        return cell
    raise ValueError(
        f"unknown matrix depth {depth!r}; expected one of "
        f"{VALID_MATRIX_DEPTHS}"
    )


def _object_type(obj: dict) -> str:
    t = obj.get("type")
    if t:
        return str(t)
    tags = obj.get("tags") or []
    if tags:
        return str(tags[0])
    return str(obj.get("name") or "Unknown")


def _effect_signature(before_obj: dict | None, after_obj: dict | None) -> str:
    """Sorted, comma-joined names of changed sprite-own fields (excluding _EFFECT_SKIP_FIELDS)."""
    if before_obj is None:
        return "born"
    if after_obj is None:
        return "gone"
    keys = set(before_obj.keys()) | set(after_obj.keys())
    changed = sorted(
        k for k in keys
        if k not in _EFFECT_SKIP_FIELDS
        and before_obj.get(k) != after_obj.get(k)
    )
    if not changed:
        return "no_change"
    return ",".join(changed)


def _pair_before_after(
    before: list[dict], after: list[dict],
) -> list[tuple[dict, dict | None]]:
    """Two-pass greedy pairing. Pass 1 matches exact (name, x, y). Pass 2
    matches any remaining same-name after-object to handle moved sprites.

    Sprite names are not unique within a state, so a simple dict keyed by
    name would collapse multiple instances. Before-objects with no match
    map to None (caller treats as "gone"). After-objects with no match are
    appended as (None, after_obj) births (caller treats as "born").
    """
    after_by_name: dict[str, list[dict]] = defaultdict(list)
    for ao in after:
        name = ao.get("name")
        if name is None:
            continue
        after_by_name[name].append(ao)
    consumed: set[int] = set()
    pairs: list[tuple[dict, dict | None]] = []

    deferred: list[int] = []
    for i, bo in enumerate(before):
        name = bo.get("name")
        if name is None or name not in after_by_name:
            pairs.append((bo, None))
            continue
        bx, by = bo.get("x"), bo.get("y")
        match = None
        for ao in after_by_name[name]:
            if id(ao) in consumed:
                continue
            if ao.get("x") == bx and ao.get("y") == by:
                match = ao
                consumed.add(id(ao))
                break
        if match is not None:
            pairs.append((bo, match))
        else:
            pairs.append((bo, None))
            deferred.append(len(pairs) - 1)

    for idx in deferred:
        bo, _ = pairs[idx]
        name = bo.get("name")
        for ao in after_by_name.get(name, ()):
            if id(ao) not in consumed:
                consumed.add(id(ao))
                pairs[idx] = (bo, ao)
                break

    for ao in after:
        if ao.get("name") is None:
            continue
        if id(ao) not in consumed:
            pairs.append((None, ao))

    return pairs


def _context_signature(before_state: list[dict], obj: dict) -> str:
    ox, oy = obj.get("x", 0), obj.get("y", 0)
    name = obj.get("name")
    neighbours: list[str] = []
    for o in before_state:
        if o.get("name") == name:
            continue
        if not o.get("visible", True):
            continue
        x, y = o.get("x", 0), o.get("y", 0)
        if max(abs(x - ox), abs(y - oy)) <= 1:
            neighbours.append(_object_type(o))
    return ",".join(sorted(neighbours))


def _augmented_context_signature(
    transition: dict,
    target_obj: dict,
    committed_features: list[dict] | None,
) -> str:
    """Base context signature augmented with committed feature values (operative ξ).
    Degrades to the base context signature when committed_features is empty.
    """
    before = transition.get("before_state") or []
    base = _context_signature(before, target_obj)
    if not committed_features:
        return base
    parts = [base]
    for feat in committed_features:
        parts.append(_xi_feature_value(feat, transition, target_obj))
    return "|".join(parts)


def compute_epistemic_matrix(
    replay_buffer: list[dict],
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    *,
    alpha_0: float = 1.0,
    beta_0: float = 1.0,
    kappa: float = 2.0,
    sort_by: str = "thompson",
    rng: Any = None,
) -> dict[str, Any]:
    """Aggregate per-(type, action) cells from the replay buffer into the
    epistemic matrix, with heuristic and Bayesian priority columns."""
    if sort_by not in _VALID_SORT_KEYS:
        raise ValueError(
            f"sort_by must be one of {_VALID_SORT_KEYS}, got {sort_by!r}"
        )

    w1, w2, w3 = weights
    cells: dict[tuple[str, int], dict[str, Any]] = defaultdict(
        lambda: {"effects": Counter(), "contexts": set()}
    )

    for t in replay_buffer:
        action_id = t["action_id"]
        before = t.get("before_state") or []
        after = t.get("after_state") or []
        for bo, ao in _pair_before_after(before, after):
            ref = bo if bo is not None else ao
            name = ref.get("name")
            if not name:
                continue
            ty = _object_type(ref)
            if not ty or ty == "Unknown":
                continue
            sig = _effect_signature(bo, ao)
            ctx = _context_signature(before, ref)
            cell = cells[(ty, action_id)]
            cell["effects"][sig] += 1
            cell["contexts"].add(ctx)

    out_cells: list[dict[str, Any]] = []
    for (ty, aid), data in cells.items():
        effects: Counter = data["effects"]
        n = sum(effects.values())
        if n == 0:
            continue
        majority_effect, majority_count = effects.most_common(1)[0]
        c = majority_count / n
        d = len(data["contexts"])
        cond = 1 if c < 1.0 else 0

        pi = w1 / (1 + n) + w2 / (1 + d) + w3 * (1 - c)

        s = majority_count
        alpha = alpha_0 + s
        beta_post = beta_0 + n - s
        ab = alpha + beta_post
        mu = alpha / ab if ab > 0 else 0.0
        denom = (ab * ab) * (ab + 1)
        sigma = math.sqrt((alpha * beta_post) / denom) if denom > 0 else 0.0

        pi_ucb = (1.0 - mu) + kappa * sigma

        if rng is not None and hasattr(rng, "beta"):
            try:
                p_tilde = float(rng.beta(alpha, beta_post))
            except Exception:
                p_tilde = mu
        else:
            p_tilde = mu
        pi_thompson = 1.0 - p_tilde

        out_cells.append({
            "type": ty,
            "action_id": aid,
            "n": n,
            "d": d,
            "c": round(c, 4),
            "cond": cond,
            "priority": round(pi, 4),
            "alpha": round(alpha, 4),
            "beta": round(beta_post, 4),
            "mu": round(mu, 4),
            "sigma": round(sigma, 4),
            "priority_ucb": round(pi_ucb, 4),
            "priority_thompson": round(pi_thompson, 4),
            "majority_effect": majority_effect,
            "effects": dict(effects),
            "contexts_seen": sorted(data["contexts"])[:20],
            "n_distinct_contexts": d,
        })

    sort_key = {
        "heuristic": "priority",
        "ucb": "priority_ucb",
        "thompson": "priority_thompson",
    }[sort_by]
    out_cells.sort(key=lambda c_: -c_[sort_key])

    types_seen = sorted({c_["type"] for c_ in out_cells})
    actions_seen = sorted({c_["action_id"] for c_ in out_cells})

    return {
        "weights": {"w1": w1, "w2": w2, "w3": w3},
        "bayes": {
            "alpha_0": alpha_0,
            "beta_0": beta_0,
            "kappa": kappa,
        },
        "sort_by": sort_by,
        "n_transitions": len(replay_buffer),
        "n_cells": len(out_cells),
        "types_seen": types_seen,
        "actions_seen": actions_seen,
        "cells": out_cells,
    }


def dump_epistemic_matrix(
    replay_buffer: list[dict],
    path: Path,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    *,
    alpha_0: float = 1.0,
    beta_0: float = 1.0,
    kappa: float = 2.0,
    sort_by: str = "thompson",
    rng: Any = None,
    depth: str = MATRIX_DEPTH_DIRICHLET,
) -> None:
    """Compute and write the matrix to path as JSON. depth controls posterior
    granularity: "binary" (KT/KF/UK), "beta" (mu/sigma), or "dirichlet" (full counts).
    """
    if depth not in VALID_MATRIX_DEPTHS:
        raise ValueError(
            f"depth must be one of {VALID_MATRIX_DEPTHS}, got {depth!r}"
        )
    matrix = compute_epistemic_matrix(
        replay_buffer,
        weights=weights,
        alpha_0=alpha_0,
        beta_0=beta_0,
        kappa=kappa,
        sort_by=sort_by,
        rng=rng,
    )
    matrix["depth"] = depth
    if depth != MATRIX_DEPTH_DIRICHLET:
        matrix["cells"] = [
            _project_to_depth(c, depth) for c in matrix["cells"]
        ]
        if depth == MATRIX_DEPTH_BINARY:
            order = {"UK": 0, "KF": 1, "KT": 2}
            matrix["cells"].sort(
                key=lambda c_: (order.get(c_["ternary"], 99), c_["n"])
            )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(matrix, indent=2, default=str))


_NULL_EFFECTS = ("no_change", "gone", "born")


def _stratify(
    replay_buffer: list[dict], type_of: Any,
    committed_features: list[dict] | None = None,
) -> dict[tuple[str, int, str], Counter]:
    """Accumulate per-(type, action, context) effect Counters.

    Per-transition dedup: each unique (type, action, context, effect_sig)
    counts at most once per transition to avoid instance-multiplicity bias.
    """
    strata: dict[tuple[str, int, str], Counter] = defaultdict(Counter)
    for t in replay_buffer:
        action_id = t.get("action_id")
        before = t.get("before_state") or []
        after = t.get("after_state") or []
        seen: set[tuple] = set()
        for bo, ao in _pair_before_after(before, after):
            ref = bo if bo is not None else ao
            name = ref.get("name")
            if not name:
                continue
            ty = type_of(ref)
            if not ty or ty == "Unknown":
                continue
            sig = _effect_signature(bo, ao)
            ctx = _augmented_context_signature(t, ref, committed_features)
            key = (str(ty), action_id, ctx, sig)
            if key in seen:
                continue
            seen.add(key)
            strata[(str(ty), action_id, ctx)][sig] += 1
    return strata


def _composite_signature(before_objs: list[dict],
                         after_objs: list[dict]) -> str:
    """Permutation/translation-invariant joint effect signature for one composite group.

    Encodes the summed-coordinate displacement and sorted member effect signatures,
    so rigid co-movers yield a constant signature regardless of fragment-identity permutation.
    """
    bx = sum(int(o.get("x", 0)) for o in before_objs)
    by = sum(int(o.get("y", 0)) for o in before_objs)
    ax = sum(int(o.get("x", 0)) for o in after_objs)
    ay = sum(int(o.get("y", 0)) for o in after_objs)
    pairs = _pair_before_after(before_objs, after_objs)
    member_sigs = sorted(
        _effect_signature(bo, ao)
        for bo, ao in pairs
        if (bo if bo is not None else ao).get("name")
    )
    return f"cd={ax - bx},{ay - by}|m={'|'.join(member_sigs)}"


def _stratify_composite(
    replay_buffer: list[dict], base_type_of: Any,
    composite_groups: list[frozenset[str]],
    committed_features: list[dict] | None = None,
) -> dict[tuple[str, int, str], Counter]:
    """Stratify with selected base-type groups re-paired as composite prediction units.

    Objects not in any group stratify identically to _stratify, so ungrouped strata
    cancel exactly in any flat-vs-candidate eta comparison.
    """
    member_label: dict[str, str] = {}
    for grp in composite_groups:
        lbl = "composite:" + "+".join(sorted(grp))
        for bt in grp:
            member_label[str(bt)] = lbl

    strata: dict[tuple[str, int, str], Counter] = defaultdict(Counter)
    for t in replay_buffer:
        action_id = t.get("action_id")
        before = t.get("before_state") or []
        after = t.get("after_state") or []

        seen: set[tuple] = set()
        for bo, ao in _pair_before_after(before, after):
            ref = bo if bo is not None else ao
            name = ref.get("name")
            if not name:
                continue
            bt = base_type_of(ref)
            if not bt or bt == "Unknown":
                continue
            if member_label.get(str(bt)) is not None:
                continue
            sig = _effect_signature(bo, ao)
            ctx = _augmented_context_signature(t, ref, committed_features)
            key = (str(bt), action_id, ctx, sig)
            if key in seen:
                continue
            seen.add(key)
            strata[(str(bt), action_id, ctx)][sig] += 1

        grp_before: dict[str, list] = defaultdict(list)
        grp_after: dict[str, list] = defaultdict(list)
        for o in before:
            if not o.get("name"):
                continue
            bt = base_type_of(o)
            lbl = member_label.get(str(bt)) if bt else None
            if lbl is not None:
                grp_before[lbl].append(o)
        for o in after:
            if not o.get("name"):
                continue
            bt = base_type_of(o)
            lbl = member_label.get(str(bt)) if bt else None
            if lbl is not None:
                grp_after[lbl].append(o)
        for lbl in set(grp_before) | set(grp_after):
            bobjs = grp_before.get(lbl, [])
            aobjs = grp_after.get(lbl, [])
            if not bobjs and not aobjs:
                continue
            sig = _composite_signature(bobjs, aobjs)
            ctx = (
                _augmented_context_signature(t, bobjs[0], committed_features)
                if bobjs else ""
            )
            strata[(lbl, action_id, ctx)][sig] += 1
    return strata


def compute_ontology_error(
    replay_buffer: list[dict],
    type_of: Any = None,
    *,
    alpha_0: float = 1.0,
    kappa: float = 2.0,
    effect_alphabet_size: int | None = None,
    top_k: int = 30,
    composite_groups: list[frozenset[str]] | None = None,
    committed_features: list[dict] | None = None,
) -> dict[str, Any]:
    """Compute eta(R, D) under representation type_of (flat _object_type if None).

    Returns the scalar eta plus the worst top_k strata with full effect counts.
    """
    tf = type_of or _object_type
    if composite_groups:
        strata = _stratify_composite(
            replay_buffer, tf, composite_groups,
            committed_features=committed_features,
        )
    else:
        strata = _stratify(
            replay_buffer, tf,
            committed_features=committed_features,
        )

    if effect_alphabet_size is None:
        seen: set[str] = set()
        for ctr in strata.values():
            seen.update(ctr.keys())
        E = max(2, len(seen))
    else:
        E = max(2, int(effect_alphabet_size))

    lnE = math.log(E)
    rows: list[dict[str, Any]] = []
    total_n = 0
    weighted = 0.0
    for (ty, aid, ctx), ctr in strata.items():
        n_g = sum(ctr.values())
        if n_g == 0:
            continue
        denom = E * alpha_0 + n_g
        ent = 0.0
        for cnt in ctr.values():
            p = (alpha_0 + cnt) / denom
            ent -= p * math.log(p)
        n_unseen = E - len(ctr)
        if n_unseen > 0 and alpha_0 > 0:
            p0 = alpha_0 / denom
            ent -= n_unseen * (p0 * math.log(p0))
        eta_g = ent / lnE
        eta_g = 0.0 if eta_g < 0.0 else (1.0 if eta_g > 1.0 else eta_g)
        s_g = ctr.most_common(1)[0][1]
        total_n += n_g
        weighted += n_g * eta_g
        rows.append({
            "type": ty, "action_id": aid, "context": ctx,
            "n": n_g, "modal_frac": round(s_g / n_g, 4),
            "eta_g": round(eta_g, 4),
            "effects": dict(ctr),
        })

    eta = (weighted / total_n) if total_n > 0 else 0.0
    rows.sort(key=lambda r: -(r["n"] * r["eta_g"]))
    return {
        "eta": round(eta, 6),
        "n_strata": len(rows),
        "n_transitions": len(replay_buffer),
        "effect_alphabet_size": E,
        "alpha_0": alpha_0,
        "estimator": "dirichlet_predictive_entropy_over_lnE",
        "worst_strata": rows[:top_k],
    }


def _covariation_pairs(
    replay_buffer: list[dict], base_type_of: Any,
) -> list[tuple[tuple[str, str], float]]:
    """Score base-type pairs by co-change containment.

    score(A,B) = co_change(A,B) / min(chg(A), chg(B)), high for rigid co-movers.
    """
    chg: Counter = Counter()
    co: Counter = Counter()
    for t in replay_buffer:
        before = t.get("before_state") or []
        after = t.get("after_state") or []
        changed: set[str] = set()
        for bo, ao in _pair_before_after(before, after):
            ref = bo if bo is not None else ao
            if not ref.get("name"):
                continue
            if _effect_signature(bo, ao) in _NULL_EFFECTS:
                continue
            ty = base_type_of(ref)
            if ty and ty != "Unknown":
                changed.add(str(ty))
        for ty in changed:
            chg[ty] += 1
        cl = sorted(changed)
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                co[(cl[i], cl[j])] += 1
    scored = []
    for (a, b), c in co.items():
        m = min(chg[a], chg[b])
        if m <= 0:
            continue
        scored.append(((a, b), c / m))
    scored.sort(key=lambda x: -x[1])
    return scored


def _active_cooccurrence_pairs(
    replay_buffer: list[dict], base_type_of: Any,
) -> list[tuple[tuple[str, str], float]]:
    """Score active base-type pairs by co-presence.

    score(a,b) = co_present(a,b) / min(present[a], present[b]), restricted to
    types that change in at least one transition (excludes inert scenery).
    """
    present: Counter = Counter()
    co_present: Counter = Counter()
    ever_change: set[str] = set()
    for t in replay_buffer:
        before = t.get("before_state") or []
        after = t.get("after_state") or []
        pres: set[str] = set()
        for o in before:
            if not o.get("name"):
                continue
            ty = base_type_of(o)
            if ty and ty != "Unknown":
                pres.add(str(ty))
        for bo, ao in _pair_before_after(before, after):
            ref = bo if bo is not None else ao
            if not ref.get("name"):
                continue
            ty = base_type_of(ref)
            if (ty and ty != "Unknown"
                    and _effect_signature(bo, ao) not in _NULL_EFFECTS):
                ever_change.add(str(ty))
        pl = sorted(pres)
        for ty in pl:
            present[ty] += 1
        for i in range(len(pl)):
            for j in range(i + 1, len(pl)):
                co_present[(pl[i], pl[j])] += 1
    scored: list[tuple[tuple[str, str], float]] = []
    for (a, b), c in co_present.items():
        if a not in ever_change or b not in ever_change:
            continue
        m = min(present[a], present[b])
        if m <= 0:
            continue
        scored.append(((a, b), c / m))
    scored.sort(key=lambda x: -x[1])
    return scored


def _merge_type_of(base_type_of: Any, merge_map: dict[str, str]) -> Any:
    def tf(obj: dict) -> str:
        bt = base_type_of(obj)
        return merge_map.get(str(bt), str(bt))
    return tf


def compute_ontology_error_with_candidates(
    replay_buffer: list[dict],
    *,
    alpha_0: float = 1.0,
    kappa: float = 2.0,
    max_candidates: int = 12,
    pair_threshold: float = 0.6,
    effect_alphabet_size: int | None = None,
    include_composite: bool = True,
    committed_features: list[dict] | None = None,
) -> dict[str, Any]:
    """Compute flat eta and eta* = min eta over label-merge and composite-repairing candidates.

    Answers whether eta is high under the flat partition and reducible by a better one.
    """
    flat = compute_ontology_error(
        replay_buffer, _object_type,
        alpha_0=alpha_0, kappa=kappa,
        effect_alphabet_size=effect_alphabet_size,
        committed_features=committed_features,
    )
    pairs = _covariation_pairs(replay_buffer, _object_type)

    candidates: list[tuple[str, Any, Any]] = []
    for (a, b), sc in pairs[:max_candidates]:
        label = f"{a}+{b}"
        candidates.append((
            f"merge[{label}|{sc:.2f}]",
            _merge_type_of(_object_type, {a: label, b: label}),
            None,
        ))
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    n_strong = 0
    for (a, b), sc in pairs:
        if sc < pair_threshold:
            break
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
            n_strong += 1
    if n_strong > 0:
        agg_map = {k: f"cluster:{find(k)}" for k in list(parent.keys())}
        candidates.append((
            f"agglomerate[>={pair_threshold},{n_strong} merges]",
            _merge_type_of(_object_type, agg_map),
            None,
        ))

    if include_composite:
        comp_pairs: list[tuple[tuple[str, str], float]] = list(pairs)
        seen_p = {p for p, _ in comp_pairs}
        for p, sc in _active_cooccurrence_pairs(replay_buffer, _object_type):
            if p not in seen_p:
                comp_pairs.append((p, sc))
                seen_p.add(p)
        comp_pairs.sort(key=lambda x: -x[1])
        for (a, b), sc in comp_pairs[:max_candidates]:
            candidates.append((
                f"composite[{a}+{b}|{sc:.2f}]",
                None, [frozenset({a, b})],
            ))
        cparent: dict[str, str] = {}

        def cfind(x: str) -> str:
            cparent.setdefault(x, x)
            while cparent[x] != x:
                cparent[x] = cparent[cparent[x]]
                x = cparent[x]
            return x

        c_strong = 0
        for (a, b), sc in comp_pairs:
            if sc < pair_threshold:
                break
            ra, rb = cfind(a), cfind(b)
            if ra != rb:
                cparent[rb] = ra
                c_strong += 1
        if c_strong > 0:
            clusters: dict[str, set] = defaultdict(set)
            for k in list(cparent.keys()):
                clusters[cfind(k)].add(k)
            comp_groups = [
                frozenset(v) for v in clusters.values() if len(v) >= 2
            ]
            if comp_groups:
                total = sum(len(g) for g in comp_groups)
                candidates.append((
                    f"composite-agglomerate[{total} in "
                    f"{len(comp_groups)}]",
                    None, comp_groups,
                ))

    eta_star = flat["eta"]
    best_label = "flat"
    cand_summ: list[dict[str, Any]] = []
    for label, tf, cgroups in candidates:
        r = compute_ontology_error(
            replay_buffer, tf,
            alpha_0=alpha_0, kappa=kappa,
            effect_alphabet_size=effect_alphabet_size,
            composite_groups=cgroups,
            committed_features=committed_features,
        )
        cand_summ.append({"candidate": label, "eta": r["eta"],
                          "n_strata": r["n_strata"]})
        if r["eta"] < eta_star:
            eta_star = r["eta"]
            best_label = label

    xi_ledger = compute_xi_candidate_ledger(
        replay_buffer, flat, alpha_0=alpha_0,
        committed_features=committed_features,
    )

    return {
        "flat_eta": flat["eta"],
        "eta_star": round(eta_star, 6),
        "eta_reduction": round(flat["eta"] - eta_star, 6),
        "best_candidate": best_label,
        "best_kind": (
            "flat" if best_label == "flat"
            else best_label.split("[", 1)[0]
        ),
        "n_candidates": len(candidates),
        "flat": flat,
        "candidates": sorted(cand_summ, key=lambda c_: c_["eta"]),
        "xi_candidate_ledger": xi_ledger,
    }


def dump_ontology_error(
    replay_buffer: list[dict],
    path: Path,
    *,
    alpha_0: float = 1.0,
    kappa: float = 2.0,
    max_candidates: int = 12,
    step: int | None = None,
    committed_features: list[dict] | None = None,
) -> dict[str, Any]:
    """Compute eta / eta* and write the full report to ``path`` (JSON).
    Returns the compact summary for the caller's running trace."""
    res = compute_ontology_error_with_candidates(
        replay_buffer, alpha_0=alpha_0, kappa=kappa,
        max_candidates=max_candidates,
        committed_features=committed_features,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = dict(res)
    if step is not None:
        out["step"] = int(step)
    path.write_text(json.dumps(out, indent=2, default=str))
    return {
        "step": step,
        "eta": res["flat_eta"],
        "eta_star": res["eta_star"],
        "eta_reduction": res["eta_reduction"],
        "best_candidate": res["best_candidate"],
        "n_strata": res["flat"]["n_strata"],
        "n_candidates": res["n_candidates"],
        "n_transitions": len(replay_buffer),
    }


def _collect_stratum_transitions(
    replay_buffer: list[dict],
    target_type: str,
    target_action: Any,
    target_context: str,
    committed_features: list[dict] | None = None,
) -> list[tuple[dict, dict, dict | None]]:
    """Return (transition, before_obj, after_obj) tuples for sprites in
    the target stratum (type, action, augmented context)."""
    out: list[tuple[dict, dict, dict | None]] = []
    for t in replay_buffer:
        if t.get("action_id") != target_action:
            continue
        before = t.get("before_state") or []
        after = t.get("after_state") or []
        for bo, ao in _pair_before_after(before, after):
            ref = bo if bo is not None else ao
            name = ref.get("name")
            if not name:
                continue
            ty = _object_type(ref)
            if ty != target_type:
                continue
            ctx = _augmented_context_signature(t, ref, committed_features)
            if ctx != target_context:
                continue
            out.append((t, bo, ao))
    return out


def _pixels_hash_bucket(pixels: Any) -> str:
    """Bucketed hash of a sprite's pixel pattern (mod 100000 to bound cardinality)."""
    if pixels is None:
        return "none"
    try:
        h = hash(tuple(tuple(int(c) for c in row) for row in pixels))
        return str(h % 100000)
    except (TypeError, ValueError):
        return "err"


def _xi_feature_value(
    feature: dict, transition: dict, target_obj: dict,
) -> str:
    """Compute the augmented context feature value for one (feature, transition, object) triple."""
    kind = feature.get("kind")
    if kind == "target_field":
        f = str(feature.get("field", ""))
        if f == "pixels_hash":
            return f"px={_pixels_hash_bucket(target_obj.get('pixels'))}"
        v = target_obj.get(f)
        return f"{f}={v}"

    before = transition.get("before_state") or []

    if kind == "neighbour_at_offset":
        dx = int(feature.get("dx", 0))
        dy = int(feature.get("dy", 0))
        ox = int(target_obj.get("x", 0)) + dx
        oy = int(target_obj.get("y", 0)) + dy
        for o in before:
            if o.get("name") == target_obj.get("name"):
                continue
            if int(o.get("x", 0)) == ox and int(o.get("y", 0)) == oy:
                return f"@{dx},{dy}={_object_type(o)}"
        return f"@{dx},{dy}=none"

    if kind == "neighbourhood_radius":
        r = int(feature.get("r", 1))
        ox = int(target_obj.get("x", 0))
        oy = int(target_obj.get("y", 0))
        name = target_obj.get("name")
        nbrs: list[str] = []
        for o in before:
            if o.get("name") == name:
                continue
            if not o.get("visible", True):
                continue
            x, y = int(o.get("x", 0)), int(o.get("y", 0))
            d = max(abs(x - ox), abs(y - oy))
            if 0 < d <= r:
                nbrs.append(_object_type(o))
        return f"R{r}=" + ",".join(sorted(nbrs))

    if kind == "click_offset":
        cx = transition.get("click_x")
        cy = transition.get("click_y")
        if cx is None or cy is None:
            return "cli=none"
        tx = int(target_obj.get("display_x", target_obj.get("x", 0)))
        ty = int(target_obj.get("display_y", target_obj.get("y", 0)))
        return f"cli=({int(cx) - tx},{int(cy) - ty})"

    return f"{kind}=?"


def _enumerate_xi_candidates(
    stratum_transitions: list[tuple[dict, dict, dict | None]],
    target_action: Any,
) -> list[dict]:
    """Enumerate xi-refinement candidates for one stratum.

    Only emits candidates with a plausible chance of de-confounding (varying fields,
    observed neighbours, radius {2,3}, click offset for action 6 only).
    """
    candidates: list[dict] = []
    if not stratum_transitions:
        return candidates

    skip_for_target_field = _EFFECT_SKIP_FIELDS | {
        "x", "y", "pixels", "collidable",
    }
    field_values: dict[str, set] = defaultdict(set)
    for _, bo, ao in stratum_transitions:
        ref = bo if bo is not None else ao
        for k, v in ref.items():
            if k in skip_for_target_field:
                continue
            try:
                hash(v)
            except TypeError:
                continue
            field_values[k].add(v)
    for field, vals in field_values.items():
        if len(vals) >= 2:
            candidates.append({"kind": "target_field", "field": field})

    pixel_buckets: set = set()
    for _, bo, ao in stratum_transitions:
        ref = bo if bo is not None else ao
        pixel_buckets.add(_pixels_hash_bucket(ref.get("pixels")))
        if len(pixel_buckets) >= 2:
            break
    if len(pixel_buckets) >= 2:
        candidates.append({"kind": "target_field", "field": "pixels_hash"})

    offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]
    for dx, dy in offsets:
        has_neighbour = False
        for trans, bo, ao in stratum_transitions:
            ref = bo if bo is not None else ao
            ox = int(ref.get("x", 0)) + dx
            oy = int(ref.get("y", 0)) + dy
            for o in trans.get("before_state") or []:
                if o.get("name") == ref.get("name"):
                    continue
                if int(o.get("x", 0)) == ox and int(o.get("y", 0)) == oy:
                    has_neighbour = True
                    break
            if has_neighbour:
                break
        if has_neighbour:
            candidates.append(
                {"kind": "neighbour_at_offset", "dx": dx, "dy": dy}
            )

    candidates.append({"kind": "neighbourhood_radius", "r": 2})
    candidates.append({"kind": "neighbourhood_radius", "r": 3})

    if target_action == 6:
        has_click = any(
            t.get("click_x") is not None and t.get("click_y") is not None
            for t, _, _ in stratum_transitions
        )
        if has_click:
            candidates.append({"kind": "click_offset"})

    return candidates


def _score_xi_candidate(
    stratum_transitions: list[tuple[dict, dict, dict | None]],
    feature: dict,
    *,
    alpha_0: float,
    effect_alphabet_size: int,
    n_min: int,
    modal_frac_min: float,
) -> dict | None:
    """Re-stratify under augmented xi = base + feature and return eta_new and identification counts.

    Returns None if no transitions matched. Per-transition dedup mirrors _stratify.
    """
    by_trans: dict[int, list] = defaultdict(list)
    for trans, bo, ao in stratum_transitions:
        by_trans[id(trans)].append((trans, bo, ao))

    sub_strata: dict[str, Counter] = defaultdict(Counter)
    for _tid, group in by_trans.items():
        seen_in_trans: set = set()
        for trans, bo, ao in group:
            target = bo if bo is not None else ao
            extra = _xi_feature_value(feature, trans, target)
            sig = _effect_signature(bo, ao)
            key = (extra, sig)
            if key in seen_in_trans:
                continue
            seen_in_trans.add(key)
            sub_strata[extra][sig] += 1

    if not sub_strata:
        return None

    E = max(2, int(effect_alphabet_size))
    lnE = math.log(E)
    total_n = 0
    weighted_eta = 0.0
    identified = 0
    for _extra, ctr in sub_strata.items():
        n_g = sum(ctr.values())
        if n_g == 0:
            continue
        denom = E * alpha_0 + n_g
        ent = 0.0
        for cnt in ctr.values():
            p = (alpha_0 + cnt) / denom
            ent -= p * math.log(p)
        n_unseen = E - len(ctr)
        if n_unseen > 0 and alpha_0 > 0:
            p0 = alpha_0 / denom
            ent -= n_unseen * (p0 * math.log(p0))
        eta_g = ent / lnE if lnE > 0 else 0.0
        eta_g = 0.0 if eta_g < 0.0 else (1.0 if eta_g > 1.0 else eta_g)
        total_n += n_g
        weighted_eta += n_g * eta_g
        modal_frac = ctr.most_common(1)[0][1] / n_g
        if n_g >= n_min and modal_frac >= modal_frac_min:
            identified += 1

    eta_new = (weighted_eta / total_n) if total_n > 0 else 1.0
    return {
        "eta_new": round(eta_new, 6),
        "n_substrata": len(sub_strata),
        "identified_substrata": identified,
        "total_n": total_n,
    }


def compute_xi_candidate_ledger(
    replay_buffer: list[dict],
    flat_report: dict,
    *,
    alpha_0: float = 1.0,
    top_k_strata: int = 3,
    n_min: int = 3,
    modal_frac_min: float = 0.95,
    min_eta_reduction: float = 0.05,
    committed_features: list[dict] | None = None,
) -> dict:
    """Score xi-refinement candidates over the top-K worst strata of flat_report.

    Candidates are scored but not applied. A candidate is accepted if eta-reduction >= min_eta_reduction
    and at least one identified sub-stratum (n >= n_min, modal_frac >= modal_frac_min).
    """
    worst_strata = flat_report.get("worst_strata") or []
    params = {
        "top_k_strata": top_k_strata,
        "n_min": n_min,
        "modal_frac_min": modal_frac_min,
        "min_eta_reduction": min_eta_reduction,
    }
    committed_summary = list(committed_features or [])
    if not worst_strata:
        return {
            "strata": [],
            "n_strata_scored": 0,
            "params": params,
            "committed_features": committed_summary,
        }
    E = max(2, int(flat_report.get("effect_alphabet_size", 2)))

    strata_entries: list[dict] = []
    for stratum in worst_strata[:top_k_strata]:
        ty = stratum["type"]
        aid = stratum["action_id"]
        ctx = stratum["context"]
        eta_old = float(stratum["eta_g"])
        transitions_in = _collect_stratum_transitions(
            replay_buffer, ty, aid, ctx,
            committed_features=committed_features,
        )
        if not transitions_in:
            continue
        cands = _enumerate_xi_candidates(transitions_in, aid)
        cands = [c for c in cands if c not in committed_summary]
        cand_entries: list[dict] = []
        for feat in cands:
            scored = _score_xi_candidate(
                transitions_in, feat,
                alpha_0=alpha_0,
                effect_alphabet_size=E,
                n_min=n_min,
                modal_frac_min=modal_frac_min,
            )
            if scored is None:
                continue
            eta_new = scored["eta_new"]
            reduction = eta_old - eta_new
            accepted = (
                reduction >= min_eta_reduction
                and scored["identified_substrata"] >= 1
            )
            cand_entries.append({
                "feature": feat,
                "eta_old": round(eta_old, 6),
                "eta_new": eta_new,
                "eta_reduction": round(reduction, 6),
                "n_substrata": scored["n_substrata"],
                "identified_substrata": scored["identified_substrata"],
                "total_n": scored["total_n"],
                "accepted": accepted,
            })
        cand_entries.sort(
            key=lambda c: (0 if c["accepted"] else 1, -c["eta_reduction"])
        )
        strata_entries.append({
            "stratum": {
                "type": ty,
                "action_id": aid,
                "context": ctx,
                "n": stratum["n"],
                "eta_g": round(eta_old, 6),
            },
            "candidates": cand_entries,
            "n_candidates_accepted": sum(
                1 for c in cand_entries if c["accepted"]
            ),
        })

    return {
        "strata": strata_entries,
        "n_strata_scored": len(strata_entries),
        "params": params,
        "committed_features": committed_summary,
    }


def _candidate_roles(aliases: dict[str, list[dict]] | None) -> list[str]:
    """Union of distinct alias names across all tags, in lexicographic order."""
    if not aliases:
        return []
    seen: set[str] = set()
    for cands in aliases.values():
        if not isinstance(cands, list):
            continue
        for c in cands:
            if not isinstance(c, dict):
                continue
            a = str(c.get("alias", "")).strip()
            if a:
                seen.add(a)
    return sorted(seen)


def _alias_prior(
    aliases: dict[str, list[dict]] | None,
    tag: str,
    candidate_roles: list[str],
    eps: float = 1e-3,
) -> dict[str, float]:
    """Categorical prior over candidate roles for tag tau.

    Listed roles get score-proportional mass. Unlisted roles get eps smoothing.
    Returns {role_name: prob} summing to 1.
    """
    K = len(candidate_roles)
    if K == 0:
        return {}
    tag_scores: dict[str, float] = {}
    if aliases and tag in aliases and isinstance(aliases[tag], list):
        for c in aliases[tag]:
            if not isinstance(c, dict):
                continue
            a = str(c.get("alias", "")).strip()
            s = float(c.get("score", 0) or 0)
            if a:
                tag_scores[a] = max(tag_scores.get(a, 0.0), s)
    eps_total = eps * K
    rem = max(0.0, 1.0 - eps_total)
    score_sum = sum(s for s in tag_scores.values() if s > 0)
    out: dict[str, float] = {}
    for r in candidate_roles:
        base = eps
        if score_sum > 0 and r in tag_scores and tag_scores[r] > 0:
            base += rem * (tag_scores[r] / score_sum)
        out[r] = base
    if score_sum <= 0:
        u = 1.0 / K
        return {r: u for r in candidate_roles}
    z = sum(out.values())
    if z > 0:
        out = {r: p / z for r, p in out.items()}
    return out


def _per_tag_effect_counts(
    replay_buffer: list[dict],
) -> dict[str, Counter]:
    """Per-tag effect-signature counts (sufficient statistic for role dynamics likelihood).

    Per-transition dedup: same-tag instances with identical effect count once.
    """
    out: dict[str, Counter] = defaultdict(Counter)
    for t in replay_buffer:
        before = t.get("before_state") or []
        after = t.get("after_state") or []
        seen: set[tuple] = set()
        for bo, ao in _pair_before_after(before, after):
            ref = bo if bo is not None else ao
            name = ref.get("name")
            if not name:
                continue
            ty = _object_type(ref)
            if not ty or ty == "Unknown":
                continue
            sig = _effect_signature(bo, ao)
            key = (ty, sig)
            if key in seen:
                continue
            seen.add(key)
            out[ty][sig] += 1
    return out


def compute_role_posterior(
    replay_buffer: list[dict],
    aliases: dict[str, list[dict]] | None = None,
    *,
    alpha_0: float = 1.0,
) -> dict[str, Any]:
    """Per-tag role posterior p(role|tag, D) via alias prior fused with a one-step EM
    dynamics likelihood (paper §4/§5).

    When aliases is None, returns H_norm=1.0 for all tags (pessimistic upper bound).
    """
    candidate_roles = _candidate_roles(aliases)
    counts = _per_tag_effect_counts(replay_buffer)

    if not candidate_roles:
        return {
            "candidate_roles": [],
            "per_tag": {
                tag: {
                    "posterior": {},
                    "H": 0.0,
                    "H_norm": 1.0,
                    "map_role": None,
                    "map_score": 0.0,
                }
                for tag in counts.keys()
            },
            "n_candidates": 0,
            "fallback": "no_aliases",
        }

    K = len(candidate_roles)
    eff_alphabet: set[str] = set()
    for ctr in counts.values():
        eff_alphabet.update(ctr.keys())
    E = max(2, len(eff_alphabet))

    tag_prior = {
        tag: _alias_prior(aliases, tag, candidate_roles) for tag in counts
    }

    role_aggregate: dict[str, Counter] = {
        r: Counter() for r in candidate_roles
    }
    for tag, prior in tag_prior.items():
        if not prior:
            continue
        argmax_role = max(prior.keys(), key=lambda r: prior[r])
        role_aggregate[argmax_role].update(counts[tag])

    per_tag_out: dict[str, dict[str, Any]] = {}
    for tag, ctr in counts.items():
        prior = tag_prior.get(tag, {r: 1.0 / K for r in candidate_roles})
        log_post: dict[str, float] = {}
        for r in candidate_roles:
            agg = role_aggregate.get(r) or Counter()
            n_k = sum(agg.values())
            denom = E * alpha_0 + n_k
            ll = 0.0
            for e, c in ctr.items():
                q = (alpha_0 + agg.get(e, 0)) / denom
                q = max(q, 1e-300)
                ll += c * math.log(q)
            log_post[r] = math.log(max(prior[r], 1e-300)) + ll
        m = max(log_post.values())
        unn = {r: math.exp(lp - m) for r, lp in log_post.items()}
        z = sum(unn.values())
        post = {r: u / z for r, u in unn.items()} if z > 0 else {
            r: 1.0 / K for r in candidate_roles
        }
        H = 0.0
        for p in post.values():
            if p > 0:
                H -= p * math.log(p)
        H_norm = H / math.log(K) if K > 1 else 0.0
        H_norm = 0.0 if H_norm < 0 else (1.0 if H_norm > 1 else H_norm)
        map_role = max(post.keys(), key=lambda r: post[r])
        per_tag_out[tag] = {
            "posterior": {r: round(post[r], 6) for r in candidate_roles},
            "H": round(H, 6),
            "H_norm": round(H_norm, 6),
            "map_role": map_role,
            "map_score": round(post[map_role], 6),
        }

    return {
        "candidate_roles": candidate_roles,
        "per_tag": per_tag_out,
        "n_candidates": K,
        "fallback": None,
        "effect_alphabet_size": E,
    }


def compute_ontology_error_extended(
    replay_buffer: list[dict],
    aliases: dict[str, list[dict]] | None = None,
    type_of: Any = None,
    *,
    alpha_0: float = 1.0,
    kappa: float = 2.0,
    effect_alphabet_size: int | None = None,
    top_k: int = 30,
    committed_features: list[dict] | None = None,
) -> dict[str, Any]:
    """Compute eta_extended (paper §5 Definition 1 / noisy-OR role unification).

    Returns both legs (eta_effect, eta_role) and the composed eta_extended.
    When aliases is None, falls back to the pessimistic upper bound (H_tilde_tau=1 everywhere).
    """
    base = compute_ontology_error(
        replay_buffer, type_of=type_of, alpha_0=alpha_0, kappa=kappa,
        effect_alphabet_size=effect_alphabet_size, top_k=top_k,
        committed_features=committed_features,
    )
    role = compute_role_posterior(
        replay_buffer, aliases, alpha_0=alpha_0,
    )

    tf = type_of or _object_type
    strata = _stratify(
        replay_buffer, tf, committed_features=committed_features,
    )
    if effect_alphabet_size is None:
        seen: set[str] = set()
        for ctr in strata.values():
            seen.update(ctr.keys())
        E = max(2, len(seen))
    else:
        E = max(2, int(effect_alphabet_size))
    lnE = math.log(E)

    per_tag_H_norm: dict[str, float] = {
        tag: float(d.get("H_norm", 1.0))
        for tag, d in role["per_tag"].items()
    }

    def _h_for(tag: str) -> float:
        return per_tag_H_norm.get(tag, 1.0)

    rows: list[dict[str, Any]] = []
    total_n = 0
    weighted_joint = 0.0
    weighted_role = 0.0
    weighted_effect = 0.0
    for (ty, aid, ctx), ctr in strata.items():
        n_g = sum(ctr.values())
        if n_g == 0:
            continue
        denom = E * alpha_0 + n_g
        ent = 0.0
        for cnt in ctr.values():
            p = (alpha_0 + cnt) / denom
            ent -= p * math.log(p)
        n_unseen = E - len(ctr)
        if n_unseen > 0 and alpha_0 > 0:
            p0 = alpha_0 / denom
            ent -= n_unseen * (p0 * math.log(p0))
        eta_g_effect = ent / lnE if lnE > 0 else 0.0
        eta_g_effect = max(0.0, min(1.0, eta_g_effect))
        H_tau = _h_for(ty)
        eta_g_joint = H_tau + (1.0 - H_tau) * eta_g_effect
        eta_g_joint = max(0.0, min(1.0, eta_g_joint))

        total_n += n_g
        weighted_joint += n_g * eta_g_joint
        weighted_role += n_g * H_tau
        weighted_effect += n_g * eta_g_effect

        rows.append({
            "type": ty,
            "action_id": aid,
            "context": ctx,
            "n": n_g,
            "H_tilde_tau": round(H_tau, 4),
            "eta_g_effect": round(eta_g_effect, 4),
            "eta_g_joint": round(eta_g_joint, 4),
        })

    eta_extended = (weighted_joint / total_n) if total_n > 0 else 0.0
    eta_role = (weighted_role / total_n) if total_n > 0 else 0.0
    eta_effect = (weighted_effect / total_n) if total_n > 0 else 0.0
    rows.sort(key=lambda r: -(r["n"] * r["eta_g_joint"]))

    return {
        "eta_extended": round(eta_extended, 6),
        "eta_effect_component": round(eta_effect, 6),
        "eta_role_component": round(eta_role, 6),
        "eta_base": base["eta"],
        "n_strata": len(rows),
        "n_transitions": len(replay_buffer),
        "n_candidate_roles": role["n_candidates"],
        "fallback": role["fallback"],
        "alpha_0": alpha_0,
        "effect_alphabet_size": E,
        "estimator": "noisy_or_role_effect_predictive_entropy",
        "role_posterior": role["per_tag"],
        "worst_strata": rows[:top_k],
    }


def dump_ontology_error_extended(
    replay_buffer: list[dict],
    path: Path,
    *,
    aliases: dict[str, list[dict]] | None = None,
    alpha_0: float = 1.0,
    step: int | None = None,
    committed_features: list[dict] | None = None,
) -> dict[str, Any]:
    """Compute eta_extended and write the full report to path as JSON.
    Returns the compact summary for the running trace."""
    res = compute_ontology_error_extended(
        replay_buffer, aliases=aliases, alpha_0=alpha_0,
        committed_features=committed_features,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = dict(res)
    if step is not None:
        out["step"] = int(step)
    path.write_text(json.dumps(out, indent=2, default=str))
    return {
        "step": step,
        "eta_extended": res["eta_extended"],
        "eta_effect_component": res["eta_effect_component"],
        "eta_role_component": res["eta_role_component"],
        "eta_base": res["eta_base"],
        "n_strata": res["n_strata"],
        "n_candidate_roles": res["n_candidate_roles"],
        "n_transitions": len(replay_buffer),
        "fallback": res["fallback"],
    }
