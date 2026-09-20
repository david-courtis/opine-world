"""Per-object epistemic substrate Σ(o).

classes(o) = an enumerable action backbone (ξ-refined) plus a mode ledger
with observed/pooled/goal provenance. Evidence is engagement-gated and
unexercised classes sit at prior uncertainty, so η(o), the attempt-weighted
mean of per-class uncertainty, reads an untouched object as unknown. C(o)
is prequential (test-then-train) forward accuracy against the live
synthesized model. LP, CAI, rel and Π feed exploration priority.

Every quantity is a deterministic function of the transition stream and the
stored vocabulary, with no LLM in the update path. The identity unit is the
sprite name, and same-name instances pool.
"""
from __future__ import annotations

import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .epistemic import (
    _augmented_context_signature,
    _effect_signature,
    _pair_before_after,
    _pixels_hash_bucket,
    _xi_feature_value,
)

SIGMA_ARTIFACT_VERSION = 1

# Passed as ``predicted_state`` when the live model raised or returned a
# non-state. Every engaged object then scores loss 1.
MODEL_ERROR = "__model_error__"

# Coordinate-free playable actions. RESET (0) and UNDO (7) are excluded as
# recovery controls, never level mechanics.
_BACKBONE_ACTIONS = (1, 2, 3, 4, 5)
_CLICK_ON = "click_on"
_CLICK_OFF = "click_off"
_MODE_PREFIX = "mode|"
_GOAL_MODE_PREFIX = "mode|goal:"

# Cap on enumerated unexercised ξ-bins per action cell. Truncation is
# recorded in the artifact, never silent.
MAX_ENUM_BINS_PER_ACTION = 128

# Alias vote counts are unbounded, softened to [0, 1) as s / (s + SOFTENING).
_REL_SCORE_SOFTENING = 3.0

# Objects not directly actionable cost more probes to engage.
_CAI_ACTIONABLE_THRESHOLD = 0.1
_COST_DIRECT = 1.0
_COST_AUTONOMOUS = 3.0

# Mirrors aliases.DECORATIVE_DEFAULT, duplicated so retro tooling can import
# sigma without the aliases module.
_DECORATIVE_HEADS = frozenset({
    "wall", "scenery", "decoration", "decorative", "border", "tile",
    "floor", "background", "hud",
})


def _dirichlet_entropy(counts: dict[str, int], alphabet_size: int,
                       alpha_0: float) -> float:
    """Normalized predictive entropy of Dir(alpha_0·1 + counts) over an
    alphabet of ``alphabet_size`` symbols, in [0, 1]. Mirrors the estimator in
    ``epistemic.compute_ontology_error`` (unseen symbols carry prior mass)."""
    E = max(2, int(alphabet_size))
    n = sum(counts.values())
    denom = E * alpha_0 + n
    ent = 0.0
    for cnt in counts.values():
        p = (alpha_0 + cnt) / denom
        ent -= p * math.log(p)
    n_unseen = E - len(counts)
    if n_unseen > 0 and alpha_0 > 0:
        p0 = alpha_0 / denom
        ent -= n_unseen * (p0 * math.log(p0))
    eta = ent / math.log(E)
    return 0.0 if eta < 0.0 else (1.0 if eta > 1.0 else eta)


def _posterior_error_rate(n_pred: int, k_correct: int, alpha_0: float) -> float:
    """Laplace-smoothed misprediction rate, used as a floor under U_c.

    A Beta-entropy reading would let a consistently-mispredicted class read
    as resolved (its rate concentrates at 0 and entropy falls); the floor
    keeps a class open until its effects are both consistent and predicted.
    """
    return (n_pred - k_correct + alpha_0) / (n_pred + 2.0 * alpha_0)


def _records_match(a: dict | None, b: dict | None) -> bool:
    """Existence plus verifier-field agreement (x, y, visible, rotation,
    pixels)."""
    return not _record_mismatch_fields(a, b)


# The grounded fluent families the verifier compares, i.e. the record
# schema; per-family prequential accuracy localizes which CPF is failing.
FLUENT_FAMILIES = ("x", "y", "visible", "rotation", "pixels", "existence")

# Forward predictions required before the prediction record may resolve a
# mixed cell (three-state U_c, engine flag sigma_model_resolved).
MODEL_RESOLVED_MIN_PRED = 3


def _record_mismatch_fields(
    a: dict | None, b: dict | None,
) -> tuple[str, ...]:
    """Fluent families on which two records disagree. Existence covers
    gone/born mismatches; empty tuple means exact match."""
    if a is None or b is None:
        return () if a is None and b is None else ("existence",)
    out = []
    if int(a.get("x", 0)) != int(b.get("x", 0)):
        out.append("x")
    if int(a.get("y", 0)) != int(b.get("y", 0)):
        out.append("y")
    if bool(a.get("visible", True)) != bool(b.get("visible", True)):
        out.append("visible")
    if int(a.get("rotation", 0) or 0) != int(b.get("rotation", 0) or 0):
        out.append("rotation")
    if a.get("pixels") != b.get("pixels"):
        out.append("pixels")
    return tuple(out)


def _mode_profile(obj: dict) -> str:
    """An object's own-mode profile: attribute configuration ignoring position.

    Position is excluded because moving is a transition, not a mode. The
    profile is (visible, rotation, pixel-pattern bucket).
    """
    vis = 1 if obj.get("visible", True) else 0
    rot = int(obj.get("rotation", 0) or 0)
    px = _pixels_hash_bucket(obj.get("pixels"))
    return f"vis={vis},rot={rot},px={px}"


def _primary_tag(obj: dict) -> str:
    tags = obj.get("tags") or []
    if tags:
        return str(tags[0])
    t = obj.get("type")
    if t:
        return str(t)
    return str(obj.get("name") or "Unknown")


def _click_xy(transition: dict) -> tuple[int, int] | None:
    cx = transition.get("click_x")
    cy = transition.get("click_y")
    if cx is None or cy is None:
        return None
    return int(cx), int(cy)


def _click_hits(obj: dict, cx: int, cy: int) -> bool:
    """Whether a display-space click lands in the object's display rectangle."""
    try:
        x = int(obj.get("display_x", obj.get("x", 0)))
        y = int(obj.get("display_y", obj.get("y", 0)))
        w = int(obj.get("display_w", obj.get("w", 1)) or 1)
        h = int(obj.get("display_h", obj.get("h", 1)) or 1)
    except (TypeError, ValueError):
        return False
    return x <= cx < x + w and y <= cy < y + h


def _chebyshev(a: dict, b: dict) -> int:
    return max(
        abs(int(a.get("x", 0)) - int(b.get("x", 0))),
        abs(int(a.get("y", 0)) - int(b.get("y", 0))),
    )


def _rect_in_counter_mask(obj: dict, mask) -> bool:
    """Whether the object's display rectangle lies entirely inside the
    validated move-counter mask (display-space (row, col) cells)."""
    if not mask:
        return False
    try:
        x = int(obj.get("display_x", obj.get("x", 0)))
        y = int(obj.get("display_y", obj.get("y", 0)))
        w = int(obj.get("display_w", obj.get("w", 1)) or 1)
        h = int(obj.get("display_h", obj.get("h", 1)) or 1)
    except (TypeError, ValueError):
        return False
    for yy in range(y, y + h):
        for xx in range(x, x + w):
            if (yy, xx) not in mask:
                return False
    return True


def _action_cell_key(action_id: int, transition: dict, target_obj: dict,
                     committed_features: list[dict] | None) -> str:
    """Full key of the action cell an engaged observation falls into: base
    action component plus the values of every committed ξ feature evaluated on
    (transition, target). Component order is canonical (sorted)."""
    if action_id == 6:
        xy = _click_xy(transition)
        base = (
            _CLICK_ON if xy is not None and _click_hits(target_obj, *xy)
            else _CLICK_OFF
        )
    else:
        base = f"a{int(action_id)}"
    parts: list[str] = []
    for feat in committed_features or []:
        try:
            parts.append(_xi_feature_value(feat, transition, target_obj))
        except Exception:
            parts.append(f"{feat.get('kind', '?')}=err")
    if not parts:
        return base
    return base + "|" + "|".join(sorted(parts))


def _split_key(key: str) -> tuple[str, frozenset[str]]:
    parts = key.split("|")
    return parts[0], frozenset(parts[1:])


def _outcome_hash(bo: dict, ao: dict | None) -> str:
    """Effect-delta symbol for CAI. Deltas, not absolute records: an
    autonomous drifter visits a fresh position every step, which would make
    each sample unique and saturate plug-in MI, while its delta is constant."""
    if ao is None:
        return "gone"
    dx = int(ao.get("x", 0)) - int(bo.get("x", 0))
    dy = int(ao.get("y", 0)) - int(bo.get("y", 0))
    dvis = int(bool(ao.get("visible", True)) != bool(bo.get("visible", True)))
    drot = (int(ao.get("rotation", 0) or 0) - int(bo.get("rotation", 0) or 0))
    dpx = int(
        _pixels_hash_bucket(ao.get("pixels"))
        != _pixels_hash_bucket(bo.get("pixels"))
    )
    return f"d{dx},{dy};v{dvis};r{drot};px{dpx}"


def _plugin_mutual_information(joint: Counter) -> float:
    """I(A; O) / log(min(|A|, |O|)) over an (action, outcome) Counter, zero
    when either marginal is degenerate."""
    n = sum(joint.values())
    if n == 0:
        return 0.0
    pa: Counter = Counter()
    po: Counter = Counter()
    for (a, o), c in joint.items():
        pa[a] += c
        po[o] += c
    if len(pa) < 2 or len(po) < 2:
        return 0.0
    mi = 0.0
    for (a, o), c in joint.items():
        p_ao = c / n
        mi += p_ao * math.log(p_ao / ((pa[a] / n) * (po[o] / n)))
    norm = math.log(min(len(pa), len(po)))
    return max(0.0, min(1.0, mi / norm))


def compute_cai(
    transitions: list[dict],
    *,
    committed_features: list[dict] | None = None,
    min_context_n: int = 3,
) -> dict[str, float]:
    """CAI(o): max over contexts (n >= min_context_n) of normalized
    I(effect-delta ; action | context).

    Uses all present-object samples, not engagement-gated, since an idle
    object's action-independence is exactly the CAI = 0 verdict. Plug-in MI
    biases up on sparse contexts and the context-size gate bounds that.

    A click's intervention content is per-object (on-target or not), so the
    action variable is a6:on / a6:off relative to each object. Otherwise a
    click-only game has a constant action marginal and CAI degenerates to 0
    for everything. For the same reason the conditioning context must be
    action-independent: click-derived features would absorb the contrast.
    """
    ctx_features = [
        f for f in (committed_features or [])
        if f.get("kind") != "click_offset"
    ]
    # name -> context -> Counter[(action_symbol, outcome_hash)]
    tables: dict[str, dict[str, Counter]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    for t in transitions:
        before = t.get("before_state") or []
        after = t.get("after_state") or []
        action_id = int(t.get("action_id", -1))
        xy = _click_xy(t) if action_id == 6 else None
        pairs = _pair_before_after(before, after)
        seen: set[tuple[str, str, str, str]] = set()
        for bo, ao in pairs[: len(before)]:
            if bo is None:
                continue
            name = bo.get("name")
            if not name:
                continue
            name = str(name)
            if xy is not None:
                act = "a6:on" if _click_hits(bo, *xy) else "a6:off"
            else:
                act = f"a{action_id}"
            ctx = _augmented_context_signature(t, bo, ctx_features)
            outcome = _outcome_hash(bo, ao)
            key = (name, ctx, act, outcome)
            if key in seen:
                continue
            seen.add(key)
            tables[name][ctx][(act, outcome)] += 1

    out: dict[str, float] = {}
    for name, contexts in tables.items():
        best = 0.0
        for joint in contexts.values():
            if sum(joint.values()) < min_context_n:
                continue
            best = max(best, _plugin_mutual_information(joint))
        out[name] = round(best, 6)
    return out


class SigmaState:
    """Incremental per-object substrate. ``observe`` one serialized
    transition at a time, then ``table``/``dump_sigma`` render the artifact."""

    def __init__(
        self,
        *,
        alpha: float = 0.9,
        alpha_0: float = 1.0,
        lp_window: int = 5,
        m_min: int = 2,
        eps: float = 0.3,
        goal_cap: int = 3,
        omega: float = 0.5,
        model_resolved: bool = False,
    ) -> None:
        if not 0.0 < float(alpha) <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha!r}")
        if float(alpha_0) <= 0.0:
            raise ValueError(f"alpha_0 must be > 0, got {alpha_0!r}")
        if int(lp_window) < 1:
            raise ValueError(f"lp_window must be >= 1, got {lp_window!r}")
        if int(m_min) < 1:
            raise ValueError(f"m_min must be >= 1, got {m_min!r}")
        if not 0.0 <= float(eps) <= 1.0:
            raise ValueError(f"eps must be in [0, 1], got {eps!r}")
        if int(goal_cap) < 0:
            raise ValueError(f"goal_cap must be >= 0, got {goal_cap!r}")
        if float(omega) < 0.0:
            raise ValueError(f"omega must be >= 0, got {omega!r}")
        self.alpha = float(alpha)          # prequential fading factor
        self.alpha_0 = float(alpha_0)      # Dirichlet/Laplace pseudo-count
        self.lp_window = int(lp_window)
        self.m_min = int(m_min)
        self.eps = float(eps)
        self.goal_cap = int(goal_cap)
        self.omega = float(omega)          # novelty weight in Π
        # three-state U_c: the prediction record resolves mixed cells (not
        # persisted; the engine re-applies its config on construction)
        self.model_resolved = bool(model_resolved)

        self._tag_of: dict[str, str] = {}
        self._action_cells: dict[str, dict[str, Counter]] = defaultdict(dict)
        self._mode_cells: dict[str, dict[str, Counter]] = defaultdict(dict)
        self._own_profiles: dict[str, set[str]] = defaultdict(set)
        self._tag_profiles: dict[str, set[str]] = defaultdict(set)
        self._k_att: dict[str, dict[str, int]] = defaultdict(dict)
        self._goal_modes: dict[str, dict[str, str]] = defaultdict(dict)
        self._alphabet: set[str] = set()

        self._available_actions: list[int] = []
        self._committed_features: list[dict] = []
        self.n_transitions = 0
        self.last_step: int | None = None

        self._preq_S: dict[str, float] = {}
        self._preq_N: dict[str, float] = {}
        # per fluent-family fading loss: which CPF is failing, not just
        # which object
        self._preq_field_S: dict[str, dict[str, float]] = {}
        self._preq_field_N: dict[str, dict[str, float]] = {}
        # admitted model-fluent refinements, wholesale-replaced per harvest
        self._model_refinements: dict[str, list[dict]] = {}
        # ring of C values over the last lp_window+1 engaged encounters
        self._c_hist: dict[str, list[float]] = {}
        # action-cell and mode-cell keys share this dict (disjoint namespaces)
        self._cell_pred: dict[str, dict[str, list[int]]] = defaultdict(dict)
        self._cai: dict[str, float] = {}
        self._reward_tags: set[str] = set()
        # transient reward lookback, reconstructed by replay after a resume
        self._last_delta_tags: set[str] = set()
        self._last_delta_tags_level: int | None = None
        # transient failed-probe anchor: names that changed last transition
        self._last_changed_names: set[str] = set()

    # ------------------------------------------------------------------ #
    # Observation                                                         #
    # ------------------------------------------------------------------ #

    def observe(
        self,
        transition: dict,
        *,
        available_actions: list[int] | None = None,
        committed_features: list[dict] | None = None,
        step: int | None = None,
        predicted_state: Any = None,
        counter_mask: Any = None,
    ) -> None:
        """Fold one serialized transition into the substrate.

        ``predicted_state`` is the live model's pre-repair prediction for
        this same transition (a state list, ``MODEL_ERROR``, or None when no
        model ran); engaged objects are scored against it prequentially.
        ``counter_mask`` is the model's validated move-counter strip: the
        verifier exempts sprites inside it, so they must not accrue
        prequential loss either.
        """
        if available_actions is not None:
            self._available_actions = sorted(
                int(a) for a in available_actions
            )
        if committed_features is not None:
            self._committed_features = list(committed_features)
        self.n_transitions += 1
        if step is not None:
            self.last_step = int(step)

        before = transition.get("before_state") or []
        after = transition.get("after_state") or []
        action_id = int(transition.get("action_id", -1))
        xy = _click_xy(transition) if action_id == 6 else None

        # Mode registration is presence-gated, not engagement-gated: seeing a
        # configuration proves the mode is real.
        for obj in list(before) + list(after):
            name = obj.get("name")
            if not name:
                continue
            name = str(name)
            tag = self._tag_of.setdefault(name, _primary_tag(obj))
            profile = _mode_profile(obj)
            self._own_profiles[name].add(profile)
            self._tag_profiles[tag].add(profile)

        pairs = _pair_before_after(before, after)

        changed_objs: list[dict] = []
        sig_of: list[tuple[dict | None, dict | None, str]] = []
        for bo, ao in pairs:
            sig = _effect_signature(bo, ao)
            sig_of.append((bo, ao, sig))
            if sig != "no_change":
                changed_objs.append(bo if bo is not None else ao)

        # The ARC-3 reward transition is also the level sweep where every
        # object reads gone or born, so reward adjacency counts only
        # attribute-delta changes and looks back one same-level transition
        # for the completing move.
        delta_tags = {
            _primary_tag(bo)
            for bo, ao in pairs
            if bo is not None and ao is not None
            and _effect_signature(bo, ao) != "no_change"
        }
        if float(transition.get("reward", 0.0) or 0.0) > 0.0:
            self._reward_tags.update(delta_tags)
            if self._last_delta_tags_level == transition.get("level"):
                self._reward_tags.update(self._last_delta_tags)
        self._last_delta_tags = delta_tags
        self._last_delta_tags_level = transition.get("level")

        # dedup per (name, cell, sig) per transition to avoid
        # same-name-multiplicity bias, mirroring epistemic._stratify
        seen: set[tuple[str, str, str]] = set()
        engaged: list[tuple[int, str, str, str | None]] = []

        for idx, (bo, ao, sig) in enumerate(sig_of):
            ref = bo if bo is not None else ao
            name = ref.get("name")
            if not name:
                continue
            name = str(name)

            responded = sig != "no_change"
            targeted = xy is not None and bo is not None and _click_hits(bo, *xy)
            proximal = False
            if not responded and not targeted and bo is not None:
                for ch in changed_objs:
                    if ch is bo:
                        continue
                    if _chebyshev(bo, ch) <= 1:
                        proximal = True
                        break
            if not (responded or targeted or proximal):
                continue  # a non-event is not evidence

            self._alphabet.add(sig)

            target = bo if bo is not None else ao
            cell = _action_cell_key(
                action_id, transition, target, self._committed_features,
            )
            key = (name, cell, sig)
            if key not in seen:
                seen.add(key)
                ctr = self._action_cells[name].setdefault(cell, Counter())
                ctr[sig] += 1
                # attempt pruning is reversible on any observation
                self._k_att[name].pop(cell, None)

            profile: str | None = None
            if bo is not None:
                profile = _mode_profile(bo)
                mkey = (name, _MODE_PREFIX + profile, sig)
                if mkey not in seen:
                    seen.add(mkey)
                    mctr = self._mode_cells[name].setdefault(
                        profile, Counter()
                    )
                    mctr[sig] += 1

            engaged.append((idx, name, cell, profile))

        # Failed-probe attempts: a coordinate-free action that changed
        # NOTHING board-wide leaves no engagement evidence, yet objects
        # adjacent to the last-active entity were plausibly subjected to it.
        # Their attempted cells decay reversibly (P4: clicks and
        # near-a-change cases already self-prune through engagement).
        if (
            action_id in _BACKBONE_ACTIONS
            and not changed_objs
            and self._last_changed_names
        ):
            anchors = [
                o for o in before
                if str(o.get("name", "")) in self._last_changed_names
            ]
            for bo, _ao, _sig in sig_of:
                if bo is None or not bo.get("name"):
                    continue
                name = str(bo["name"])
                if name in self._last_changed_names:
                    continue
                if any(_chebyshev(bo, a) <= 1 for a in anchors):
                    self.note_attempt(name, _action_cell_key(
                        action_id, transition, bo,
                        self._committed_features,
                    ))
        self._last_changed_names = {
            str((bo if bo is not None else ao).get("name", ""))
            for bo, ao, sig in sig_of if sig != "no_change"
        } - {""}

        if predicted_state is not None and engaged:
            scoreable = [
                e for e in engaged
                if not _rect_in_counter_mask(
                    (pairs[e[0]][0] or pairs[e[0]][1]) or {}, counter_mask,
                )
            ]
            if scoreable:
                self._score_predictions(
                    before, pairs, predicted_state, scoreable,
                )

    def _score_predictions(
        self,
        before: list[dict],
        pairs_actual: list[tuple[dict | None, dict | None]],
        predicted_state: Any,
        engaged: list[tuple[int, str, str, str | None]],
    ) -> None:
        """Fold 0/1 forward losses for every engaged object into the fading
        accumulators and the per-class prediction counts."""
        if isinstance(predicted_state, str):
            if predicted_state != MODEL_ERROR:
                raise ValueError(
                    "predicted_state must be a state list, MODEL_ERROR, or "
                    f"None, got string {predicted_state!r}"
                )
            mismatches = {
                idx: FLUENT_FAMILIES for idx, _, _, _ in engaged
            }
        else:
            mismatches = self._pairwise_mismatches(
                before, pairs_actual, list(predicted_state), engaged,
            )
        losses = {idx: (1 if fields else 0) for idx, fields in
                  mismatches.items()}

        # per (name, cell) dedup takes the max loss across instances
        cell_loss: dict[tuple[str, str], int] = {}
        name_loss: dict[str, int] = {}
        for idx, name, cell, profile in engaged:
            loss = losses[idx]
            name_loss[name] = max(name_loss.get(name, 0), loss)
            cell_loss[(name, cell)] = max(cell_loss.get((name, cell), 0), loss)
            if profile is not None:
                mkey = _MODE_PREFIX + profile
                cell_loss[(name, mkey)] = max(
                    cell_loss.get((name, mkey), 0), loss,
                )
        for (name, cell), loss in cell_loss.items():
            n_pred, k_correct = self._cell_pred[name].get(cell, (0, 0))
            self._cell_pred[name][cell] = [n_pred + 1, k_correct + (1 - loss)]
        name_fields: dict[str, set[str]] = {}
        for idx, name, _cell, _profile in engaged:
            name_fields.setdefault(name, set()).update(mismatches[idx])
        for name, loss in name_loss.items():
            s = loss + self.alpha * self._preq_S.get(name, 0.0)
            n = 1.0 + self.alpha * self._preq_N.get(name, 0.0)
            self._preq_S[name] = s
            self._preq_N[name] = n
            hist = self._c_hist.setdefault(name, [])
            hist.append(round(1.0 - s / n, 6))
            del hist[: -(self.lp_window + 1)]
            missed = name_fields.get(name, set())
            fs = self._preq_field_S.setdefault(name, {})
            fn = self._preq_field_N.setdefault(name, {})
            for fam in FLUENT_FAMILIES:
                fs[fam] = (
                    (1.0 if fam in missed else 0.0)
                    + self.alpha * fs.get(fam, 0.0)
                )
                fn[fam] = 1.0 + self.alpha * fn.get(fam, 0.0)

    @staticmethod
    def _pairwise_mismatches(
        before: list[dict],
        pairs_actual: list[tuple[dict | None, dict | None]],
        predicted_state: list,
        engaged: list[tuple[int, str, str, str | None]],
    ) -> dict[int, tuple[str, ...]]:
        """Mismatched fluent families per engaged pair index (empty tuple
        means the forward prediction was exact).

        ``_pair_before_after`` emits one entry per before-object in order,
        then births, so the first ``len(before)`` entries of the actual and
        predicted pairings align index for index. Actual births match
        predicted births by name and record agreement; an unmatched birth
        is an existence miss.
        """
        predicted = [o for o in predicted_state if isinstance(o, dict)]
        pairs_pred = _pair_before_after(before, predicted)
        n_before = len(before)
        pred_borns = [ao for _, ao in pairs_pred[n_before:] if ao is not None]

        out: dict[int, tuple[str, ...]] = {}
        for idx, _, _, _ in engaged:
            if idx < n_before:
                out[idx] = _record_mismatch_fields(
                    pairs_actual[idx][1], pairs_pred[idx][1],
                )
            else:
                born = pairs_actual[idx][1]
                same_name = [
                    p for p in pred_borns
                    if p.get("name") == born.get("name")
                ]
                if any(_records_match(born, p) for p in same_name):
                    out[idx] = ()
                elif same_name:
                    out[idx] = _record_mismatch_fields(born, same_name[0])
                else:
                    out[idx] = ("existence",)
        return out

    # ------------------------------------------------------------------ #
    # Attempt pruning and goal modes                                      #
    # ------------------------------------------------------------------ #

    def _class_uncertainty(
        self, entropy_u: float, n_pred: int, k_pred: int,
    ) -> tuple[float, str | None]:
        """U_c with its resolution provenance.

        Default composition is the max of the two one-sided alarms. With
        ``model_resolved`` on and enough scored forward predictions, the
        prediction record is authoritative: a mixed cell the model predicts
        exactly is a representational confound (the cell key cannot express
        the model's fingerprint), not an epistemic unknown, and it resolves
        instead of reading unknown forever. A mispredicted cell stays open
        under both regimes.
        """
        if n_pred <= 0:
            return entropy_u, None
        floor_u = _posterior_error_rate(n_pred, k_pred, self.alpha_0)
        if self.model_resolved and n_pred >= MODEL_RESOLVED_MIN_PRED:
            if entropy_u > self.eps >= floor_u:
                return floor_u, "model"
            return floor_u, None
        return max(entropy_u, floor_u), None

    def set_model_fluents(self, report: dict | None) -> None:
        """Absorb a fluents.harvest_and_dispose report: admitted fluents'
        bins become model-provenance classes and their parent cells stop
        double counting. Wholesale replacement per harvest; the whole
        structure is recomputable from (buffer, current code)."""
        refs: dict[str, list[dict]] = {}
        for fname, entry in ((report or {}).get("fluents") or {}).items():
            if entry.get("status") != "admitted":
                continue
            for adm in entry.get("admitted", []):
                refs.setdefault(str(adm["tag"]), []).append({
                    "fluent": str(fname),
                    "cell": str(adm["cell"]),
                    "bins": [dict(b) for b in adm.get("bins", [])],
                    "holes": [str(h) for h in adm.get("holes", [])],
                })
        self._model_refinements = refs

    def evidence_signature(self) -> tuple[int, int, int]:
        """Cheap progress fingerprint: distinct exercised action cells,
        distinct engaged mode cells, alphabet size. Unchanged across steps
        means no new KIND of evidence has arrived, whatever the step count."""
        n_cells = sum(len(c) for c in self._action_cells.values())
        n_modes = sum(len(m) for m in self._mode_cells.values())
        return (n_cells, n_modes, len(self._alphabet))

    def briefing(
        self,
        *,
        reason: str = "",
        aliases: dict | None = None,
        max_items: int = 4,
    ) -> str:
        """Condensed hypothesis briefing for the acting agent: LP movers and
        discrete holes, one line each with the claim it makes. Never a raw
        eta sort (raw eta ranks the frozen ceiling, not the frontier).
        Empty string when there is nothing worth saying."""
        lines: list[str] = []

        movers = []
        for name, hist in self._c_hist.items():
            if len(hist) >= 2 and abs(hist[-1] - hist[0]) > 0.0:
                movers.append((abs(hist[-1] - hist[0]), name, hist))
        for _lp, name, hist in sorted(movers, reverse=True)[:max_items]:
            verb = "fell" if hist[-1] < hist[0] else "rose"
            lines.append(
                f"- {name} ({self._tag_of.get(name, '?')}): model accuracy "
                f"{verb} {hist[0]:.2f}->{hist[-1]:.2f} over its last "
                f"engagements. The frontier is here: still being learned, "
                f"or broken by the last revision."
            )

        n_model_holes = 0
        for tag, refs in sorted(self._model_refinements.items()):
            for ref in refs:
                for hv in ref.get("holes", []):
                    if n_model_holes >= max_items:
                        break
                    n_model_holes += 1
                    lines.append(
                        f"- {tag}: the model's own {ref['fluent']}={hv} "
                        f"under {ref['cell']} has never been tested. One "
                        f"probe either confirms its rule or yields a "
                        f"counterexample."
                    )

        n_goal = 0
        for name, modes in sorted(self._goal_modes.items()):
            for mode, pid in sorted(modes.items()):
                if n_goal >= max_items:
                    break
                n_goal += 1
                lines.append(
                    f"- {name}: the goal hypothesis requires mode '{mode}' "
                    f"(predicate {pid}); no observed state has ever "
                    f"satisfied it."
                )

        n_pooled = 0
        for name in sorted(self._tag_of):
            if n_pooled >= max_items:
                break
            tag = self._tag_of[name]
            unseen = (
                self._tag_profiles.get(tag, set())
                - self._own_profiles.get(name, set())
            )
            if unseen:
                n_pooled += 1
                lines.append(
                    f"- {name} ({tag}): same-tag instances exhibit "
                    f"{len(unseen)} configuration(s) this one never has. A "
                    f"reachable state of it likely remains unvisited."
                )

        if not lines:
            return ""
        header = (
            "EPISTEMIC BRIEFING"
            + (f" ({reason})" if reason else "")
            + ": mechanically tracked leads, not truths. Each line names "
            "an untested claim and the probe that would settle it."
        )
        footer = (
            "Full tables: sigma.json (per-class evidence), "
            "model_fluents.json (the model's declared predicates and their "
            "untested values). The tracker only sees situations its "
            "vocabulary expresses; treat these as leads to challenge, "
            "alongside your own reading of the board."
        )
        return header + "\n" + "\n".join(lines) + "\n" + footer

    def forward_record(
        self, tag: Any, action_id: Any,
    ) -> dict[str, int] | None:
        """Aggregate forward-prediction record over every instance of a tag
        in the action's cells: how often the live model was tested there and
        how often it was exactly right. None when never tested."""
        try:
            aid = int(action_id)
        except (TypeError, ValueError):
            return None
        bases = (
            {_CLICK_ON, _CLICK_OFF} if aid == 6 else {f"a{aid}"}
        )
        n_sum, k_sum = 0, 0
        for name, cells in self._cell_pred.items():
            if self._tag_of.get(name) != str(tag):
                continue
            for cell, (n_pred, k_pred) in cells.items():
                if cell.split("|", 1)[0] in bases:
                    n_sum += int(n_pred)
                    k_sum += int(k_pred)
        if n_sum == 0:
            return None
        return {"n_pred": n_sum, "k_pred": k_sum}

    def note_attempt(self, object_name: str, class_key: str) -> None:
        """Record a targeted probe of an unexercised action cell that induced
        nothing. Mode cells never accrue attempt debt."""
        name = str(object_name)
        if class_key.startswith(_MODE_PREFIX):
            return
        if class_key in self._action_cells.get(name, {}):
            return  # exercised: attempts are moot
        cur = self._k_att[name].get(class_key, 0)
        self._k_att[name][class_key] = cur + 1

    def inject_goal_modes(self, requirements: list[dict]) -> dict[str, Any]:
        """Add goal-stratum mode classes, capped at ``goal_cap`` per object.
        Each entry: {"object": name_or_tag, "mode": label, "predicate_id": id}.

        The table is keyed by sprite name, but grounding predicates usually
        name a tag. A tag resolves to every instance carrying it. An object
        matching neither a known name nor a known tag is counted as
        unmatched, never stored under a dead key that no row would render.
        """
        summary = {"added": 0, "capped": 0, "duplicates": 0, "unmatched": 0}
        for entry in requirements or []:
            obj = str(entry.get("object", "") or "")
            mode = str(entry.get("mode", "") or "")
            pid = str(entry.get("predicate_id", "") or "")
            if not obj or not mode:
                continue
            if obj in self._tag_of:
                names = [obj]
            else:
                names = [
                    n for n, tag in self._tag_of.items() if tag == obj
                ]
            if not names:
                summary["unmatched"] += 1
                continue
            for name in names:
                modes = self._goal_modes[name]
                if mode in modes:
                    summary["duplicates"] += 1
                    continue
                if len(modes) >= self.goal_cap:
                    summary["capped"] += 1
                    continue
                modes[mode] = pid
                summary["added"] += 1
        return summary

    def set_cai(self, cai: dict[str, float]) -> None:
        """Replace the CAI map. ``compute_cai`` needs the full buffer, so it
        runs batchwise on the engine cadence rather than inside observe."""
        self._cai = {str(n): float(v) for n, v in (cai or {}).items()}

    def _rel_of(self, tag: str, aliases: dict | None) -> float:
        """Goal-relevance proxy: 1 for tags that changed at a reward step,
        else the softened top committed-alias score, zeroed for decorative
        commitments. A soft acquisition prior, never a verified quantity."""
        if tag in self._reward_tags:
            return 1.0
        entries = (aliases or {}).get(tag) or []
        best_alias, best_score = None, 0
        for e in entries:
            if not isinstance(e, dict):
                continue
            a = str(e.get("alias", ""))
            s = int(e.get("score", 0) or 0)
            if a.startswith("unknown_") or s <= 0:
                continue
            if s > best_score:
                best_alias, best_score = a, s
        if best_alias is None:
            return 0.0
        head = best_alias.strip().lower().split("_", 1)[0]
        if head in _DECORATIVE_HEADS:
            return 0.0
        return round(best_score / (best_score + _REL_SCORE_SOFTENING), 6)

    def retire_goal_modes(self, predicate_ids: list[str]) -> int:
        """Dispose goal modes whose predicates were dropped (goal-hypothesis
        generalization) or proven unnecessary (reward fired without them)."""
        drop = {str(p) for p in predicate_ids or []}
        removed = 0
        for name in list(self._goal_modes.keys()):
            modes = self._goal_modes[name]
            for mode in [m for m, pid in modes.items() if pid in drop]:
                del modes[mode]
                removed += 1
        return removed

    def goal_predicate_ids(self) -> list[str]:
        return sorted({
            pid
            for modes in self._goal_modes.values()
            for pid in modes.values()
            if pid
        })

    # ------------------------------------------------------------------ #
    # Enumeration and the measure                                         #
    # ------------------------------------------------------------------ #

    def _backbone_cells(self) -> tuple[list[str], bool]:
        """Enumerable coarse action cells under the current vocabulary:
        available actions crossed with closed-codomain ξ bins
        (``neighbour_at_offset`` over known tags ∪ {none}). Returns
        (cells, truncated_flag)."""
        bases: list[str] = [
            f"a{a}" for a in self._available_actions
            if a in _BACKBONE_ACTIONS
        ]
        if 6 in self._available_actions:
            bases.extend([_CLICK_ON, _CLICK_OFF])

        closed_bins: list[list[str]] = []
        tags = sorted(self._tag_profiles.keys())
        for feat in self._committed_features:
            if feat.get("kind") != "neighbour_at_offset":
                continue  # open codomains contribute observed bins only
            dx, dy = int(feat.get("dx", 0)), int(feat.get("dy", 0))
            closed_bins.append(
                [f"@{dx},{dy}={t}" for t in tags] + [f"@{dx},{dy}=none"]
            )

        if not closed_bins:
            return bases, False

        n_product = 1
        for b in closed_bins:
            n_product *= len(b)
        truncated = n_product > MAX_ENUM_BINS_PER_ACTION

        cells: list[str] = []
        for base in bases:
            combos = itertools.islice(
                itertools.product(*closed_bins), MAX_ENUM_BINS_PER_ACTION,
            )
            for combo in combos:
                cells.append(base + "|" + "|".join(sorted(combo)))
        return cells, truncated

    def _object_rows(
        self, aliases: dict | None = None,
    ) -> tuple[list[dict], bool]:
        coarse_cells, truncated = self._backbone_cells()
        E = max(2, len(self._alphabet))
        rows: list[dict] = []

        for name in sorted(self._tag_of.keys()):
            tag = self._tag_of[name]
            observed = self._action_cells.get(name, {})
            classes: list[dict] = []
            # Directly actionable = some exercised action cell produced a
            # real effect (or CAI clears the threshold, checked below).
            direct = False

            # Exercised action cells. Once forward predictions exist, U_c is
            # floored by the class's misprediction rate (see
            # _posterior_error_rate): consistent effects alone do not make a
            # class known if the model cannot predict them.
            preds = self._cell_pred.get(name, {})
            for cell, ctr in sorted(observed.items()):
                if any(sig != "no_change" for sig in ctr):
                    direct = True
                n_c = sum(ctr.values())
                entropy_u = _dirichlet_entropy(ctr, E, self.alpha_0)
                n_pred, k_pred = preds.get(cell, (0, 0))
                u_c, resolved = self._class_uncertainty(
                    entropy_u, n_pred, k_pred,
                )
                cls = {
                    "key": cell,
                    "provenance": "backbone",
                    "n": n_c,
                    "n_pred": n_pred,
                    "k_pred": k_pred,
                    "k_att": 0,
                    "w": 1.0,
                    "U": round(u_c, 6),
                    "modal_effect": ctr.most_common(1)[0][0],
                    "seen": True,
                }
                if resolved:
                    cls["resolved_by"] = resolved
                classes.append(cls)

            # Unexercised coarse cells: a coarse cell is covered when some
            # observed cell shares its base and includes all its components.
            obs_split = [_split_key(k) for k in observed.keys()]
            for cell in coarse_cells:
                base, comps = _split_key(cell)
                covered = any(
                    ob == base and comps <= oc for ob, oc in obs_split
                )
                if covered:
                    continue
                k_att = int(self._k_att.get(name, {}).get(cell, 0))
                classes.append({
                    "key": cell,
                    "provenance": "backbone",
                    "n": 0,
                    "n_pred": 0,
                    "k_pred": 0,
                    "k_att": k_att,
                    "w": round(1.0 / (1.0 + k_att), 6),
                    "U": 1.0,
                    "modal_effect": None,
                    "seen": False,
                })

            # Mode cells: own + pooled profiles, then goal-injected labels.
            own = self._own_profiles.get(name, set())
            pooled = self._tag_profiles.get(tag, set())
            engaged_modes = self._mode_cells.get(name, {})
            for profile in sorted(own | pooled):
                ctr = engaged_modes.get(profile, Counter())
                n_c = sum(ctr.values())
                entropy_u = (
                    _dirichlet_entropy(ctr, E, self.alpha_0)
                    if n_c > 0 else 1.0
                )
                mkey = _MODE_PREFIX + profile
                n_pred, k_pred = preds.get(mkey, (0, 0))
                u_c, resolved = self._class_uncertainty(
                    entropy_u, n_pred, k_pred,
                )
                cls = {
                    "key": mkey,
                    "provenance": (
                        "observed" if profile in own else "pooled"
                    ),
                    "n": n_c,
                    "n_pred": n_pred,
                    "k_pred": k_pred,
                    "k_att": 0,
                    "w": 1.0,
                    "U": round(u_c, 6),
                    "modal_effect": (
                        ctr.most_common(1)[0][0] if n_c > 0 else None
                    ),
                    "seen": profile in own,
                }
                if resolved:
                    cls["resolved_by"] = resolved
                classes.append(cls)
            for mode, pid in sorted(self._goal_modes.get(name, {}).items()):
                classes.append({
                    "key": _GOAL_MODE_PREFIX + mode,
                    "provenance": "goal",
                    "predicate_id": pid,
                    "n": 0,
                    "n_pred": 0,
                    "k_pred": 0,
                    "k_att": 0,
                    "w": 1.0,
                    "U": 1.0,
                    "modal_effect": None,
                    "seen": False,
                })

            # Admitted model-fluent refinements: bins replace their parent
            # cell in the aggregate (the parent stays displayed at w=0 with
            # a refined_by mark), produced-but-untested bins are holes.
            # Bin evidence is tag-level: pooled over same-tag instances.
            for ref in self._model_refinements.get(tag, []):
                fname, cell = ref["fluent"], ref["cell"]
                parent = next(
                    (c for c in classes if c["key"] == cell
                     and c["provenance"] == "backbone"),
                    None,
                )
                if parent is not None:
                    parent["w"] = 0.0
                    parent.setdefault("refined_by", []).append(fname)
                for b in ref.get("bins", []):
                    classes.append({
                        "key": f"{cell}|model:{fname}={b['value']}",
                        "provenance": "model",
                        "n": int(b.get("n", 0)),
                        "n_pred": 0,
                        "k_pred": 0,
                        "k_att": 0,
                        "w": 1.0,
                        "U": float(b.get("U", 1.0)),
                        "modal_effect": b.get("modal_outcome"),
                        "seen": True,
                        "tag_level": True,
                    })
                for hv in ref.get("holes", []):
                    classes.append({
                        "key": f"{cell}|model:{fname}={hv}",
                        "provenance": "model",
                        "n": 0,
                        "n_pred": 0,
                        "k_pred": 0,
                        "k_att": 0,
                        "w": 1.0,
                        "U": 1.0,
                        "modal_effect": None,
                        "seen": False,
                        "tag_level": True,
                    })

            active = [c for c in classes if c["w"] > 0]
            K = len(active)
            if K == 0:
                continue
            w_sum = sum(c["w"] for c in active)
            eta = (
                sum(c["w"] * c["U"] for c in active) / w_sum
                if w_sum > 0 else 1.0
            )
            eta = 0.0 if eta < 0.0 else (1.0 if eta > 1.0 else eta)
            n_total = sum(c["n"] for c in active)
            cov = sum(
                1 for c in active
                if c["n"] >= self.m_min and c["U"] <= self.eps
            ) / K
            nu = (1.0 + n_total) ** -0.5

            n_eff = self._preq_N.get(name, 0.0)
            c_val = (
                round(1.0 - self._preq_S[name] / n_eff, 6)
                if n_eff > 0 else None
            )
            field_s = self._preq_field_S.get(name, {})
            field_n = self._preq_field_N.get(name, {})
            c_fields = {
                fam: (
                    round(1.0 - field_s.get(fam, 0.0) / field_n[fam], 6)
                    if field_n.get(fam, 0.0) > 0 else None
                )
                for fam in FLUENT_FAMILIES
            }
            hist = self._c_hist.get(name, [])
            lp_val = (
                round(abs(hist[-1] - hist[0]), 6)
                if len(hist) >= 2 else None
            )

            cai_val = self._cai.get(name)
            rel_val = self._rel_of(tag, aliases)
            # Measured CAI overrides the any-effect heuristic in both
            # directions. An autonomous drifter has real effects in its
            # action cells yet is not directly actionable.
            if cai_val is not None:
                direct = cai_val >= _CAI_ACTIONABLE_THRESHOLD
            cost = _COST_DIRECT if direct else _COST_AUTONOMOUS
            # novelty is additive outside the rel product so a rel
            # false-negative cannot starve discovery
            pi_val = (rel_val * (lp_val or 0.0) + self.omega * nu) / cost

            rows.append({
                "name": name,
                "tag": tag,
                "K": K,
                "eta": round(eta, 6),
                "cov": round(cov, 6),
                "nu": round(nu, 6),
                "n_total": n_total,
                "C": c_val,
                "C_n": round(n_eff, 3),
                "C_fields": c_fields,
                "LP": lp_val,
                "CAI": cai_val,
                "rel": rel_val,
                "Pi": round(pi_val, 6),
                "classes": sorted(
                    classes, key=lambda c: (-c["U"], c["key"]),
                ),
            })
            if self.model_resolved:
                rows[-1]["eta_structural"] = round(
                    sum(
                        c["w"] for c in active
                        if c.get("resolved_by") == "model"
                    ) / w_sum if w_sum > 0 else 0.0,
                    6,
                )

        rows.sort(key=lambda r: (-r["eta"], r["name"]))
        return rows, truncated

    def eta(self, object_name: str) -> float | None:
        """η(o) for one object, or None if unknown."""
        rows, _ = self._object_rows()
        for r in rows:
            if r["name"] == str(object_name):
                return r["eta"]
        return None

    def table(self, *, aliases: dict | None = None) -> dict[str, Any]:
        """Render the Σ artifact payload. ``aliases`` is the analyzer's
        ``type_aliases`` map, consumed only by the rel(o) proxy."""
        rows, truncated = self._object_rows(aliases)
        return {
            "version": SIGMA_ARTIFACT_VERSION,
            "n_transitions": self.n_transitions,
            "last_step": self.last_step,
            "effect_alphabet": sorted(self._alphabet),
            "effect_alphabet_size": max(2, len(self._alphabet)),
            "available_actions": list(self._available_actions),
            "committed_features": list(self._committed_features),
            "truncated_enumeration": truncated,
            "params": {
                "alpha": self.alpha,
                "alpha_0": self.alpha_0,
                "lp_window": self.lp_window,
                "m_min": self.m_min,
                "eps": self.eps,
                "goal_cap": self.goal_cap,
                "omega": self.omega,
            },
            "objects": rows,
        }

    # ------------------------------------------------------------------ #
    # Persistence                                                         #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": SIGMA_ARTIFACT_VERSION,
            "params": {
                "alpha": self.alpha,
                "alpha_0": self.alpha_0,
                "lp_window": self.lp_window,
                "m_min": self.m_min,
                "eps": self.eps,
                "goal_cap": self.goal_cap,
                "omega": self.omega,
            },
            "tag_of": dict(self._tag_of),
            "action_cells": {
                n: {c: dict(ctr) for c, ctr in cells.items()}
                for n, cells in self._action_cells.items()
            },
            "mode_cells": {
                n: {p: dict(ctr) for p, ctr in cells.items()}
                for n, cells in self._mode_cells.items()
            },
            "own_profiles": {
                n: sorted(v) for n, v in self._own_profiles.items()
            },
            "tag_profiles": {
                t: sorted(v) for t, v in self._tag_profiles.items()
            },
            "k_att": {n: dict(v) for n, v in self._k_att.items()},
            "goal_modes": {n: dict(v) for n, v in self._goal_modes.items()},
            "alphabet": sorted(self._alphabet),
            "available_actions": list(self._available_actions),
            "committed_features": list(self._committed_features),
            "n_transitions": self.n_transitions,
            "last_step": self.last_step,
            "preq_S": dict(self._preq_S),
            "preq_N": dict(self._preq_N),
            "preq_field_S": {
                n: dict(v) for n, v in self._preq_field_S.items()
            },
            "preq_field_N": {
                n: dict(v) for n, v in self._preq_field_N.items()
            },
            "model_refinements": {
                t: [dict(r) for r in refs]
                for t, refs in self._model_refinements.items()
            },
            "c_hist": {n: list(v) for n, v in self._c_hist.items()},
            "cell_pred": {
                n: {c: list(pair) for c, pair in cells.items()}
                for n, cells in self._cell_pred.items()
            },
            "cai": dict(self._cai),
            "reward_tags": sorted(self._reward_tags),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SigmaState":
        p = d.get("params") or {}
        state = cls(
            alpha=p.get("alpha", 0.9),
            alpha_0=p.get("alpha_0", 1.0),
            lp_window=p.get("lp_window", 5),
            m_min=p.get("m_min", 2),
            eps=p.get("eps", 0.3),
            goal_cap=p.get("goal_cap", 3),
            omega=p.get("omega", 0.5),
        )
        state._tag_of = dict(d.get("tag_of") or {})
        for n, cells in (d.get("action_cells") or {}).items():
            state._action_cells[n] = {
                c: Counter(ctr) for c, ctr in cells.items()
            }
        for n, cells in (d.get("mode_cells") or {}).items():
            state._mode_cells[n] = {
                pr: Counter(ctr) for pr, ctr in cells.items()
            }
        for n, v in (d.get("own_profiles") or {}).items():
            state._own_profiles[n] = set(v)
        for t, v in (d.get("tag_profiles") or {}).items():
            state._tag_profiles[t] = set(v)
        for n, v in (d.get("k_att") or {}).items():
            state._k_att[n] = {c: int(k) for c, k in v.items()}
        for n, v in (d.get("goal_modes") or {}).items():
            state._goal_modes[n] = dict(v)
        state._alphabet = set(d.get("alphabet") or [])
        state._available_actions = [
            int(a) for a in d.get("available_actions") or []
        ]
        state._committed_features = list(d.get("committed_features") or [])
        state.n_transitions = int(d.get("n_transitions") or 0)
        state.last_step = d.get("last_step")
        state._preq_S = {
            k: float(v) for k, v in (d.get("preq_S") or {}).items()
        }
        state._preq_N = {
            k: float(v) for k, v in (d.get("preq_N") or {}).items()
        }
        state._preq_field_S = {
            n: {f: float(v) for f, v in fams.items()}
            for n, fams in (d.get("preq_field_S") or {}).items()
        }
        state._preq_field_N = {
            n: {f: float(v) for f, v in fams.items()}
            for n, fams in (d.get("preq_field_N") or {}).items()
        }
        state._model_refinements = {
            t: [dict(r) for r in refs]
            for t, refs in (d.get("model_refinements") or {}).items()
        }
        state._c_hist = {
            n: [float(x) for x in v]
            for n, v in (d.get("c_hist") or {}).items()
        }
        for n, cells in (d.get("cell_pred") or {}).items():
            state._cell_pred[n] = {
                c: [int(pair[0]), int(pair[1])] for c, pair in cells.items()
            }
        state._cai = {
            n: float(v) for n, v in (d.get("cai") or {}).items()
        }
        state._reward_tags = set(d.get("reward_tags") or [])
        return state

    @classmethod
    def from_transitions(
        cls,
        transitions: list[dict],
        *,
        available_actions: list[int] | None = None,
        committed_features: list[dict] | None = None,
        **params: Any,
    ) -> "SigmaState":
        """Rebuild the buffer-derivable substrate by replaying ``observe``.

        k_att, goal modes, and prequential accumulators are not
        buffer-derivable and start empty here. The checkpoint carries them.
        """
        state = cls(**params)
        for i, t in enumerate(transitions):
            state.observe(
                t,
                available_actions=available_actions,
                committed_features=committed_features,
                step=t.get("timestep", i),
            )
        return state


def dump_sigma(
    state: SigmaState, path: str | Path, *, aliases: dict | None = None,
) -> dict[str, Any]:
    """Write the Σ artifact to ``path`` as JSON and return the payload (so
    callers can derive the per-step trace line without recomputing)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = state.table(aliases=aliases)
    path.write_text(json.dumps(payload, indent=2, default=str))
    return payload


def trace_record_from(payload: dict, step: int | None = None) -> dict:
    """Compact per-step trace line derived from a Σ artifact payload,
    appended to ``sigma_trace.jsonl`` by the engine."""
    return {
        "step": step if step is not None else payload.get("last_step"),
        "n_transitions": payload.get("n_transitions"),
        "objects": {
            r["name"]: {
                "eta": r["eta"],
                "C": r["C"],
                "LP": r["LP"],
                "cov": r["cov"],
                "Pi": r["Pi"],
                "n_total": r["n_total"],
            }
            for r in payload.get("objects", [])
        },
    }
