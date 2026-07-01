"""Ontology state: structural-commitment phase, candidate-search hyperparameters, and the eta_t/eta*_t trace (formalism §6)."""
from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from .epistemic import (
    compute_ontology_error_extended,
    compute_ontology_error_with_candidates,
)


class Phase(str, Enum):
    FLAT = "flat"
    CRYSTALLIZED = "crystallized"


class Ontology:
    """Owns the structural-commitment phase, candidate-search hyperparameters, and the eta_t/eta*_t trace.

    kappa matches the epistemic UCB confidence width by design. Pass epistemic_kappa and alpha_0 directly.
    """

    def __init__(
        self,
        *,
        alpha_0: float = 1.0,
        kappa: float = 2.0,
        max_candidates: int = 12,
    ) -> None:
        self.phase: Phase = Phase.FLAT
        self.alpha_0 = float(alpha_0)
        self.kappa = float(kappa)
        self.max_candidates = int(max_candidates)
        self._trace: list[dict[str, Any]] = []
        self._latest: dict[str, Any] | None = None
        self._committed_features: list[dict[str, Any]] = []


    def measure(
        self,
        step: int,
        replay_buffer: list[dict],
        aliases: dict[str, list[dict]] | None = None,
    ) -> dict[str, Any]:
        """Compute eta and eta* on the current buffer and append to the trace.

        When aliases is provided, also computes eta_extended (paper §5, Def. 1).
        """
        res = compute_ontology_error_with_candidates(
            replay_buffer,
            alpha_0=self.alpha_0,
            kappa=self.kappa,
            max_candidates=self.max_candidates,
            committed_features=self._committed_features or None,
        )
        rec = {
            "step": int(step),
            "phase": self.phase.value,
            "eta": res["flat_eta"],
            "eta_star": res["eta_star"],
            "eta_reduction": res["eta_reduction"],
            "best_candidate": res["best_candidate"],
            "n_strata": res["flat"]["n_strata"],
            "n_candidates": res["n_candidates"],
            "n_transitions": len(replay_buffer),
            "n_committed_features": len(self._committed_features),
        }
        if aliases is not None:
            ext = compute_ontology_error_extended(
                replay_buffer, aliases=aliases, alpha_0=self.alpha_0,
                committed_features=self._committed_features or None,
            )
            res["extended"] = ext
            rec["eta_extended"] = ext["eta_extended"]
            rec["eta_effect_component"] = ext["eta_effect_component"]
            rec["eta_role_component"] = ext["eta_role_component"]
            rec["n_candidate_roles"] = ext["n_candidate_roles"]
            rec["role_fallback"] = ext["fallback"]
        self._trace.append(rec)
        self._latest = res
        return res


    _XI_FEATURE_SCHEMA: dict[str, tuple[str, ...]] = {
        "target_field": ("field",),
        "neighbour_at_offset": ("dx", "dy"),
        "neighbourhood_radius": ("r",),
        "click_offset": (),
    }

    def _validate_feature(self, feat: Any) -> dict | None:
        if not isinstance(feat, dict):
            return None
        kind = feat.get("kind")
        if kind not in self._XI_FEATURE_SCHEMA:
            return None
        required = self._XI_FEATURE_SCHEMA[kind]
        for k in required:
            if k not in feat:
                return None
        norm: dict[str, Any] = {"kind": str(kind)}
        for k in required:
            norm[k] = feat[k]
        return norm

    def apply_xi_updates(
        self, updates: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply synth-emitted xi-refinement "add" entries. It is monotonically additive and returns a summary.

        Invalid entries are skipped and recorded under "errors". Duplicates are no-ops.
        """
        summary = {"added": 0, "skipped_invalid": 0, "duplicates": 0,
                   "errors": []}
        adds = updates.get("add", []) if isinstance(updates, dict) else []
        if not isinstance(adds, list):
            summary["errors"].append("'add' is not a list")
            return summary
        for entry in adds:
            norm = self._validate_feature(entry)
            if norm is None:
                summary["skipped_invalid"] += 1
                summary["errors"].append(f"invalid feature: {entry!r}")
                continue
            if norm in self._committed_features:
                summary["duplicates"] += 1
                continue
            self._committed_features.append(norm)
            summary["added"] += 1
        return summary

    def committed_features(self) -> list[dict]:
        return list(self._committed_features)


    def dump(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "phase": self.phase.value,
            "alpha_0": self.alpha_0,
            "kappa": self.kappa,
            "max_candidates": self.max_candidates,
            "committed_features": list(self._committed_features),
            "trace": self._trace,
            "latest": self._latest,
        }
        path.write_text(json.dumps(payload, indent=2, default=str))

    def append_trace_line(self, path: Path, rec: dict[str, Any]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")


    def rehydrate(self, jsonl_path: Path) -> int:
        """Restore the in-memory trace from the append-only JSONL. Returns the number of records restored.

        The trace is not in the checkpoint pickle, since it is recomputable from the buffer.
        This restores full history so the in-memory trace is not truncated on resume.
        """
        jsonl_path = Path(jsonl_path)
        if not jsonl_path.exists():
            return 0
        restored: list[dict[str, Any]] = []
        try:
            for line in jsonl_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    restored.append(json.loads(line))
                except Exception:
                    continue
        except Exception:
            return 0
        if restored:
            self._trace = restored
        return len(restored)
