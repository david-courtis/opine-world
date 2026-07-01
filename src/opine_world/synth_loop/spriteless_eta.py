"""Synth-fed object abstraction for frames-only ETA diagnostics.

Frames-only runs do not receive engine sprite records.  When the synthesized
``game_engine.py`` exports ``extract_objects(frame)``, this module applies that
extractor to the observed frame replay and converts the result into the same
list-of-object-dicts schema used by the object-centric epistemic machinery.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import pickle
import signal
from collections import Counter
from pathlib import Path
from typing import Any

from .epistemic import (
    MATRIX_DEPTH_DIRICHLET,
    compute_epistemic_matrix,
    compute_ontology_error_with_candidates,
)

MAX_OBJECTS_PER_FRAME = 512
EXTRACT_TIMEOUT_S = 2
ARTIFACT_NAME = "spriteless_object_abstraction.json"
SPRITELESS_REPLAY_NAME = "spriteless_replay_buffer.pkl"


def _load_game_engine(code_path: Path) -> tuple[Any | None, str | None]:
    spec = importlib.util.spec_from_file_location("_spriteless_game_engine", code_path)
    if spec is None or spec.loader is None:
        return None, f"could not load import spec for {code_path}"
    module = importlib.util.module_from_spec(spec)
    module.copy = copy
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    return module, None


def _frame_to_lists(frame: Any) -> list[list[int]] | None:
    if frame is None:
        return None
    try:
        if hasattr(frame, "tolist"):
            frame = frame.tolist()
        return [[int(v) for v in row] for row in frame]
    except Exception:
        return None


def _slice_pixels(
    frame: list[list[int]] | None, x: int, y: int, w: int, h: int,
) -> list[list[int]]:
    if frame is None:
        return []
    out: list[list[int]] = []
    for yy in range(max(0, y), min(len(frame), y + h)):
        row = frame[yy]
        out.append([int(row[xx]) for xx in range(max(0, x), min(len(row), x + w))])
    return out


def _normalize_mask(mask: Any) -> list[list[int]] | None:
    if mask is None:
        return None
    try:
        if hasattr(mask, "tolist"):
            mask = mask.tolist()
        return [[1 if bool(v) else 0 for v in row] for row in mask]
    except Exception:
        return None


def _apply_mask(
    pixels: list[list[int]], mask: list[list[int]] | None,
) -> list[list[int]]:
    if mask is None:
        return pixels
    out: list[list[int]] = []
    for y, row in enumerate(pixels):
        mask_row = mask[y] if y < len(mask) else []
        out_row: list[int] = []
        for x, value in enumerate(row):
            keep = bool(mask_row[x]) if x < len(mask_row) else False
            out_row.append(int(value) if keep else -1)
        out.append(out_row)
    return out


def _coerce_tags(value: Any, fallback: str) -> list[str]:
    if isinstance(value, str):
        tags = [value]
    elif isinstance(value, (list, tuple, set)):
        tags = [str(v) for v in value if str(v)]
    else:
        tags = []
    return tags or [fallback]


def _get_field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def normalize_objects(
    raw_objects: Any,
    frame: Any,
    *,
    max_objects: int = MAX_OBJECTS_PER_FRAME,
) -> tuple[list[dict], list[str]]:
    """Normalize synth-returned objects into the sprite-record-like schema.

    Accepts dictionaries or simple objects with attributes.  Invalid entries are
    skipped. The returned warnings are for diagnostics only.
    """
    warnings: list[str] = []
    frame_list = _frame_to_lists(frame)
    if raw_objects is None:
        return [], ["extract_objects returned None"]
    if not isinstance(raw_objects, (list, tuple)):
        return [], [f"extract_objects returned {type(raw_objects).__name__}, not list"]

    out: list[dict] = []
    for i, obj in enumerate(list(raw_objects)[:max_objects]):
        try:
            ty = (
                _get_field(obj, "type")
                or _get_field(obj, "kind")
                or _get_field(obj, "role")
                or _get_field(obj, "tag")
                or "visual_object"
            )
            ty = str(ty)[:80] or "visual_object"
            raw_name = (
                _get_field(obj, "name")
                or _get_field(obj, "id")
                or _get_field(obj, "key")
            )
            name = str(raw_name)[:120] if raw_name is not None else f"{ty}_{i:03d}"
            tags = _coerce_tags(_get_field(obj, "tags"), ty)

            if _get_field(obj, "bbox") is not None:
                bbox = list(_get_field(obj, "bbox"))
                if len(bbox) >= 4:
                    x, y, w, h = bbox[:4]
                else:
                    x = y = 0
                    w = h = 1
            else:
                x = _get_field(obj, "x", _get_field(obj, "col", 0))
                y = _get_field(obj, "y", _get_field(obj, "row", 0))
                w = _get_field(obj, "w", _get_field(obj, "width", 1))
                h = _get_field(obj, "h", _get_field(obj, "height", 1))
            x, y = int(x), int(y)
            w, h = max(1, int(w)), max(1, int(h))

            mask = _normalize_mask(
                _get_field(obj, "mask", _get_field(obj, "alpha_mask", None))
            )
            pixels = _get_field(obj, "pixels")
            pixels_source = "synth"
            if pixels is None:
                pixels = _slice_pixels(frame_list, x, y, w, h)
                pixels_source = "flattened_frame_slice"
            else:
                try:
                    if hasattr(pixels, "tolist"):
                        pixels = pixels.tolist()
                    pixels = [[int(v) for v in row] for row in pixels]
                except Exception:
                    warnings.append(f"{name}: invalid pixels; sliced from frame")
                    pixels = _slice_pixels(frame_list, x, y, w, h)
                    pixels_source = "flattened_frame_slice"
            pixels = _apply_mask(pixels, mask)

            layer = int(_get_field(obj, "layer", 0) or 0)
            rotation = int(_get_field(obj, "rotation", 0) or 0)
            visible = bool(_get_field(obj, "visible", True))
            color = _get_field(obj, "color", _get_field(obj, "palette", None))
            rec = {
                "name": name,
                "type": ty,
                "tags": tags,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "display_x": x,
                "display_y": y,
                "display_w": w,
                "display_h": h,
                "visible": visible,
                "collidable": bool(_get_field(obj, "collidable", True)),
                "layer": layer,
                "rotation": rotation,
                "pixels": pixels,
                "pixels_source": pixels_source,
            }
            if mask is not None:
                rec["mask"] = mask
            if color is not None:
                rec["color"] = str(color)
            out.append(rec)
        except Exception as exc:
            warnings.append(f"object {i}: {type(exc).__name__}: {exc}")

    if len(raw_objects) > max_objects:
        warnings.append(
            f"truncated {len(raw_objects)} objects to max_objects={max_objects}"
        )
    return out, warnings


def _call_with_timeout(fn: Any, frame: Any) -> tuple[Any, str | None]:
    old_handler = None
    alarm_set = False
    try:
        def _timeout(*_args):
            raise TimeoutError("extract_objects timed out")

        old_handler = signal.signal(signal.SIGALRM, _timeout)
        signal.alarm(EXTRACT_TIMEOUT_S)
        alarm_set = True
    except Exception:
        alarm_set = False
    try:
        return fn(copy.deepcopy(frame)), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    finally:
        if alarm_set:
            try:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            except Exception:
                pass


def _frame_key(frame: Any) -> str:
    data = _frame_to_lists(frame)
    if data is None:
        return "none"
    return json.dumps(data, separators=(",", ":"))


def _extract_cached(
    extractor: Any,
    frame: Any,
    cache: dict[str, list[dict]],
    errors: list[dict],
) -> list[dict]:
    key = _frame_key(frame)
    if key in cache:
        return copy.deepcopy(cache[key])
    raw, err = _call_with_timeout(extractor, frame)
    if err is not None:
        errors.append({"error": err})
        cache[key] = []
        return []
    objs, warnings = normalize_objects(raw, frame)
    if warnings:
        errors.append({"warnings": warnings[:6]})
    cache[key] = objs
    return copy.deepcopy(objs)


def build_spriteless_replay(
    replay_buffer: list[dict],
    extractor: Any,
) -> tuple[list[dict], dict]:
    cache: dict[str, list[dict]] = {}
    errors: list[dict] = []
    out: list[dict] = []
    counts: list[int] = []
    type_counter: Counter[str] = Counter()

    for trans in replay_buffer:
        before_frame = trans.get("before_frame")
        after_frame = trans.get("after_frame")
        if before_frame is None or after_frame is None:
            continue
        before_state = _extract_cached(extractor, before_frame, cache, errors)
        after_state = _extract_cached(extractor, after_frame, cache, errors)
        counts.extend([len(before_state), len(after_state)])
        for obj in before_state + after_state:
            type_counter[str(obj.get("type") or "visual_object")] += 1
        d = {
            "before_state": before_state,
            "action_id": trans.get("action_id"),
            "action_name": trans.get("action_name"),
            "after_state": after_state,
            "diff_text": trans.get("diff_text", ""),
            "reward": trans.get("reward", 0.0),
            "done": trans.get("done", False),
            "timestep": trans.get("timestep"),
            "level": trans.get("level"),
        }
        if "click_x" in trans and "click_y" in trans:
            d["click_x"] = trans["click_x"]
            d["click_y"] = trans["click_y"]
        out.append(d)

    summary = {
        "n_frame_transitions": len(replay_buffer),
        "n_object_transitions": len(out),
        "n_unique_frames_extracted": len(cache),
        "object_count_min": min(counts) if counts else 0,
        "object_count_max": max(counts) if counts else 0,
        "object_count_mean": (
            round(sum(counts) / len(counts), 3) if counts else 0.0
        ),
        "types_seen": sorted(type_counter.keys()),
        "n_types_seen": len(type_counter),
        "errors": errors[:20],
        "n_errors": len(errors),
    }
    return out, summary


def refresh_spriteless_diagnostics(
    *,
    replay_buffer: list[dict],
    code_path: Path,
    output_dir: Path,
    step: int,
    synthesis_count: int,
    alpha_0: float,
    beta_0: float,
    kappa: float,
    sort_by: str,
    rng: Any = None,
    max_candidates: int = 12,
) -> dict[str, Any]:
    """Compute frame-object epistemic artifacts from ``extract_objects``.

    Returns a payload with ``ok`` and optional ``trace_record`` /
    ``ontology_latest`` entries.  All failures are represented as artifacts,
    never raised, so a bad extractor cannot stop exploration.
    """
    output_dir = Path(output_dir)
    code_path = Path(code_path)
    artifact_path = output_dir / ARTIFACT_NAME
    status: dict[str, Any] = {
        "ok": False,
        "source": "spriteless_synth_extractor",
        "step": int(step),
        "synthesis_count": int(synthesis_count),
        "code_path": str(code_path),
    }
    module, err = _load_game_engine(code_path)
    if err is not None:
        status["error"] = f"load failed: {err}"
        artifact_path.write_text(json.dumps(status, indent=2, default=str))
        return status
    extractor = getattr(module, "extract_objects", None)
    if not callable(extractor):
        status["error"] = "game_engine.py does not define extract_objects(frame)"
        artifact_path.write_text(json.dumps(status, indent=2, default=str))
        return status

    spriteless_replay, summary = build_spriteless_replay(
        replay_buffer, extractor,
    )
    status.update(summary)
    if not spriteless_replay or summary.get("n_types_seen", 0) <= 0:
        status["error"] = "extract_objects produced no usable objects"
        artifact_path.write_text(json.dumps(status, indent=2, default=str))
        return status

    matrix = compute_epistemic_matrix(
        spriteless_replay,
        alpha_0=alpha_0,
        beta_0=beta_0,
        kappa=kappa,
        sort_by=sort_by,
        rng=rng,
    )
    matrix["depth"] = MATRIX_DEPTH_DIRICHLET
    matrix["source"] = "spriteless_synth_extractor"
    matrix["synthesis_count"] = int(synthesis_count)
    matrix["object_abstraction_artifact"] = ARTIFACT_NAME
    (output_dir / "epistemic_matrix.json").write_text(
        json.dumps(matrix, indent=2, default=str)
    )

    ontology = compute_ontology_error_with_candidates(
        spriteless_replay,
        alpha_0=alpha_0,
        kappa=kappa,
        max_candidates=max_candidates,
    )
    ontology["source"] = "spriteless_synth_extractor"
    ontology["synthesis_count"] = int(synthesis_count)
    ontology["object_abstraction_artifact"] = ARTIFACT_NAME

    trace_record = {
        "step": int(step),
        "phase": "flat",
        "source": "spriteless_synth_extractor",
        "eta": ontology["flat_eta"],
        "eta_star": ontology["eta_star"],
        "eta_reduction": ontology["eta_reduction"],
        "best_candidate": ontology["best_candidate"],
        "n_strata": ontology["flat"]["n_strata"],
        "n_candidates": ontology["n_candidates"],
        "n_transitions": len(spriteless_replay),
        "n_induced_types": summary["n_types_seen"],
        "n_unique_frames_extracted": summary["n_unique_frames_extracted"],
        "synthesis_count": int(synthesis_count),
    }

    status["ok"] = True
    status["trace_record"] = trace_record
    artifact_path.write_text(json.dumps(status, indent=2, default=str))
    with open(output_dir / SPRITELESS_REPLAY_NAME, "wb") as f:
        pickle.dump(spriteless_replay, f)

    status["ontology_latest"] = ontology
    status["epistemic_n_cells"] = matrix.get("n_cells", 0)
    return status
