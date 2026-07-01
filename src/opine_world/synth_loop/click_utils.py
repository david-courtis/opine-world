"""Click-action helpers for ARC-AGI-3 ACTION6.

Actions are polymorphic: int for coordinate-free ids (1-5, 7), or
{"action_id": 6, "x": int, "y": int} for clicks. Click memory is keyed by
connected-component id so clicks on the same sprite collapse to one entry.
"""
from __future__ import annotations

from typing import Any


def is_click(action: Any) -> bool:
    if isinstance(action, dict):
        return int(action.get("action_id", -1)) == 6
    return False


def action_id_of(action: Any) -> int:
    if isinstance(action, int):
        return action
    if isinstance(action, dict) and "action_id" in action:
        return int(action["action_id"])
    raise ValueError(f"unrecognised action form: {action!r}")


def action_xy(action: Any) -> tuple[int, int] | None:
    if not is_click(action):
        return None
    return int(action["x"]), int(action["y"])


def action_to_dict(action: Any) -> dict:
    if isinstance(action, int):
        return {"action_id": action}
    if isinstance(action, dict):
        out = {"action_id": int(action["action_id"])}
        if "x" in action:
            out["x"] = int(action["x"])
        if "y" in action:
            out["y"] = int(action["y"])
        return out
    raise ValueError(f"unrecognised action form: {action!r}")


def action_label(action: Any, frame: Any = None) -> str:
    """Return a short label for the run log: RESET, ACTION<n>, or ACTION6(x=..., y=..., ...)."""
    if isinstance(action, str) and action.upper() == "RESET":
        return "RESET"
    aid = action_id_of(action)
    if aid == 0:
        return "RESET"
    if not is_click(action):
        return f"ACTION{aid}"
    x, y = action["x"], action["y"]
    if frame is None:
        return f"ACTION6(x={x}, y={y})"
    try:
        grid = frame.tolist() if hasattr(frame, "tolist") else frame
        label, _ = get_click_info(grid, int(y), int(x))
        return f"ACTION6(x={x}, y={y}, {label})"
    except Exception:
        return f"ACTION6(x={x}, y={y})"


def click_memory_key(action: Any, frame: Any = None) -> str:
    """Return the state-action memory key: stringified action id, or click_<comp_id> for clicks.

    Multiple clicks on the same connected component collapse to one key for retry-avoidance.
    """
    aid = action_id_of(action)
    if not is_click(action):
        return str(aid)
    x, y = int(action["x"]), int(action["y"])
    if frame is None:
        return f"click_{x}_{y}"
    try:
        grid = frame.tolist() if hasattr(frame, "tolist") else frame
        _label, comp_id = get_click_info(grid, y, x)
        return f"click_{comp_id}"
    except Exception:
        return f"click_{x}_{y}"


def find_connected_components(grid: list[list[int]]) -> dict[tuple, int]:
    """BFS flood-fill over equal-value 4-neighbours. Returns {(row, col): comp_id}."""
    if not grid:
        return {}
    rows, cols = len(grid), len(grid[0])
    comp_map: dict[tuple, int] = {}
    comp_id = 0

    def bfs(start_r: int, start_c: int, value: int) -> None:
        nonlocal comp_id
        queue = [(start_r, start_c)]
        while queue:
            r, c = queue.pop(0)
            if (r, c) in comp_map:
                continue
            if r < 0 or r >= rows or c < 0 or c >= cols:
                continue
            if grid[r][c] != value:
                continue
            comp_map[(r, c)] = comp_id
            queue.extend([(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)])

    for r in range(rows):
        for c in range(cols):
            if (r, c) not in comp_map:
                bfs(r, c, grid[r][c])
                comp_id += 1
    return comp_map


def get_click_info(grid: list[list[int]], row: int, col: int) -> tuple[str, str]:
    """Return (label, component_id) for the cell at (row, col), or ("?", "invalid") if out-of-bounds."""
    if not grid or row < 0 or row >= len(grid) or col < 0 or col >= len(grid[0]):
        return "?", "invalid"
    value = grid[row][col]
    comp_map = find_connected_components(grid)
    cid = comp_map.get((row, col), -1)
    size = sum(1 for v in comp_map.values() if v == cid)
    return f"val={value},comp_size={size}", f"val{value}_comp{cid}"


def candidate_click_targets(
    grid: list[list[int]], state: list[dict], max_targets: int = 32,
) -> list[tuple[int, int]]:
    """Return up to max_targets (x, y) click candidates from structured-state object centroids."""
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for o in state or []:
        if not o.get("visible", True):
            continue
        try:
            x = int(o.get("x", 0))
            y = int(o.get("y", 0))
            w = int(o.get("w", 1))
            h = int(o.get("h", 1))
        except (TypeError, ValueError):
            continue
        cx, cy = x + w // 2, y + h // 2
        if (cx, cy) not in seen:
            out.append((cx, cy))
            seen.add((cx, cy))
        if len(out) >= max_targets:
            break
    return out
