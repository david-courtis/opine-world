#!/usr/bin/env python3
"""Build the OPINE-World results site data from run artifacts.

Reads each game directory in the results archive (run_log.txt, analyzer_logs/,
synthesis_curve.jsonl, synthesis/run_NNN/game_engine.py) and emits:

  replay_data/<game>.json.gz   delta-compressed replay bundle per game
  assets/site_data.js          card stats (D), first-frame thumbnails (THUMBS)

Bundle schema:
  { game_id, frame_count,
    frames:  [{s, g|d, a}],           g = full 64x64 grid, d = [[x,y,v],...] delta
    level_steps: [frameIdx,...],      frames where a level was cleared (reward 1.0)
    plans:   [{f, plan, reasoning}],  analyzer move-sets, f = frame of the call
    synth:   [{f, round, acc, e}],    synthesis rounds, e = index into engines
    engines: [{t}|{o,a},...] }        t = full text, o = [[start,delCount,[lines]],...]
                                      line-diff vs previous version, a = [[j1,j2],...]
                                      added-line ranges in the new version

Usage:
    python3 build_site_data.py [--results /path/to/opine-world-results] [--games g1,g2]
"""

import argparse
import difflib
import gzip
import json
import re
from pathlib import Path

from PIL import Image

# Paper-verified constants, copied from ARC-3-D3M-Model/docs/iclr/make_tables.py.
# HUMAN: per-level human baseline actions. D3M: OPINE-World per-level actions.
# CLEARED: levels cleared. ACT: game totals as published (tab:per-game).
HUMAN = {
    'tu93': [19, 16, 34, 42, 123, 80, 14, 23, 111], 'sb26': [18, 28, 18, 19, 31, 23, 58, 18],
    'lp85': [17, 38, 31, 16, 41, 60, 26, 159], 'ar25': [32, 50, 75, 37, 89, 159, 233, 73],
    'tr87': [54, 58, 40, 45, 71, 146], 'r11l': [22, 33, 51, 26, 52, 49], 'ft09': [43, 12, 23, 28, 65, 37],
    'cd82': [55, 8, 41, 21, 23, 23], 'cn04': [29, 54, 85, 300, 208, 113], 'su15': [22, 42, 26, 115, 36, 31, 8, 40, 41],
    're86': [26, 42, 86, 108, 189, 139, 424, 241], 'tn36': [32, 72, 26, 40, 30, 55, 62],
    'vc33': [7, 18, 44, 61, 131, 34, 152], 'm0r0': [30, 111, 203, 26, 500, 237],
    'sc25': [36, 6, 32, 83, 143, 50], 'sp80': [39, 58, 25, 148, 96, 152],
    'wa30': [71, 119, 183, 98, 368, 68, 79, 442, 415], 'g50t': [78, 175, 179, 230, 96, 54, 67],
    'ls20': [22, 123, 73, 84, 96, 192, 186], 'ka59': [28, 109, 51, 51, 33, 132, 326],
    'dc22': [59, 102, 67, 98, 324, 578], 'sk48': [61, 177, 101, 103, 230, 181, 125, 92],
    'lf52': [32, 81, 60, 71, 205, 148, 244, 109, 164, 225], 'bp35': [21, 48, 44, 38, 33, 87, 86, 131, 163],
    's5i5': [20, 89, 106, 54, 162, 38, 86, 83]}
D3M = {
    'tu93': [22, 32, 19, 37, 31, 36, 14, 25, 55], 'sb26': [13, 47, 31, 15, 17, 39, 35, 17],
    'lp85': [8, 12, 17, 16, 11, 20, 8, 17], 'ar25': [17, 16, 75, 24, 34, 55, 112, 47],
    'tr87': [33, 29, 26, 27, 22, 74], 'r11l': [7, 13, 35, 16, 23, 33], 'ft09': [6, 7, 14, 16, 55, 13],
    'cd82': [74, 6, 20, 23, 20, 17], 'cn04': [32, 50, 32, 39, 49, 61], 'su15': [23, 37, 16, 99, 24, 18, 7, 59, 51],
    're86': [21, 36, 52, 65, 71, 70, 214, 321], 'tn36': [13, 23, 9, 111, 136, 62, 63],
    'vc33': [10, 11, 28, 175, 91, 36, 75], 'm0r0': [19, 35, 76, 16, 56, 57],
    'sc25': [32, 5, 36, 32, 50, 101], 'sp80': [10, 30, 56, 43, 85, 145],
    'wa30': [49, 169, 80, 71, 245, 54, 53, 440, 304], 'g50t': [90, 78, 75, 134, 158, 122, 100],
    'ls20': [17, 75, 103, 93, 129, 359, 183], 'ka59': [45, 76, 66, 165, 27, 104, 593],
    'dc22': [132, 74, 53, 120, 280, 820], 'sk48': [18, 116, 64, 236, 66], 'lf52': [11, 267, 104, 118],
    'bp35': [19, 343, 141], 's5i5': [23, 74, 125, 40, 376]}
CLEARED = {
    'tu93': 9, 'sb26': 8, 'lp85': 8, 'ar25': 8, 'tr87': 6, 'r11l': 6, 'ft09': 6, 'cd82': 6,
    'cn04': 6, 'su15': 9, 're86': 8, 'tn36': 7, 'vc33': 7, 'm0r0': 6, 'sc25': 6, 'sp80': 6,
    'wa30': 9, 'g50t': 7, 'ls20': 7, 'ka59': 6, 'dc22': 6, 'sk48': 4, 'lf52': 3, 'bp35': 2, 's5i5': 4}
B1 = {
    'tu93': [18, 16, 19, 18, 29, 28, 14, 21, 29], 'sb26': [13, 15, 15, 15, 17, 19, 39, 17],
    'lp85': [14, 138, 19, 13, 10, 20, 5, 7], 'ar25': [17, 14, 41, 22, 30, 56, 37, 47],
    'tr87': [47, 190, 47, 36, 47, 173], 'r11l': [5, 22, 20, 13, 90, 77], 'ft09': [4, 7, 14, 86, 23, 23],
    'cd82': [16, 6, 65, 14, 13, 16], 'cn04': [14, 199, 22, 29, 124, 60], 'su15': [18, 89, 19, 14, 7, 58, 6, 54, 19],
    're86': [23, 38, 124, 64, 1505, 0, 0, 0], 'tn36': [11, 22, 14, 137, 264, 1012, 1598],
    'vc33': [3, 10, 54, 1529, 0, 0, 0], 'm0r0': [20, 534, 79, 13, 408, 1519],
    'sc25': [20, 5, 63, 1019, 0, 0], 'sp80': [6, 780, 0, 0, 0, 0],
    'wa30': [429, 121, 181, 97, 233, 48, 42, 139, 205], 'g50t': [58, 138, 85, 99, 55, 48, 71],
    'ls20': [22, 97, 74, 101, 76, 216, 128], 'ka59': [112, 66, 35, 61, 93, 552, 180],
    'dc22': [66, 46, 92, 114, 1524, 0], 'sk48': [14, 411, 83, 558, 76, 1681, 0, 0],
    'lf52': [8, 209, 81, 58, 91, 151, 2048, 0, 0, 0], 'bp35': [20, 183, 53, 36, 50, 0, 0, 0, 0],
    's5i5': [170, 948, 396, 1514, 0, 0, 0, 0]}
CLEARED_B1 = {
    'tu93': 9, 'sb26': 8, 'lp85': 8, 'ar25': 8, 'tr87': 6, 'r11l': 6, 'ft09': 6, 'cd82': 6,
    'cn04': 6, 'su15': 9, 're86': 4, 'tn36': 6, 'vc33': 3, 'm0r0': 5, 'sc25': 3, 'sp80': 1,
    'wa30': 9, 'g50t': 7, 'ls20': 7, 'ka59': 7, 'dc22': 4, 'sk48': 5, 'lf52': 6, 'bp35': 4, 's5i5': 3}
ACT = {
    'tu93': 272, 'sb26': 214, 'lp85': 110, 'ar25': 381, 'tr87': 212, 'r11l': 128, 'ft09': 111,
    'cd82': 161, 'cn04': 263, 'su15': 334, 're86': 850, 'tn36': 417, 'vc33': 427, 'm0r0': 259,
    'sc25': 256, 'sp80': 369, 'wa30': 1465, 'g50t': 757, 'ls20': 959, 'ka59': 1076, 'dc22': 1479,
    'sk48': 596, 'lf52': 593, 'bp35': 512, 's5i5': 638}
CAP = {g: 3000 if g == 'm0r0' else 2000 for g in HUMAN}
# non-wins all ended on API errors / rate limits before the move cap, not on budget
ERRORED = {g for g in HUMAN if CLEARED[g] < len(HUMAN[g])}

RGB = {
    0: (0xFF, 0xFF, 0xFF), 1: (0xCC, 0xCC, 0xCC), 2: (0x99, 0x99, 0x99), 3: (0x66, 0x66, 0x66),
    4: (0x33, 0x33, 0x33), 5: (0x00, 0x00, 0x00), 6: (0xE5, 0x3A, 0xA3), 7: (0xFF, 0x7B, 0xCC),
    8: (0xF9, 0x3C, 0x31), 9: (0x1E, 0x93, 0xFF), 10: (0x88, 0xD8, 0xF1), 11: (0xFF, 0xDC, 0x00),
    12: (0xFF, 0x85, 0x1B), 13: (0x92, 0x12, 0x31), 14: (0x4F, 0xCC, 0x30), 15: (0xA3, 0x56, 0xD6)}
RGB_INV = {v: k for k, v in RGB.items()}
PNG_SCALE = 15


def rhae(human, ai, cleared):
    """ARC-AGI-3 game score: weighted mean of per-level min(1.15,(human/ai)^2) over
    completed levels, weights = 1-indexed level number, capped at 100."""
    L = len(human)
    den = L * (L + 1) // 2
    num = 0.0
    for l in range(L):
        if l < cleared and l < len(ai) and ai[l] > 0:
            num += (l + 1) * min(1.15, (human[l] / ai[l]) ** 2)
    return min(100.0, 100.0 * num / den)


def level_scores(human, ai, cleared):
    out = []
    for l in range(len(human)):
        if l < cleared and l < len(ai) and ai[l] > 0:
            out.append(round(100.0 * min(1.15, (human[l] / ai[l]) ** 2), 3))
        else:
            out.append(0.0)
    return out


STEP_RE = re.compile(r"^\[STEP (\d+)\]")
ACTION_RE = re.compile(r"^\[ACTION\] action_id=(\d+) action_name=(.*)")
REWARD_RE = re.compile(r"^\[REWARD\] ([0-9.]+) done=(\w+)")
LEVEL_RE = re.compile(r"^\[LEVEL\] (\d+)")
NOTE_ANALYZER_RE = re.compile(r"^\[NOTE step=(\d+) source=analyzer\]")
CLICK_RE = re.compile(r"^ACTION6\(x=(\d+), y=(\d+)")


def action_label(action_id, action_name):
    if action_id == 0:
        return "RESET"
    m = CLICK_RE.match(action_name)
    if m:
        return f"click({m.group(1)},{m.group(2)})"
    return action_name.split("(")[0]


def parse_run_log(path):
    """Single pass over run_log.txt.

    Returns (steps, analyzer_steps):
      steps: list of {step, action_id, action_name, reward, level, grid} in step order
      analyzer_steps: env step of each source=analyzer NOTE, in file order
    """
    steps = []
    analyzer_steps = []
    cur = None
    grid_rows_left = 0
    seen = set()

    with open(path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if grid_rows_left:
                cur["grid"].append([int(c, 16) for c in line])
                grid_rows_left -= 1
                continue
            m = STEP_RE.match(line)
            if m:
                n = int(m.group(1))
                cur = {"step": n, "reward": 0.0, "grid": []}
                # snapshot-resume replays a step verbatim at segment boundaries;
                # keep the first occurrence only
                if n not in seen:
                    seen.add(n)
                    steps.append(cur)
                continue
            if cur is not None:
                m = ACTION_RE.match(line)
                if m:
                    cur["action_id"] = int(m.group(1))
                    cur["action_name"] = m.group(2).strip()
                    continue
                m = REWARD_RE.match(line)
                if m:
                    cur["reward"] = float(m.group(1))
                    continue
                m = LEVEL_RE.match(line)
                if m:
                    cur["level"] = int(m.group(1))
                    continue
                if line == "[ASCII_FRAME]":
                    grid_rows_left = 64
                    continue
            m = NOTE_ANALYZER_RE.match(line)
            if m:
                analyzer_steps.append(int(m.group(1)))
                continue
    return steps, analyzer_steps


def compute_delta(prev, curr):
    delta = []
    for y in range(64):
        prow, crow = prev[y], curr[y]
        for x in range(64):
            if prow[x] != crow[x]:
                delta.append([x, y, crow[x]])
    return delta


def plan_to_str(plan):
    # entries are ints, {'action_id': 6, 'x', 'y'} or {'action': 'ACTION6', 'x', 'y'}
    parts = []
    for p in plan:
        if isinstance(p, dict):
            name = p.get("action") or f"ACTION{p.get('action_id')}"
            if "x" in p and "y" in p:
                verb = "click" if name == "ACTION6" else name
                parts.append(f"{verb}({p['x']},{p['y']})")
            else:
                parts.append(name)
        else:
            parts.append(f"ACTION{p}" if p != 0 else "RESET")
    return ",".join(parts)


def load_plans(game_dir, analyzer_steps, frame_of):
    logs = sorted((game_dir / "analyzer_logs").glob("call_*.next_actions.json"))
    assert len(logs) == len(analyzer_steps), (
        f"{game_dir.name}: {len(logs)} analyzer call files vs {len(analyzer_steps)} NOTE lines")
    plans = []
    for step, path in zip(analyzer_steps, logs):
        data = json.loads(path.read_text())
        plans.append({
            "f": frame_of(step),
            "plan": plan_to_str(data.get("plan", [])),
            "reasoning": (data.get("reasoning") or "").strip(),
        })
    return plans


def decode_png_grid(path, ox, oy):
    im = Image.open(path).convert("RGB")
    s = PNG_SCALE
    grid = []
    for y in range(64):
        row = []
        for x in range(64):
            v = RGB_INV.get(im.getpixel((ox + x * s + s // 2, oy + y * s + s // 2)))
            if v is None:
                return None
            row.append(v)
        grid.append(row)
    return grid


def calibrate_png_offset(path, ref_grid):
    """Find the pixel origin of the 64x64 cell grid inside a rendered frame PNG."""
    im = Image.open(path).convert("RGB")
    W, H = im.size
    s = PNG_SCALE
    for ox, oy in [(9, 9)] + [(a, b) for a in range(0, W - 64 * s + 1)
                              for b in range(0, H - 64 * s + 1)]:
        row0 = [RGB_INV.get(im.getpixel((ox + x * s + s // 2, oy + s // 2))) for x in range(64)]
        if row0 != ref_grid[0]:
            continue
        if decode_png_grid(path, ox, oy) == ref_grid:
            return ox, oy
    return None


def load_tick_grids(game_dir, steps):
    """Intermediate animation frames per step, decoded from the tick PNGs.

    Returns {step: [grid, ...]} in tick order. The recorded final_frame PNG of one
    event is calibrated against the same step's ASCII grid; a game whose render
    offset cannot be recovered contributes no ticks.
    """
    events_path = game_dir / "frames" / "animation_events.jsonl"
    if not events_path.exists():
        return {}
    events = []
    for line in events_path.read_text().splitlines():
        if line.strip():
            events.append(json.loads(line))
    by_step = {s["step"]: s for s in steps}
    offset = None
    for ev in events:
        final = game_dir / "frames" / Path(ev.get("final_frame", "")).name
        if ev["step"] in by_step and final.exists():
            offset = calibrate_png_offset(final, by_step[ev["step"]]["grid"])
            if offset:
                break
    if not offset:
        if events:
            print(f"  note {game_dir.name}: could not calibrate tick PNGs, skipping animations")
        return {}
    ticks = {}
    missing = 0
    for ev in events:
        grids = []
        for t in ev.get("tick_frames", []):
            p = game_dir / "frames" / Path(t).name
            if not p.exists():
                missing += 1
                continue
            g = decode_png_grid(p, *offset)
            if g is None:
                missing += 1
                continue
            grids.append(g)
        if grids and ev["step"] in by_step:
            ticks[ev["step"]] = grids
    if missing:
        print(f"  note {game_dir.name}: {missing} tick frames missing/undecodable")
    return ticks


def diff_ops(prev_lines, new_lines):
    """Line diff as splice ops on the previous version plus added ranges in the new one.

    Ops are [start, delCount, [addedLines]] in previous-version coordinates and
    must be applied in reverse order so earlier indices stay valid.
    """
    ops = []
    added = []
    sm = difflib.SequenceMatcher(a=prev_lines, b=new_lines, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        ops.append([i1, i2 - i1, new_lines[j1:j2]])
        if j2 > j1:
            added.append([j1, j2])
    return ops, added


def apply_ops(prev_lines, ops):
    out = list(prev_lines)
    for start, del_count, lines in reversed(ops):
        out[start:start + del_count] = lines
    return out


def load_engines(game_dir, synth_curve, frame_of):
    """Engine code versions (deduped) + synthesis event list."""
    synth = []
    engines = []
    prev_lines = None
    for rec in synth_curve:
        run_dir = game_dir / "synthesis" / f"run_{rec['round']:03d}"
        engine_path = run_dir / "game_engine.py"
        assert engine_path.exists(), f"{game_dir.name}: missing {engine_path}"
        text = engine_path.read_text()
        lines = text.split("\n")
        if prev_lines is None:
            engines.append({"t": text})
            prev_lines = lines
        elif lines != prev_lines:
            ops, added = diff_ops(prev_lines, lines)
            engines.append({"o": ops, "a": added})
            assert apply_ops(prev_lines, ops) == lines, (
                f"{game_dir.name}: diff round-trip failed at round {rec['round']}")
            prev_lines = lines
        acc = rec.get("accuracy_pct")
        synth.append({
            "f": frame_of(rec["step"]),
            "round": rec["round"],
            "acc": round(acc, 1) if isinstance(acc, (int, float)) else None,
            "e": len(engines) - 1,
        })
    return synth, engines


def process_game(game, game_dir):
    steps, analyzer_steps = parse_run_log(game_dir / "run_log.txt")
    assert steps, f"{game}: empty run_log"
    assert [s["step"] for s in steps] == list(range(len(steps))), f"{game}: steps not contiguous"
    for s in steps:
        assert len(s["grid"]) == 64 and all(len(r) == 64 for r in s["grid"]), (
            f"{game}: bad grid at step {s['step']}")

    ticks = load_tick_grids(game_dir, steps)

    # one frame group per step: animation ticks first (action label on the first
    # emitted frame of the group), then the authoritative ASCII final frame
    frames = []
    first_f = {}
    final_f = {}
    prev = None
    n_ticks = 0
    for i, s in enumerate(steps):
        label = action_label(s.get("action_id", -1), s.get("action_name", ""))
        group = [(g, True) for g in ticks.get(s["step"], [])] + [(s["grid"], False)]
        first_f[i] = len(frames)
        for g, is_tick in group:
            fr = {"s": i}
            if prev is None:
                fr["g"] = g
            else:
                d = compute_delta(prev, g)
                if d:
                    fr["d"] = d
            if label:
                fr["a"] = label
                label = ""
            if is_tick:
                n_ticks += 1
            frames.append(fr)
            prev = g
        final_f[i] = len(frames) - 1

    # A level is complete where [LEVEL] increments on the next step. The winning
    # completion has no next step and its reward line is missing from some logs
    # (lp85, r11l), so wins always close at the last frame. Reward lines alone
    # undercount: a resume boundary can swallow one (ar25, cd82).
    completion_steps = [i for i in range(len(steps) - 1)
                        if steps[i + 1].get("level", 0) > steps[i].get("level", 0)]
    if CLEARED[game] == len(HUMAN[game]):
        completion_steps.append(len(steps) - 1)
    assert len(completion_steps) == CLEARED[game], (
        f"{game}: {len(completion_steps)} level completions vs CLEARED={CLEARED[game]}")
    level_steps = [final_f[i] for i in completion_steps]

    # frames must replay to the final grid exactly
    replay = [row[:] for row in frames[0]["g"]]
    for fr in frames[1:]:
        for x, y, v in fr.get("d", []):
            replay[y][x] = v
    assert replay == steps[-1]["grid"], f"{game}: delta replay mismatch"

    max_step = len(steps) - 1
    plans = load_plans(game_dir, analyzer_steps,
                       lambda st: first_f[min(st, max_step)])

    curve_path = game_dir / "synthesis_curve.jsonl"
    # the archive redacts some token counts to literal <placeholders> (re86)
    synth_curve = [json.loads(re.sub(r": <[a-z_]+>", ": null", l))
                   for l in curve_path.read_text().splitlines() if l.strip()]
    run_nums = {int(d.name[4:]) for d in (game_dir / "synthesis").iterdir()
                if d.is_dir() and d.name.startswith("run_") and d.name[4:].isdigit()}
    curve_rounds = {r["round"] for r in synth_curve}
    assert curve_rounds <= run_nums, (
        f"{game}: curve rounds missing run dirs: {sorted(curve_rounds - run_nums)}")
    if run_nums - curve_rounds:
        print(f"  note {game}: run dirs without curve rows (interrupted): "
              f"{sorted(run_nums - curve_rounds)}")
    synth, engines = load_engines(game_dir, synth_curve,
                                  lambda st: final_f[min(st, max_step)])

    eta_trace = []
    trace_path = game_dir / "ontology_error_trace.jsonl"
    if trace_path.exists():
        for line in trace_path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(re.sub(r": <[a-z_]+>", ": null", line))
            if isinstance(rec.get("eta"), (int, float)):
                eta_trace.append([min(rec["step"], max_step), round(rec["eta"], 4)])
    eta_meta = {
        "trace": eta_trace,
        "completions": completion_steps,
        "total": len(steps),
    }

    bundle = {
        "game_id": game,
        "frame_count": len(frames),
        "level_steps": level_steps,
        "frames": frames,
        "plans": plans,
        "synth": synth,
        "engines": engines,
    }
    return bundle, steps[0]["grid"], eta_meta, n_ticks


def card(game):
    lv, tl = CLEARED[game], len(HUMAN[game])
    return {
        "g": game,
        "s": round(rhae(HUMAN[game], D3M[game], lv), 1),
        "lv": lv, "tl": tl,
        "act": ACT[game], "cap": CAP[game],
        "w": lv == tl,
        "err": game in ERRORED,
        "bl": HUMAN[game], "la": D3M[game],
        "b1": B1[game], "b1lv": CLEARED_B1[game],
        "ls": level_scores(HUMAN[game], D3M[game], lv),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path.home() / "git" / "opine-world-results")
    ap.add_argument("--games", type=str, default="")
    args = ap.parse_args()

    docs = Path(__file__).parent
    out_replay = docs / "replay_data"
    out_assets = docs / "assets"
    out_replay.mkdir(exist_ok=True)
    out_assets.mkdir(exist_ok=True)

    games = args.games.split(",") if args.games else sorted(HUMAN)
    thumbs = {}
    eta = {}
    total = 0
    for game in games:
        game_dir = args.results / game
        if not game_dir.is_dir():
            print(f"  SKIP {game}: no directory")
            continue
        bundle, first_grid, eta_meta, n_ticks = process_game(game, game_dir)
        thumbs[game] = first_grid
        eta[game] = eta_meta
        raw = json.dumps(bundle, separators=(",", ":")).encode()
        gz = gzip.compress(raw, 9)
        (out_replay / f"{game}.json.gz").write_bytes(gz)
        total += len(gz)
        flag = "  <-- LARGE" if len(gz) > 2.5e6 else ""
        print(f"  {game}: {bundle['frame_count']} frames ({n_ticks} anim), "
              f"{len(bundle['plans'])} plans, {len(bundle['synth'])} rounds, "
              f"{len(bundle['engines'])} engine versions, {len(gz)/1024:.0f} KB gz{flag}")

    if not args.games:
        d = [card(g) for g in sorted(HUMAN)]
        js = (
            "// Generated by build_site_data.py — do not edit by hand.\n"
            f"const D={json.dumps(d, separators=(',', ':'))};\n"
            f"const THUMBS={json.dumps(thumbs, separators=(',', ':'))};\n"
        )
        (out_assets / "site_data.js").write_text(js)
        ejs = (
            "// Generated by build_site_data.py — do not edit by hand.\n"
            f"const ETA={json.dumps(eta, separators=(',', ':'))};\n"
        )
        (out_assets / "eta_data.js").write_text(ejs)
        print(f"assets/site_data.js: {len(js)/1024:.0f} KB, assets/eta_data.js: {len(ejs)/1024:.0f} KB")
    print(f"Total replay data: {total/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()
