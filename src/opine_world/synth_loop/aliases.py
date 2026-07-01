"""Structured hypothesis space for role names assigned to obfuscated sprite tags.

Per-tag ranked candidate list updated by the analyzer via alias_updates.json.
Always contains at least MIN_CANDIDATES entries. The unknown_n placeholders pad short lists.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


MIN_CANDIDATES = 4


def seed(type_tag: str) -> list[dict[str, Any]]:
    return [
        {"alias": f"unknown_{i+1}", "score": 0}
        for i in range(MIN_CANDIDATES)
    ]


def ensure_seeded(
    aliases: dict[str, list[dict]], type_tag: str,
) -> None:
    if type_tag not in aliases:
        aliases[type_tag] = seed(type_tag)


def _normalize(entries: list[dict]) -> list[dict]:
    used = {e["alias"] for e in entries}
    entries = sorted(entries, key=lambda e: -int(e.get("score", 0)))
    n = 1
    while len(entries) < MIN_CANDIDATES:
        candidate = f"unknown_{n}"
        while candidate in used:
            n += 1
            candidate = f"unknown_{n}"
        entries.append({"alias": candidate, "score": 0})
        used.add(candidate)
        n += 1
    return entries


def apply_updates(
    aliases: dict[str, list[dict]],
    updates: dict[str, Any],
) -> dict[str, Any]:
    """Apply analyzer-emitted updates from alias_updates.json. Returns an apply-summary.

    Keys: "add" (score 0, noop if exists), "upvote" (increment by "by", default 1),
    "remove" (drop candidate, re-pad if needed). All keys optional.
    """
    summary = {"added": 0, "upvoted": 0, "removed": 0, "errors": []}

    for entry in updates.get("add", []):
        try:
            tag = str(entry["type"])
            alias = str(entry["alias"]).strip()
            if not alias:
                continue
            ensure_seeded(aliases, tag)
            if not any(e["alias"] == alias for e in aliases[tag]):
                aliases[tag].append({"alias": alias, "score": 0})
                summary["added"] += 1
        except Exception as e:
            summary["errors"].append(f"add: {e}")

    for entry in updates.get("upvote", []):
        try:
            tag = str(entry["type"])
            alias = str(entry["alias"]).strip()
            by = int(entry.get("by", 1))
            if not alias:
                continue
            ensure_seeded(aliases, tag)
            existing = next(
                (e for e in aliases[tag] if e["alias"] == alias), None
            )
            if existing is None:
                aliases[tag].append({"alias": alias, "score": by})
                summary["added"] += 1
            else:
                existing["score"] = int(existing.get("score", 0)) + by
            summary["upvoted"] += 1
        except Exception as e:
            summary["errors"].append(f"upvote: {e}")

    for entry in updates.get("remove", []):
        try:
            tag = str(entry["type"])
            alias = str(entry["alias"]).strip()
            if tag not in aliases or not alias:
                continue
            before = len(aliases[tag])
            aliases[tag] = [
                e for e in aliases[tag] if e["alias"] != alias
            ]
            if len(aliases[tag]) < before:
                summary["removed"] += 1
        except Exception as e:
            summary["errors"].append(f"remove: {e}")

    for tag in list(aliases.keys()):
        aliases[tag] = _normalize(aliases[tag])

    return summary


def write_workspace_artifact(
    aliases: dict[str, list[dict]], path: Path,
) -> None:
    path.write_text(json.dumps(aliases, indent=2, sort_keys=True))


def read_updates_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def format_for_world_model_doc(
    aliases: dict[str, list[dict]],
    *,
    top_k: int = 3,
) -> str:
    """Render a markdown section showing top-k role candidates per tag for inclusion in world_model_doc."""
    if not aliases:
        return ""
    lines = ["## Type-role hypotheses (analyzer-maintained)",
             "Ordered desc by score (analyzer upvotes a candidate on "
             "consistent observation). Top-1 may differ from ground "
             "truth, treat as a *prior*, not a fact.\n"]
    for tag in sorted(aliases.keys()):
        ranked = aliases[tag][:top_k]
        formatted = " | ".join(
            f"{e['alias']}({e['score']})" for e in ranked
        )
        lines.append(f"  - {tag}: {formatted}")
    return "\n".join(lines)


def best_alias(aliases: dict[str, list[dict]], type_tag: str) -> str | None:
    if type_tag not in aliases:
        return None
    for e in aliases[type_tag]:
        if not e["alias"].startswith("unknown_") and e.get("score", 0) > 0:
            return e["alias"]
    return None


DECORATIVE_DEFAULT: frozenset[str] = frozenset({
    "wall",
    "scenery",
    "decoration",
    "decorative",
    "border",
    "tile",
    "floor",
    "background",
    "hud",
})


def is_decorative(alias: str | None,
                  decorative_set: frozenset[str] = DECORATIVE_DEFAULT) -> bool:
    if not alias:
        return False
    head = alias.strip().lower().split("_", 1)[0]
    return head in decorative_set


def nondecorative_committed(
    aliases: dict[str, list[dict]],
    *,
    min_score: int = 5,
    min_margin: int = 3,
    decorative_set: frozenset[str] = DECORATIVE_DEFAULT,
) -> dict[str, str]:
    """Return {tag: top_alias} for tags that are confidently committed and non-decorative (paper Prop. 5).

    A tag qualifies when top_score >= min_score, top beats next-best by >= min_margin,
    top alias is not an unknown_ placeholder, and the alias is not in the decorative set.
    """
    out: dict[str, str] = {}
    for tag, entries in aliases.items():
        if not isinstance(entries, list) or not entries:
            continue
        ranked = sorted(
            entries, key=lambda e: -int(e.get("score", 0) or 0),
        )
        top = ranked[0]
        top_alias = str(top.get("alias", ""))
        top_score = int(top.get("score", 0) or 0)
        if top_alias.startswith("unknown_"):
            continue
        if top_score < int(min_score):
            continue
        next_score = (
            int(ranked[1].get("score", 0) or 0) if len(ranked) > 1 else 0
        )
        if top_score - next_score < int(min_margin):
            continue
        if is_decorative(top_alias, decorative_set):
            continue
        out[str(tag)] = top_alias
    return out


def annotate_text(text: str, aliases: dict[str, list[dict]]) -> str:
    """Inject tag=role annotations into text for every tag that has a committed alias.

    Longest tags are matched first to avoid partial replacement of shared prefixes.
    """
    if not aliases or not text:
        return text
    pairs = []
    for tag, candidates in aliases.items():
        if not isinstance(candidates, list):
            continue
        for cand in candidates:
            alias = cand.get("alias", "") if isinstance(cand, dict) else ""
            if (not alias.startswith("unknown_")
                    and int(cand.get("score", 0) or 0) > 0):
                pairs.append((tag, alias))
                break
    if not pairs:
        return text
    pairs.sort(key=lambda p: -len(p[0]))
    role_map = {t: r for t, r in pairs}
    import re
    pattern = re.compile("|".join(re.escape(t) for t, _ in pairs))
    def repl(m):
        tag = m.group(0)
        end = m.end()
        if end < len(text) and text[end] == "=":
            return tag
        return f"{tag}={role_map[tag]}"
    return pattern.sub(repl, text)
