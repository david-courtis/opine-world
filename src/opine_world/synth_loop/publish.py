"""Finalize a run directory into its clean, shareable form.

A run produces the full operational output while it executes: timestamped logs,
diagnostics that the agents read as inputs each call, checkpoints and snapshots
for resume, and raw stream-json transcripts. Unless debug mode is on, the engine
calls ``finalize_clean`` once the run completes to reduce the directory to the
substantive record: it removes timestamped logs, raw stream-json transcripts, and
resume infrastructure, rewrites the analyzer transcripts into readable Markdown,
and scrubs absolute filesystem paths and stray timestamps from the text it keeps.
The derived signals the agents read as inputs (the epistemic matrix, the
ontology-error report and trace, the synth status handoff, planner verification,
the object abstraction) are kept.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

DROP_FILES = (
    "engine.log", "nohup.out",
    "spriteless_replay_buffer.pkl",
    "analyzer_failure.json", "checkpoint.pkl", "checkpoint.pkl.tmp",
)
DROP_DIRS = ("snapshots", "consumer_workspace", "animation_analysis")
SYNTH_DROP_FILES = (
    "claude_chat.jsonl", "claude_prompt.txt", "claude_prompt_escalation.txt",
    "codex_quota_wait.txt", "codex_retry_tracker.jsonl",
    "requires_critique_response.flag",
)
SYNTH_DROP_GLOBS = ("*_stdout.jsonl", "*_stdout.txt", "*_stderr.txt")
TEXT_SUFFIXES = (".txt", ".md", ".json", ".jsonl")
_ISO = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?")
_TMP = re.compile(r"/tmp/claude-[A-Za-z0-9][^\s\"'\\)]*")
_ALT_CFG = re.compile(r"\.claude-[A-Za-z0-9_]+")
_SWEEP = re.compile(r"round[-_]robin[A-Za-z0-9_-]*")
_ACCT = re.compile(r"acct[0-9]+")
_DATE = re.compile(
    r"(?<!\d)20\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])(?:T\d{4,6})?(?!\d)"
)


def _blocks(content):
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return content
    return []


def transcript_md(chat_jsonl: Path) -> str:
    """Render an analyzer stream-json transcript as readable Markdown, keeping
    reasoning text, tool calls, and tool results and dropping timestamps, ids,
    and usage metadata."""
    out = ["# Analyzer call transcript", ""]
    for line in chat_jsonl.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        etype = ev.get("type")
        msg = ev.get("message") if isinstance(ev.get("message"), dict) else {}
        if etype == "assistant":
            for b in _blocks(msg.get("content")):
                bt = b.get("type")
                if bt == "text" and b.get("text", "").strip():
                    out += ["## Assistant", b["text"].rstrip(), ""]
                elif bt == "tool_use":
                    inp = json.dumps(b.get("input", {}), ensure_ascii=False)
                    out += [f"**tool call** `{b.get('name', '?')}`",
                            "```json", inp, "```", ""]
        elif etype == "user":
            for b in _blocks(msg.get("content")):
                if b.get("type") == "tool_result":
                    txt = b.get("content")
                    if isinstance(txt, list):
                        txt = "\n".join(
                            p.get("text", "") for p in txt if isinstance(p, dict)
                        )
                    if str(txt).strip():
                        out += ["## Tool result", "```", str(txt).rstrip(),
                                "```", ""]
    return "\n".join(out).rstrip() + "\n"


def _clean_analyzer_logs(al: Path) -> None:
    for f in al.glob("*.chat.jsonl"):
        try:
            (al / f.name.replace(".chat.jsonl", ".transcript.md")).write_text(
                transcript_md(f)
            )
        except Exception:
            pass
        f.unlink(missing_ok=True)
    for f in al.glob("*.stderr.txt"):
        f.unlink(missing_ok=True)


def _sanitize_summary(p: Path) -> None:
    try:
        d = json.loads(p.read_text())
    except Exception:
        return
    if isinstance(d, dict) and "replay_path" in d:
        d["replay_path"] = os.path.basename(str(d.get("replay_path") or ""))
        p.write_text(json.dumps(d, indent=2))


def _sanitize_text(od: Path, path_subs) -> None:
    subs = list(path_subs) + [(str(Path.home()), "<home>")]
    subs = [(old, new) for old, new in subs if old]
    for p in od.rglob("*"):
        if not p.is_file() or p.suffix not in TEXT_SUFFIXES:
            continue
        try:
            text = p.read_text(errors="ignore")
        except Exception:
            continue
        original = text
        for old, new in subs:
            text = text.replace(old, new)
        text = _TMP.sub("<tmp>", text)
        text = _ALT_CFG.sub(".claude", text)
        text = _SWEEP.sub("<sweep>", text)
        text = _ACCT.sub("acct", text)
        text = _DATE.sub("<ts>", text)
        text = _ISO.sub("<ts>", text)
        if text != original:
            p.write_text(text)


def finalize_clean(output_dir: str | Path, path_subs=()) -> None:
    """Reduce a finished run directory to its shareable form, in place.

    ``path_subs`` is an ordered list of (absolute_prefix, placeholder) string
    replacements applied to kept text files (most specific first); the running
    user's home directory is always scrubbed as well.
    """
    od = Path(output_dir)
    for name in DROP_FILES:
        (od / name).unlink(missing_ok=True)
    for name in DROP_DIRS:
        shutil.rmtree(od / name, ignore_errors=True)
    synth = od / "synthesis"
    if synth.is_dir():
        for run in synth.glob("run_*"):
            for name in SYNTH_DROP_FILES:
                (run / name).unlink(missing_ok=True)
            for pat in SYNTH_DROP_GLOBS:
                for f in run.glob(pat):
                    f.unlink(missing_ok=True)
    al = od / "analyzer_logs"
    if al.is_dir():
        _clean_analyzer_logs(al)
    for cache in od.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)
    _sanitize_summary(od / "summary.json")
    _sanitize_text(od, path_subs)
