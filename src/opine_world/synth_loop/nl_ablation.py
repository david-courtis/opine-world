"""Natural-language ablation: strip the NL intermediate representation.

The dual-agent structure carries a natural-language world model. The
exploration agent is explicitly told to keep an interpretable NL account, the
synthesizer writes prose handoffs, and the code itself carries comments and
docstrings. Testing whether that NL representation earns its place cannot be
done by removing one agent, because the dual-agent structure confers other
advantages and the comparison would confound them. This module removes the NL
instead, leaving the architecture untouched: every artifact the agents persist
between invocations must be Python, and comments and docstrings are stripped
before anything is carried forward.

What that scopes the claim to. Each synthesis agent is freshly initialized, so
what survives a round is exactly what the next agent reads. Stripping on
persist therefore removes the NL representation from the loop, even though a
single turn can still reason in prose internally. The honest statement is that
no natural-language intermediate representation persists across agent
invocations, which is the representational claim, not a claim about the
model's private reasoning.

What is deliberately NOT ablated: the engine's own NL descriptions of
observations (diff text in context.txt, divergence feedback, the task prompt
itself). Those are the environment interface, identical in every arm. Ablating
them would change what the agent can observe rather than how it is allowed to
represent what it learned.
"""
from __future__ import annotations

import ast
import io
import re
import tokenize

# Docstring position: a string expression as the first statement of a module,
# function, or class. Every other string literal is data the code may need.
_DOCSTRING_PARENTS = (
    ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
)


class StripResult:
    """Stripped source plus what was removed, so the engine can log it and
    the arm can be shown to have actually bound."""

    __slots__ = ("source", "n_comments", "n_docstrings", "error")

    def __init__(
        self,
        source: str,
        n_comments: int = 0,
        n_docstrings: int = 0,
        error: str | None = None,
    ) -> None:
        self.source = source
        self.n_comments = n_comments
        self.n_docstrings = n_docstrings
        self.error = error

    @property
    def stripped_anything(self) -> bool:
        return bool(self.n_comments or self.n_docstrings)

    def as_dict(self) -> dict:
        return {
            "n_comments": self.n_comments,
            "n_docstrings": self.n_docstrings,
            "error": self.error,
        }


def _strip_comments(source: str) -> tuple[str, int]:
    """Remove comment text, leaving the rest of each line untouched.

    Line-oriented rather than a token re-serialization: re-emitting the token
    stream has to guess at whitespace, and guessing wrong ADDS lines. Since
    the engine re-strips carried-forward code every round, any expansion
    compounds. Comment token positions come from tokenize, so a '#' inside a
    string literal is never mistaken for a comment.
    """
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return source, 0
    cuts: dict[int, int] = {}
    for tok in tokens:
        if tok.type != tokenize.COMMENT:
            continue
        row, col = tok.start
        # Earliest comment on the line wins, so a trailing comment containing
        # a '#' cannot re-extend the cut.
        if row not in cuts or col < cuts[row]:
            cuts[row] = col
    if not cuts:
        return source, 0
    lines = source.splitlines()
    for row, col in cuts.items():
        if 1 <= row <= len(lines):
            lines[row - 1] = lines[row - 1][:col].rstrip()
    out = "\n".join(lines)
    if source.endswith("\n"):
        out += "\n"
    return out, len(cuts)


def _docstring_line_spans(tree: ast.AST) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, _DOCSTRING_PARENTS):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if not isinstance(first, ast.Expr):
            continue
        value = first.value
        if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
            continue
        if len(body) == 1:
            # Sole statement: it is load bearing as a body, so it must be
            # replaced rather than deleted.
            spans.append((first.lineno, first.end_lineno or first.lineno))
        else:
            spans.append((first.lineno, first.end_lineno or first.lineno))
    return spans


# Removing a comment or docstring leaves the line behind. Left uncollapsed
# those blanks accumulate every time the engine re-strips carried-forward
# code, which is once per synthesis round.
MAX_CONSECUTIVE_BLANKS = 2


def _collapse_blank_runs(source: str) -> str:
    """Cap runs of blank lines so the strip cannot grow the file.

    This is what makes the strip idempotent: without it, N rounds of
    re-stripping inflate a module without bound.
    """
    out: list[str] = []
    run = 0
    for line in source.splitlines():
        if line.strip():
            run = 0
            out.append(line)
            continue
        run += 1
        if run <= MAX_CONSECUTIVE_BLANKS:
            out.append("")
    while out and not out[-1].strip():
        out.pop()
    text = "\n".join(out)
    return text + "\n" if text else ""


def strip_natural_language(source: str) -> StripResult:
    """Remove comments and docstrings, keeping the module importable.

    A docstring that is a function's only statement becomes ``pass`` at the
    same indentation, since deleting it would leave an empty body.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return StripResult(source, error=f"syntax error: {exc}")

    sole_body_lines = set()
    for node in ast.walk(tree):
        if not isinstance(node, _DOCSTRING_PARENTS):
            continue
        body = getattr(node, "body", None)
        if not body or len(body) != 1:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                and isinstance(first.value.value, str):
            if not isinstance(node, ast.Module):
                sole_body_lines.add(first.lineno)

    spans = _docstring_line_spans(tree)
    lines = source.splitlines()
    drop: set[int] = set()
    replace: dict[int, str] = {}
    for start, end in spans:
        if start in sole_body_lines:
            indent = re.match(r"[ \t]*", lines[start - 1]).group(0)
            replace[start] = indent + "pass"
            for ln in range(start + 1, end + 1):
                drop.add(ln)
        else:
            for ln in range(start, end + 1):
                drop.add(ln)

    kept: list[str] = []
    for i, line in enumerate(lines, 1):
        if i in replace:
            kept.append(replace[i])
        elif i in drop:
            continue
        else:
            kept.append(line)
    no_docs = "\n".join(kept)
    if source.endswith("\n"):
        no_docs += "\n"

    stripped, n_comments = _strip_comments(no_docs)
    stripped = _collapse_blank_runs(stripped)
    # A strip that breaks the file is not a strip. Fall back rather than hand
    # the verifier something that will not import.
    try:
        ast.parse(stripped)
    except SyntaxError as exc:
        return StripResult(
            no_docs, n_comments=0, n_docstrings=len(spans),
            error=f"comment strip broke the module, kept comments: {exc}",
        )
    return StripResult(stripped, n_comments=n_comments, n_docstrings=len(spans))


def natural_language_violations(source: str) -> list[str]:
    """Prose that survived, for logging. Empty means the arm bound."""
    result = strip_natural_language(source)
    out: list[str] = []
    if result.error:
        out.append(result.error)
    if result.n_comments:
        out.append(f"{result.n_comments} comment(s)")
    if result.n_docstrings:
        out.append(f"{result.n_docstrings} docstring(s)")
    return out


def enforce_python_artifact(text: str) -> tuple[str, str | None]:
    """Make a persisted handoff artifact Python, or say why it is not.

    Returns (enforced_text, error). The shared documents keep their filenames
    across arms so that staging, snapshotting, and the analyzer handoff are
    byte-identical code paths in both, and only the content is manipulated.
    """
    if not text.strip():
        return text, None
    try:
        ast.parse(text)
    except SyntaxError as exc:
        return text, f"artifact is not valid Python: {exc}"
    result = strip_natural_language(text)
    if result.error:
        return text, result.error
    return result.source, None
