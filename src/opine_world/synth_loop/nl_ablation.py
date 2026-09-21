"""Natural-language ablation. Removes comments and docstrings from what the agents keep."""

from __future__ import annotations

import ast
import io
import re
import tokenize

_DOCSTRING_PARENTS = (
    ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
)


class StripResult:
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
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return source, 0
    cuts: dict[int, int] = {}
    for tok in tokens:
        if tok.type != tokenize.COMMENT:
            continue
        row, col = tok.start
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
            spans.append((first.lineno, first.end_lineno or first.lineno))
        else:
            spans.append((first.lineno, first.end_lineno or first.lineno))
    return spans


MAX_CONSECUTIVE_BLANKS = 2


def _collapse_blank_runs(source: str) -> str:
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
    """Remove comments and docstrings and keep the code importable."""
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
    try:
        ast.parse(stripped)
    except SyntaxError as exc:
        return StripResult(
            no_docs, n_comments=0, n_docstrings=len(spans),
            error=f"comment strip broke the module, kept comments: {exc}",
        )
    return StripResult(stripped, n_comments=n_comments, n_docstrings=len(spans))


def natural_language_violations(source: str) -> list[str]:
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
    """Return the text if it is Python, or the reason it is not."""
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
