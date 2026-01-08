from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Iterable, List, Set


_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9+./#-]*")


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    violations: List[str]
    new_numbers: List[str]
    new_tools: List[str]


def _normalize_number(token: str) -> str | None:
    raw = token.strip().rstrip("%")
    if not raw:
        return None
    try:
        return str(Decimal(raw).normalize())
    except InvalidOperation:
        return None


def _extract_numbers(text: str) -> Set[str]:
    values: Set[str] = set()
    for token in _NUMBER_RE.findall(text or ""):
        norm = _normalize_number(token)
        if norm is not None:
            values.add(norm)
    return values


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _extract_tool_tokens(text: str, allowlist: Set[str]) -> Set[str]:
    tokens = _tokenize(text)
    tool_like: Set[str] = set()
    for token in tokens:
        if token in allowlist:
            tool_like.add(token)
        elif re.search(r"[0-9+./#-]", token):
            tool_like.add(token)
    return tool_like


def _balanced_braces(text: str) -> bool:
    depth = 0
    for ch in text:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def validate_rewrite(
    original_text: str, rewritten_text: str, allowlist: Iterable[str]
) -> ValidationResult:
    """Validate a rewritten bullet against safety rules."""
    allow = {t.lower() for t in allowlist if t}
    original_tokens = set(_tokenize(original_text))
    allowed_terms = allow | original_tokens

    original_numbers = _extract_numbers(original_text)
    rewritten_numbers = _extract_numbers(rewritten_text)
    new_numbers = sorted(rewritten_numbers - original_numbers)

    rewritten_tools = _extract_tool_tokens(rewritten_text, allow)
    new_tools = sorted(t for t in rewritten_tools if t not in allowed_terms)

    violations: List[str] = []
    if new_numbers:
        violations.append("new_numbers")
    if new_tools:
        violations.append("new_tools")
    if not _balanced_braces(rewritten_text):
        violations.append("unbalanced_braces")
    if (rewritten_text or "").rstrip().endswith("\\"):
        violations.append("dangling_backslash")

    return ValidationResult(
        ok=not violations,
        violations=violations,
        new_numbers=new_numbers,
        new_tools=new_tools,
    )
