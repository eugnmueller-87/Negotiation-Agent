"""Shared number-word parsing — one list, imported by both the guard and (v1) the
extractor, so the two never drift.

Deliberately finite and modest: it covers the spelled numbers that appear in real
negotiation prose ("net sixty", "twelve months", "ninety-six euros") up to a few
hundred. It is **defense-in-depth**, not a proof — the guard's honesty boundary
(``docs/peitho-v2-architecture.md`` §4.5) states plainly that open-ended paraphrase
is best-effort.
"""

from __future__ import annotations

import re

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_SCALES = {"hundred": 100, "thousand": 1000}

_ALL_WORDS = set(_UNITS) | set(_TENS) | set(_SCALES) | {"and"}
_WORD_RE = re.compile(r"[a-z]+", re.I)


def _words_to_int(tokens: list[str]) -> int | None:
    """Parse a run of number-words into an int, or None if not a clean number."""
    if not tokens:
        return None
    total = 0
    current = 0
    seen = False
    for tok in tokens:
        if tok == "and":
            continue
        if tok in _UNITS:
            current += _UNITS[tok]
            seen = True
        elif tok in _TENS:
            current += _TENS[tok]
            seen = True
        elif tok == "hundred":
            current = (current or 1) * 100
            seen = True
        elif tok == "thousand":
            total += (current or 1) * 1000
            current = 0
            seen = True
        else:
            return None
    return (total + current) if seen else None


def spelled_numbers(text: str) -> list[int]:
    """Every spelled-out integer found in ``text`` (best-effort, finite vocabulary).

    Groups consecutive number-words (allowing "and") and parses each run. Hyphenated
    forms like "forty-five" are handled because the tokenizer splits on non-letters.
    """
    tokens = _WORD_RE.findall(text.lower())
    out: list[int] = []
    run: list[str] = []
    for tok in tokens:
        if tok in _ALL_WORDS:
            run.append(tok)
        else:
            if run:
                val = _words_to_int(run)
                if val is not None:
                    out.append(val)
                run = []
    if run:
        val = _words_to_int(run)
        if val is not None:
            out.append(val)
    return out
