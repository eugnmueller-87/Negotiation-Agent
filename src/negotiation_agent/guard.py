"""The numeric guard — the mechanism that makes the product's claim TRUE.

Claim: *the LLM cannot cause an unapproved figure to leave the server.* This module
is where that becomes mechanical rather than aspirational. The old demo annotated a
message *after* committing it (theater); here, ``check`` runs *before* release and a
violating draft has exactly two fates — redraft, or fall through to a deterministic
template. There is no code path where a violating string is returned.

Three deterministic layers ship (numeric, spelled-number, mechanism-leak). A fourth
semantic layer is out of scope here and is documented as best-effort in
``docs/peitho-v2-architecture.md`` §4.5. The honest headline is "the model can't emit
an unapproved **figure**" — never "the model can't cheat."
"""

from __future__ import annotations

import re

from negotiation_agent.numbers import spelled_numbers

# A numeric literal: digits with an optional decimal and thousands separators.
_NUM_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:[.,]\d+)?")

# Mechanism-leak bigrams — SPECIFIC internal-jargon phrases, not bare words. Bare
# "utility"/"floor"/"target"/"beta" false-positive on ordinary buyer prose ("your
# target go-live", "the floor on volume"), so we deny only phrases that could only
# be the engine narrating itself.
_LEAK_PHRASES = (
    "reservation utility",
    "target utility",
    "walk-away point",
    "walk away point",
    "boulware",
    "concession curve",
    "acceptance threshold",
    "utility score",
    # deadline / room-of-movement leakage (audit #9): the schedule pressure and floor
    # proximity are buyer-internal leverage the counterparty must not read in prose.
    "final round",
    "last round",
    "our final offer",
    "our walk-away",
    "our floor",
    "room to move before",
    "rounds remaining",
    "rounds left",
)

# First-person GIVING frames — a concession of value not in the allowlist. Scoped to
# giving, so "can you improve the rebate?" (an ask) is not flagged.
_CONCESSION_FRAMES = (
    "we'll waive",
    "we will waive",
    "we can include",
    "we'll include",
    "we can throw in",
    "we'll throw in",
    "at no cost to you",
    "free of charge",
    "no charge",
    "on the house",
)


def _to_number(raw: str) -> float:
    """Parse a matched numeric literal, resolving thousands vs decimal separators."""
    cleaned = raw.replace(",", "") if re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", raw) else raw
    return float(cleaned.replace(",", "."))


def _approved_forms(approved: dict[str, float]) -> list[float]:
    return [float(v) for v in approved.values()]


def _matches_approved(value: float, approved_values: list[float], *, tol: float) -> bool:
    """True if ``value`` equals an approved value within display tolerance.

    Integers must match exactly (tol collapses to 0 for whole numbers); continuous
    values match within half the display ULP so "€96.00" == 96.0 but "96.5" != 96.0.
    """
    for a in approved_values:
        if a == int(a):  # integer-valued approved number -> exact
            if value == a:
                return True
        elif abs(value - a) <= tol:
            return True
    return False


def check(text: str, approved: dict[str, float], *, tol: float = 0.005) -> list[str]:
    """Return the list of guard violations in ``text``. Empty list == clean.

    ``approved`` is ``EngineDecision.approved_numbers`` (the allowlist). On ESCALATE
    it is empty, so *any* figure is a violation and the message must be figure-free.
    """
    approved_values = _approved_forms(approved)
    violations: list[str] = []

    # Layer 1 — numeric literals.
    for m in _NUM_RE.finditer(text):
        raw = m.group(0)
        value = _to_number(raw)
        if not _matches_approved(value, approved_values, tol=tol):
            violations.append(raw)

    # Layer 2 — spelled-out numbers.
    for value in spelled_numbers(text):
        if not _matches_approved(float(value), approved_values, tol=tol):
            violations.append(f"{value} (spelled)")

    # Layer 3 — mechanism-leak phrases.
    low = text.lower()
    for phrase in _LEAK_PHRASES:
        if phrase in low:
            violations.append(f"leak:{phrase}")

    # Concession frames giving unlisted value.
    for frame in _CONCESSION_FRAMES:
        if frame in low:
            violations.append(f"concession:{frame}")

    return violations


def is_clean(text: str, approved: dict[str, float], *, tol: float = 0.005) -> bool:
    return not check(text, approved, tol=tol)
