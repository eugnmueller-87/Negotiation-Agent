"""Contract intake — extract the starting position from an existing contract.

A negotiation usually doesn't start from nothing: there's a current contract with
a price, payment terms, a length, a renewal date. Uploading that document and
pulling the terms out gives the agent its **opening state** — the supplier's
current position — and the supplier's name to research.

This is the same extractor seam the v1 design uses for supplier email replies,
pointed at a document instead. The contract text is **untrusted input** (it may
contain prompt-injection), so extraction runs through an interface with no tool
access, and every field carries a confidence the caller can gate on.

``ContractExtractor`` is the seam: a deterministic stub for tests/offline use,
and an LLM-backed implementation (v1) that plugs in behind the same Protocol —
mirroring the ``SupplierAgent`` pattern in the simulator.
"""

from __future__ import annotations

import re
from typing import Protocol

from pydantic import BaseModel, Field

from negotiation_agent.envelope import Offer


class ExtractedTerm(BaseModel):
    """One term pulled from the contract, with the span it came from and a score."""

    model_config = {"frozen": True}

    name: str
    value: float
    quote: str = ""  # verbatim span the value came from (grounding)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)


class ContractExtraction(BaseModel):
    """The structured result of reading a contract."""

    model_config = {"frozen": True}

    supplier_name: str | None = None
    category: str | None = None
    terms: list[ExtractedTerm] = Field(default_factory=list)
    renewal_date: str | None = None
    warnings: list[str] = Field(default_factory=list)

    def to_offer(self, term_names: list[str]) -> Offer:
        """Build an :class:`Offer` for the given envelope term names.

        Only terms that were extracted are set; the caller merges the rest from
        the envelope's worst/opening values, exactly as the engine merges partial
        supplier offers. Never invents a value for a term it didn't find.
        """
        found = {t.name: t.value for t in self.terms}
        return Offer(terms={n: found[n] for n in term_names if n in found})

    def low_confidence(self, threshold: float = 0.6) -> list[str]:
        """Names of terms below the confidence threshold — the caller should
        confirm these with a human before trusting them as the opening state."""
        return [t.name for t in self.terms if t.confidence < threshold]


class ContractExtractor(Protocol):
    """Reads contract text into a structured :class:`ContractExtraction`.

    The v1 LLM-backed extractor implements this behind the same seam; it has no
    tool access and its only output is the validated structure.
    """

    def extract(self, contract_text: str) -> ContractExtraction: ...


# Contract text is untrusted (upload / injection surface). Cap the input so no
# regex can be driven quadratic by a pathological megabyte of digits, and bound
# every numeric quantifier so a single token stays linear. Real figures are short.
_MAX_CONTRACT_CHARS = 200_000  # ~200 KB of text; real contracts are far smaller

# Term-name → the units/patterns we recognise in a contract. Kept as data so the
# regex stub and the LLM prompt (v1) share one vocabulary.
_PRICE_RE = re.compile(
    r"(?:€|eur\s*)\s*([0-9]{1,12}(?:[.,][0-9]{1,6})?)"
    r"|([0-9]{1,12}(?:[.,][0-9]{1,6})?)\s*(?:€|eur)(?!\w)",
    re.I,
)
_PAYMENT_RE = re.compile(r"\bnet[\s-]?([0-9]{1,3})|([0-9]{1,3})\s*days?\b", re.I)
_MONTHS_RE = re.compile(r"([0-9]{1,3})\s*(?:month|months|mo)\b", re.I)
_VOLUME_RE = re.compile(r"([0-9][0-9.,]{1,15})\s*(?:units|pcs|pieces|stück)\b", re.I)
_REBATE_RE = re.compile(r"([0-9]{1,6}(?:[.,][0-9]{1,4})?)\s*%\s*(?:rebate|discount|rabatt)", re.I)
_SUPPLIER_RE = re.compile(
    r"(?:supplier|vendor|lieferant|seller)\s*[:\-]?\s*"
    r"([A-Z][\w&.\- ]{2,60}?(?:GmbH|AG|Ltd|Inc|B\.V\.|S\.A\.|SE|KG))",
    re.I,
)


def _num(m: re.Match[str] | None) -> float | None:
    """Parse a captured number, handling both decimal and thousands separators.

    Ambiguity between European decimal comma (``11,50``) and English thousands
    comma (``40,000``) is resolved by the grouping: a separator followed by
    exactly three digits and no further separators is treated as thousands;
    otherwise as a decimal point. Dots are handled symmetrically.
    """
    if not m:
        return None
    g = next((x for x in m.groups() if x), None)
    if not g:
        return None
    # thousands: 40,000 or 40.000 (sep + exactly 3 trailing digits, whole number)
    raw = g.replace(".", "").replace(",", "") if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", g) else None
    val = float(raw) if raw is not None else float(g.replace(",", "."))
    # A pathological long digit run can float() to inf; never emit a non-finite value.
    return val if val == val and val not in (float("inf"), float("-inf")) else None


class RegexContractExtractor:
    """A deterministic, dependency-free extractor for tests, demos, and offline use.

    It reads the common contract figures with regexes. Confidence is a coarse
    signal (a matched pattern is 0.8, nothing is not emitted) — the real
    per-field calibration is the LLM extractor's job in v1. It never guesses:
    a term with no match is simply absent.
    """

    def extract(self, contract_text: str) -> ContractExtraction:
        # Cap untrusted input before any regex runs (ReDoS defense-in-depth).
        text = (contract_text or "")[:_MAX_CONTRACT_CHARS]
        terms: list[ExtractedTerm] = []

        def add(name: str, pattern: re.Pattern[str]) -> None:
            m = pattern.search(text)
            if m is None:
                return
            v = _num(m)
            if v is not None:
                terms.append(
                    ExtractedTerm(name=name, value=v, quote=m.group(0).strip(), confidence=0.8)
                )

        add("price", _PRICE_RE)
        add("payment_days", _PAYMENT_RE)
        add("contract_months", _MONTHS_RE)
        add("volume_units", _VOLUME_RE)
        add("rebate_pct", _REBATE_RE)

        sm = _SUPPLIER_RE.search(text)
        supplier = sm.group(1).strip() if sm else None

        warnings: list[str] = []
        if not terms:
            warnings.append("No negotiable terms were recognised in the document.")
        if supplier is None:
            warnings.append("No supplier legal name was recognised — confirm before research.")

        return ContractExtraction(supplier_name=supplier, terms=terms, warnings=warnings)


def extract_contract(
    contract_text: str, extractor: ContractExtractor | None = None
) -> ContractExtraction:
    """Convenience: extract with the deterministic extractor by default.

    Pass an LLM-backed extractor (v1) to swap the implementation behind the seam.
    """
    return (extractor or RegexContractExtractor()).extract(contract_text)
