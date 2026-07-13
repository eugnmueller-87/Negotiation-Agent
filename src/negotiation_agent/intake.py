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
# 2 MB of TEXT comfortably covers any real contract's words plus schedules/exhibits
# (a 20 MB uploaded PDF is mostly images; its extracted text is a fraction of that).
# If input still exceeds this, we truncate AND warn — never a silent cut.
_MAX_CONTRACT_CHARS = 2_000_000  # ~2 MB of text

# Term-name → the units/patterns we recognise in a contract. Kept as data so the
# regex stub and the LLM prompt (v1) share one vocabulary.
# A well-formed number token: optional thousands groups then an optional decimal tail.
# Matches 11 · 11.50 · 1.234,56 · 1,234.56 · 40,000 — but not 1.2.3 (that fails _num too).
_NUM = r"[0-9]{1,3}(?:[.,][0-9]{3})*(?:[.,][0-9]{1,6})?|[0-9]{1,12}(?:[.,][0-9]{1,6})?"

_PRICE_RE = re.compile(
    rf"(?:€|eur\s*)\s*({_NUM})|(?<![\d.,])({_NUM})\s*(?:€|eur)(?!\w)",
    re.I,
)
# net-N is the authoritative payment term. The bare "N days" fallback requires a payment
# context word nearby and a leading \b, so "90 days notice" (a termination clause) is not
# misread as payment_days. net-N is preferred by being the first alternative searched.
# The third alternative catches "due/payable within thirty (30) days" — real contracts spell the
# number and give the digit in parentheses; the parenthesized digit is authoritative. A bounded
# [^.\n]{0,40} context window keeps it linear and stops it reaching across sentences.
_PAYMENT_RE = re.compile(
    r"\bnet[\s-]?([0-9]{1,3})\b"
    r"|\b(?:payment|due|invoice|payable|zahlungsziel|zahlbar)[^.\n]{0,40}?\bwithin\b"
    r"[^.\n]{0,20}?\(([0-9]{1,3})\)\s*days?\b"
    r"|\b(?:payment|due|invoice|payable|zahlungsziel|zahlbar)[^.\n]{0,40}?\b([0-9]{1,3})\s*days?\b",
    re.I,
)
# Contract length in months. Prefer a parenthesized digit ("twenty-four (24) months") when
# present — spelled numbers are common in the term clause and the digit is authoritative.
_MONTHS_RE = re.compile(
    r"\(([0-9]{1,3})\)\s*(?:month|months)\b|\b([0-9]{1,3})\s*(?:month|months|mo)\b", re.I
)

# "Total contract value: EUR 194,920" is a TOTAL/annual figure, not a per-unit price. Extracting
# it as `price` (the per-unit lever) is wrong — it produces a nonsense per-unit target/floor. Pull
# it as a distinct `total_value` term the caller can label honestly and NOT treat as per-unit.
# Value nouns exclude "price/cost/amount" — those are the per-unit and liability-cap vectors
# ("total price per unit", "total liability ... amount"). Currency is MANDATORY.
#
# Two guards on EACH captured number, in this order:
#  1. (?![\d.,]?\d) — a TOKEN-BOUNDARY guard. Without it, when the per-unit lookahead below
#     rejects a full figure ("400,000 per annum"), the engine BACKTRACKS _NUM digit by digit
#     until it finds a shorter capture the lookahead accepts ("400,00" followed by "0") —
#     silently reading EUR 400,000 as 400.00 (a 1000x corruption whose quote contradicted its
#     value). The guard forbids ending a capture right before a digit (or separator+digit), so
#     the alternative fails cleanly instead of truncating. (Fable-5 red-team, verified 2026-07-13.)
#  2. the per-UNIT-rate ban ("EUR 12 per unit" is a unit price, not a total). "per annum"/"per
#     year" are EXEMPTED (an annual total IS a total); "per unit"/"per piece" stay rejected.
_PER_UNIT_BAN = r"(?!\s*(?:per(?!\s+(?:annum|year)\b)|each|/|pro)\b)"
_NUM_END = r"(?![\d.,]?\d)"
# The captured number, then the token-boundary guard, then the per-unit-rate ban (see above).
_TOTAL_NUM = "(" + _NUM + ")" + _NUM_END + _PER_UNIT_BAN
_TOTAL_VALUE_RE = re.compile(
    r"\b(?:total|aggregate|annual)\b[^.\n]{0,30}?"
    r"\b(?:contract|subscription|order|deal)?\s*(?:value|fees?|spend)\b"
    r"[^.\n]{0,20}?(?:€|eur|usd|\$)\s*" + _TOTAL_NUM
    + r"|(?:€|eur|usd|\$)\s*" + _TOTAL_NUM
    + r"[^.\n]{0,20}?\b(?:total|in total|per annum|per year|annually|for the (?:initial )?term)\b",
    re.I,
)
# A liability/penalty/indemnity CAP figure ("total liability shall not exceed EUR 50,000") is
# neither a unit price nor a contract value — it's a legal cap. _TOTAL_VALUE_RE already excludes
# it; the plausibility-ceiling relabel below must too, or a large cap gets mislabeled total_value.
# This matches a liability cue within a bounded window before a currency figure, so we can
# recognise the cap figure and refuse to treat it as price OR total_value. Bounded window = linear.
_LIABILITY_CAP_RE = re.compile(
    r"\b(?:liabilit(?:y|ies)|indemnit(?:y|ies)|penalt(?:y|ies)|damages|liquidated)\b"
    r"[^.\n]{0,40}?(?:€|eur|usd|\$)\s*(" + _NUM + r")",
    re.I,
)
# the (?<![\d.,]) stops a match from starting mid-token, so "1.2.3 units" is rejected
# whole (no spurious "2.3") rather than backtracking into the malformed tail. It is ALSO the
# ReDoS guard: without it an unanchored _NUM re-tries its greedy consume-then-fail at every
# position over a long digit/comma run — O(N²), ~28 s at 64 KB. The lookbehind fails a
# match-start mid-token in O(1), collapsing the scan to linear (verified 28 s → 4 ms).
_VOLUME_RE = re.compile(rf"(?<![\d.,])({_NUM})\s*(?:units|pcs|pieces|stück)\b", re.I)
_REBATE_RE = re.compile(rf"(?<![\d.,])({_NUM})\s*%\s*(?:rebate|discount|rabatt)", re.I)
# A legal-entity name: a capitalised run ending in a company suffix. Bounded length; the suffix
# anchors the end so the run can't sprawl. Reused across the supplier patterns below.
_ENTITY = (
    r"[A-Z][\w&.,\- ]{2,60}?"
    r"(?:GmbH|AG|Ltd|LLC|Inc|Corp|Co|B\.V\.|N\.V\.|S\.A\.|S\.R\.L\.|SE|KG|plc|Pty)\b\.?"
)

# Supplier name, most-specific pattern first:
#  1. an explicit "Supplier:/Vendor: <Entity>" label
#  2. "entered into between <Entity>, ..." — the classic MSA preamble (the FIRST party named is
#     usually the supplier/provider). We take the first entity after "between".
#  3. "by <Entity> ('defined term')" / "<Entity> ('Provider')" — the provider defined-term form.
_SUPPLIER_RES = (
    # 1. A LABELLED field "Supplier: <Entity>". The delimiter (:/-/=) is REQUIRED and the entity
    #    must start immediately after it — otherwise prose like "The Supplier shall provide services
    #    to Acme Buyer GmbH" would grab the BUYER (the label word treated as running text).
    re.compile(
        rf"(?:supplier|vendor|lieferant|seller|provider|licensor)\s*[:\-=]\s*({_ENTITY})", re.I
    ),
    re.compile(rf"\bentered into\b[^.\n]{{0,40}}?\bbetween\s+({_ENTITY})", re.I),
    re.compile(rf"\bby and between\s+({_ENTITY})", re.I),
    # "by <Entity> ("DefinedTerm")" — anchor on "by " so the entity can't greedily swallow the
    # preceding words. \bby\s+ then a NON-space-leading name run bounded before the suffix.
    re.compile(rf"\bby\s+({_ENTITY})\s*[,;]?\s*\(\s*[\"'“]", re.I),
)


def _num(m: re.Match[str] | None) -> float | None:
    """Parse a captured number, handling European and English separators. Never raises.

    Handles the four real forms seen in German/English procurement contracts:
      - plain              ``11.50`` / ``11,50`` → one separator = the decimal mark
      - thousands only     ``40,000`` / ``40.000`` → sep + groups of exactly 3, whole
      - full EU            ``1.234,56`` → dots group thousands, comma is the decimal
      - full English       ``1,234.56`` → commas group thousands, dot is the decimal
    When both separators appear, the LAST one is the decimal mark and the others are
    thousands groups. Anything malformed (``1.2.3``, ``1,000,``) returns ``None`` rather
    than raising — untrusted contract/supplier text must never 500 the extractor.
    """
    if not m:
        return None
    g = next((x for x in m.groups() if x), None)
    if not g:
        return None
    normalized = _normalize_number(g)
    if normalized is None:
        return None
    try:
        val = float(normalized)
    except ValueError:
        return None
    # A pathological long digit run can float() to inf; never emit a non-finite value.
    return val if val == val and val not in (float("inf"), float("-inf")) else None


def _normalize_number(g: str) -> str | None:
    """Turn a captured number token into a plain ``float``-parseable string, or ``None``.

    ``None`` for any token that isn't a well-formed number so the caller can treat it as
    'no match' — the extractor's honest contract (a term with no clean value is absent).
    """
    has_dot = "." in g
    has_comma = "," in g
    if has_dot and has_comma:
        # both present: the LAST separator is the decimal, the rest are thousands groups
        dec = "," if g.rfind(",") > g.rfind(".") else "."
        thou = "." if dec == "," else ","
        return g.replace(thou, "").replace(dec, ".")
    sep = "." if has_dot else "," if has_comma else ""
    if not sep:
        return g if g.isdigit() else None
    parts = g.split(sep)
    if not all(p.isdigit() for p in parts) or "" in parts:
        return None  # malformed: "1,000," or "1..2" → no clean value
    # thousands grouping: 2+ separators, first group 1–3 digits, every later group exactly 3
    if len(parts) > 2 and len(parts[0]) <= 3 and all(len(p) == 3 for p in parts[1:]):
        return "".join(parts)
    # exactly one separator: sep + exactly 3 trailing digits reads as thousands (40,000);
    # anything else is a decimal mark (11,50 → 11.5). Matches the documented v0 rule.
    if len(parts) == 2:
        if len(parts[0]) <= 3 and len(parts[1]) == 3:
            return parts[0] + parts[1]
        return parts[0] + "." + parts[1]
    return None


class RegexContractExtractor:
    """A deterministic, dependency-free extractor for tests, demos, and offline use.

    It reads the common contract figures with regexes. Confidence is a coarse
    signal (a matched pattern is 0.8, nothing is not emitted) — the real
    per-field calibration is the LLM extractor's job in v1. It never guesses:
    a term with no match is simply absent.
    """

    def extract(self, contract_text: str) -> ContractExtraction:
        # Cap untrusted input before any regex runs (ReDoS defense-in-depth). A cut is
        # rare (real contract TEXT is far under 2 MB) but must be surfaced, not silent.
        full = contract_text or ""
        text = full[:_MAX_CONTRACT_CHARS]
        truncated = len(full) > _MAX_CONTRACT_CHARS
        terms: list[ExtractedTerm] = []

        def find(pattern: re.Pattern[str]) -> tuple[float, str] | None:
            m = pattern.search(text)
            if m is None:
                return None
            v = _num(m)
            return (v, m.group(0).strip()) if v is not None else None

        def add(name: str, pattern: re.Pattern[str]) -> None:
            hit = find(pattern)
            if hit is not None:
                terms.append(
                    ExtractedTerm(name=name, value=hit[0], quote=hit[1], confidence=0.8)
                )

        # Total contract value first — if the ONLY euro figure is a total, don't also mislabel it as
        # a per-unit price. Emit `total_value` (a distinct term) and suppress `price` when the price
        # match is the same amount as the total (i.e. the total IS the only money in the document).
        total = find(_TOTAL_VALUE_RE)
        price = find(_PRICE_RE)

        # A per-unit price above this is almost never a unit price — it's a total/annual/lump-sum
        # fee the _TOTAL_VALUE_RE phrasing missed (e.g. "annual subscription fee is EUR 400,000").
        # Labeling it `price` produces a nonsense per-unit target/floor and, downstream, a mandate
        # the engine can't negotiate. When _PRICE_RE grabs a figure this large and no explicit total
        # was found, treat it AS the total, never an implausible `price`. Real per-unit procurement
        # prices are small (cents to low thousands); machinery is the rare high end.
        _MAX_PLAUSIBLE_UNIT_PRICE = 10_000.0
        price_implausible = price is not None and price[0] > _MAX_PLAUSIBLE_UNIT_PRICE
        # ...but a large figure in a LIABILITY/PENALTY context is a cap, not a contract value.
        # "Total liability shall not exceed EUR 50,000" must NOT become a total_value. The total
        # regex already excludes it; the plausibility relabel must respect the same boundary or it
        # re-introduces the exact mislabel. If the price figure is the amount a liability cue caps,
        # drop it — it's neither a unit price nor a contract value.
        cap = _num(_LIABILITY_CAP_RE.search(text)) if price is not None else None
        price_is_liability_cap = price is not None and cap is not None and cap == price[0]

        if total is not None:
            terms.append(
                ExtractedTerm(name="total_value", value=total[0], quote=total[1], confidence=0.8)
            )
        elif price_implausible and not price_is_liability_cap:
            # No labelled total, but the only money figure is too large to be per-unit — it's the
            # total. Emit it as total_value (lower confidence: phrasing wasn't an explicit total)
            # and warn. This is what stops "EUR 194,920" showing up as "194.920 € per unit".
            terms.append(
                ExtractedTerm(name="total_value", value=price[0], quote=price[1], confidence=0.5)
            )

        # keep `price` only if it's a plausible per-unit figure AND distinct from the total — a real
        # per-unit price alongside the total. An implausibly large figure is never a `price`, and a
        # liability-cap figure is never a price either.
        if (
            price is not None
            and not price_implausible
            and not price_is_liability_cap
            and (total is None or price[0] != total[0])
        ):
            terms.append(
                ExtractedTerm(name="price", value=price[0], quote=price[1], confidence=0.8)
            )

        add("payment_days", _PAYMENT_RE)
        add("contract_months", _MONTHS_RE)
        add("volume_units", _VOLUME_RE)
        add("rebate_pct", _REBATE_RE)

        supplier: str | None = None
        for pattern in _SUPPLIER_RES:
            sm = pattern.search(text)
            if sm:
                supplier = sm.group(1).strip().rstrip(",").strip()
                break

        warnings: list[str] = []
        if total is None and price_implausible:
            warnings.append(
                f"The only monetary figure found ({price[0]:,.0f}) is too large to be a per-unit "
                f"price — it was read as a total/annual value, not a unit price. Set a per-unit "
                f"target and floor yourself, or confirm the unit price."
            )
        if truncated:
            warnings.append(
                f"Document text exceeded {_MAX_CONTRACT_CHARS // 1000} KB and was truncated for "
                f"analysis — only the first part was read. Split it or extract the key clauses."
            )
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
