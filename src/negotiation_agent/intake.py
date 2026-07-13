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

# A scale word right after a money figure multiplies it: "EUR 2.4 million" is 2,400,000, not 2.4.
# Ignoring it is a silent 1e3–1e9x corruption. Matched at the match site (see `find`), anchored to
# ^ against a short slice after the number, so _NUM/_num stay pure and it can't add ReDoS surface.
# Bare "m" is deliberately excluded (ambiguous with metres); only unambiguous words/abbrevs scale.
# No ^ anchor: pattern.match(text, pos) already anchors the attempt AT pos (like \A there), and ^
# without re.MULTILINE would only ever match at offset 0. \s* then the scale word.
_SCALE_RE = re.compile(
    r"\s*(million|mio\.?|mrd\.?|milliarde[n]?|billion|thousand|tsd\.?)\b",
    re.I,
)
_SCALE_FACTOR = {
    "million": 1e6,
    "mio": 1e6,
    "mio.": 1e6,
    "mrd": 1e9,
    "mrd.": 1e9,
    "milliarde": 1e9,
    "milliarden": 1e9,
    "billion": 1e9,
    "thousand": 1e3,
    "tsd": 1e3,
    "tsd.": 1e3,
}

_PRICE_RE = re.compile(
    rf"(?:€|eur\s*)\s*({_NUM})|(?<![\d.,])({_NUM})\s*(?:€|eur)(?!\w)",
    re.I,
)
# An explicit per-unit CONTEXT: "EUR 48,500 per unit", "unit price of EUR 12,500", "Stückpreis".
# A machinery unit price can legitimately exceed the plausibility ceiling, so when this cue sits
# right on the price figure we KEEP it as a unit price instead of relabeling it a total. Matched
# against a bounded window around the price span (see extract), so it stays linear.
_PER_UNIT_CONTEXT_RE = re.compile(
    r"\bper\s+(?:unit|piece|item|pc|stück|stueck)s?\b|\beach\b|/\s*(?:unit|pc|stück)\b"
    r"|\bunit\s+price\b|\bprice\s+per\b|\bst(?:ü|ue)ckpreis\b",
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
# A figure that is NEITHER a unit price NOR a contract value: a liability/indemnity/penalty CAP
# ("total liability shall not exceed EUR 50,000"), an insurance-coverage minimum ("insurance with
# coverage of EUR 5,000,000"), or a security deposit/bond/retention ("security deposit of EUR
# 50,000"). All three are legal/financial obligations, not the deal's price or spend. The total
# regex excludes them by phrasing; the plausibility-ceiling relabel below and the price slot must
# too, or a large cap/cover/deposit gets mislabeled. Matches a cue within a bounded window before a
# currency figure. Window is 60 (not 40): "liability insurance with coverage of ... EUR" needs it,
# still bounded = still linear.
_NON_CONTRACT_FIGURE_RE = re.compile(
    r"\b(?:liabilit(?:y|ies)|indemnif(?:y|ies|ied|ication)|indemnit(?:y|ies)|penalt(?:y|ies)"
    r"|damages|liquidated|insur(?:ance|ed)|coverage|cover|deposit|retention|bond|guarantee)\b"
    r"[^.\n]{0,60}?(?:€|eur|usd|\$)\s*(" + _NUM + r")",
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


def _scaled_value(m: re.Match[str], text: str) -> tuple[float, str] | None:
    """(value, verbatim quote) for a money match, applying a trailing scale word ("2.4 million" →
    2_400_000). Returns None if the number can't be parsed. The scale word is matched immediately
    after the captured number group against a short bounded slice, so ``_NUM``/``_num`` stay pure
    and this adds no ReDoS surface."""
    v = _num(m)
    if v is None:
        return None
    num_end = next((m.end(i) for i, g in enumerate(m.groups(), 1) if g), None)
    quote = m.group(0).strip()
    if num_end is not None:
        sm = _SCALE_RE.match(text, num_end, num_end + 16)
        if sm is not None:
            v *= _SCALE_FACTOR[sm.group(1).lower()]
            quote = text[m.start() : sm.end()].strip()
    return (v, quote)


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
            return _scaled_value(m, text) if m is not None else None

        def add(name: str, pattern: re.Pattern[str]) -> None:
            hit = find(pattern)
            if hit is not None:
                terms.append(
                    ExtractedTerm(name=name, value=hit[0], quote=hit[1], confidence=0.8)
                )

        _MAX_PLAUSIBLE_UNIT_PRICE = 10_000.0

        # Figures that are NEITHER a price nor a contract value: liability/indemnity caps, insurance
        # minimums, deposits/bonds. Collect them all so the price/total search skips them, else
        # "Liquidated damages of EUR 50,000 ... unit price EUR 12.50" loses the real 12.50 to the
        # cap (or mislabels the 50,000 as a total).
        non_contract_amounts = {
            v for v in (_num(m) for m in _NON_CONTRACT_FIGURE_RE.finditer(text)) if v is not None
        }

        # Total contract value first — if the ONLY euro figure is a total, don't also mislabel it as
        # a per-unit price.
        total = find(_TOTAL_VALUE_RE)

        def _has_per_unit_context(span: tuple[int, int]) -> bool:
            """True if an explicit per-unit cue sits on the price figure at ``span`` — then a large
            figure is a legit (machinery) unit price, not a total the plausibility ceiling should
            relabel. Bounded window around the span keeps this linear."""
            lo, hi = max(0, span[0] - 40), min(len(text), span[1] + 20)
            return _PER_UNIT_CONTEXT_RE.search(text, lo, hi) is not None

        # Pick the price by scanning ALL price matches, not just the leftmost: skip figures that are
        # really caps/deposits/insurance, and skip a figure that IS the total's amount (dedup), so a
        # real per-unit price alongside a total/cap still surfaces. Remember the first implausible,
        # non-cap figure separately — with no labelled total it becomes the total_value fallback.
        price: tuple[float, str] | None = None
        implausible_fallback: tuple[float, str] | None = None
        for m in _PRICE_RE.finditer(text):
            hit = _scaled_value(m, text)
            if hit is None:
                continue
            value, quote = hit
            if value in non_contract_amounts:
                continue  # a cap/deposit/insurance figure is neither price nor total
            if total is not None and value == total[0]:
                continue  # same figure as the total — not a separate unit price
            if value > _MAX_PLAUSIBLE_UNIT_PRICE and not _has_per_unit_context(m.span()):
                # too large to be a unit price and no per-unit cue → a total the phrasing missed;
                # keep the FIRST such as the total_value fallback, keep scanning for a real price
                if implausible_fallback is None:
                    implausible_fallback = (value, quote)
                continue
            price = (value, quote)
            break

        if total is not None:
            terms.append(
                ExtractedTerm(name="total_value", value=total[0], quote=total[1], confidence=0.8)
            )
        elif implausible_fallback is not None:
            # No labelled total, but a lone figure too large to be per-unit (not a cap): it's the
            # total. Emit as total_value at lower confidence + warn. Stops "EUR 194,920 per unit".
            terms.append(
                ExtractedTerm(
                    name="total_value",
                    value=implausible_fallback[0],
                    quote=implausible_fallback[1],
                    confidence=0.5,
                )
            )

        if price is not None:
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
        if total is None and implausible_fallback is not None:
            warnings.append(
                f"The only monetary figure found ({implausible_fallback[0]:,.0f}) is too large to "
                f"be a per-unit price — it was read as a total/annual value, not a unit price. Set "
                f"a per-unit target and floor yourself, or confirm the unit price."
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
