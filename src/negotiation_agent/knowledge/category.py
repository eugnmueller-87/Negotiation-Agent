"""Procurement category detection — read a contract, name its category, deterministically.

The buyer agent should negotiate a cloud contract with cloud tactics and an HR-agency
contract with staffing tactics — not generic advice. That means knowing the *category* of
the deal in front of it. This classifies contract text into a fixed category vocabulary by
weighted keyword scoring: fully automatic (no human picks a category), pure, and explainable
(the matched terms are returned so a mis-detect is legible, not silent).

The vocabulary is deliberately the procurement spend categories the user named plus the
common indirect ones. ``UNKNOWN`` is a first-class result — better to say "I don't know this
category, using general strategy" than to force a wrong one (the honesty rule).
"""

from __future__ import annotations

import re
from typing import Literal

Category = Literal[
    "cloud_infrastructure",
    "software_licenses",
    "marketing",
    "hr_services",
    "hr_staffing_agency",
    "legal_services",
    "facility_services",
    "professional_services",
    "logistics",
    "unknown",
]

# Human labels for the UI / prompts.
CATEGORY_LABELS: dict[str, str] = {
    "cloud_infrastructure": "Cloud & infrastructure",
    "software_licenses": "Software licenses / SaaS",
    "marketing": "Marketing & agencies",
    "hr_services": "HR services",
    "hr_staffing_agency": "HR staffing / temp agency",
    "legal_services": "Legal services",
    "facility_services": "Facility services",
    "professional_services": "Professional / consulting services",
    "logistics": "Logistics & freight",
    "unknown": "Uncategorised",
}

# Weighted signal terms per category. A strong, unambiguous term scores 3; a supporting
# term scores 1. Matched as whole words, case-insensitive. Ordering of the dict is the
# tie-break preference (more specific categories first), so e.g. an HR *staffing agency*
# contract isn't swallowed by generic "hr_services".
_SIGNALS: dict[str, dict[str, int]] = {
    "hr_staffing_agency": {
        "staffing": 3,
        "temp agency": 3,
        "temporary staff": 3,
        "contingent": 3,
        "recruitment agency": 3,
        "placement fee": 3,
        "contractor": 1,
        "headcount": 1,
        "temp": 1,
        "agency worker": 3,
        "personaldienstleister": 3,
        "zeitarbeit": 3,
    },
    "cloud_infrastructure": {
        "cloud": 3,
        "aws": 3,
        "azure": 3,
        "gcp": 3,
        "compute": 2,
        "vcpu": 3,
        "vm": 1,
        "kubernetes": 2,
        "iaas": 3,
        "paas": 3,
        "hosting": 2,
        "data center": 2,
        "datacenter": 2,
        "bandwidth": 1,
        "storage": 1,
        "region": 1,
        "egress": 2,
    },
    "software_licenses": {
        "license": 3,
        "licence": 3,
        "saas": 3,
        "subscription": 2,
        "seat": 2,
        "seats": 2,
        "per user": 2,
        "named user": 3,
        "annual recurring": 1,
        "software": 1,
        "renewal": 1,
        "entitlement": 2,
        "true-up": 3,
        "maintenance": 1,
        "on-premise": 1,
    },
    "marketing": {
        "marketing": 3,
        "advertising": 3,
        "campaign": 2,
        "media buy": 3,
        "creative": 1,
        "agency of record": 3,
        "impressions": 2,
        "cpm": 2,
        "brand": 1,
        "seo": 2,
        "ppc": 2,
        "influencer": 2,
        "content production": 2,
    },
    "legal_services": {
        "legal": 3,
        "law firm": 3,
        "counsel": 2,
        "attorney": 3,
        "litigation": 3,
        "outside counsel": 3,
        "billable hour": 3,
        "retainer": 2,
        "solicitor": 3,
        "kanzlei": 3,
        "rechtsanwalt": 3,
    },
    "facility_services": {
        "facility": 3,
        "cleaning": 3,
        "janitorial": 3,
        "maintenance service": 2,
        "security guard": 3,
        "catering": 2,
        "landscaping": 2,
        "hvac": 2,
        "waste": 1,
        "gebäudereinigung": 3,
        "facility management": 3,
        "fm services": 3,
    },
    "professional_services": {
        "consulting": 3,
        "consultancy": 3,
        "advisory": 2,
        "statement of work": 2,
        "sow": 2,
        "deliverable": 1,
        "day rate": 2,
        "implementation": 1,
        "system integrator": 2,
    },
    "logistics": {
        "freight": 3,
        "logistics": 3,
        "shipping": 2,
        "carrier": 2,
        "incoterms": 3,
        "warehousing": 2,
        "customs": 2,
        "pallet": 1,
        "haulage": 3,
        "last mile": 2,
    },
    "hr_services": {
        "payroll": 3,
        "hris": 3,
        "benefits administration": 3,
        "human resources": 2,
        "onboarding": 1,
        "employee": 1,
        "hr platform": 3,
        "talent management": 2,
    },
}


def _score(text: str, terms: dict[str, int]) -> tuple[int, list[str]]:
    score = 0
    hits: list[str] = []
    for term, weight in terms.items():
        # whole-word / phrase match; escape the term, allow word boundaries
        if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text):
            score += weight
            hits.append(term)
    return score, hits


# A UNSPSC code in the text — "UNSPSC 80111600", "unspsc: 43211500", or a bare 8-digit code
# alongside the word. Authoritative when present: a spend-system-coded contract beats keywords.
_UNSPSC_RE = re.compile(r"unspsc\D{0,6}(\d{2,8})", re.IGNORECASE)


def detect_category(text: str, *, hint: str | None = None) -> tuple[Category, list[str]]:
    """Classify contract/negotiation text into a procurement category.

    If the text carries a UNSPSC code (the standard real spend systems assign), that wins —
    a coded contract is authoritative. Otherwise ``hint`` (e.g. the setup form's free-text
    category) is scored alongside the contract body; it nudges but never overrides a strong
    contract signal. Returns the category and the matched terms (legible, auditable).
    ``unknown`` when nothing scores.
    """
    # UNSPSC code takes priority — the standards-based path, and the bridge to spend systems.
    m = _UNSPSC_RE.search(text) or (_UNSPSC_RE.search(hint) if hint else None)
    if m:
        from negotiation_agent.knowledge.unspsc import category_from_unspsc

        mapped = category_from_unspsc(m.group(1))
        if mapped != "unknown":
            return mapped, [f"UNSPSC {m.group(1)}"]

    haystack = f"{text}\n{hint or ''}".lower()
    best: Category = "unknown"
    best_score = 0
    best_hits: list[str] = []
    for category, terms in _SIGNALS.items():
        score, hits = _score(haystack, terms)
        if score > best_score:
            best, best_score, best_hits = category, score, hits  # type: ignore[assignment]
    # Require a minimum signal so a stray keyword doesn't force a category.
    if best_score < 3:
        return "unknown", []
    return best, best_hits
