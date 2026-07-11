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
    # expanded 2026-07-11 to cover the full CPV map (the buyer negotiates every category)
    "financial_insurance",
    "it_hardware",
    "repair_maintenance",
    "engineering_construction",
    "real_estate",
    "telecoms_equipment",
    "telecom_services",
    "training_education",
    "travel_catering",
    "installation",
    "other_business",
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
    "financial_insurance": "Financial & insurance services",
    "it_hardware": "IT hardware & office equipment",
    "repair_maintenance": "Repair & maintenance services",
    "engineering_construction": "Engineering / architecture / inspection",
    "real_estate": "Real estate & property",
    "telecoms_equipment": "Telecoms & AV equipment",
    "telecom_services": "Postal & telecom services",
    "training_education": "Training & education services",
    "travel_catering": "Travel, hotel & catering",
    "installation": "Installation services",
    "other_business": "Other business & personal services",
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
    "financial_insurance": {
        "insurance": 3,
        "premium": 2,
        "underwrit": 3,
        "actuar": 3,
        "banking": 3,
        "credit facility": 3,
        "brokerage": 3,
        "policy": 1,
        "liability cover": 3,
        "reinsurance": 3,
        "treasury": 2,
        "pension fund": 3,
        "loan": 2,
        "hedging": 2,
    },
    "it_hardware": {
        "laptop": 3,
        "desktop": 2,
        "workstation": 3,
        "server hardware": 3,
        "monitor": 1,
        "peripheral": 2,
        "printer": 1,
        "toner": 2,
        "device fleet": 3,
        "endpoint device": 3,
        "warranty": 1,
        "rma": 2,
        "office equipment": 2,
        "docking station": 2,
    },
    "repair_maintenance": {
        "repair": 2,
        "maintenance contract": 3,
        "preventive maintenance": 3,
        "spare parts": 2,
        "breakdown": 2,
        "service level": 1,
        "uptime": 2,
        "fleet maintenance": 3,
        "planned maintenance": 3,
        "callout": 2,
        "wartungsvertrag": 3,
    },
    "engineering_construction": {
        "engineering": 2,
        "architect": 3,
        "structural": 2,
        "civil engineering": 3,
        "construction management": 3,
        "quantity surveying": 3,
        "inspection": 2,
        "site supervision": 3,
        "cad": 1,
        "feasibility study": 2,
        "surveying": 2,
    },
    "real_estate": {
        "lease": 3,
        "tenancy": 3,
        "landlord": 3,
        "property management": 3,
        "rent": 2,
        "square metre": 1,
        "office space": 2,
        "sublet": 3,
        "real estate": 3,
        "service charge": 2,
        "dilapidations": 3,
        "mietvertrag": 3,
    },
    "telecoms_equipment": {
        "switchboard": 3,
        "pabx": 3,
        "fibre-optic": 3,
        "handset": 2,
        "network cabling": 3,
        "satellite dish": 3,
        "av equipment": 3,
        "loudspeaker": 2,
        "transmitter": 2,
        "router hardware": 2,
    },
    "telecom_services": {
        "mobile plan": 3,
        "data plan": 3,
        "sim": 2,
        "roaming": 3,
        "call minutes": 3,
        "connectivity": 2,
        "isp": 3,
        "leased line": 3,
        "postal service": 3,
        "courier": 2,
        "tariff": 2,
        "mvno": 3,
        "carrier contract": 3,
    },
    "training_education": {
        "training": 2,
        "course": 2,
        "e-learning": 3,
        "certification course": 3,
        "seminar": 2,
        "workshop": 1,
        "curriculum": 2,
        "learning management": 3,
        "instructor": 2,
        "upskilling": 2,
        "schulung": 3,
    },
    "travel_catering": {
        "hotel": 3,
        "catering": 3,
        "canteen": 3,
        "travel management": 3,
        "airfare": 3,
        "per diem": 3,
        "accommodation": 2,
        "banquet": 2,
        "meal": 1,
        "corporate travel": 3,
        "room rate": 2,
        "gbt": 2,
    },
    "installation": {
        "installation": 2,
        "commissioning": 3,
        "fit-out": 3,
        "wiring": 2,
        "mounting": 2,
        "rollout": 2,
        "deployment on-site": 3,
        "hardware installation": 3,
        "cabling": 2,
    },
    "other_business": {
        "translation": 3,
        "interpreting": 3,
        "printing": 2,
        "scanning": 2,
        "archiving": 2,
        "records management": 3,
        "reprographic": 3,
        "mailroom": 2,
        "document management service": 3,
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


# A standards code in the text. UNSPSC = the global spend standard; CPV = the EU procurement
# standard (German Vergabe / TED). Authoritative when present: a coded contract beats keywords.
_UNSPSC_RE = re.compile(r"unspsc\D{0,6}(\d{2,8})", re.IGNORECASE)
_CPV_RE = re.compile(r"cpv\D{0,6}(\d{2,8})", re.IGNORECASE)


def detect_category(text: str, *, hint: str | None = None) -> tuple[Category, list[str]]:
    """Classify contract/negotiation text into a procurement category.

    If the text carries a UNSPSC or CPV code (the standards real spend systems / EU tenders
    assign), that wins — a coded contract is authoritative. Otherwise ``hint`` (e.g. the setup
    form's free-text category) is scored alongside the contract body; it nudges but never
    overrides a strong contract signal. Returns the category and the matched terms (legible,
    auditable). ``unknown`` when nothing scores.
    """
    both = f"{text}\n{hint or ''}"
    # CPV first (this is an EU/German procurement tool), then UNSPSC — a coded contract wins.
    cpv = _CPV_RE.search(both)
    if cpv:
        from negotiation_agent.knowledge.cpv import category_from_cpv

        mapped = category_from_cpv(cpv.group(1))
        if mapped != "unknown":
            return mapped, [f"CPV {cpv.group(1)}"]
    uns = _UNSPSC_RE.search(both)
    if uns:
        from negotiation_agent.knowledge.unspsc import category_from_unspsc

        mapped = category_from_unspsc(uns.group(1))
        if mapped != "unknown":
            return mapped, [f"UNSPSC {uns.group(1)}"]

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
