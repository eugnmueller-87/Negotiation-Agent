"""CPV division → negotiation-strategy category crosswalk.

CPV (Common Procurement Vocabulary) is the EU procurement taxonomy — what German ``Vergabe``
and EU TED tenders code spend in (the European counterpart to UNSPSC). A CPV code's leading
2 digits are the *division*; a few 3-digit *groups* negotiate differently enough to override
the division.

The buyer negotiates across the whole CPV map, so this maps every service/indirect division
the engine has a strategy bucket for. Like ``unspsc``, this is a small, stable table — not a
dependency on the 4,600-code CPV file. A coded contract routes to the right playbook without
re-running keyword detection, and the keyword detector and the EU standard agree.
"""

from __future__ import annotations

from negotiation_agent.knowledge.category import Category

# CPV division (2-digit) -> engine strategy category. Divisions with no engine bucket are
# absent (resolve to "unknown" -> the coverage-gap flag fires, honestly).
_DIVISION_TO_CATEGORY: dict[str, Category] = {
    "48": "software_licenses",  # Software package and information systems
    "72": "cloud_infrastructure",  # IT services: consulting, software dev, internet, support
    "30": "it_hardware",  # Office and computing machinery, equipment and supplies
    "32": "telecoms_equipment",  # Radio, TV, communication, telecom and related equipment
    "50": "repair_maintenance",  # Repair and maintenance services
    "51": "installation",  # Installation services (except software)
    "55": "travel_catering",  # Hotel, restaurant and retail trade services
    "60": "logistics",  # Transport services (excl. waste transport)
    "63": "logistics",  # Supporting/auxiliary transport; travel agency services
    "64": "telecom_services",  # Postal and telecommunications services
    "66": "financial_insurance",  # Financial and insurance services
    "70": "real_estate",  # Real estate services
    "71": "engineering_construction",  # Architectural, construction, engineering, inspection
    "80": "training_education",  # Education and training services
    "90": "facility_services",  # Sewage, refuse, cleaning and environmental services
    "92": "marketing",  # Recreational, cultural and sporting services (media/events)
    "98": "other_business",  # Other community, social and personal services
}

# Finer 3-digit CPV groups whose negotiation differs from their broad division. CPV 79 is
# "Business services: law, marketing, consulting, recruitment, printing, security" — a mixed
# division that must be split to route correctly.
_GROUP_TO_CATEGORY: dict[str, Category] = {
    "791": "legal_services",  # Legal services
    "792": "professional_services",  # Accounting, auditing, fiscal
    "793": "marketing",  # Market research; advertising and marketing
    "794": "professional_services",  # Business and management consultancy
    "795": "other_business",  # Office-support services
    "796": "hr_staffing_agency",  # Recruitment / supply of personnel incl. temp staff
    "797": "facility_services",  # Investigation and security services (guarding)
    "798": "other_business",  # Printing and related services
    "799": "other_business",  # Miscellaneous business services
    "722": "cloud_infrastructure",  # IT systems/technical consultancy, hosting
}


def _digits(code: str) -> str:
    return "".join(ch for ch in code if ch.isdigit())


def category_from_cpv(code: str) -> Category:
    """Map a CPV code (any length ≥2 digits) to an engine strategy category.

    Checks the more specific 3-digit group first (e.g. 796 recruitment beats 79 business
    services), then the 2-digit division. Returns ``unknown`` for anything unmapped — which
    correctly triggers the coverage-gap flag rather than guessing a playbook.
    """
    digits = _digits(code)
    if len(digits) >= 3 and digits[:3] in _GROUP_TO_CATEGORY:
        return _GROUP_TO_CATEGORY[digits[:3]]
    if len(digits) >= 2 and digits[:2] in _DIVISION_TO_CATEGORY:
        return _DIVISION_TO_CATEGORY[digits[:2]]
    return "unknown"
