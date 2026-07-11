"""UNSPSC segment → negotiation-strategy category crosswalk.

UNSPSC (United Nations Standard Products and Services Code) is the taxonomy real
procurement systems (SAP Ariba, Coupa, and this project's own SpendLens/Hermes) classify
spend into. It is hierarchical: an 8-digit code's leading 2 digits are the *segment*.

The negotiation engine doesn't need 1.2M leaf codes — it needs the ~9 coarse strategy
buckets (``category.Category``). This maps the public UNSPSC segments to those buckets, so:
  - a contract already coded by a spend system can be routed to the right playbook without
    re-running keyword detection, and
  - the keyword detector and the standard agree on the same vocabulary.

Only service/indirect segments the engine has playbooks for are mapped; a direct-materials
or unmapped segment returns ``unknown`` (honest — no playbook, so the gap flag fires).
This is a small, stable, standards-based table — not a dependency on any external taxonomy
service. Segment titles are the official UNSPSC segment names (public standard).
"""

from __future__ import annotations

from negotiation_agent.knowledge.category import Category

# UNSPSC segment code (2-digit, as string) -> engine strategy category. Segments with no
# engine playbook are deliberately absent (they resolve to "unknown").
_SEGMENT_TO_CATEGORY: dict[str, Category] = {
    "43": "cloud_infrastructure",  # Information Technology Broadcasting and Telecommunications
    "81": "professional_services",  # Engineering, Research & Technology Based Services
    "80": "professional_services",  # Management and Business Professionals and Admin Services
    "82": "marketing",  # Editorial, Design, Graphic and Fine Art Services
    "83": "facility_services",  # Public Utilities and Public Sector Related Services
    "76": "facility_services",  # Industrial Cleaning Services
    "72": "facility_services",  # Building, Facility Construction and Maintenance Services
    "78": "logistics",  # Transportation, Storage and Mail Services
    "84": "legal_services",  # Financial and Insurance Services (legal/advisory overlap)
    "93": "legal_services",  # Politics and Civic Affairs Services (legal/regulatory)
    "86": "hr_services",  # Education and Training Services
    "85": "hr_services",  # Healthcare Services (benefits/HR admin overlap)
}

# A finer override: some 4-digit families sit under a broad segment but negotiate very
# differently. UNSPSC family 80.11 is Human resources services (incl. temp staffing).
_FAMILY_TO_CATEGORY: dict[str, Category] = {
    "8011": "hr_staffing_agency",  # Human resources services (temp/contingent staffing)
    "8010": "professional_services",  # Management advisory services
    "8014": "marketing",  # Trade shows and exhibits (events/marketing)
    "8216": "marketing",  # Advertising
    "8111": "cloud_infrastructure",  # Computer services
    "8112": "software_licenses",  # Data services / software
    "7811": "logistics",  # Mail and cargo transport
    "8012": "professional_services",  # Human resources development (training/consulting)
}


def _clean(code: str) -> str:
    return "".join(ch for ch in code if ch.isdigit())


def category_from_unspsc(code: str) -> Category:
    """Map a UNSPSC code (any length ≥2 digits) to an engine strategy category.

    Checks the more specific 4-digit family first (e.g. 8011 staffing beats 80 professional
    services), then the 2-digit segment. Returns ``unknown`` for anything unmapped — which
    correctly triggers the coverage-gap flag rather than guessing a playbook.
    """
    digits = _clean(code)
    if len(digits) >= 4 and digits[:4] in _FAMILY_TO_CATEGORY:
        return _FAMILY_TO_CATEGORY[digits[:4]]
    if len(digits) >= 2 and digits[:2] in _SEGMENT_TO_CATEGORY:
        return _SEGMENT_TO_CATEGORY[digits[:2]]
    return "unknown"
