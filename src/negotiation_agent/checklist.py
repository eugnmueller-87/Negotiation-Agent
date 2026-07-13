"""The senior-counsel review checklist — what an experienced reviewer never forgets to look for.

A junior reviewer reads the clauses that are present and flags the bad ones. A SENIOR counsel
works from a mental checklist: they know the protections a well-drafted contract *should* contain,
so they catch the missing DPA, the absent audit right, the auto-renewal with no notice window — the
risks of OMISSION a clause-by-clause reader misses. This module is that checklist, as data.

Two uses:
  - :func:`checklist_prompt_block` renders the checklist into the extraction prompt so the LLM hunts
    the full list per category AND is told to file a finding when a listed protection is ABSENT.
  - each :class:`ChecklistItem` carries a heuristic ``position`` — the negotiation stance a buyer's
    counsel would take. These are HEURISTICS ("a common supplier ask you should push back on"),
    deliberately NOT market-benchmark claims ("the standard is X"): we never assert an unsourced
    fact about the market. The wording is a defensible position, not a citation.

The checklist is advisory (it shapes what the LLM looks for and how findings read). The hard floor —
what the AI cannot talk down — lives in :mod:`negotiation_agent.gate`. Checklist informs; gate
decides.
"""

from __future__ import annotations

from pydantic import BaseModel

from .gate import Category


class ChecklistItem(BaseModel):
    """One thing a senior counsel checks for in a given category. ``look_for`` guides extraction;
    ``position`` is the heuristic negotiation stance (never an unsourced market-standard claim)."""

    model_config = {"frozen": True}

    key: str  # stable slug, e.g. "dpa"
    label: str  # short human label, e.g. "Art. 28 data-processing agreement"
    look_for: str  # what the reviewer checks — presence AND absence
    position: str  # the buyer-side stance if it's missing/adverse — a HEURISTIC, not a benchmark


# The checklist per category. Positions are worded as negotiation guidance ("push for", "a common
# supplier ask") — deliberately never "the market standard is N months", which would be an
# unsourced factual claim (banned by the no-fabrication rule).
CHECKLIST: dict[Category, tuple[ChecklistItem, ...]] = {
    "legal": (
        ChecklistItem(
            key="liability_cap",
            label="Liability cap level",
            look_for="Is aggregate liability capped, and at what level relative to annual fees? A "
            "cap set at only a few months' fees is low; no cap at all is worse.",
            position="Push for a cap at 12 months' fees; treat a few-months cap as a supplier "
            "opening position to negotiate up, not a settled term.",
        ),
        ChecklistItem(
            key="liability_carveouts",
            label="Liability carve-outs",
            look_for="Are data-breach, confidentiality, and IP-infringement damages carved OUT of "
            "the cap (uncapped for the buyer's benefit), or swallowed by it?",
            position="Carve data-protection and confidentiality breaches out of the cap, or add a "
            "super-cap for them; a single incident can exceed a low general cap.",
        ),
        ChecklistItem(
            key="indemnity_mutual",
            label="Mutual indemnity",
            look_for="Is the indemnity mutual, or does only the buyer indemnify the supplier? Is "
            "there a supplier IP-infringement indemnity?",
            position="Make the indemnity mutual and subject to the cap; require a supplier "
            "IP-infringement indemnity.",
        ),
        ChecklistItem(
            key="ip_ownership",
            label="IP ownership of deliverables",
            look_for="Who owns IP in deliverables, custom work, and buyer data/derived data? Any "
            "unexpected assignment of buyer IP or broad licence-back?",
            position="Buyer retains ownership of its data and any bespoke deliverables it pays for; "
            "resist broad licences over buyer data.",
        ),
        ChecklistItem(
            key="termination_convenience",
            label="Termination for convenience",
            look_for="Can the buyer exit for convenience, and on what notice? Is exit blocked "
            "during an initial term?",
            position="Secure a termination-for-convenience right on reasonable notice; avoid being "
            "locked in with no exit during a long initial term.",
        ),
        ChecklistItem(
            key="warranty",
            label="Warranty and 'as is' disclaimers",
            look_for="Are services warranted to conform to documentation, or provided 'as is' with "
            "all warranties disclaimed?",
            position="Require a conformance warranty with a remedy; resist a blanket 'as is' "
            "disclaimer for a paid service.",
        ),
        ChecklistItem(
            key="governing_law",
            label="Governing law and venue",
            look_for="Is governing law and dispute venue acceptable, or a jurisdiction inconvenient "
            "/ unfavourable to the buyer?",
            position="Prefer a neutral or buyer-home jurisdiction; flag a venue that raises the "
            "cost of enforcing the contract.",
        ),
    ),
    "gdpr": (
        ChecklistItem(
            key="dpa",
            label="Art. 28 data-processing agreement",
            look_for="Is there a data-processing agreement with the Art. 28(3) mandatory terms "
            "(processing only on instructions, confidentiality, security, deletion/return)?",
            position="Require a signed Art. 28 DPA with the mandatory processor obligations before "
            "any personal data is processed.",
        ),
        ChecklistItem(
            key="subprocessors",
            label="Sub-processor consent/notice",
            look_for="Are sub-processors restricted — prior notice, a right to object, flow-down of "
            "equivalent obligations — or engaged freely?",
            position="Require prior notice of new sub-processors with a right to object and to "
            "terminate, and flow-down of the DPA terms.",
        ),
        ChecklistItem(
            key="transfer_mechanism",
            label="International transfer mechanism",
            look_for="If data leaves the EEA/UK, is a valid transfer mechanism named (adequacy, SCCs "
            "+ TIA, UK IDTA)?",
            position="Name a valid transfer mechanism and location; no transfer to an inadequate "
            "country without SCCs and a transfer risk assessment.",
        ),
        ChecklistItem(
            key="breach_72h",
            label="72-hour breach notification",
            look_for="Does the processor commit to notify the controller of a personal-data breach "
            "without undue delay, in time to meet the 72-hour duty?",
            position="Require breach notification without undue delay and within a fixed short "
            "window, with enough detail to meet the buyer's own 72-hour duty.",
        ),
        ChecklistItem(
            key="retention_deletion",
            label="Retention and deletion",
            look_for="Is personal data deleted or returned on termination, and retention limited to "
            "what's necessary?",
            position="Require deletion or return of personal data on termination and a defined, "
            "limited retention period.",
        ),
    ),
    "infosec": (
        ChecklistItem(
            key="breach_notice_sla",
            label="Security-incident notification SLA",
            look_for="Is there a firm incident-notification deadline, or only 'reasonable efforts' "
            "with no time commitment?",
            position="Require notification of a security incident within a fixed short window (e.g. "
            "24-48h); resist 'reasonable efforts' with no deadline.",
        ),
        ChecklistItem(
            key="audit_right",
            label="Audit / assurance right",
            look_for="Can the buyer audit security, or receive an independent assurance report "
            "(SOC 2 / ISO 27001)? Or is there no assurance at all?",
            position="Secure an audit right or a recognised assurance report (SOC 2 Type II / ISO "
            "27001) refreshed annually.",
        ),
        ChecklistItem(
            key="encryption",
            label="Encryption of data",
            look_for="Is buyer data encrypted in transit and at rest, or is there no encryption "
            "commitment?",
            position="Require encryption in transit and at rest as a baseline security control.",
        ),
        ChecklistItem(
            key="security_standard",
            label="Security standard / measures",
            look_for="Are concrete security measures committed (a named standard, TOMs), or only "
            "'commercially reasonable' measures with no substance?",
            position="Require adherence to a named security standard with defined technical and "
            "organisational measures, not just 'reasonable' language.",
        ),
    ),
    "coc": (
        ChecklistItem(
            key="coc_flowdown",
            label="Code-of-Conduct flow-down",
            look_for="Is the supplier bound by the buyer's Supplier Code of Conduct, and required to "
            "cascade equivalent obligations to sub-suppliers?",
            position="Bind the supplier to the buyer Code of Conduct and require cascade to "
            "sub-suppliers; without it there is no contractual basis for due-diligence expectations.",
        ),
        ChecklistItem(
            key="supply_chain_dd",
            label="Supply-chain due diligence (LkSG/CSDDD)",
            look_for="Does the contract support the buyer's LkSG/CSDDD duties — human-rights and "
            "environmental due diligence, remediation, and reporting?",
            position="Add human-rights/environmental due-diligence obligations, an audit right, and "
            "a remediation duty to support LkSG/CSDDD compliance.",
        ),
        ChecklistItem(
            key="anti_corruption",
            label="Anti-bribery / sanctions compliance",
            look_for="Are anti-bribery, anti-corruption, and sanctions-compliance obligations "
            "present, with a right to terminate on breach?",
            position="Require anti-bribery/anti-corruption and sanctions-compliance warranties with "
            "a termination right on breach.",
        ),
    ),
    "commercial": (
        ChecklistItem(
            key="price_escalation",
            label="Price escalation / indexation cap",
            look_for="Is annual price escalation capped (a fixed % or index), or open-ended (e.g. "
            "CPI plus a margin with no ceiling)?",
            position="Cap annual escalation at a fixed ceiling or index-only; treat uncapped "
            "'index + margin' as a term to negotiate a hard cap onto.",
        ),
        ChecklistItem(
            key="auto_renewal",
            label="Auto-renewal and notice window",
            look_for="Does the contract auto-renew, and is the non-renewal notice window short or "
            "onerous (an evergreen trap)?",
            position="Prefer no auto-renewal, or a short, clearly-flagged non-renewal notice "
            "window; an auto-renew with a long notice period is an evergreen trap.",
        ),
        ChecklistItem(
            key="minimum_commit",
            label="Minimum commitment vs. forecast",
            look_for="Is there a minimum volume/spend commitment above a realistic forecast, billed "
            "regardless of usage?",
            position="Right-size the minimum commit to forecast, or add carry-forward of unused "
            "volume; a commit above real demand is pay-for-nothing.",
        ),
        ChecklistItem(
            key="overage_pricing",
            label="Overage / burst pricing",
            look_for="Is usage above the commitment priced with tiers/discounts, or billed linearly "
            "with no protection on a spike?",
            position="Negotiate tiered overage pricing and a true-up mechanism; flat per-unit "
            "overage exposes the buyer on a demand spike.",
        ),
        ChecklistItem(
            key="benchmarking",
            label="Benchmarking / most-favoured pricing",
            look_for="Is there a right to benchmark pricing against the market, or a most-favoured "
            "-customer assurance, over a multi-year term?",
            position="Add a benchmarking right on a multi-year deal so pricing can be re-tested "
            "against the market mid-term.",
        ),
        ChecklistItem(
            key="exit_transition",
            label="Exit / transition assistance",
            look_for="On termination, is there transition assistance and data-return/portability, "
            "or is the buyer stranded?",
            position="Require exit/transition assistance and data export in a usable format on "
            "termination to avoid lock-in.",
        ),
    ),
}


_CATEGORY_TITLE: dict[Category, str] = {
    "legal": "LEGAL",
    "gdpr": "GDPR / DATA PROTECTION",
    "infosec": "INFORMATION SECURITY",
    "coc": "CODE OF CONDUCT / COMPLIANCE",
    "commercial": "COMMERCIAL",
}


def checklist_prompt_block() -> str:
    """Render the checklist as a prompt section so the extraction LLM hunts the full list per
    category and files a finding when a listed protection is ABSENT (the senior-counsel behaviour).
    Kept compact — labels + what to look for, not the negotiation positions (those are attached to
    findings deterministically after extraction)."""
    lines = ["CHECKLIST — for EACH category, check every item; file a finding for any that is "
             "adverse OR ABSENT (word an absence with 'no'/'missing'/'without'):"]
    for category, items in CHECKLIST.items():
        lines.append(f"\n{_CATEGORY_TITLE[category]}:")
        for item in items:
            lines.append(f"  - {item.label}: {item.look_for}")
    return "\n".join(lines)
