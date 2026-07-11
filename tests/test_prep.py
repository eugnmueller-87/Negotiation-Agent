"""Category detection, tone inference, coverage gaps, and the contract store.

The three preparation layers the agent uses before negotiating: WHAT category the deal is
(so it pulls the right strategy), HOW the counterpart writes (so it mirrors tone), and an
honest flag when the KB lacks a playbook. Plus the swappable contract store.
"""

from __future__ import annotations

import pytest

from negotiation_agent.knowledge.category import CATEGORY_LABELS, detect_category
from negotiation_agent.knowledge.contracts import SampleContractStore
from negotiation_agent.knowledge.tone import detect_register, greeting_for


# ── category detection ──────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Azure reserved vCPU compute, egress bandwidth, multi-region IaaS hosting",
            "cloud_infrastructure",
        ),
        (
            "500 named-user software licences, SaaS subscription, true-up at renewal",
            "software_licenses",
        ),
        (
            "temporary staffing agency, placement fee, contingent worker, Zeitarbeit",
            "hr_staffing_agency",
        ),
        ("outside counsel litigation retainer, billable hour, law firm", "legal_services"),
        (
            "janitorial cleaning, HVAC maintenance service, security guard, facility",
            "facility_services",
        ),
        ("agency of record, media buy, advertising campaign, CPM impressions", "marketing"),
        ("freight forwarding, incoterms DAP, warehousing, last mile carrier", "logistics"),
    ],
)
def test_detect_category(text, expected):
    cat, hits = detect_category(text)
    assert cat == expected
    assert hits  # the matched terms are returned for a legible result


def test_detect_category_unknown_on_weak_signal():
    # a category we have no vocabulary for -> unknown, not a forced wrong label
    cat, hits = detect_category("Corrugated packaging annual supply, price per box")
    assert cat == "unknown"
    assert hits == []


def test_detect_category_hint_nudges_but_contract_wins():
    # a cloud contract body outweighs a misleading 'legal' hint
    cat, _ = detect_category("Azure vCPU compute IaaS hosting egress", hint="legal services")
    assert cat == "cloud_infrastructure"


def test_every_category_has_a_label():
    for cat, _ in [(c, None) for c in CATEGORY_LABELS]:
        assert CATEGORY_LABELS[cat]


# ── tone inference ──────────────────────────────────────────────────────────────
def test_register_defaults_formal_with_no_messages():
    assert detect_register([]) == "formal"


def test_register_informal_on_casual_opener():
    assert detect_register(["Hi Eugen, thanks! Sounds good, let's talk."]) == "informal"


def test_register_stays_formal_on_formal_message():
    assert detect_register(["Dear Sir, please find our proposal. Kind regards."]) == "formal"


def test_register_neutral_message_stays_formal():
    # a plain offer with no register signal defaults to the safe (formal) side
    assert detect_register(["We can offer 92 EUR per unit, net-30, 24 months."]) == "formal"


def test_greeting_matches_register():
    assert greeting_for("formal", contact="Mr. Schmidt", supplier="X") == "Dear Mr. Schmidt,"
    assert greeting_for("informal", contact="Anna", supplier="X") == "Hello Anna,"
    assert greeting_for("formal", contact="", supplier="Nordwerk") == "Dear Nordwerk team,"
    assert greeting_for("informal", contact="", supplier="Nordwerk") == "Hi Nordwerk team,"


# ── coverage gap (uses the shipped index) ────────────────────────────────────────
def test_has_category_playbook_reflects_index():
    from negotiation_agent.knowledge.retrieve import has_category_playbook

    # the shipped KB has cloud + license material; it has no HR-staffing-agency material
    assert has_category_playbook("cloud_infrastructure") is True
    assert has_category_playbook("hr_staffing_agency") is False
    assert has_category_playbook("unknown") is False


# ── the contract store (simulated, swappable) ────────────────────────────────────
def test_sample_store_lists_and_gets():
    store = SampleContractStore()
    contracts = store.list_contracts()
    assert len(contracts) >= 5
    one = contracts[0]
    assert store.get(one.contract_id) == one
    assert store.get("does-not-exist") is None


def test_sample_contracts_detect_to_their_intended_category():
    # each sample contract's text should classify to the category it's labelled with
    store = SampleContractStore()
    for c in store.list_contracts():
        detected, _ = detect_category(c.text)
        assert detected == c.category, f"{c.contract_id}: {detected} != {c.category}"
