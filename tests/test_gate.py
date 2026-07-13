"""Severity escalation + the legal-review gate — deterministic code overruling the LLM.

The guarantee: the LLM proposes a severity, deterministic rules can only RAISE it, and the gate
that pulls legal in fires on named, human-owned thresholds — carrying the rule that fired, not
just a boolean. These tests prove a model can't talk a missing DPA down to "low", can't be talked
up past a rule floor either (raise-never-lower), and that an unverified finding can't trigger a
review on its own (the anchor layer's trust chains into the gate).
"""

from __future__ import annotations

import pytest

from negotiation_agent import gate


def _finding(category, severity, title, quote="", verified=True):
    return gate.RiskFinding(
        category=category, severity=severity, title=title, quote=quote, verified=verified
    )


def _missing_dpa(verified=True):
    """The recurring 'model called it low, but the DPA is missing' finding."""
    return _finding("gdpr", "low", "Missing DPA", "no data-processing agreement", verified=verified)


# ── escalation raises to the rule floor ──────────────────────────────────────────
@pytest.mark.parametrize(
    ("category", "title", "quote", "expected", "rule_id"),
    [
        ("gdpr", "Missing DPA", "the agreement contains no data-processing agreement", "critical",
         "R-GDPR-NO-DPA"),
        ("gdpr", "Sub-processors unrestricted", "sub-processor consent not required", "high",
         "R-GDPR-SUBPROCESSOR"),
        ("legal", "Uncapped liability", "the Supplier's liability shall be uncapped", "critical",
         "R-LEGAL-UNCAPPED-LIABILITY"),
        ("legal", "Liability cap", "shall not exceed the fees paid in the three (3) months", "high",
         "R-LEGAL-LOW-LIABILITY-CAP"),
        ("infosec", "No breach notice", "no breach notification obligation is stated", "high",
         "R-INFOSEC-NO-BREACH-NOTICE"),
        # ── the senior-counsel expansion (CoC + Commercial + more Legal/GDPR/InfoSec) ──
        ("gdpr", "No transfer mechanism",
         "data is transferred internationally with no standard contractual clauses", "high",
         "R-GDPR-NO-TRANSFER-MECHANISM"),
        ("legal", "IP assigned away",
         "all intellectual property in the deliverables shall be assigned to the Supplier", "high",
         "R-LEGAL-IP-ASSIGNMENT"),
        ("legal", "Locked in",
         "the Customer has no right to terminate for convenience during the initial term", "high",
         "R-LEGAL-NO-EXIT"),
        ("infosec", "No audit",
         "the Customer has no right to audit the Supplier's security controls", "medium",
         "R-INFOSEC-NO-AUDIT-RIGHT"),
        ("coc", "No CoC flow-down",
         "does not require the Supplier to comply with the Customer's Supplier Code of Conduct",
         "medium", "R-COC-NO-FLOWDOWN"),
        ("coc", "No supply-chain DD",
         "there is no human rights due diligence obligation", "high",
         "R-COC-NO-SUPPLY-CHAIN-DD"),
        ("commercial", "Uncapped escalation",
         "the annual fee may increase by CPI plus 3% with no ceiling", "high",
         "R-COMMERCIAL-UNCAPPED-ESCALATION"),
        ("commercial", "Auto-renewal",
         "this Agreement automatically renews for successive one-year terms", "high",
         "R-COMMERCIAL-AUTO-RENEWAL"),
        ("commercial", "Minimum commit",
         "a minimum annual volume of 3,000,000 units regardless of actual usage", "medium",
         "R-COMMERCIAL-MINIMUM-COMMIT"),
    ],
)
def test_rule_raises_low_finding_to_its_floor(category, title, quote, expected, rule_id):
    e = gate.escalate_severity(_finding(category, "low", title, quote))
    assert e.severity == expected and rule_id in e.raised_by


# The other side of the coin: a well-drafted clause must NOT trip a rule (no false-positives —
# a rule that fires on a GOOD clause would cry wolf and erode trust in the gate).
@pytest.mark.parametrize(
    ("category", "title", "quote"),
    [
        ("gdpr", "Transfer covered",
         "cross-border transfers are governed by the standard contractual clauses"),
        ("legal", "Buyer owns IP",
         "the Customer owns all intellectual property in the deliverables"),
        ("legal", "Has exit", "either party may terminate for convenience on 30 days' notice"),
        ("infosec", "Audit allowed", "the Customer may audit the Supplier once per year"),
        ("coc", "CoC bound",
         "the Supplier shall comply with the Customer's Supplier Code of Conduct"),
        ("commercial", "Capped escalation", "annual increases are capped at 3 percent"),
        ("commercial", "No renewal", "this Agreement expires at the end of the term"),
    ],
)
def test_good_clause_does_not_false_fire(category, title, quote):
    e = gate.escalate_severity(_finding(category, "low", title, quote))
    assert e.raised_by == () and e.severity == "low"


# Regressions from the adversarial red-team of the rules (verified real defects, now fixed).
def test_supply_chain_dd_matches_hyphenated_human_rights():
    # a hyphen must not let the clause slip through on punctuation alone
    e = gate.escalate_severity(
        _finding("coc", "low", "No DD", "there is no human-rights due diligence obligation")
    )
    assert e.severity == "high" and "R-COC-NO-SUPPLY-CHAIN-DD" in e.raised_by


def test_minimum_commit_does_not_fire_on_an_sla_uptime_clause():
    # "uptime committed regardless of usage" is an SLA, not a spend commitment — must NOT fire
    sla = "uptime of 99.9% is committed regardless of actual usage"
    e = gate.escalate_severity(_finding("commercial", "low", "SLA", sla))
    assert e.raised_by == ()


def test_minimum_commit_still_fires_on_a_real_take_or_pay():
    e = gate.escalate_severity(
        _finding("commercial", "low", "Commit",
                 "a minimum spend of EUR 500,000 is payable regardless of actual consumption")
    )
    assert "R-COMMERCIAL-MINIMUM-COMMIT" in e.raised_by


def test_escalation_records_the_rule_that_raised_it():
    # audit trail: raised_by names the rule, so the escalation is defensible
    e = gate.escalate_severity(_missing_dpa())
    assert "R-GDPR-NO-DPA" in e.raised_by


# ── raise, NEVER lower ───────────────────────────────────────────────────────────
def test_rule_floor_below_llm_severity_does_not_lower_it():
    # sub-processor rule floors at "high"; a finding the model already called "critical" stays
    e = gate.escalate_severity(_finding("gdpr", "critical", "Sub-processors unrestricted"))
    assert e.severity == "critical"


def test_a_rule_that_did_not_raise_is_not_credited():
    # if the floor was already met, no rule id is added — raised_by reflects real escalations only
    e = gate.escalate_severity(_finding("gdpr", "critical", "Sub-processors unrestricted"))
    assert e.raised_by == ()


def test_finding_in_unrelated_category_is_untouched():
    # a commercial finding has no legal/gdpr rule — severity and raised_by unchanged
    e = gate.escalate_severity(_finding("commercial", "high", "CPI + 3% uncapped indexation"))
    assert e.severity == "high" and e.raised_by == ()


# ── the legal gate ───────────────────────────────────────────────────────────────
def test_critical_in_legal_or_gdpr_forces_review():
    v = gate.legal_gate(gate.escalate_all([_missing_dpa()]))
    assert v.review_required is True


def test_gate_verdict_names_the_rule_that_fired():
    v = gate.legal_gate(gate.escalate_all([_missing_dpa()]))
    assert "Critical" in v.rule and "gdpr" in v.rule


def test_three_highs_force_review_even_without_a_critical():
    findings = [_finding("commercial", "high", f"high risk {i}") for i in range(3)]
    v = gate.legal_gate(findings)
    assert v.review_required is True and "High" in v.rule


def test_two_highs_do_not_force_review():
    findings = [_finding("commercial", "high", f"high risk {i}") for i in range(2)]
    v = gate.legal_gate(findings)
    assert v.review_required is False


def test_unverified_finding_cannot_trigger_review_on_its_own():
    # a Critical finding whose quote the anchor layer could NOT verify must not pull legal in
    v = gate.legal_gate(gate.escalate_all([_missing_dpa(verified=False)]))
    assert v.review_required is False and v.ignored_unverified == 1


def test_policy_owner_is_carried_on_the_verdict():
    policy = gate.GatePolicy(owner="E. Procurement")
    v = gate.legal_gate([], policy)
    assert v.owner == "E. Procurement"


def test_high_count_threshold_is_configurable():
    # a stricter owner sets the bar at 2 Highs
    policy = gate.GatePolicy(high_count_threshold=2)
    findings = [_finding("commercial", "high", f"high risk {i}") for i in range(2)]
    v = gate.legal_gate(findings, policy)
    assert v.review_required is True
