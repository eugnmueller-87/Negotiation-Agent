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
    ],
)
def test_rule_raises_low_finding_to_its_floor(category, title, quote, expected, rule_id):
    e = gate.escalate_severity(_finding(category, "low", title, quote))
    assert e.severity == expected


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
