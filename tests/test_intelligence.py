"""Contract intelligence — schema + the pure finding→adjustment rule engine.

The rules must fire ONLY on confirmed document facts / real sourced findings, and a
sample-sourced brief must shape nothing (it can't forge a compliance date).
"""

from __future__ import annotations

import datetime as dt

from negotiation_agent.envelope import Direction, Envelope, TermSpec, TermType
from negotiation_agent.intake import ContractExtraction, ExtractedTerm
from negotiation_agent.intelligence import (
    ContractIntelligence,
    ContractLifecycle,
    DocumentGrounded,
    LegalFlags,
    derive_assurance,
    propose_adjustments,
)
from negotiation_agent.research import SupplierBrief, sample_brief
from negotiation_agent.shaper import AddTerm, ShiftTarget, apply_adjustments

TODAY = dt.date(2026, 7, 10)


def _intel(**kw):
    extraction = kw.pop("extraction", ContractExtraction(supplier_name="Acme GmbH"))
    return ContractIntelligence(extraction=extraction, **kw)


def _hades_brief(**kw):
    base = dict(
        company="Acme GmbH",
        source="hades",
        risk_level="Medium",
        recommendation="Conditional Approval",
        sanctioned=False,
        registry_status="active",
        lksg_signal="no_findings",
    )
    base.update(kw)
    return SupplierBrief(**base)


def test_derive_assurance_thresholds():
    assert derive_assurance(0.9, True) == "confirmed"
    assert derive_assurance(0.9, False) == "probable"  # high conf but quote not verified
    assert derive_assurance(0.7, True) == "probable"
    assert derive_assurance(0.4, True) == "unknown"


def test_sample_brief_shapes_nothing():
    # a sample brief is display-only — it can't forge a compliance date, so no rule fires
    adjustments, blocked, _ = propose_adjustments(_intel(), sample_brief("Acme GmbH"), today=TODAY)
    assert blocked is False
    assert adjustments == []


def test_sanctions_hit_blocks():
    brief = _hades_brief(sanctioned=True, recommendation="Block")
    adjustments, blocked, reason = propose_adjustments(_intel(), brief, today=TODAY)
    assert blocked is True
    assert adjustments == []  # a block emits no shaping
    assert "clear" in reason.lower()


def test_dissolved_registry_blocks():
    brief = _hades_brief(registry_status="dissolved 2024")
    _, blocked, reason = propose_adjustments(_intel(), brief, today=TODAY)
    assert blocked is True
    assert "registry" in reason.lower()


def test_lksg_redflag_anchors_and_gates():
    brief = _hades_brief(lksg_signal="red_flag")
    adjustments, blocked, _ = propose_adjustments(_intel(), brief, today=TODAY)
    assert blocked is False
    rule_ids = {a.rule_id for a in adjustments}
    assert "R-LKSG-REDFLAG" in rule_ids  # raises target
    assert "R-LKSG-REDFLAG-GATE" in rule_ids  # adds a required gate
    shift = next(a for a in adjustments if a.rule_id == "R-LKSG-REDFLAG")
    assert isinstance(shift.delta, ShiftTarget) and shift.delta.target_delta > 0


def test_dpa_missing_fires_only_on_confirmed_false():
    legal_confirmed = LegalFlags(has_dpa=DocumentGrounded(value="false", assurance="confirmed"))
    adj, _, _ = propose_adjustments(_intel(legal=legal_confirmed), None, today=TODAY)
    assert any(a.rule_id == "R-DPA-MISSING" for a in adj)

    # unknown assurance must NOT trigger — a false "no DPA" would wrongly add a gate
    legal_unknown = LegalFlags(has_dpa=DocumentGrounded(value="false", assurance="unknown"))
    adj2, _, _ = propose_adjustments(_intel(legal=legal_unknown), None, today=TODAY)
    assert not any(a.rule_id == "R-DPA-MISSING" for a in adj2)

    # None (could-not-determine) must NOT trigger
    adj3, _, _ = propose_adjustments(_intel(legal=LegalFlags()), None, today=TODAY)
    assert not any(a.rule_id == "R-DPA-MISSING" for a in adj3)


def test_expiring_soon_lowers_floor():
    exp = DocumentGrounded(value="2026-07-20", assurance="confirmed")  # 10 days out
    life = ContractLifecycle(expiration_date=exp)
    adj, _, _ = propose_adjustments(_intel(lifecycle=life), None, today=TODAY)
    soon = next(a for a in adj if a.rule_id == "R-EXPIRING-SOON")
    assert isinstance(soon.delta, ShiftTarget)
    assert soon.delta.reservation_delta < 0  # lowers the floor


def test_no_rebate_adds_give_term():
    extraction = ContractExtraction(
        supplier_name="Acme", terms=[ExtractedTerm(name="volume_units", value=40000)]
    )
    adj, _, _ = propose_adjustments(_intel(extraction=extraction), None, today=TODAY)
    rebate = next(a for a in adj if a.rule_id == "R-NO-REBATE")
    assert isinstance(rebate.delta, AddTerm)
    assert rebate.delta.spec.name == "rebate_pct"
    assert rebate.role == "give"


def test_sourced_finding_requires_a_date():
    # the compliance rule as a type: no retrieved_at -> construction fails
    import pytest
    from pydantic import ValidationError

    from negotiation_agent.intelligence import SourcedFinding

    with pytest.raises(ValidationError):
        SourcedFinding(claim="sanctioned", source_ref="OFAC", provider="hades")  # no retrieved_at


def test_rules_flow_through_the_shaper_to_a_valid_envelope():
    # end-to-end: findings -> adjustments -> apply_adjustments -> valid Envelope
    base = Envelope(
        negotiation_id="e",
        version=1,
        signed_by="t",
        target_utility=0.90,
        reservation_utility=0.60,
        terms=[
            TermSpec(
                name="price",
                term_type=TermType.PRICE,
                direction=Direction.MINIMIZE,
                best=92.0,
                worst=108.0,
                weight=0.5,
            ),
            TermSpec(
                name="payment_days",
                term_type=TermType.PAYMENT_DAYS,
                direction=Direction.MAXIMIZE,
                best=60.0,
                worst=30.0,
                weight=0.25,
            ),
            TermSpec(
                name="volume_units",
                term_type=TermType.VOLUME_UNITS,
                direction=Direction.MINIMIZE,
                best=10000.0,
                worst=50000.0,
                weight=0.25,
            ),
        ],
    )
    extraction = ContractExtraction(
        supplier_name="Acme", terms=[ExtractedTerm(name="volume_units", value=40000)]
    )
    exp = DocumentGrounded(value="2026-07-20", assurance="confirmed")
    intel = _intel(extraction=extraction, lifecycle=ContractLifecycle(expiration_date=exp))
    adj, blocked, _ = propose_adjustments(intel, _hades_brief(), today=TODAY)
    assert not blocked
    # apply only the envelope-affecting deltas (gates don't touch the envelope)
    shaped, appetite = apply_adjustments(base, adj)
    assert abs(sum(t.weight for t in shaped.terms) - 1.0) < 1e-6
    assert shaped.reservation_utility < shaped.target_utility
