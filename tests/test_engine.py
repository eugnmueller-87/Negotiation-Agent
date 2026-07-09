"""Deal engine — Boulware schedule, accept rule, and the decide() state machine."""

from __future__ import annotations

import pytest

from negotiation_agent.engine import (
    DealEngine,
    EngineConfig,
    NegotiationState,
    Outcome,
)
from negotiation_agent.envelope import Offer
from negotiation_agent.supplier_model import SupplierModel


def _engine(env, **cfg):
    return DealEngine(env, SupplierModel.uniform(env), EngineConfig(**cfg))


def test_boulware_endpoints(simple_envelope):
    eng = _engine(simple_envelope, max_rounds=8, beta=4.0)
    assert eng.threshold(0) == pytest.approx(simple_envelope.target_utility)
    assert eng.threshold(8) == pytest.approx(simple_envelope.reservation_utility)


def test_boulware_concedes_late(simple_envelope):
    # beta>1 => small concession at half-time, most of it near the deadline.
    eng = _engine(simple_envelope, max_rounds=8, beta=4.0)
    span = simple_envelope.target_utility - simple_envelope.reservation_utility
    conceded_at_half = simple_envelope.target_utility - eng.threshold(4)
    assert conceded_at_half < 0.1 * span  # < 10% conceded at half-time


def test_boulware_monotone_decreasing(simple_envelope):
    eng = _engine(simple_envelope)
    vals = [eng.threshold(t) for t in range(9)]
    assert all(a >= b for a, b in zip(vals, vals[1:], strict=False))


def test_opening_anchor_is_counter_at_target(simple_envelope):
    eng = _engine(simple_envelope)
    decision, state = eng.decide(NegotiationState(), incoming=None)
    assert decision.outcome is Outcome.COUNTER
    assert decision.counter_utility >= simple_envelope.target_utility - 1e-9
    assert state.round_index == 1
    # approved_numbers carries exactly the offered term values (numeric guard input)
    assert set(decision.approved_numbers) == set(simple_envelope.term_map)


def test_accept_when_incoming_clears_threshold(simple_envelope):
    eng = _engine(simple_envelope)
    _, state = eng.decide(NegotiationState(), incoming=None)
    # A generous supplier offer above the round threshold -> ACCEPT.
    good = Offer(terms={"price": 9.0, "rebate_pct": 8.0})  # U = 1.0
    decision, _ = eng.decide(state, good)
    assert decision.outcome is Outcome.ACCEPT
    assert decision.approved_numbers == good.terms


def test_counter_when_below_threshold(simple_envelope):
    eng = _engine(simple_envelope)
    _, state = eng.decide(NegotiationState(), incoming=None)
    weak = Offer(terms={"price": 12.0, "rebate_pct": 0.0})  # U = 0.0
    decision, _ = eng.decide(state, weak)
    assert decision.outcome is Outcome.COUNTER


def test_unknown_terms_escalate(simple_envelope):
    eng = _engine(simple_envelope, on_unknown_terms="escalate")
    _, state = eng.decide(NegotiationState(), incoming=None)
    weird = Offer(terms={"price": 10.0, "rebate_pct": 4.0, "surprise_clause": 1.0})
    decision, _ = eng.decide(state, weird)
    assert decision.outcome is Outcome.ESCALATE
    assert "unmodeled_terms" in decision.reason


def test_partial_offer_merges_from_standing_counter(simple_envelope):
    eng = _engine(simple_envelope)
    d0, state = eng.decide(NegotiationState(), incoming=None)
    # Supplier addresses only price; rebate should inherit the standing counter.
    partial = Offer(terms={"price": 9.0})
    decision, _ = eng.decide(state, partial)
    # Must not raise KeyError; it scores a full merged package.
    assert decision.outcome in (Outcome.ACCEPT, Outcome.COUNTER)


def test_stall_escalates(simple_envelope):
    eng = _engine(simple_envelope, stall_rounds=3)
    _, state = eng.decide(NegotiationState(), incoming=None)
    weak = Offer(terms={"price": 12.0, "rebate_pct": 0.0})
    # Feed the identical weak offer repeatedly -> stall guard fires.
    outcome = None
    for _ in range(6):
        decision, state = eng.decide(state, weak)
        outcome = decision.outcome
        if outcome is Outcome.ESCALATE:
            break
    assert outcome is Outcome.ESCALATE
    assert decision.reason == "supplier_stalled"


def test_deadline_escalates_without_deal(simple_envelope):
    eng = _engine(simple_envelope, max_rounds=3, stall_rounds=99)
    state = NegotiationState()
    decision, state = eng.decide(state, incoming=None)
    # Alternate distinct-but-weak offers so stall never trips; ride past deadline.
    prices = [11.9, 11.8, 11.7, 11.6, 11.5]
    for p in prices:
        weak = Offer(terms={"price": p, "rebate_pct": 0.0})
        decision, state = eng.decide(state, weak)
        if decision.outcome is Outcome.ESCALATE:
            break
    assert decision.outcome is Outcome.ESCALATE
    assert decision.reason.startswith("deadline_no_deal")


def test_decide_is_pure(simple_envelope):
    eng = _engine(simple_envelope)
    _, state = eng.decide(NegotiationState(), incoming=None)
    offer = Offer(terms={"price": 10.5, "rebate_pct": 4.0})
    d1, s1 = eng.decide(state, offer)
    d2, s2 = eng.decide(state, offer)
    assert d1.model_dump() == d2.model_dump()
    assert s1.model_dump() == s2.model_dump()
