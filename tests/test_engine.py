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


def test_firm_but_acceptable_offer_is_accepted_not_stalled(simple_envelope):
    # A supplier who holds FIRM (repeats) at an offer that clears the decaying threshold must
    # be ACCEPTED, not escalated as a stall (audit issue #6). Before the fix, the stall guard
    # fired first and threw this closable deal to a human.
    eng = _engine(simple_envelope, max_rounds=8, beta=2.0, stall_rounds=3)
    _, state = eng.decide(NegotiationState(), incoming=None)
    firm = Offer(terms={"price": 9.2, "rebate_pct": 7.0})  # u ~ 0.92, held firm every round
    decision = None
    for _ in range(8):
        decision, state = eng.decide(state, firm)
        if decision.outcome is not Outcome.COUNTER:
            break
    assert decision.outcome is Outcome.ACCEPT
    assert decision.reason == "accept_threshold"


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


# ── adaptive "senior negotiator" mode ────────────────────────────────────────────
def _adaptive(env, **cfg):
    return DealEngine(env, SupplierModel.uniform(env), EngineConfig(adaptive=True, **cfg))


def test_adaptive_off_by_default_preserves_the_fixed_engine(simple_envelope):
    # a default engine and an explicitly-non-adaptive one must decide identically
    plain = DealEngine(simple_envelope, SupplierModel.uniform(simple_envelope))
    state = NegotiationState()
    offer = Offer(terms={"price": 10.5, "rebate_pct": 4.0})
    d, _ = plain.decide(state, offer)
    assert d.tactic == "" and d.base_threshold is None  # no strategy annotations when off


def test_adaptive_opening_is_annotated_anchor(simple_envelope):
    eng = _adaptive(simple_envelope)
    d, _ = eng.decide(NegotiationState(), None)
    assert d.tactic == "anchor" and d.base_threshold is not None


def test_adaptive_threshold_never_drops_below_reservation(simple_envelope):
    # feed a stream of supplier offers; every decision's threshold must stay >= the floor
    eng = _adaptive(simple_envelope, max_rounds=8)
    state = NegotiationState()
    _, state = eng.decide(state, None)
    res = simple_envelope.reservation_utility
    # a supplier that moves a lot (max reciprocity) is the case that could push the bar down
    for price in (11.9, 11.0, 10.0, 9.5, 9.1):
        offer = Offer(terms={"price": price, "rebate_pct": 8.0})
        d, state = eng.decide(state, offer)
        assert d.threshold >= res - 1e-9


def test_adaptive_records_supplier_history(simple_envelope):
    eng = _adaptive(simple_envelope)
    state = NegotiationState()
    _, state = eng.decide(state, None)
    o1 = Offer(terms={"price": 11.5, "rebate_pct": 2.0})
    _, state = eng.decide(state, o1)
    assert state.supplier_history == [o1]


def test_adaptive_is_deterministic(simple_envelope):
    eng = _adaptive(simple_envelope)
    state = NegotiationState()
    _, state = eng.decide(state, None)
    offer = Offer(terms={"price": 11.0, "rebate_pct": 4.0})
    d1, s1 = eng.decide(state, offer)
    d2, s2 = eng.decide(state, offer)
    assert d1.model_dump() == d2.model_dump() and s1.model_dump() == s2.model_dump()
