"""The round-index fold invariant — the load-bearing rule the v2 server rests on.

The v2 backend is stateless: it re-derives the engine state by folding
``DealEngine.decide`` over the transcript every request (see
``docs/peitho-v2-architecture.md`` §3.2). That only stays correct if the
round-index semantics are pinned exactly, because a live LLM/UI never sees the
engine's internal ``round_index`` — the server must reconstruct it.

The rule (engine.py:119-124, 250-252):
  - Round 0 is the BUYER'S opening anchor: ``decide(state, None)`` runs ``_counter``,
    emits a decision with ``round_index == 0``, and advances ``next_state`` to 1.
  - The supplier's FIRST offer is therefore scored at ``threshold(1)``, not
    ``threshold(0)``.
  - ACCEPT and ESCALATE do NOT advance ``round_index``.

The v1 design prose ("store the opening offer as turn 0, score at round 0") was
off by one — it double-counted the anchor. These tests exist so that bug can
never reach the server.
"""

from __future__ import annotations

import pytest

from negotiation_agent.engine import DealEngine, EngineConfig, NegotiationState, Outcome
from negotiation_agent.envelope import Offer
from negotiation_agent.supplier_model import SupplierModel


def _engine(env, **cfg):
    return DealEngine(env, SupplierModel.uniform(env), EngineConfig(**cfg))


def _fold(engine, supplier_offers):
    """Replay the negotiation exactly as the stateless server will.

    Returns the list of (decision, threshold_scored_at) for the anchor followed
    by each supplier offer — the sequence the server reconstructs per request.
    """
    state = NegotiationState()
    steps = []
    decision, state = engine.decide(state, None)  # round-0 buyer anchor
    steps.append((decision, decision.threshold))
    for offer in supplier_offers:
        decision, state = engine.decide(state, offer)
        steps.append((decision, decision.threshold))
    return steps, state


def test_anchor_is_round_zero_first_supplier_offer_is_round_one(simple_envelope):
    eng = _engine(simple_envelope, max_rounds=8, beta=4.0)
    # a weak supplier offer that will be countered (not accepted), so the fold advances
    weak = Offer(terms={"price": 12.0, "rebate_pct": 0.0})  # U = 0.0
    steps, _ = _fold(eng, [weak])

    anchor_decision, anchor_theta = steps[0]
    first_decision, first_theta = steps[1]

    # the anchor is the buyer's move at round 0, scored at threshold(0) == target
    assert anchor_decision.outcome is Outcome.COUNTER
    assert anchor_decision.round_index == 0
    assert anchor_theta == pytest.approx(eng.threshold(0))
    assert anchor_theta == pytest.approx(simple_envelope.target_utility)

    # the FIRST supplier offer is scored at threshold(1) — the off-by-one guard
    assert first_decision.round_index == 1
    assert first_theta == pytest.approx(eng.threshold(1))
    assert first_theta != pytest.approx(eng.threshold(0))


def test_fold_scores_each_supplier_offer_at_the_expected_threshold(simple_envelope):
    eng = _engine(simple_envelope, max_rounds=8, beta=4.0)
    # five weak offers, each distinct so the stall guard never fires, all countered
    offers = [Offer(terms={"price": 12.0, "rebate_pct": float(i) * 0.001}) for i in range(5)]
    steps, _ = _fold(eng, offers)

    # steps[0] is the anchor (round 0); steps[k] scores the k-th supplier offer at round k
    for k in range(1, len(steps)):
        decision, theta = steps[k]
        assert decision.round_index == k, f"supplier offer {k} scored at wrong round"
        assert theta == pytest.approx(eng.threshold(k)), f"offer {k} scored at wrong threshold"


def test_accept_does_not_advance_round_index(simple_envelope):
    eng = _engine(simple_envelope, max_rounds=8, beta=4.0)
    _, state = eng.decide(NegotiationState(), None)  # anchor -> state.round_index == 1
    assert state.round_index == 1
    good = Offer(terms={"price": 9.0, "rebate_pct": 8.0})  # U = 1.0 -> ACCEPT at threshold(1)
    decision, next_state = eng.decide(state, good)
    assert decision.outcome is Outcome.ACCEPT
    assert decision.round_index == 1  # scored at round 1
    assert next_state.round_index == 1  # ACCEPT does not advance


def test_escalate_does_not_advance_round_index(simple_envelope):
    # an unmodeled term escalates immediately, leaving the round index untouched
    eng = _engine(simple_envelope, max_rounds=8, beta=4.0, on_unknown_terms="escalate")
    _, state = eng.decide(NegotiationState(), None)  # round_index -> 1
    bad = Offer(terms={"price": 10.0, "rebate_pct": 4.0, "warranty_years": 5.0})
    decision, next_state = eng.decide(state, bad)
    assert decision.outcome is Outcome.ESCALATE
    assert next_state.round_index == 1  # ESCALATE does not advance


def test_recalibrated_default_schedule_matches_design(mixed_envelope):
    """Pin the v2 default Boulware schedule (docs §5.1) so a config drift is caught.

    Design defaults: max_rounds=6, beta=2.5, target=0.90, reservation=0.60.
    Verified schedule: r0=.9000 r1=.8966 r2=.8808 r3=.8470 r4=.7911 r5=.7098 r6=.6000
    (mixed_envelope has target 0.95/res 0.55, so we assert the SHAPE with the design
    params against a purpose-built envelope below rather than the fixture's utilities.)
    """
    from negotiation_agent.envelope import Direction, Envelope, TermSpec, TermType

    env = Envelope(
        negotiation_id="v2-default",
        version=1,
        signed_by="design",
        target_utility=0.90,
        reservation_utility=0.60,
        terms=[
            TermSpec(name="price", term_type=TermType.PRICE, direction=Direction.MINIMIZE,
                     best=92.0, worst=108.0, weight=0.50),
            TermSpec(name="payment_days", term_type=TermType.PAYMENT_DAYS,
                     direction=Direction.MAXIMIZE, best=60.0, worst=30.0, weight=0.25),
            TermSpec(name="contract_months", term_type=TermType.CONTRACT_MONTHS,
                     direction=Direction.MINIMIZE, best=12.0, worst=24.0, weight=0.25),
        ],
    )
    eng = _engine(env, max_rounds=6, beta=2.5)
    expected = [0.9000, 0.8966, 0.8808, 0.8470, 0.7911, 0.7098, 0.6000]
    got = [eng.threshold(t) for t in range(7)]
    for t, (g, e) in enumerate(zip(got, expected, strict=True)):
        assert g == pytest.approx(e, abs=5e-4), f"threshold({t}) drifted: {g:.4f} vs {e:.4f}"
