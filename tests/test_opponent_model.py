"""Opponent modelling — inferring supplier priorities from observed moves.

The negotiator's edge: a supplier that DEFENDS a term values it (high appetite → the buyer should
trade for movement there, not spend concessions on it), while a term the supplier CONCEDES freely is
low-appetite (spend concessions there, take the free ground). These tests pin that inference and the
guards that keep it honest — a feint backward can't lower inferred appetite, a single offer yields
no information, and the belief is a pure function of the observed offers.
"""

from __future__ import annotations

import pytest

from negotiation_agent import opponent_model as om
from negotiation_agent.envelope import Direction, Envelope, Offer, TermSpec, TermType


@pytest.fixture
def env() -> Envelope:
    return Envelope(
        negotiation_id="t", version=1, signed_by="e",
        target_utility=0.9, reservation_utility=0.5,
        terms=[
            TermSpec(name="price", term_type=TermType.PRICE, direction=Direction.MINIMIZE,
                     best=90, worst=110, weight=0.6),
            TermSpec(name="payment_days", term_type=TermType.PAYMENT_DAYS,
                     direction=Direction.MAXIMIZE, best=90, worst=30, weight=0.4),
        ],
    )


def _offers(pairs):
    return [Offer(terms={"price": p, "payment_days": d}) for p, d in pairs]


def test_single_offer_gives_no_information():
    # can't infer priorities from one observation — belief stays uniform
    env = Envelope(negotiation_id="t", version=1, signed_by="e", target_utility=0.9,
                   reservation_utility=0.5,
                   terms=[TermSpec(name="price", term_type=TermType.PRICE,
                                   direction=Direction.MINIMIZE, best=90, worst=110, weight=1.0)])
    belief = om.infer_appetite(env, _offers([(110, 30)]))
    assert belief.appetite == {"price": 1.0}  # uniform.appetite is 1.0 per term


def test_defended_term_gets_high_appetite(env):
    # supplier holds price (110→108, ~10%) but concedes payment fully (30→90)
    belief = om.infer_appetite(env, _offers([(110, 30), (109, 60), (108, 90)]))
    assert belief.appetite["price"] > belief.appetite["payment_days"]


def test_fully_conceded_term_gets_near_zero_appetite(env):
    # payment conceded across its whole range → appetite ≈ 0 (they don't value it)
    belief = om.infer_appetite(env, _offers([(110, 30), (110, 90)]))
    assert belief.appetite["payment_days"] == pytest.approx(0.0, abs=0.01)


def test_stubborn_supplier_keeps_all_appetite_high(env):
    # concedes nothing → every term reads as defended
    belief = om.infer_appetite(env, _offers([(110, 30), (110, 30)]))
    assert all(v == pytest.approx(1.0, abs=0.01) for v in belief.appetite.values())


def test_walking_an_offer_backward_cannot_lower_inferred_appetite(env):
    # a supplier feinting AWAY from the buyer (price up) must not register as a concession
    forward = om.infer_appetite(env, _offers([(110, 30), (108, 30)]))
    backward = om.infer_appetite(env, _offers([(110, 30), (112, 30)]))  # walked price the wrong way
    # backward movement is clamped to 0 → price appetite stays at the defended (high) end
    assert backward.appetite["price"] >= forward.appetite["price"]


def test_priorities_route_concessions_to_the_low_appetite_term(env):
    # the engine spends concessions where the supplier gives ground: payment here
    belief = om.infer_appetite(env, _offers([(110, 30), (108, 90)]))
    pri = belief.priorities(env)
    assert pri["payment_days"] < pri["price"]


def test_held_firm_true_when_supplier_defends(env):
    assert om.held_firm(env, _offers([(110, 30), (110, 30)]), term="price") is True


def test_held_firm_false_when_supplier_concedes(env):
    assert om.held_firm(env, _offers([(110, 30), (110, 90)]), term="payment_days") is False


def test_reciprocity_zero_for_a_staller(env):
    assert om.reciprocity(env, _offers([(110, 30), (110, 30)])) == pytest.approx(0.0)


def test_reciprocity_positive_when_supplier_moves(env):
    assert om.reciprocity(env, _offers([(110, 30), (100, 60)])) > 0.0


def test_inference_is_pure(env):
    offers = _offers([(110, 30), (108, 60), (106, 90)])
    assert om.infer_appetite(env, offers).appetite == om.infer_appetite(env, offers).appetite
