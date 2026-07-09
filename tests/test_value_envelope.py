"""Value functions and envelope utility — the scoring foundation."""

from __future__ import annotations

import pytest

from negotiation_agent.envelope import Direction, Envelope, Offer, TermSpec, TermType
from negotiation_agent.value import linear_inverse, linear_value


def test_linear_value_endpoints_and_clamp():
    assert linear_value(9.0, best=9.0, worst=12.0) == 1.0  # best -> 1
    assert linear_value(12.0, best=9.0, worst=12.0) == 0.0  # worst -> 0
    assert linear_value(10.5, best=9.0, worst=12.0) == pytest.approx(0.5)
    # Beyond the endpoints clamps, never goes out of [0,1].
    assert linear_value(8.0, best=9.0, worst=12.0) == 1.0
    assert linear_value(13.0, best=9.0, worst=12.0) == 0.0


def test_linear_value_maximize_direction():
    assert linear_value(90, best=90, worst=30) == 1.0
    assert linear_value(30, best=90, worst=30) == 0.0
    assert linear_value(60, best=90, worst=30) == pytest.approx(0.5)


def test_linear_inverse_roundtrip():
    for v in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = linear_inverse(v, best=9.0, worst=12.0)
        assert linear_value(x, best=9.0, worst=12.0) == pytest.approx(v)


def test_value_function_rejects_degenerate():
    with pytest.raises(ValueError):
        linear_value(1.0, best=5.0, worst=5.0)


def test_termspec_direction_validation():
    with pytest.raises(ValueError):
        # MINIMIZE requires best < worst.
        TermSpec(name="p", term_type=TermType.PRICE, direction=Direction.MINIMIZE,
                 best=12.0, worst=9.0, weight=1.0)


def test_envelope_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        Envelope(
            negotiation_id="x", version=1, signed_by="t",
            terms=[
                TermSpec(name="a", term_type=TermType.PRICE, direction=Direction.MINIMIZE,
                         best=9.0, worst=12.0, weight=0.5),
                TermSpec(name="b", term_type=TermType.REBATE_PCT, direction=Direction.MAXIMIZE,
                         best=8.0, worst=0.0, weight=0.4),
            ],
        )


def test_envelope_reservation_below_target():
    with pytest.raises(ValueError):
        Envelope(
            negotiation_id="x", version=1, signed_by="t",
            target_utility=0.5, reservation_utility=0.6,
            terms=[TermSpec(name="a", term_type=TermType.PRICE, direction=Direction.MINIMIZE,
                            best=9.0, worst=12.0, weight=1.0)],
        )


def test_utility_ideal_and_reservation(simple_envelope):
    assert simple_envelope.utility(simple_envelope.ideal_offer()) == pytest.approx(1.0)
    assert simple_envelope.utility(simple_envelope.reservation_offer()) == pytest.approx(0.0)


def test_utility_weighted_sum(simple_envelope):
    # price at midpoint (v=0.5, w=0.7), rebate at best (v=1.0, w=0.3) -> 0.35 + 0.3
    offer = Offer(terms={"price": 10.5, "rebate_pct": 8.0})
    assert simple_envelope.utility(offer) == pytest.approx(0.65)


def test_utility_missing_term_raises(simple_envelope):
    with pytest.raises(KeyError):
        simple_envelope.utility(Offer(terms={"price": 10.0}))


def test_supplier_over_concession_clamps(simple_envelope):
    # Supplier offers a price better than the buyer's `best` — must clamp to v=1,
    # never dominate the package with >1 contribution.
    offer = Offer(terms={"price": 5.0, "rebate_pct": 0.0})
    u = simple_envelope.utility(offer)
    assert u == pytest.approx(0.7)  # price v=1 *0.7 + rebate v=0 *0.3
