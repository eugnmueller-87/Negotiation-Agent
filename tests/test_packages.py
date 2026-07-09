"""Logrolling package search — the core IP. These tests are the proof.

Key properties under test:
  * realized utility always clears the threshold (the guarantee integer snap
    must not break);
  * concessions route to high-supplier-appetite / low-buyer-weight terms
    (logrolling), not split evenly;
  * concessions nest monotonically as the threshold decays;
  * infeasible caps raise rather than silently under-deliver.
"""

from __future__ import annotations

import pytest

from negotiation_agent.packages import InfeasiblePackage, fill_package
from negotiation_agent.supplier_model import SupplierModel


def _priorities(env, **appetite):
    return SupplierModel(appetite=appetite).priorities(env)


def test_package_clears_threshold_continuous(simple_envelope):
    prio = _priorities(simple_envelope, price=1.0, rebate_pct=1.0)
    offer, realized, _ = fill_package(simple_envelope, 0.75, prio)
    assert realized == pytest.approx(0.75)  # continuous term absorbs exactly
    assert simple_envelope.utility(offer) >= 0.75 - 1e-9


def test_logrolling_concedes_on_high_appetite_term(simple_envelope):
    # Buyer weights price heavily (0.7). Supplier appetite entirely on rebate.
    # The engine should HOLD price near best and give ground on rebate.
    prio = _priorities(simple_envelope, price=0.01, rebate_pct=1.0)
    offer, _, _ = fill_package(simple_envelope, 0.70, prio)
    price_v = simple_envelope.term_map["price"].value(offer.terms["price"])
    rebate_v = simple_envelope.term_map["rebate_pct"].value(offer.terms["rebate_pct"])
    # Price held high, rebate conceded low.
    assert price_v > 0.9
    assert rebate_v < 0.2


def test_logrolling_flips_with_appetite(simple_envelope):
    # Flip the appetite: now supplier cares about price -> engine holds rebate.
    prio = _priorities(simple_envelope, price=1.0, rebate_pct=0.01)
    offer, _, _ = fill_package(simple_envelope, 0.70, prio)
    price_v = simple_envelope.term_map["price"].value(offer.terms["price"])
    rebate_v = simple_envelope.term_map["rebate_pct"].value(offer.terms["rebate_pct"])
    assert rebate_v > 0.9
    assert price_v < 0.7  # price is heavy so some is held, but it's the concede target


def test_integer_snap_never_drops_below_threshold(mixed_envelope):
    # The critical guarantee: rounding integer terms toward `best` keeps U >= theta.
    prio = _priorities(mixed_envelope, price=1.0, payment_days=1.0, volume_units=1.0)
    for theta in [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        offer, realized, _ = fill_package(mixed_envelope, theta, prio)
        assert realized >= theta - 1e-9, f"theta={theta} realized={realized}"
        # integer terms are whole numbers
        assert offer.terms["payment_days"] == round(offer.terms["payment_days"])
        assert offer.terms["volume_units"] == round(offer.terms["volume_units"])


def test_concessions_nest_monotonically(simple_envelope):
    # As threshold decays, no term's offered value increases (concessions don't retract).
    prio = _priorities(simple_envelope, price=0.2, rebate_pct=1.0)
    prev = None
    for theta in [0.95, 0.85, 0.75, 0.65, 0.55]:
        _, _, v = fill_package(simple_envelope, theta, prio)
        if prev is not None:
            for name in v:
                assert v[name] <= prev[name] + 1e-9, f"{name} increased at theta={theta}"
        prev = v


def test_caps_enforce_monotonicity(simple_envelope):
    prio = _priorities(simple_envelope, price=1.0, rebate_pct=1.0)
    _, _, v1 = fill_package(simple_envelope, 0.9, prio)
    # Cap at the round-1 values; a lower threshold must stay within caps.
    _, _, v2 = fill_package(simple_envelope, 0.7, prio, caps=v1)
    for name in v2:
        assert v2[name] <= v1[name] + 1e-9


def test_infeasible_caps_raise(simple_envelope):
    prio = _priorities(simple_envelope, price=1.0, rebate_pct=1.0)
    tight = {"price": 0.1, "rebate_pct": 0.1}  # max U = 0.7*0.1 + 0.3*0.1 = 0.1
    with pytest.raises(InfeasiblePackage):
        fill_package(simple_envelope, 0.8, prio, caps=tight)


def test_single_term_envelope_degenerate():
    from negotiation_agent.envelope import Direction, Envelope, TermSpec, TermType

    env = Envelope(
        negotiation_id="x",
        version=1,
        signed_by="t",
        terms=[
            TermSpec(
                name="price",
                term_type=TermType.PRICE,
                direction=Direction.MINIMIZE,
                best=9.0,
                worst=12.0,
                weight=1.0,
            )
        ],
    )
    prio = SupplierModel(appetite={"price": 1.0}).priorities(env)
    offer, realized, _ = fill_package(env, 0.6, prio)
    assert realized == pytest.approx(0.6)
