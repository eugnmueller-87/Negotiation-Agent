"""The logrolling proof: appetite-aware packages beat uniform price-splitting.

This is the test the headline claim rests on. At matched buyer utility, the
logrolling package must leave the supplier materially better off than a naive
split-the-difference package. A near-tie would mean the pitch is unproven.
"""

from __future__ import annotations

import pytest

from negotiation_agent.baseline import compare_at_threshold, uniform_split_package
from negotiation_agent.simulator.scenarios import _true_priorities, reference_matrix
from negotiation_agent.supplier_model import SupplierModel


def _oracle(scenario):
    return SupplierModel(appetite=_true_priorities(scenario.supplier_envelope))


def test_split_package_hits_threshold():
    sc = reference_matrix()[0]
    env = sc.buyer_envelope
    for th in [0.55, 0.65, 0.75, 0.85, 0.95]:
        offer = uniform_split_package(env, th)
        assert env.utility(offer) >= th - 1e-9


def test_logrolling_beats_split_at_matched_buyer_cost():
    """Core claim, made falsifiable. At each threshold the two packages deliver
    (near-)identical BUYER utility, but logrolling gives the SUPPLIER a large,
    consistent gain — because it concedes on the terms the supplier values."""
    sc = reference_matrix()[0]
    buyer, supplier = sc.buyer_envelope, sc.supplier_envelope
    oracle = _oracle(sc)

    gains = []
    for th in [0.55, 0.65, 0.75, 0.85, 0.95]:
        c = compare_at_threshold(buyer, supplier, th, oracle)
        # Buyer cost is matched (both clear the threshold, neither overshoots much).
        assert c.buyer_utility_logroll >= th - 1e-9
        assert abs(c.buyer_utility_logroll - c.buyer_utility_split) < 0.02
        # Logrolling never leaves the supplier worse off...
        assert c.supplier_gain >= 0.0
        gains.append(c.supplier_gain)

    # ...and on average the gain is large and unambiguous, not a near-tie.
    mean_gain = sum(gains) / len(gains)
    assert mean_gain > 0.15, f"logrolling edge collapsed to {mean_gain:.3f}"


def test_split_ignores_supplier_belief():
    """The split baseline must be belief-invariant — that's what makes it a
    fair control for 'what if we ignored supplier priorities'."""
    sc = reference_matrix()[0]
    env = sc.buyer_envelope
    a = uniform_split_package(env, 0.75)
    b = uniform_split_package(env, 0.75)
    assert a.terms == b.terms


def test_comparison_is_deterministic():
    sc = reference_matrix()[0]
    oracle = _oracle(sc)
    c1 = compare_at_threshold(sc.buyer_envelope, sc.supplier_envelope, 0.7, oracle)
    c2 = compare_at_threshold(sc.buyer_envelope, sc.supplier_envelope, 0.7, oracle)
    assert c1.model_dump() == c2.model_dump()


@pytest.mark.parametrize("th", [0.55, 0.75, 0.95])
def test_supplier_gain_property(th):
    sc = reference_matrix()[0]
    c = compare_at_threshold(sc.buyer_envelope, sc.supplier_envelope, th, _oracle(sc))
    assert c.supplier_gain == pytest.approx(c.supplier_utility_logroll - c.supplier_utility_split)
