"""Shared fixtures."""

from __future__ import annotations

import pytest

from negotiation_agent.envelope import Direction, Envelope, TermSpec, TermType


@pytest.fixture
def simple_envelope() -> Envelope:
    """Two continuous terms (price heavy, rebate light) — easy to reason about."""
    return Envelope(
        negotiation_id="test",
        version=1,
        signed_by="tester",
        target_utility=0.95,
        reservation_utility=0.55,
        terms=[
            TermSpec(name="price", term_type=TermType.PRICE, direction=Direction.MINIMIZE,
                     best=9.0, worst=12.0, weight=0.7),
            TermSpec(name="rebate_pct", term_type=TermType.REBATE_PCT,
                     direction=Direction.MAXIMIZE, best=8.0, worst=0.0, weight=0.3),
        ],
    )


@pytest.fixture
def mixed_envelope() -> Envelope:
    """Continuous + integer terms — exercises the integer snap."""
    return Envelope(
        negotiation_id="test",
        version=1,
        signed_by="tester",
        target_utility=0.95,
        reservation_utility=0.55,
        terms=[
            TermSpec(name="price", term_type=TermType.PRICE, direction=Direction.MINIMIZE,
                     best=9.0, worst=12.0, weight=0.4),
            TermSpec(name="payment_days", term_type=TermType.PAYMENT_DAYS,
                     direction=Direction.MAXIMIZE, best=90, worst=30, weight=0.3),
            TermSpec(name="volume_units", term_type=TermType.VOLUME_UNITS,
                     direction=Direction.MINIMIZE, best=10000, worst=50000, weight=0.3),
        ],
    )
