"""Negotiation strategy — adaptive concession + tactic selection.

The one invariant that MUST hold no matter what: the adapted acceptance threshold can never leave
the mandate band ``[reservation, target]`` — a supplier cannot, by any pattern of moves, coax the
engine into accepting below its walk-away floor. These tests prove that with a full sweep, plus the
nudge direction (engage → ease, stall → firm) and the tactic labels.
"""

from __future__ import annotations

import pytest

from negotiation_agent import strategy as st

RES, TGT = 0.5, 0.9


# ── the hard invariant: never past the floor (or above target) ───────────────────
@pytest.mark.parametrize("reciprocity", [i / 50 for i in range(0, 51)])
@pytest.mark.parametrize("base", [0.5, 0.6, 0.7, 0.8, 0.9])
def test_adapted_threshold_never_leaves_the_mandate_band(reciprocity, base):
    adj = st.adapt_threshold(base, reciprocity=reciprocity, reservation=RES, target=TGT)
    assert RES - 1e-12 <= adj <= TGT + 1e-12


def test_base_at_reservation_with_full_engagement_stays_at_reservation():
    # the extreme case: even a fully-engaging supplier can't push the threshold below the floor
    adj = st.adapt_threshold(RES, reciprocity=1.0, reservation=RES, target=TGT)
    assert adj == pytest.approx(RES)


# ── the nudge direction ──────────────────────────────────────────────────────────
def test_engaging_supplier_eases_the_threshold_down():
    assert st.adapt_threshold(0.8, reciprocity=1.0, reservation=RES, target=TGT) < 0.8


def test_stalling_supplier_firms_the_threshold_up():
    assert st.adapt_threshold(0.7, reciprocity=0.0, reservation=RES, target=TGT) > 0.7


def test_neutral_reciprocity_leaves_the_threshold_unchanged():
    adj = st.adapt_threshold(0.7, reciprocity=0.1, reservation=RES, target=TGT)
    assert adj == pytest.approx(0.7)


def test_adaptation_is_monotone_in_reciprocity():
    # more engagement → threshold eased at least as much (never firmer)
    lo = st.adapt_threshold(0.8, reciprocity=0.3, reservation=RES, target=TGT)
    hi = st.adapt_threshold(0.8, reciprocity=0.9, reservation=RES, target=TGT)
    assert hi <= lo


# ── tactic selection ─────────────────────────────────────────────────────────────
def _tactic(**kw):
    defaults = dict(is_opening=False, incoming_utility=0.6, adjusted_threshold=0.7,
                    reservation=RES, reciprocity=0.3, supplier_held_terms=[], at_deadline=False)
    defaults.update(kw)
    return st.choose_tactic(**defaults)[0]


def test_opening_is_anchor():
    assert _tactic(is_opening=True, incoming_utility=None) == "anchor"


def test_deadline_below_floor_is_walk():
    assert _tactic(incoming_utility=0.4, adjusted_threshold=0.5, at_deadline=True) == "walk"


def test_offer_clearing_the_bar_is_concede():
    assert _tactic(incoming_utility=0.85, adjusted_threshold=0.7) == "concede"


def test_supplier_defending_a_term_is_trade():
    assert _tactic(supplier_held_terms=["price"]) == "trade"


def test_staller_is_hold():
    assert _tactic(reciprocity=0.0, supplier_held_terms=[]) == "hold"
