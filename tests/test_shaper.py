"""The deterministic mandate transform — the thesis applied to mandate construction.

The load-bearing property: for every accepted subset, apply_adjustments returns a
VALID Envelope or raises MandateConflict — never an invalid mandate, never a silently
dropped rule. That property is the audit guarantee.
"""

from __future__ import annotations

import datetime as dt
import itertools

import pytest

from negotiation_agent.envelope import Direction, Envelope, Offer, TermSpec, TermType
from negotiation_agent.shaper import (
    AddTerm,
    MandateConflict,
    ProposedAdjustment,
    ShiftTarget,
    TightenBounds,
    WeightBump,
    _add_term_spec,
    apply_adjustments,
    days_until,
    incumbent_scores_below_floor,
)


def _base():
    return Envelope(
        negotiation_id="n",
        version=1,
        signed_by="tester",
        target_utility=0.90,
        reservation_utility=0.60,
        terms=[
            TermSpec(
                name="price",
                term_type=TermType.PRICE,
                direction=Direction.MINIMIZE,
                best=92.0,
                worst=108.0,
                weight=0.50,
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
                name="contract_months",
                term_type=TermType.CONTRACT_MONTHS,
                direction=Direction.MINIMIZE,
                best=12.0,
                worst=24.0,
                weight=0.25,
            ),
        ],
    )


def _adj(rule, delta, role="hedge", sev="medium"):
    return ProposedAdjustment(rule_id=rule, severity=sev, role=role, delta=delta, rationale="x")


ALL_ADJ = [
    _adj("R-WEIGHT", WeightBump(term_name="price", delta=0.10)),
    _adj("R-EXPIRING-SOON", ShiftTarget(target_delta=-0.03, reservation_delta=-0.02)),
    _adj("R-EXPIRING-FAR", ShiftTarget(target_delta=0.02)),
    _adj("R-TIGHTEN", TightenBounds(term_name="contract_months", new_worst=18.0)),
    _adj(
        "R-ADD-REBATE",
        AddTerm(
            spec=_add_term_spec(
                "rebate_pct", TermType.REBATE_PCT, Direction.MAXIMIZE, 8.0, 0.0, 0.06
            ),
            appetite=0.8,
        ),
        role="give",
    ),
]


@pytest.mark.parametrize("n", range(len(ALL_ADJ) + 1))
def test_every_subset_validates_or_conflicts(n):
    # THE audit property: over all subsets of size n, apply never emits an invalid
    # envelope — it returns a valid Envelope or raises MandateConflict.
    base = _base()
    for combo in itertools.combinations(ALL_ADJ, n):
        try:
            shaped, _ = apply_adjustments(base, list(combo))
        except MandateConflict:
            continue  # a valid outcome — the rules conflict, human deselects
        # if it returned, it must be a real, re-validated Envelope with weights summing to 1
        assert abs(sum(t.weight for t in shaped.terms) - 1.0) < 1e-6
        assert shaped.reservation_utility < shaped.target_utility


def test_empty_accepted_returns_base_unchanged_shape():
    base = _base()
    shaped, appetite = apply_adjustments(base, [])
    assert abs(sum(t.weight for t in shaped.terms) - 1.0) < 1e-6
    assert shaped.target_utility == base.target_utility
    assert shaped.reservation_utility == base.reservation_utility
    assert shaped.version == base.version + 1


def test_weight_bump_renormalises_to_one():
    base = _base()
    shaped, _ = apply_adjustments(base, [_adj("R", WeightBump(term_name="price", delta=0.30))])
    assert abs(sum(t.weight for t in shaped.terms) - 1.0) < 1e-9
    # price got relatively heavier
    price = next(t for t in shaped.terms if t.name == "price")
    assert price.weight > 0.50


def test_expiring_soon_lowers_the_floor_not_raises_it():
    # THE corrected sign bug: urgency LOWERS reservation (accept a worse deal, no-deal
    # is worse). A rule that raised the floor would make the agent walk away MORE.
    base = _base()
    shaped, _ = apply_adjustments(
        base, [_adj("R-EXPIRING-SOON", ShiftTarget(target_delta=-0.03, reservation_delta=-0.02))]
    )
    assert shaped.reservation_utility < base.reservation_utility
    assert shaped.target_utility < base.target_utility


def test_add_term_and_appetite_flow_through():
    base = _base()
    give = _adj(
        "R-ADD-REBATE",
        AddTerm(
            spec=_add_term_spec(
                "rebate_pct", TermType.REBATE_PCT, Direction.MAXIMIZE, 8.0, 0.0, 0.06
            ),
            appetite=0.8,
        ),
        role="give",
    )
    shaped, appetite = apply_adjustments(base, [give], supplier_appetite={"price": 0.15})
    assert any(t.name == "rebate_pct" for t in shaped.terms)
    assert appetite["rebate_pct"] == 0.8  # the give-term's appetite hint is carried
    assert appetite["price"] == 0.15


def test_tighten_bounds_stays_direction_valid():
    base = _base()
    # MINIMIZE contract_months (best=12<worst=24) tightened to worst=18 — must stay > best
    shaped, _ = apply_adjustments(
        base, [_adj("R", TightenBounds(term_name="contract_months", new_worst=18.0))]
    )
    t = next(x for x in shaped.terms if x.name == "contract_months")
    assert t.best < t.worst <= 24.0
    assert t.worst == pytest.approx(18.0)


def test_tighten_clamps_illegal_worst():
    base = _base()
    # trying to tighten past `best` (worst below best on a MINIMIZE term) must clamp,
    # never produce an invalid TermSpec that fails the direction validator
    shaped, _ = apply_adjustments(
        base, [_adj("R", TightenBounds(term_name="contract_months", new_worst=5.0))]
    )
    t = next(x for x in shaped.terms if x.name == "contract_months")
    assert t.worst > t.best  # still a valid MINIMIZE span


def test_tight_spread_clamps_gracefully_keeping_invariant():
    # a downward target shift on a tiny gap is handled by the clamp (reservation follows
    # target down, staying strictly below) — a valid, safer outcome than erroring.
    tight = Envelope(
        negotiation_id="t",
        version=1,
        signed_by="x",
        target_utility=0.62,
        reservation_utility=0.60,
        terms=[
            TermSpec(
                name="price",
                term_type=TermType.PRICE,
                direction=Direction.MINIMIZE,
                best=92.0,
                worst=108.0,
                weight=1.0,
            )
        ],
    )
    shaped, _ = apply_adjustments(
        tight, [_adj("R", ShiftTarget(target_delta=-0.05, reservation_delta=0.0))]
    )
    assert shaped.reservation_utility < shaped.target_utility  # invariant held by clamp
    assert shaped.target_utility == pytest.approx(0.57)


def test_unresolvable_shift_raises_conflict():
    # driving target to 0 while reservation is pushed up produces an impossible spread
    # the clamp can't rescue → MandateConflict, never an invalid envelope
    tight = Envelope(
        negotiation_id="t",
        version=1,
        signed_by="x",
        target_utility=0.10,
        reservation_utility=0.05,
        terms=[
            TermSpec(
                name="price",
                term_type=TermType.PRICE,
                direction=Direction.MINIMIZE,
                best=92.0,
                worst=108.0,
                weight=1.0,
            )
        ],
    )
    with pytest.raises(MandateConflict):
        apply_adjustments(
            tight, [_adj("R", ShiftTarget(target_delta=-0.10, reservation_delta=0.0))]
        )


def test_determinism_same_subset_same_output():
    base = _base()
    subset = [ALL_ADJ[0], ALL_ADJ[3]]
    a, _ = apply_adjustments(base, subset)
    b, _ = apply_adjustments(base, subset)
    assert a.model_dump() == b.model_dump()  # byte-identical


def test_incumbent_below_floor_flag():
    base = _base()
    shaped, _ = apply_adjustments(base, [])
    # an incumbent at the worst of every term scores 0 → below the 0.60 floor
    bad = Offer(terms={"price": 108.0, "payment_days": 30.0, "contract_months": 24.0})
    assert incumbent_scores_below_floor(shaped, bad) is True
    # an incumbent at best scores 1.0 → above the floor
    good = Offer(terms={"price": 92.0, "payment_days": 60.0, "contract_months": 12.0})
    assert incumbent_scores_below_floor(shaped, good) is False
    assert incumbent_scores_below_floor(shaped, None) is False


@pytest.mark.parametrize(
    "text,expected_sign",
    [
        ("2099-01-01", 1),  # far future → positive
        ("01.01.2099", 1),
        ("garbage", None),
        ("", None),
        (None, None),
    ],
)
def test_date_parser(text, expected_sign):
    today = dt.date(2026, 7, 10)
    d = days_until(text, today=today)
    if expected_sign is None:
        assert d is None
    else:
        assert d is not None and (d > 0) == (expected_sign > 0)
