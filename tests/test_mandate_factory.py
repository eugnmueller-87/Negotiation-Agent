"""mandate_factory — compiling a portfolio row into a baseline-scaled envelope pair.

The load-bearing test is the SCALE FIX: a real-price contract (e.g. €272) must never
fabricate savings by clamping a synthetic offer to utility 1.0. Both envelopes' price
bands are re-based onto the row's own baseline, so a settled price is arithmetically
bounded within the ±pct the human signed.
"""

from __future__ import annotations

import pytest

from negotiation_agent.engine import DealEngine, EngineConfig
from negotiation_agent.mandate_factory import (
    LOW_CONFIDENCE_THRESHOLD,
    ContractRow,
    compile_row,
    settled_savings,
)
from negotiation_agent.simulator.loop import run_negotiation
from negotiation_agent.simulator.personas import COOPERATIVE
from negotiation_agent.simulator.supplier import ParametricSupplier
from negotiation_agent.supplier_model import SupplierModel


def test_scale_fix_no_fabrication():
    # a €272 contract with a ±5% renewal mandate: the settled price must land INSIDE the
    # baseline-scaled band and the saving must not exceed the signed 5% — the old bug scored
    # €272 against a 9..12 band, clamped the price utility, and "saved" ~96%.
    row = ContractRow(
        row_id="r", instruction="renew", renew_pct=5.0,
        baseline_price=272.0, annual_spend_eur=250000.0,
    )
    cm = compile_row(row, signed_by="cm")
    assert cm.route == "negotiate" and cm.price_scaled
    be, se = cm.buyer_envelope, cm.supplier_envelope
    lo, hi = 272.0 * 0.95, 272.0 * 1.05
    assert be.term_map["price"].best == pytest.approx(lo)  # buyer MINIMIZE: best = low end
    assert be.term_map["price"].worst == pytest.approx(hi)

    engine = DealEngine(be, SupplierModel.uniform(be), EngineConfig(max_rounds=8, beta=4.0))
    result = run_negotiation(be, engine, ParametricSupplier(se, COOPERATIVE), supplier_envelope=se)
    assert result.final_deal is not None
    settled = result.final_deal.terms["price"]
    assert lo <= settled <= hi  # inside the signed band — not a clamped fantasy

    ratio, eur = settled_savings(
        baseline_price=272.0, settled_price=settled, annual_spend_eur=250000.0
    )
    assert abs(ratio) <= 0.05 + 1e-9  # the signed ±5% is a HARD ceiling


def test_savings_math_exact_and_negative():
    assert settled_savings(baseline_price=100.0, settled_price=95.0, annual_spend_eur=40000.0) == (
        0.05,
        2000.0,
    )
    # a price INCREASE inside the band is reported as negative, never clipped to zero
    ratio, eur = settled_savings(
        baseline_price=100.0, settled_price=103.0, annual_spend_eur=40000.0
    )
    assert ratio == -0.03 and eur == -1200.0


def test_savings_ratio_only_without_spend():
    ratio, eur = settled_savings(baseline_price=100.0, settled_price=90.0, annual_spend_eur=None)
    assert ratio == 0.10 and eur is None


def test_settled_savings_rejects_nonpositive_baseline():
    with pytest.raises(ValueError):
        settled_savings(baseline_price=0.0, settled_price=1.0, annual_spend_eur=1.0)


def test_cancel_routes_to_terminate_no_envelopes():
    cm = compile_row(ContractRow(row_id="r", instruction="cancel"), signed_by="cm")
    assert cm.route == "terminate"
    assert cm.buyer_envelope is None and cm.supplier_envelope is None


def test_low_confidence_renew_queues_for_human():
    row = ContractRow(
        row_id="r", instruction="renew", renew_pct=8.0,
        baseline_price=100.0, extraction_confidence=LOW_CONFIDENCE_THRESHOLD - 0.1,
    )
    cm = compile_row(row, signed_by="cm")
    assert cm.route == "human_confirm"  # a guessed baseline is never negotiated


def test_no_baseline_is_utility_only():
    row = ContractRow(row_id="r", instruction="renew", renew_pct=10.0)
    cm = compile_row(row, signed_by="cm")
    assert cm.route == "negotiate" and cm.price_scaled is False


def test_renew_band_geometry_mirrors_and_sums_to_one():
    cm = compile_row(
        ContractRow(row_id="r", instruction="renew", renew_pct=10.0, baseline_price=100.0),
        signed_by="cm",
    )
    be, se = cm.buyer_envelope, cm.supplier_envelope
    assert (be.term_map["price"].best, be.term_map["price"].worst) == pytest.approx((90.0, 110.0))
    # supplier MAXIMISEs price: best = the HIGH end (the exact mirror)
    assert (se.term_map["price"].best, se.term_map["price"].worst) == pytest.approx((110.0, 90.0))
    assert sum(t.weight for t in be.terms) == pytest.approx(1.0)
    assert sum(t.weight for t in se.terms) == pytest.approx(1.0)


def test_cancel_rejects_renew_pct():
    with pytest.raises(ValueError):
        ContractRow(row_id="r", instruction="cancel", renew_pct=5.0)


def test_renew_requires_pct():
    with pytest.raises(ValueError):
        ContractRow(row_id="r", instruction="renew", baseline_price=100.0)
