"""Coverage for the demo entry point and the v1 LLM seams.

These exercise the CLI (the one-command demo a reviewer runs), the
``SupplierModel.from_intents`` classifier seam, and the supplier bot's
None-offer / stall paths — the surfaces the core-logic tests skip.
"""

from __future__ import annotations

import json

from negotiation_agent.envelope import Direction, Envelope, Offer, TermSpec, TermType
from negotiation_agent.simulator.cli import main
from negotiation_agent.simulator.personas import EVASIVE
from negotiation_agent.simulator.scenarios import reference_matrix
from negotiation_agent.simulator.supplier import ParametricSupplier
from negotiation_agent.supplier_model import SupplierModel


def _envelope() -> Envelope:
    return Envelope(
        negotiation_id="seam",
        version=1,
        signed_by="t",
        terms=[
            TermSpec(
                name="price",
                term_type=TermType.PRICE,
                direction=Direction.MINIMIZE,
                best=9.0,
                worst=12.0,
                weight=0.5,
            ),
            TermSpec(
                name="payment_days",
                term_type=TermType.PAYMENT_DAYS,
                direction=Direction.MAXIMIZE,
                best=90,
                worst=30,
                weight=0.5,
            ),
        ],
    )


# --- CLI: the one-command demo ------------------------------------------------


def test_cli_default_report_runs(capsys):
    rc = main([])
    assert rc == 0
    out = capsys.readouterr().out
    assert "NEGOTIATION AGENT v0" in out
    assert "OVERALL" in out
    assert "by belief condition" in out


def test_cli_json_is_valid_and_shaped(capsys):
    rc = main(["--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "batch" in payload and "per_negotiation" in payload
    assert payload["batch"]["n"] == len(reference_matrix())


def test_cli_respects_flags(capsys):
    rc = main(["--json", "--max-rounds", "4", "--beta", "6"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    # Every negotiation is bounded by the tighter deadline.
    assert all(p["rounds_used"] <= 4 for p in payload["per_negotiation"])


def test_cli_transcript_renders(capsys):
    rc = main(["--transcript", "ref/aggressive/oracle"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "TRANSCRIPT  ref/aggressive/oracle" in out
    assert "OUTCOME:" in out
    # Header columns present.
    assert "threshold" in out and "buyerU" in out and "supU" in out


def test_cli_transcript_unknown_scenario_errors(capsys):
    rc = main(["--transcript", "does/not/exist"])
    assert rc == 2
    assert "unknown scenario" in capsys.readouterr().err


def test_cli_baseline_shows_supplier_gain(capsys):
    rc = main(["--baseline"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "LOGROLLING vs PRICE-SPLIT" in out
    assert "mean supplier gain" in out


# --- from_intents: the v1 LLM-classifier seam --------------------------------


def test_from_intents_maps_labels_to_terms():
    env = _envelope()
    # "cash_flow" -> PAYMENT_DAYS; "margin" -> PRICE (+ rebate, absent here).
    model = SupplierModel.from_intents(env, {"cash_flow": 0.9, "margin": 0.3})
    prio = model.priorities(env)
    assert model.source == "llm"
    # Payment days got the higher appetite, so it normalizes higher than price.
    assert prio["payment_days"] > prio["price"]


def test_from_intents_ignores_unknown_labels():
    env = _envelope()
    model = SupplierModel.from_intents(env, {"nonsense_label": 1.0})
    # No mapped terms -> all-zero appetite -> priorities fall back to uniform.
    prio = model.priorities(env)
    assert prio["price"] == prio["payment_days"]


def test_uniform_belief_is_flat():
    env = _envelope()
    prio = SupplierModel.uniform(env).priorities(env)
    assert prio["price"] == prio["payment_days"] == 0.5


# --- supplier bot edge paths -------------------------------------------------


def test_supplier_holds_on_none_offer():
    sup = ParametricSupplier(_envelope(), EVASIVE)
    move = sup.respond(0, None)
    assert move.kind == "offer"
    assert move.offer is None  # nothing offered yet, so it holds "nothing"


def test_supplier_stall_repeats_previous_offer():
    env = _envelope()
    sup = ParametricSupplier(env, EVASIVE)  # stall_period=3
    weak = Offer(terms={"price": 12.0, "payment_days": 30})
    first = sup.respond(0, weak)  # round 0 % 3 == 0 -> real counter
    assert first.kind == "offer"
    stalled = sup.respond(1, weak)  # round 1 % 3 != 0 -> repeat
    assert stalled.offer == first.offer
