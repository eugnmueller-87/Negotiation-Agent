"""Cross-negotiation learning — outcome logging + priors that warm-start the engine.

The learning is OFF-HANDS: history feeds priors that seed the engine's starting point; it never
decides, never changes the mandate, and cold-start (no history) is byte-identical to today. These
tests pin that, plus the privacy invariant (no raw prices / PII in a stored record) and the store's
crash-tolerant read.
"""

from __future__ import annotations

import json

import pytest

from negotiation_agent import outcomes as oc
from negotiation_agent import priors as pr
from negotiation_agent.envelope import Direction, Envelope, TermSpec, TermType
from negotiation_agent.supplier_model import SupplierModel


@pytest.fixture
def env() -> Envelope:
    return Envelope(
        negotiation_id="t", version=1, signed_by="e", target_utility=0.9, reservation_utility=0.5,
        terms=[
            TermSpec(name="price", term_type=TermType.PRICE, direction=Direction.MINIMIZE,
                     best=90, worst=110, weight=0.6),
            TermSpec(name="payment_days", term_type=TermType.PAYMENT_DAYS,
                     direction=Direction.MAXIMIZE, best=90, worst=30, weight=0.4),
        ],
    )


def _accepted(conceded, settled=0.7):
    return oc.NegotiationOutcome(
        category="pkg", outcome="accepted", rounds=5, settled_utility=settled,
        settled_term_utilities={"price": 0.6, "payment_days": 0.9},
        conceded_terms=conceded, target_utility=0.9, reservation_utility=0.5,
    )


# ── cold start = today's behaviour ───────────────────────────────────────────────
def test_no_history_yields_empty_prior():
    prior = pr.learn_category_prior("pkg", [])
    assert prior.samples == 0 and not prior.confident and prior.appetite_prior == {}


def test_cold_start_seed_equals_uniform(env):
    # the whole point: with no (or thin) history the engine behaves exactly as it does today
    prior = pr.learn_category_prior("pkg", [])
    assert pr.seed_supplier_model(env, prior).appetite == SupplierModel.uniform(env).appetite


def test_thin_history_is_not_trusted(env):
    # below the confidence threshold, the seed still falls back to uniform (small samples overfit)
    prior = pr.learn_category_prior("pkg", [_accepted(["payment_days"])])  # 1 sample
    assert not prior.confident
    assert pr.seed_supplier_model(env, prior).appetite == SupplierModel.uniform(env).appetite


# ── the learned signal ───────────────────────────────────────────────────────────
def test_prior_learns_which_term_suppliers_concede(env):
    # 6 deals where suppliers held price, conceded payment → payment appetite LOW, price higher
    history = [_accepted(["payment_days"]) for _ in range(6)]
    prior = pr.learn_category_prior("pkg", history)
    assert prior.confident
    assert prior.appetite_prior["payment_days"] < 0.5  # conceded often → low appetite


def test_seeded_belief_routes_concessions_to_the_conceded_term(env):
    history = [_accepted(["payment_days"]) for _ in range(6)]
    seed = pr.seed_supplier_model(env, pr.learn_category_prior("pkg", history))
    pri = seed.priorities(env)
    assert pri["payment_days"] < pri["price"]  # engine spends concessions where suppliers give


def test_typical_settled_utility_is_the_mean_of_accepted(env):
    history = [_accepted(["payment_days"], settled=0.6), _accepted(["payment_days"], settled=0.8)]
    history += [_accepted(["payment_days"], settled=0.7) for _ in range(4)]
    prior = pr.learn_category_prior("pkg", history)
    assert prior.typical_settled_utility == pytest.approx(0.7, abs=0.01)


def test_escalation_rate_counts_failures(env):
    history = [_accepted(["payment_days"]) for _ in range(4)]
    history += [
        oc.NegotiationOutcome(category="pkg", outcome="escalated", rounds=8,
                              target_utility=0.9, reservation_utility=0.5)
        for _ in range(2)
    ]
    assert pr.learn_category_prior("pkg", history).escalation_rate == pytest.approx(2 / 6, abs=1e-3)


# ── privacy: no raw prices / PII in a stored record ──────────────────────────────
def test_stored_record_has_no_raw_prices():
    # a settled record carries dimensionless utilities, never a euro amount like 194920 or 11.5
    rec = _accepted(["payment_days"])
    raw = json.dumps(rec.model_dump(mode="json"))
    # the only floats present are in [0,1] utilities + the mandate context — assert no big number
    for tok in ("194920", "11.5", "110", "90.0"):
        assert tok not in raw


def test_no_supplier_name_field_exists():
    # the schema has no place to put a supplier name / contact — PII can't leak by construction
    assert "supplier_name" not in oc.NegotiationOutcome.model_fields
    assert "supplier" not in oc.NegotiationOutcome.model_fields


# ── the store ────────────────────────────────────────────────────────────────────
def test_store_round_trips(tmp_path):
    store = oc.OutcomeStore(tmp_path / "out.jsonl")
    store.append(_accepted(["payment_days"]))
    store.append(_accepted(["price"]))
    loaded = store.load()
    assert len(loaded) == 2 and loaded[0].conceded_terms == ["payment_days"]


def test_store_load_category_filters(tmp_path):
    store = oc.OutcomeStore(tmp_path / "out.jsonl")
    store.append(_accepted(["payment_days"]))
    store.append(oc.NegotiationOutcome(category="other", outcome="escalated", rounds=3,
                                       target_utility=0.9, reservation_utility=0.5))
    assert len(store.load_category("pkg")) == 1


def test_store_tolerates_a_malformed_trailing_line(tmp_path):
    # a crash mid-write can leave a partial last line — it must be skipped, not break the read
    path = tmp_path / "out.jsonl"
    store = oc.OutcomeStore(path)
    store.append(_accepted(["payment_days"]))
    with path.open("a", encoding="utf-8") as f:
        f.write('{"category": "pkg", "outcome": "acce')  # truncated JSON
    assert len(store.load()) == 1  # the good record survives


def test_missing_store_file_loads_empty(tmp_path):
    assert oc.OutcomeStore(tmp_path / "nope.jsonl").load() == []
