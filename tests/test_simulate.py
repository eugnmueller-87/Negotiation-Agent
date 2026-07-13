"""POST /portfolio/simulate — $0 batch negotiation against a SIMULATED counterparty.

Gated on FastAPI. No LLM on this path in any mode, so no secret is needed. Proves: every
result is method="simulated" (never "exact"), the scale fix keeps savings inside the signed
band, cancel cost-avoidance is separate and gated on a servable window, low-confidence rows
queue, the N-cap 413s, and identical requests are byte-identical (deterministic, no RNG).
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from negotiation_agent import api  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    api._rate_hits.clear()
    yield
    api._rate_hits.clear()


@pytest.fixture
def client():
    return TestClient(api.app)


def _sim(client, rows, **over):
    body = {"signed_by": "e.mueller", "buyer_name": "Acme Buyer", "rows": rows, **over}
    return client.post("/portfolio/simulate", json=body)


def test_renew_savings_stay_inside_the_signed_band(client):
    r = _sim(client, [{
        "row_id": "r1", "instruction": "renew", "renew_pct": 10.0,
        "baseline_price": 272.0, "annual_spend_eur": 250000.0,
    }])
    assert r.status_code == 200, r.text
    row = r.json()["rows"][0]
    assert row["action"] == "negotiated" and row["method"] == "simulated"
    assert row["savings_basis"] == "exact_vs_baseline"
    assert abs(row["saving_ratio"]) <= 0.10 + 1e-9  # never exceeds the signed ±10%
    # settled inside the baseline-scaled band, never a clamped fantasy
    assert 272.0 * 0.90 <= row["settled_price"] <= 272.0 * 1.10


def test_every_row_is_method_simulated(client):
    rows = [
        {"row_id": "a", "instruction": "renew", "renew_pct": 5.0, "baseline_price": 100.0},
        {"row_id": "b", "instruction": "cancel"},
        {"row_id": "c", "instruction": "renew", "renew_pct": 5.0, "baseline_price": 50.0,
         "extraction_confidence": 0.3},
    ]
    d = _sim(client, rows).json()
    assert d["method"] == "simulated"
    assert all(row["method"] == "simulated" for row in d["rows"])


def test_cancel_cost_avoidance_is_separate_and_gated(client):
    # a cancel with NO contract text has no servable-window evidence → NO avoidance booked,
    # and it is never summed into negotiated savings.
    d = _sim(client, [{"row_id": "x", "instruction": "cancel", "annual_spend_eur": 50000.0}]).json()
    row = d["rows"][0]
    assert row["action"] == "terminate_notice"
    assert row["cost_avoidance_eur"] is None  # no clock → no fabricated year of spend
    assert d["total_cost_avoidance_eur"] == 0.0
    assert d["total_saved_eur"] == 0.0  # cancel is never in the savings total


def test_low_confidence_row_is_queued_not_negotiated(client):
    row = _sim(client, [{
        "row_id": "q", "instruction": "renew", "renew_pct": 8.0,
        "baseline_price": 100.0, "extraction_confidence": 0.4,
    }]).json()["rows"][0]
    assert row["action"] == "queued_human_confirm"
    assert row["saving_ratio"] is None and row["saved_eur"] is None


def test_no_baseline_is_utility_only(client):
    d = _sim(client, [{"row_id": "u", "instruction": "renew", "renew_pct": 10.0}]).json()
    row = d["rows"][0]
    assert row["action"] == "negotiated"
    assert row["savings_basis"] == "utility_only"
    assert row["saving_ratio"] is None and row["saved_eur"] is None


def test_n_cap_413s_before_running(client, monkeypatch):
    monkeypatch.setattr(api, "_PORTFOLIO_MAX_ROWS", 3)
    rows = [{"row_id": f"r{i}", "instruction": "renew", "renew_pct": 5.0, "baseline_price": 100.0}
            for i in range(4)]
    r = _sim(client, rows)
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "too_many_rows"


def test_batch_is_deterministic(client):
    rows = [{"row_id": f"r{i}", "instruction": "renew", "renew_pct": 7.0,
             "baseline_price": 100.0 + i, "annual_spend_eur": 10000.0} for i in range(5)]
    a = _sim(client, rows).json()
    api._rate_hits.clear()
    b = _sim(client, rows).json()
    assert a == b  # splitmix persona pick + pure engine → byte-identical


def test_cancel_rejects_renew_pct_at_the_boundary(client):
    r = _sim(client, [{"row_id": "r", "instruction": "cancel", "renew_pct": 5.0}])
    assert r.status_code == 422  # the ContractRow validator rejects the mismatched pairing
