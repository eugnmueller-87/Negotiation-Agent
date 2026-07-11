"""The /intel + /reshape endpoints — contract intelligence over HTTP.

Gated on FastAPI. No network: /intel runs research=False (pure regex + rules), and
/reshape is pure. Proves the upload→shape→reshape loop the frontend drives.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from negotiation_agent import api  # noqa: E402

CONTRACT = (
    "SUPPLY AGREEMENT\n"
    "Supplier: Nordwerk Verpackung GmbH\n"
    "Unit price EUR 11.50 per unit. Net-30 days. Expires 2026-07-25 unless renewed.\n"
    "Minimum 40,000 units per year. Both parties enter a non-disclosure agreement."
)

BASE = {
    "negotiation_id": "n", "version": 1, "signed_by": "e",
    "target_utility": 0.90, "reservation_utility": 0.60,
    "terms": [
        {"name": "price", "term_type": "price", "direction": "minimize",
         "best": 11.0, "worst": 13.0, "weight": 0.5},
        {"name": "payment_days", "term_type": "payment_days", "direction": "maximize",
         "best": 60.0, "worst": 30.0, "weight": 0.25},
        {"name": "volume_units", "term_type": "volume_units", "direction": "minimize",
         "best": 10000.0, "worst": 50000.0, "weight": 0.25},
    ],
}


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    api._rate_hits.clear()  # module-global counters — isolate the strict Hades cap per test
    yield
    api._rate_hits.clear()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("PEITHO_MANDATE_SECRET", "t")
    return TestClient(api.app)


def test_prepare_degrades_when_research_is_misconfigured(client, monkeypatch):
    # a bad HADES_URL makes HadesClient() raise on construction; /prepare must still return
    # the extraction with brief=None, never 500 the whole endpoint (audit issue #9).
    monkeypatch.setenv("HADES_URL", "http://not-https.example")  # non-https → ResearchUnavailable
    r = client.post("/prepare", json={"contract_text": CONTRACT, "research": True})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["extraction"]["supplier_name"] == "Nordwerk Verpackung GmbH"
    assert body["brief"] is None  # research degraded, extraction survived


def test_intel_extracts_and_proposes(client):
    r = client.post("/intel", json={"contract_text": CONTRACT, "research": False})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["intelligence"]["extraction"]["supplier_name"] == "Nordwerk Verpackung GmbH"
    assert d["intelligence"]["lifecycle"]["expiration_date"]["value"] == "2026-07-25"
    rule_ids = {a["rule_id"] for a in d["adjustments"]}
    assert "R-EXPIRING-SOON" in rule_ids  # expiry parsed and rule fired
    assert "R-NO-REBATE" in rule_ids  # volume present, no rebate
    assert d["blocked"] is False


def test_reshape_applies_accepted_adjustments(client):
    intel = client.post("/intel", json={"contract_text": CONTRACT, "research": False}).json()
    accepted = [a["rule_id"] for a in intel["adjustments"]]
    r = client.post("/reshape", json={
        "base_envelope": BASE, "adjustments": intel["adjustments"],
        "accepted_rule_ids": accepted, "supplier_appetite": {"price": 0.15},
    })
    assert r.status_code == 200, r.text
    s = r.json()
    # expiring-soon lowered the floor; rebate give-term was added
    assert s["shaped_envelope"]["reservation_utility"] < 0.60
    assert any(t["name"] == "rebate_pct" for t in s["shaped_envelope"]["terms"])
    # weights still sum to 1.0 (re-validated envelope)
    assert abs(sum(t["weight"] for t in s["shaped_envelope"]["terms"]) - 1.0) < 1e-6


def test_reshape_with_no_accepted_is_base_shape(client):
    intel = client.post("/intel", json={"contract_text": CONTRACT, "research": False}).json()
    r = client.post("/reshape", json={
        "base_envelope": BASE, "adjustments": intel["adjustments"],
        "accepted_rule_ids": [],  # accept nothing
    })
    assert r.status_code == 200
    s = r.json()
    # nothing accepted -> target/reservation unchanged, no rebate term
    assert s["shaped_envelope"]["target_utility"] == 0.90
    assert not any(t["name"] == "rebate_pct" for t in s["shaped_envelope"]["terms"])


def test_reshape_bad_input_is_400(client):
    r = client.post("/reshape", json={
        "base_envelope": {"garbage": True}, "adjustments": [], "accepted_rule_ids": [],
    })
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "bad_reshape_input"
