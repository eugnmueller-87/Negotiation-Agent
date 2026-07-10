"""The /terminate endpoint — the notice clock + drafted notice over HTTP.

Gated on FastAPI. Pure and offline: regex Zone-B extraction + deterministic date math,
no network. Proves the upload → clock → draft loop the termination UI drives.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from negotiation_agent import api  # noqa: E402

# A contract with an expiry the regex extractor reaches; notice period supplied by the client.
CONTRACT = (
    "SUPPLY AGREEMENT\n"
    "Supplier: Nordwerk Verpackung GmbH\n"
    "Unit price EUR 11.50 per unit. Net-30 days. Expires 2099-12-31 unless renewed.\n"
    "This agreement is governed by German law."
)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("PEITHO_MANDATE_SECRET", "t")
    return TestClient(api.app)


def test_terminate_computes_clock_and_drafts_notice(client):
    r = client.post(
        "/terminate",
        json={
            "contract_text": CONTRACT,
            "buyer_name": "ACME Buyer",
            "termination_notice_days": 90,
        },
    )
    assert r.status_code == 200, r.text
    d = r.json()
    # expiry 2099 is far out -> window OPEN, deadline = expiry - 90 days
    assert d["clock"]["window_status"] == "OPEN"
    assert d["clock"]["notice_period_days"] == 90
    assert d["clock"]["notice_deadline"] == "2099-10-02"
    # the draft is grounded in the contract's own facts
    assert "Nordwerk Verpackung GmbH" in d["notice_draft"]
    assert "2099-12-31" in d["notice_draft"]
    assert "verify" in d["notice_draft"].lower()  # the mandatory disclaimer


def test_terminate_without_notice_period_is_no_deadline(client):
    r = client.post("/terminate", json={"contract_text": CONTRACT, "buyer_name": "ACME"})
    assert r.status_code == 200, r.text
    assert r.json()["clock"]["window_status"] == "NO_DEADLINE"


def test_terminate_rejects_bad_intent(client):
    r = client.post("/terminate", json={"contract_text": CONTRACT, "intent": "delete_everything"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "bad_intent"
