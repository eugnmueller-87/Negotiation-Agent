"""The /negotiate endpoints — the full loop over HTTP, with a fake drafter.

Gated on FastAPI being installed (the ``[web]`` extra). No network: the drafter is
a fake and PEITHO_MANDATE_SECRET is set to a test value. Proves the guard-with-
redraft loop, mandate tamper rejection, and the closed-game gate over the wire.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from negotiation_agent import api  # noqa: E402

SECRET = "test-mandate-secret"

MANDATE = {
    "envelope": {
        "negotiation_id": "neg-1",
        "version": 1,
        "signed_by": "buyer",
        "target_utility": 0.90,
        "reservation_utility": 0.60,
        "terms": [
            {
                "name": "price",
                "term_type": "price",
                "direction": "minimize",
                "best": 92.0,
                "worst": 108.0,
                "weight": 0.5,
            },
            {
                "name": "payment_days",
                "term_type": "payment_days",
                "direction": "maximize",
                "best": 60.0,
                "worst": 30.0,
                "weight": 0.25,
            },
            {
                "name": "contract_months",
                "term_type": "contract_months",
                "direction": "minimize",
                "best": 12.0,
                "worst": 24.0,
                "weight": 0.25,
            },
        ],
    },
    "supplier_appetite": {"price": 0.15, "payment_days": 0.85, "contract_months": 0.70},
    "config": {"max_rounds": 6, "beta": 2.5, "stall_rounds": 3, "on_unknown_terms": "escalate"},
}


class _FakeDrafter:
    """Drafts using only the approved figures — always guard-clean."""

    def draft_buyer(self, brief, thread, advice=None, correspondents=None):
        nums = " ".join(f"{v:g}" for v in brief.approved_numbers.values())
        return f"Our position: {nums}. Looking forward to working together."

    def draft_supplier(self, persona, thread, company, category):
        return "..."


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("PEITHO_MANDATE_SECRET", SECRET)
    monkeypatch.setattr(api, "draft_client_factory", lambda: _FakeDrafter())
    return TestClient(api.app)


def _open(client):
    r = client.post("/negotiate/open", json={"mandate": MANDATE, "session_id": "s1"})
    assert r.status_code == 200, r.text
    return r.json()


def test_open_signs_mandate_and_drafts_anchor(client):
    body = _open(client)
    assert body["signed_mandate"]["sig"]
    assert body["turn"]["outcome"] == "counter"
    assert body["turn"]["buyer_message"]
    # opening anchor is at the buyer's target — approved figures present, message clean
    assert body["turn"]["approved_numbers"]


def test_step_folds_and_counters(client):
    signed = _open(client)["signed_mandate"]
    r = client.post(
        "/negotiate/step",
        json={
            "signed_mandate": signed,
            "transcript": {"turns": []},
            "supplier_input": {
                "mode": "bot",
                "raw_text": "We can do €105.00, net-30, 24 months.",
                "persona": "aggressive",
            },
            "session_id": "s1",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["turn"]["round_index"] == 1  # first supplier offer scored at round 1
    assert body["turn"]["guard"]["released_by"] in ("model", "fallback")
    # no buyer-internal block without the god-view gate
    assert body["turn"]["internal"] is None
    # the redacted supplier view carries no reservation floor
    assert "0.6" not in r.text or "reservation" not in r.text


def test_step_reports_misconfigured_not_tampered_when_secret_missing(client, monkeypatch):
    # a signed mandate exists, but the server loses PEITHO_MANDATE_SECRET (e.g. a redeploy
    # drops the env var). /step must report a server misconfiguration (500), NOT blame the
    # client for tampering (400) — the ops signal must point at the server (audit issue #8).
    signed = _open(client)["signed_mandate"]
    monkeypatch.delenv("PEITHO_MANDATE_SECRET", raising=False)
    r = client.post(
        "/negotiate/step",
        json={
            "signed_mandate": signed,
            "transcript": {"turns": []},
            "supplier_input": {"mode": "bot", "raw_text": "€105.00, net-30, 24 months."},
            "session_id": "s1",
        },
    )
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "misconfigured"


def test_open_wraps_letter_with_correspondents_on_fallback(monkeypatch):
    # force the fallback path so the deterministic letter-wrap is exercised over HTTP
    class _Cheat:
        def draft_buyer(self, brief, thread, advice=None, correspondents=None):
            return "We'll pay €1.23 per unit."  # never approved -> falls back

        def draft_supplier(self, persona, thread, company, category):
            return "..."

    monkeypatch.setenv("PEITHO_MANDATE_SECRET", SECRET)
    monkeypatch.setattr(api, "draft_client_factory", lambda: _Cheat())
    client = TestClient(api.app)
    r = client.post(
        "/negotiate/open",
        json={
            "mandate": MANDATE,
            "session_id": "s1",
            "correspondents": {
                "supplier_name": "Nordwerk GmbH",
                "supplier_contact": "Mr. Schmidt",
                "buyer_signature": "E. Müller",
            },
        },
    )
    assert r.status_code == 200, r.text
    msg = r.json()["turn"]["buyer_message"]
    assert msg.startswith("Dear Mr. Schmidt,")
    assert "Best regards,\nE. Müller" in msg


def test_tampered_mandate_is_rejected(client):
    signed = _open(client)["signed_mandate"]
    # attacker drops the reservation floor to extract a below-floor deal
    signed["mandate"]["envelope"]["reservation_utility"] = 0.0
    r = client.post(
        "/negotiate/step",
        json={
            "signed_mandate": signed,
            "transcript": {"turns": []},
            "supplier_input": {
                "mode": "bot",
                "raw_text": "€105, net-30, 24 months",
                "persona": "aggressive",
            },
            "session_id": "s1",
        },
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "mandate_tampered"


def test_unparseable_supplier_message_is_422(client):
    signed = _open(client)["signed_mandate"]
    r = client.post(
        "/negotiate/step",
        json={
            "signed_mandate": signed,
            "transcript": {"turns": []},
            "supplier_input": {"mode": "human", "raw_text": "hello there", "persona": "aggressive"},
            "session_id": "s1",
        },
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "offer_unparseable"


def test_human_buyer_off_mandate_number_is_caught(client):
    signed = _open(client)["signed_mandate"]
    r = client.post(
        "/negotiate/step",
        json={
            "signed_mandate": signed,
            "transcript": {"turns": []},
            "supplier_input": {
                "mode": "bot",
                "raw_text": "€105, net-30, 24 months",
                "persona": "aggressive",
            },
            "buyer_input": {"mode": "human", "raw_text": "We'll pay exactly €50.00 per unit."},
            "session_id": "s1",
        },
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "buyer_text_off_mandate"


def test_godview_gate_double_locked(client, monkeypatch):
    signed = _open(client)["signed_mandate"]
    step = {
        "signed_mandate": signed,
        "transcript": {"turns": []},
        "supplier_input": {
            "mode": "bot",
            "raw_text": "€105, net-30, 24 months",
            "persona": "aggressive",
        },
        "session_id": "s1",
    }
    # header alone, no server env -> still redacted
    r = client.post("/negotiate/step", json=step, headers={"X-Peitho-Godview": "1"})
    assert r.json()["turn"]["internal"] is None
    # both header AND env -> internal present
    monkeypatch.setenv("DEMO_GODVIEW", "1")
    r = client.post("/negotiate/step", json=step, headers={"X-Peitho-Godview": "1"})
    assert r.json()["turn"]["internal"] is not None
    assert "threshold" in r.json()["turn"]["internal"]


def test_health_reports_models(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["buyer_model"] == "claude-opus-4-8"
    assert r.json()["supplier_model"] == "claude-haiku-4-5"
