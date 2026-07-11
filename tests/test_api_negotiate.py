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


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    # the rate-limit counters are module-global; clear them so one test's requests can't
    # trip the limiter in the next (the window is IP-keyed and all tests share one IP).
    api._rate_hits.clear()
    yield
    api._rate_hits.clear()


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


def test_open_is_rate_limited_per_ip_not_session(client, monkeypatch):
    # cost-bearing /open is IP-rate-limited; rotating session_id must NOT dodge the cap
    # (the old limiter keyed on the client-chosen session_id — audit issue #4).
    monkeypatch.setattr(api, "_RATE_PER_MIN", 3)
    api._rate_hits.clear()
    codes = [
        client.post(
            "/negotiate/open", json={"mandate": MANDATE, "session_id": f"rotate-{i}"}
        ).status_code
        for i in range(6)
    ]
    assert codes.count(429) >= 1  # rotating the session_id did not escape the per-IP cap
    assert 429 in codes[3:]  # the cap kicks in after the limit


def test_rate_limit_not_bypassable_by_spoofed_xff(client, monkeypatch):
    # audit SEC-4: X-Forwarded-For is append-only; a spoofed LEFTMOST value must not create a
    # fresh rate bucket. The real client is the rightmost (trusted-proxy) entry, so rotating
    # the spoofed prefix must still hit the same bucket and get 429'd.
    monkeypatch.setattr(api, "_RATE_PER_MIN", 3)
    monkeypatch.setattr(api, "_TRUSTED_PROXIES", 1)
    api._rate_hits.clear()
    codes = [
        client.post(
            "/negotiate/open",
            json={"mandate": MANDATE, "session_id": "s1"},
            # attacker rotates the spoofed prefix; Railway appends the same real IP "9.9.9.9"
            headers={"X-Forwarded-For": f"{i}.{i}.{i}.{i}, 9.9.9.9"},
        ).status_code
        for i in range(6)
    ]
    assert 429 in codes  # the rotating prefix did not escape the per-real-IP cap


def test_open_rejects_unbounded_max_rounds(client):
    # a client-authored mandate can't sign an enormous max_rounds to blow up the fold CPU —
    # the wire cap (le=64) rejects it at validation (audit issue #16).
    import copy

    bad = copy.deepcopy(MANDATE)
    bad["config"] = {**MANDATE["config"], "max_rounds": 10_000_000}
    r = client.post("/negotiate/open", json={"mandate": bad, "session_id": "s1"})
    assert r.status_code == 422  # pydantic validation rejects the oversized config


def test_step_rejects_overlong_transcript(client):
    # a transcript longer than the mandate's round budget is rejected, not folded over —
    # the fold input is provably bounded by the signed mandate (audit issue #16).
    signed = _open(client)["signed_mandate"]
    turns = [{"terms": {}, "raw_text": "€105, net-30, 24 months."} for _ in range(20)]
    r = client.post(
        "/negotiate/step",
        json={
            "signed_mandate": signed,
            "transcript": {"turns": turns},  # 20 > max_rounds (6)
            "supplier_input": {"mode": "bot", "raw_text": "€105, net-30, 24 months."},
            "session_id": "s1",
        },
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "transcript_too_long"


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


def _godview_step(signed):
    return {
        "signed_mandate": signed,
        "transcript": {"turns": []},
        "supplier_input": {"mode": "bot", "raw_text": "€105, net-30, 24 months"},
        "session_id": "s1",
    }


def test_godview_needs_the_secret_token_not_a_public_header(client, monkeypatch):
    # audit SEC-3: the buyer-internal block (reservation floor!) must unlock ONLY on the
    # server-side secret token, never on a public header any caller can send.
    signed = _open(client)["signed_mandate"]
    step = _godview_step(signed)
    monkeypatch.setenv("PEITHO_GODVIEW_TOKEN", "s3cr3t-token")

    # the old public value "1" must NOT unlock internals any more
    r = client.post("/negotiate/step", json=step, headers={"X-Peitho-Godview": "1"})
    assert r.json()["turn"]["internal"] is None
    # a wrong token stays locked
    r = client.post("/negotiate/step", json=step, headers={"X-Peitho-Godview": "wrong"})
    assert r.json()["turn"]["internal"] is None
    # only the exact secret token unlocks it
    r = client.post("/negotiate/step", json=step, headers={"X-Peitho-Godview": "s3cr3t-token"})
    assert r.json()["turn"]["internal"] is not None
    assert "reservation_utility" in r.json()["turn"]["internal"]


def test_godview_fails_closed_with_no_token_configured(client, monkeypatch):
    # no PEITHO_GODVIEW_TOKEN set → god-view is OFF whatever header arrives (fail-closed),
    # so an internet-exposed instance with no token can never leak the reservation floor.
    monkeypatch.delenv("PEITHO_GODVIEW_TOKEN", raising=False)
    signed = _open(client)["signed_mandate"]
    r = client.post(
        "/negotiate/step", json=_godview_step(signed), headers={"X-Peitho-Godview": "1"}
    )
    assert r.json()["turn"]["internal"] is None


def test_security_headers_present_on_responses(client):
    # audit #10: defense-in-depth headers on every response
    r = client.get("/health")
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["X-Frame-Options"] == "DENY"
    assert "Content-Security-Policy" in r.headers
    assert "frame-ancestors 'none'" in r.headers["Content-Security-Policy"]


def test_health_is_minimal_for_anonymous_callers(client):
    # audit #12: anonymous /health returns only liveness + the (non-sensitive) mode —
    # no model IDs / KB size to fingerprint
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body == {"status": "ok", "mode": "demo"}
    assert "buyer_model" not in body


def test_health_reports_details_under_the_godview_token(client, monkeypatch):
    # the diagnostic detail is released only to a caller holding the god-view token
    monkeypatch.setenv("PEITHO_GODVIEW_TOKEN", "tok")
    r = client.get("/health", headers={"X-Peitho-Godview": "tok"})
    assert r.status_code == 200
    assert r.json()["buyer_model"] == "claude-opus-4-8"
    assert r.json()["supplier_model"] == "claude-haiku-4-5"
