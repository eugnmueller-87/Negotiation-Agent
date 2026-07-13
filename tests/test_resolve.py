"""The /negotiate/resolve endpoint — a human closing out an escalated negotiation.

Gated on FastAPI (the ``[web]`` extra). No network, no LLM: resolve is templated in every
mode. Proves: approve closes at the supplier's OWN raw figures (never the clamped fold view),
a below-floor approve needs an explicit override + actor, the acceptance passes the guard
against the supplier's allowlist, takeover records a human handover, a still-live negotiation
can't be resolved, and no OutcomeStore write happens on a human close.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from negotiation_agent import api  # noqa: E402

SECRET = "test-mandate-secret"

# A price-heavy mandate: worst price is 108, reservation 0.60. A supplier holding at/below the
# worst end escalates, giving us a terminal transcript to resolve.
MANDATE = {
    "envelope": {
        "negotiation_id": "neg-r",
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
                "weight": 0.6,
            },
            {
                "name": "payment_days",
                "term_type": "payment_days",
                "direction": "maximize",
                "best": 60.0,
                "worst": 30.0,
                "weight": 0.2,
            },
            {
                "name": "contract_months",
                "term_type": "contract_months",
                "direction": "minimize",
                "best": 12.0,
                "worst": 24.0,
                "weight": 0.2,
            },
        ],
    },
    "supplier_appetite": {"price": 0.15, "payment_days": 0.85, "contract_months": 0.70},
    "config": {"max_rounds": 2, "beta": 2.5, "stall_rounds": 2, "on_unknown_terms": "escalate"},
}


class _FakeDrafter:
    def draft_buyer(self, brief, thread, advice=None, correspondents=None):
        nums = " ".join(f"{v:g}" for v in brief.approved_numbers.values())
        return f"Our position: {nums}."

    def draft_supplier(self, persona, thread, company, category):
        return "..."


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    api._rate_hits.clear()
    yield
    api._rate_hits.clear()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("PEITHO_MANDATE_SECRET", SECRET)
    monkeypatch.setattr(api, "draft_client_factory", lambda: _FakeDrafter())
    return TestClient(api.app)


def _signed(client):
    r = client.post("/negotiate/open", json={"mandate": MANDATE, "session_id": "s1"})
    assert r.status_code == 200, r.text
    return r.json()["signed_mandate"]


# A supplier holding firm at a below-floor position across both rounds → escalate at the deadline.
_FIRM_OFFER = "We can do €107.00 per unit, net-30, 24 months. That's our best."
_ESCALATING_TRANSCRIPT = {"turns": [{"terms": {}, "raw_text": _FIRM_OFFER}] * 2}


def _resolve(client, signed, **over):
    body = {
        "signed_mandate": signed,
        "transcript": _ESCALATING_TRANSCRIPT,
        "session_id": "s1",
        "action": "approve",
        "resolved_by": "e.mueller",
        **over,
    }
    return client.post("/negotiate/resolve", json=body)


def test_approve_closes_at_supplier_raw_figures(client):
    signed = _signed(client)
    r = _resolve(client, signed, override_below_floor=True)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["action"] == "approve"
    # the settlement is the supplier's OWN stated 107.00 — not the clamped 108 (envelope worst)
    assert body["accepted_numbers"]["price"] == 107.0
    assert body["resolved_by"] == "e.mueller"
    assert body["guard"]["released_by"] == "human"
    assert body["guard"]["resolved_by"] == "e.mueller"


def test_acceptance_message_states_only_the_supplier_figures(client):
    signed = _signed(client)
    r = _resolve(client, signed, override_below_floor=True)
    msg = r.json()["message"]
    # the acceptance names the supplier's own numbers and reads like a real letter
    assert "107" in msg
    assert "net-30" in msg
    # never the clamped worst (108) the supplier never stated
    assert "108" not in msg


def test_below_floor_approve_requires_override(client):
    signed = _signed(client)
    r = _resolve(client, signed)  # no override
    assert r.status_code == 409, r.text
    assert r.json()["error"]["code"] == "below_floor"


def test_below_floor_utility_is_gated_behind_god_view(client):
    # settled_utility is buyer-private and below_floor is a floor oracle: an UNGATED caller gets
    # neither (the server still enforces the override from its private value — that's tested above).
    signed = _signed(client)
    body = _resolve(client, signed, override_below_floor=True).json()
    assert body["settled_utility"] is None
    assert body["below_floor"] is False


def test_god_view_reveals_the_below_floor_utility(client, monkeypatch):
    # WITH the god-view token, the buyer-private figure is revealed (the same rule the /step path
    # applies to InternalState) — proving the gate, not a permanent suppression.
    monkeypatch.setenv("PEITHO_GODVIEW_TOKEN", "gv-secret")
    signed = _signed(client)
    r = client.post(
        "/negotiate/resolve",
        json={
            "signed_mandate": signed,
            "transcript": _ESCALATING_TRANSCRIPT,
            "session_id": "s1",
            "action": "approve",
            "resolved_by": "e.mueller",
            "override_below_floor": True,
        },
        headers={"X-Peitho-Godview": "gv-secret"},
    )
    body = r.json()
    assert body["below_floor"] is True
    assert body["settled_utility"] is not None and body["settled_utility"] < 0.60


def test_takeover_hands_off_to_the_human(client):
    signed = _signed(client)
    r = _resolve(client, signed, action="takeover")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["action"] == "takeover"
    assert body["message"] == ""  # the UI composes; the engine is out
    assert body["guard"]["released_by"] == "human"
    assert body["guard"]["resolved_by"] == "e.mueller"
    assert body["accepted_numbers"] == {}


def test_cannot_resolve_a_still_live_negotiation(client):
    signed = _signed(client)
    # a single in-band offer at round 1 is a COUNTER, not terminal
    r = client.post(
        "/negotiate/resolve",
        json={
            "signed_mandate": signed,
            "transcript": {
                "turns": [{"terms": {}, "raw_text": "We can do €100.00, net-45, 18 months."}]
            },
            "session_id": "s1",
            "action": "approve",
            "resolved_by": "e.mueller",
        },
    )
    assert r.status_code == 409, r.text
    assert r.json()["error"]["code"] == "not_terminal"


def test_resolve_requires_an_actor_name(client):
    signed = _signed(client)
    r = client.post(
        "/negotiate/resolve",
        json={
            "signed_mandate": signed,
            "transcript": _ESCALATING_TRANSCRIPT,
            "session_id": "s1",
            "action": "approve",
            "resolved_by": "",  # empty actor is a validation error (min_length=1)
        },
    )
    assert r.status_code == 422  # pydantic rejects the empty actor


def test_resolve_does_not_write_to_outcome_store(client, monkeypatch):
    # a human/below-floor close must NEVER feed the PII-free learning store (would poison priors)
    from negotiation_agent import outcomes

    calls = []
    monkeypatch.setattr(outcomes.OutcomeStore, "append", lambda self, o: calls.append(o))
    signed = _signed(client)
    r = _resolve(client, signed, override_below_floor=True)
    assert r.status_code == 200
    assert calls == []  # nothing written


def test_resolve_rejects_session_id_not_matching_the_mandate(client):
    # a top-level session_id that differs from the signed mandate's is rejected (the field must
    # enforce a check, not be dead surface). The HMAC already blocks real replay; this is d-i-d.
    signed = _signed(client)  # bound to session "s1"
    r = client.post(
        "/negotiate/resolve",
        json={
            "signed_mandate": signed,
            "transcript": _ESCALATING_TRANSCRIPT,
            "session_id": "s-DIFFERENT",
            "action": "approve",
            "resolved_by": "e.mueller",
            "override_below_floor": True,
        },
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "session_mismatch"
