"""Demo vs. full mode — the cost-free invariant.

The public/portfolio version runs the real deterministic engine but must make ZERO paid
calls: no Opus/Haiku drafting, no Hades research. The strongest possible check is a drafter
and a Hades client that RAISE if ever touched — in demo mode the endpoints must still succeed,
proving the paid path was never entered. Full mode (the secret PEITHO_FULL_TOKEN) unlocks it.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from negotiation_agent import api  # noqa: E402

FULL_TOKEN = "full-secret-token"

MANDATE = {
    "envelope": {
        "negotiation_id": "n", "version": 1, "signed_by": "e",
        "target_utility": 0.9, "reservation_utility": 0.6,
        "terms": [
            {"name": "price", "term_type": "price", "direction": "minimize",
             "best": 92.0, "worst": 108.0, "weight": 1.0},
        ],
    },
    "supplier_appetite": {"price": 0.15},
    "config": {"max_rounds": 6, "beta": 2.5, "stall_rounds": 3, "on_unknown_terms": "escalate"},
}

CONTRACT = "SUPPLY AGREEMENT. Supplier: Acme GmbH. Unit price EUR 11.50 per unit. Net-30."


class _ExplodingDrafter:
    """A drafter that must NEVER be called in demo mode — it raises if it is."""

    def draft_buyer(self, brief, thread, advice=None, correspondents=None):
        raise AssertionError("LLM drafter was called in demo mode — that would cost money")

    def draft_supplier(self, persona, thread, company, category):
        raise AssertionError("LLM drafter was called in demo mode")


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setenv("PEITHO_MANDATE_SECRET", "t")
    api._rate_hits.clear()
    # any real drafter construction explodes, so a passing demo test PROVES no LLM call happened
    monkeypatch.setattr(api, "draft_client_factory", lambda: _ExplodingDrafter())
    yield
    api._rate_hits.clear()


@pytest.fixture
def client():
    return TestClient(api.app)


# ── demo mode makes no paid call ─────────────────────────────────────────────────
def test_open_in_demo_mode_never_calls_the_llm(client):
    # no full token -> demo mode -> the exploding drafter must never run, yet /open succeeds
    r = client.post("/negotiate/open", json={"mandate": MANDATE, "session_id": "s1"})
    assert r.status_code == 200, r.text
    assert r.json()["turn"]["buyer_message"]  # a real (templated) message came back


def test_step_in_demo_mode_never_calls_the_llm(client):
    signed = client.post(
        "/negotiate/open", json={"mandate": MANDATE, "session_id": "s1"}
    ).json()["signed_mandate"]
    r = client.post(
        "/negotiate/step",
        json={
            "signed_mandate": signed,
            "transcript": {"turns": []},
            "supplier_input": {"mode": "bot", "raw_text": "105 EUR per unit"},
            "session_id": "s1",
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["turn"]["buyer_message"]


def test_intel_in_demo_mode_skips_hades(client, monkeypatch):
    # Hades construction explodes; demo /intel with research=True must still succeed (no call)
    def _boom(*a, **k):
        raise AssertionError("Hades was called in demo mode — that would cost money")

    monkeypatch.setattr(api, "HadesClient", _boom)
    r = client.post("/intel", json={"contract_text": CONTRACT, "research": True})
    assert r.status_code == 200, r.text
    assert r.json()["brief"] is None  # research off in demo


def test_prepare_in_demo_mode_skips_hades(client, monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("Hades was called in demo mode")

    monkeypatch.setattr(api, "HadesClient", _boom)
    r = client.post("/prepare", json={"contract_text": CONTRACT, "research": True})
    assert r.status_code == 200, r.text
    assert r.json()["brief"] is None


# ── full mode unlocks the paid path ──────────────────────────────────────────────
class _FakeDrafter:
    def draft_buyer(self, brief, thread, advice=None, correspondents=None):
        nums = " ".join(f"{v:g}" for v in brief.approved_numbers.values())
        return f"Our position: {nums}."

    def draft_supplier(self, persona, thread, company, category):
        return "..."


def test_full_token_unlocks_the_injected_drafter(client, monkeypatch):
    # with the token, the real (here, fake) drafter IS used — proves the seam flips
    monkeypatch.setenv("PEITHO_FULL_TOKEN", FULL_TOKEN)
    monkeypatch.setattr(api, "draft_client_factory", lambda: _FakeDrafter())
    r = client.post(
        "/negotiate/open",
        json={"mandate": MANDATE, "session_id": "s1"},
        headers={"X-Peitho-Full": FULL_TOKEN},
    )
    assert r.status_code == 200, r.text
    assert "Our position:" in r.json()["turn"]["buyer_message"]  # the fake drafter ran


def test_wrong_full_token_stays_in_demo_mode(client, monkeypatch):
    monkeypatch.setenv("PEITHO_FULL_TOKEN", FULL_TOKEN)
    # exploding drafter still injected; a wrong token must NOT unlock it
    r = client.post(
        "/negotiate/open",
        json={"mandate": MANDATE, "session_id": "s1"},
        headers={"X-Peitho-Full": "wrong"},
    )
    assert r.status_code == 200, r.text  # demo path ran, no explosion


def test_no_token_configured_is_demo_even_with_header(client, monkeypatch):
    # fail-closed: no PEITHO_FULL_TOKEN on the server -> every request is demo, whatever header
    monkeypatch.delenv("PEITHO_FULL_TOKEN", raising=False)
    r = client.post(
        "/negotiate/open",
        json={"mandate": MANDATE, "session_id": "s1"},
        headers={"X-Peitho-Full": "anything"},
    )
    assert r.status_code == 200, r.text  # exploding drafter never ran


def test_health_reports_mode(client, monkeypatch):
    assert client.get("/health").json()["mode"] == "demo"
    monkeypatch.setenv("PEITHO_FULL_TOKEN", FULL_TOKEN)
    r = client.get("/health", headers={"X-Peitho-Full": FULL_TOKEN})
    assert r.json()["mode"] == "full"
