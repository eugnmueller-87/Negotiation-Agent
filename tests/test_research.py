"""Supplier research (Hades client) — mapping + graceful failure.

No live network call is ever made here: the client's ``urlopen`` is monkeypatched
and the mapping is tested against a recorded response fixture. CI needs no API key.
"""

from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path

import pytest

from negotiation_agent.research import (
    HadesClient,
    ResearchUnavailable,
    SupplierBrief,
    brief_from_hades_response,
    sample_brief,
)

FIXTURE = Path(__file__).parent / "fixtures" / "hades_response.json"


@pytest.fixture
def hades_payload() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_brief_maps_the_recorded_response(hades_payload):
    brief = brief_from_hades_response(hades_payload)
    assert brief.company == "Nordwerk Verpackung GmbH"
    assert brief.risk_score == pytest.approx(3.4)
    assert brief.risk_level == "Medium"
    assert brief.recommendation == "Conditional Approval"
    assert brief.sanctioned is False
    assert brief.registry_status == "active"
    assert brief.lksg_signal == "needs_monitoring"
    assert brief.esg_rating == "neutral"
    assert len(brief.next_steps) == 2
    assert brief.source == "hades"


def test_brief_headline_and_blocking(hades_payload):
    brief = brief_from_hades_response(hades_payload)
    assert "Medium" in brief.headline() and "3.4/10" in brief.headline()
    assert brief.is_blocking is False

    blocked = brief.model_copy(update={"recommendation": "Block"})
    assert blocked.is_blocking is True


def test_brief_tolerates_missing_keys():
    # A sparse/degraded response must not raise — fields default to None.
    brief = brief_from_hades_response({"company": "X", "report": {}})
    assert brief.company == "X"
    assert brief.risk_score is None
    assert brief.recommendation is None
    assert brief.next_steps == []


def test_investigate_maps_a_mocked_200(monkeypatch, hades_payload):
    client = HadesClient(api_key="test-key")

    class FakeResp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        # the key must be sent in the header and never elsewhere
        assert req.headers.get("X-api-key") == "test-key"
        return FakeResp(json.dumps(hades_payload).encode("utf-8"))

    monkeypatch.setattr("negotiation_agent.research.urllib.request.urlopen", fake_urlopen)
    brief = client.investigate("Nordwerk Verpackung GmbH", category="Packaging")
    assert brief.risk_level == "Medium"
    assert brief.source == "hades"


def test_investigate_without_key_is_unavailable():
    client = HadesClient(api_key="")
    with pytest.raises(ResearchUnavailable) as e:
        client.investigate("ACME")
    # the message is safe to show a buyer and never leaks the (absent) key
    assert "not configured" in str(e.value).lower()


def test_investigate_rate_limited(monkeypatch):
    client = HadesClient(api_key="k")

    def raise_429(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", {}, None)

    monkeypatch.setattr("negotiation_agent.research.urllib.request.urlopen", raise_429)
    with pytest.raises(ResearchUnavailable) as e:
        client.investigate("ACME")
    assert "rate-limited" in str(e.value).lower()


def test_investigate_unreachable(monkeypatch):
    client = HadesClient(api_key="k")

    def raise_urlerror(req, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("negotiation_agent.research.urllib.request.urlopen", raise_urlerror)
    with pytest.raises(ResearchUnavailable) as e:
        client.investigate("ACME")
    assert "unreachable" in str(e.value).lower()


def test_investigate_bad_json(monkeypatch):
    client = HadesClient(api_key="k")

    class FakeResp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(
        "negotiation_agent.research.urllib.request.urlopen",
        lambda req, timeout=None: FakeResp(b"not json"),
    )
    with pytest.raises(ResearchUnavailable):
        client.investigate("ACME")


def test_error_never_contains_the_key(monkeypatch):
    client = HadesClient(api_key="super-secret-key-value")

    def raise_500(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 500, "boom", {}, None)

    monkeypatch.setattr("negotiation_agent.research.urllib.request.urlopen", raise_500)
    with pytest.raises(ResearchUnavailable) as e:
        client.investigate("ACME")
    assert "super-secret-key-value" not in str(e.value)


def test_sample_brief_is_labelled():
    b = sample_brief()
    assert isinstance(b, SupplierBrief)
    assert b.source == "sample"  # can never be mistaken for a live compliance result
    assert b.risk_level == "Medium"
