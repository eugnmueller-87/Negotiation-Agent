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


class _FakeResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _patch_opener(monkeypatch, *, returns=None, raises=None, on_open=None):
    """Patch build_opener so opener.open() returns a body, raises, or runs a hook.

    The client uses ``build_opener(_NoRedirect).open(...)``, so this is the seam
    the mocks must intercept — not the module-level ``urlopen``.
    """

    class FakeOpener:
        def open(self, req, timeout=None):
            if on_open is not None:
                on_open(req)
            if raises is not None:
                raise raises(req)
            return _FakeResp(returns if returns is not None else b"{}")

    monkeypatch.setattr(
        "negotiation_agent.research.urllib.request.build_opener", lambda *a: FakeOpener()
    )


def test_investigate_maps_a_mocked_200(monkeypatch, hades_payload):
    client = HadesClient(api_key="test-key")
    seen = {}

    def check_header(req):
        seen["key"] = req.headers.get("X-api-key")

    _patch_opener(
        monkeypatch, returns=json.dumps(hades_payload).encode("utf-8"), on_open=check_header
    )
    brief = client.investigate("Nordwerk Verpackung GmbH", category="Packaging")
    assert seen["key"] == "test-key"  # key sent only in the header
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
    _patch_opener(
        monkeypatch,
        raises=lambda req: urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", {}, None),
    )
    with pytest.raises(ResearchUnavailable) as e:
        client.investigate("ACME")
    assert "rate-limited" in str(e.value).lower()


def test_investigate_unreachable(monkeypatch):
    client = HadesClient(api_key="k")
    _patch_opener(monkeypatch, raises=lambda req: urllib.error.URLError("connection refused"))
    with pytest.raises(ResearchUnavailable) as e:
        client.investigate("ACME")
    assert "unreachable" in str(e.value).lower()


def test_investigate_bad_json(monkeypatch):
    client = HadesClient(api_key="k")
    _patch_opener(monkeypatch, returns=b"not json")
    with pytest.raises(ResearchUnavailable):
        client.investigate("ACME")


def test_error_never_contains_the_key(monkeypatch):
    client = HadesClient(api_key="super-secret-key-value")
    _patch_opener(
        monkeypatch,
        raises=lambda req: urllib.error.HTTPError(req.full_url, 500, "boom", {}, None),
    )
    with pytest.raises(ResearchUnavailable) as e:
        client.investigate("ACME")
    assert "super-secret-key-value" not in str(e.value)


def test_sample_brief_is_labelled():
    b = sample_brief()
    assert isinstance(b, SupplierBrief)
    assert b.source == "sample"  # can never be mistaken for a live compliance result
    assert b.risk_level == "Medium"


# ---- security hardening (Fable 5 review) ----


def test_non_https_url_is_rejected():
    # http/file/gopher must never carry the API key.
    for bad in ("http://hades.example", "file:///etc/passwd", "gopher://x"):
        with pytest.raises(ResearchUnavailable):
            HadesClient(base_url=bad, api_key="k")


def test_redirect_is_refused_not_followed(monkeypatch):
    # A redirect on the authenticated POST would forward X-API-Key to the target.
    # The no-redirect opener turns it into an HTTPError → ResearchUnavailable.
    client = HadesClient(api_key="secret")
    _patch_opener(
        monkeypatch,
        raises=lambda req: urllib.error.HTTPError(req.full_url, 308, "redirect refused", {}, None),
    )
    with pytest.raises(ResearchUnavailable) as e:
        client.investigate("ACME")
    assert "secret" not in str(e.value)


def test_non_object_json_is_unavailable(monkeypatch):
    client = HadesClient(api_key="k")
    _patch_opener(monkeypatch, returns=b"[1, 2, 3]")  # valid JSON, but not an object
    with pytest.raises(ResearchUnavailable):
        client.investigate("ACME")


def test_deeply_nested_json_does_not_crash(monkeypatch):
    client = HadesClient(api_key="k")
    bomb = (b"[" * 100_000) + (b"]" * 100_000)  # RecursionError, not JSONDecodeError
    _patch_opener(monkeypatch, returns=bomb)
    with pytest.raises(ResearchUnavailable):
        client.investigate("ACME")


def test_oversized_response_is_rejected(monkeypatch):
    client = HadesClient(api_key="k")
    huge = b'{"company":"' + (b"a" * (6 * 1024 * 1024)) + b'"}'
    _patch_opener(monkeypatch, returns=huge)
    with pytest.raises(ResearchUnavailable) as e:
        client.investigate("ACME")
    assert "oversized" in str(e.value).lower()


def test_overflow_and_nonfinite_scores_are_dropped():
    # a giant int or "nan"/"inf" in a scored field must never reach risk_score
    for bad in ("1e999", "nan", "inf", "9" * 400):
        brief = brief_from_hades_response({"report": {"overall_risk_score": bad}})
        assert brief.risk_score is None
