"""The /dossier/scan endpoint — the paid live-scan path, gated so the demo stays $0.

The headline guarantee: in DEMO mode the extraction client is NEVER constructed or called — proven
by injecting a factory that RAISES if touched and asserting the endpoint still returns 403. The
token flips the seam; a scanned/no-text PDF fails (422) before any extraction; the rate limit bites.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("fitz")

from fastapi.testclient import TestClient  # noqa: E402

from negotiation_agent import api, scan  # noqa: E402
from negotiation_agent.dossier import _render_sample_pdf  # noqa: E402

FULL_TOKEN = "scan-full-token"


class _ExplodingExtractor:
    """Must never be constructed OR called in demo mode — raises if it is."""

    def __init__(self):
        raise AssertionError("extractor constructed in demo mode — that would cost money")


class _FakeExtractor:
    """A stand-in extractor: finds one anchored finding in the sample's DPA clause."""

    def extract_findings(self, window, run_id):
        dpa = next((b for b in window.blocks if "sub-processors may be engaged" in b.text), None)
        if dpa is None:
            return [], 10, 5
        quote = "sub-processors may be engaged by the Supplier as required to deliver the services"
        f = scan.LlmFinding(
            category="gdpr", severity="low", title="No DPA present",
            quote=quote, anchor_id=dpa.anchor_id,
        )
        return [f], 100, 50


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setenv("PEITHO_MANDATE_SECRET", "t")
    api._rate_hits.clear()
    yield
    api._rate_hits.clear()


@pytest.fixture
def client():
    return TestClient(api.app)


def _upload(pdf=None):
    return {"file": ("contract.pdf", pdf or _render_sample_pdf(), "application/pdf")}


# ── the $0-demo guarantee ────────────────────────────────────────────────────────
def test_demo_mode_is_403_and_never_builds_the_extractor(client, monkeypatch):
    # the factory explodes if called; a passing 403 proves demo mode never touched the paid path
    monkeypatch.setattr(api, "extraction_client_factory", lambda: _ExplodingExtractor())
    r = client.post("/dossier/scan", files=_upload())
    assert r.status_code == 403 and r.json()["error"]["code"] == "full_mode_only"


def test_no_full_token_configured_is_demo_even_with_header(client, monkeypatch):
    monkeypatch.delenv("PEITHO_FULL_TOKEN", raising=False)
    monkeypatch.setattr(api, "extraction_client_factory", lambda: _ExplodingExtractor())
    r = client.post("/dossier/scan", files=_upload(), headers={"X-Peitho-Full": "anything"})
    assert r.status_code == 403


# ── full mode runs the (injected) scan ───────────────────────────────────────────
def test_full_token_runs_the_injected_scan(client, monkeypatch):
    monkeypatch.setenv("PEITHO_FULL_TOKEN", FULL_TOKEN)
    monkeypatch.setattr(api, "extraction_client_factory", lambda: _FakeExtractor())
    r = client.post("/dossier/scan", files=_upload(), headers={"X-Peitho-Full": FULL_TOKEN})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dossier"]["is_example"] is False
    # the injected finding anchored and the gdpr rule escalated it to critical
    findings = body["dossier"]["findings"]
    assert any(f["severity"] == "critical" and f["category"] == "gdpr" for f in findings)


def test_scanned_pdf_fails_before_any_extraction(client, monkeypatch):
    # a no-text PDF must 422 at the anchor step, before the extractor is ever built
    import fitz

    doc = fitz.open()
    doc.new_page()  # blank, no text
    blank = doc.tobytes()
    doc.close()
    monkeypatch.setenv("PEITHO_FULL_TOKEN", FULL_TOKEN)
    monkeypatch.setattr(api, "extraction_client_factory", lambda: _ExplodingExtractor())
    r = client.post(
        "/dossier/scan", files=_upload(blank), headers={"X-Peitho-Full": FULL_TOKEN}
    )
    assert r.status_code == 422 and r.json()["error"]["code"] == "no_text_layer"


def test_empty_file_is_rejected(client, monkeypatch):
    monkeypatch.setenv("PEITHO_FULL_TOKEN", FULL_TOKEN)
    monkeypatch.setattr(api, "extraction_client_factory", lambda: _ExplodingExtractor())
    r = client.post(
        "/dossier/scan",
        files={"file": ("empty.pdf", b"", "application/pdf")},
        headers={"X-Peitho-Full": FULL_TOKEN},
    )
    assert r.status_code == 400 and r.json()["error"]["code"] == "empty_file"
