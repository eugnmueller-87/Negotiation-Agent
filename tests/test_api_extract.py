"""The /extract-text endpoint — upload a PDF/DOCX and get back the negotiable text.

Proves the upload → text path the frontend uses so a PDF flows into /intel. Failure
cases (scanned PDF, unsupported type) return a typed 4xx, never a silent empty string.
"""

from __future__ import annotations

import io

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("pypdf")

from fastapi.testclient import TestClient  # noqa: E402

from negotiation_agent import api  # noqa: E402

CONTRACT = "SUPPLY AGREEMENT. Supplier: Acme GmbH. Unit price EUR 11.50 per unit."


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    api._rate_hits.clear()  # module-global counters — isolate the per-IP cap per test
    yield
    api._rate_hits.clear()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("PEITHO_MANDATE_SECRET", "t")
    return TestClient(api.app)


def test_extract_pdf_returns_text(client, make_text_pdf):
    files = {"file": ("contract.pdf", make_text_pdf(CONTRACT), "application/pdf")}
    r = client.post("/extract-text", files=files)
    assert r.status_code == 200, r.text
    d = r.json()
    assert "Acme GmbH" in d["text"]
    assert d["filename"] == "contract.pdf"
    assert d["chars"] > 0


def test_scanned_pdf_is_422_not_empty(client):
    # a blank page (no text layer) -> the endpoint reports it, never returns "" as success
    from pypdf import PdfWriter

    w = PdfWriter()
    w.add_blank_page(width=595, height=842)
    buf = io.BytesIO()
    w.write(buf)
    files = {"file": ("scan.pdf", buf.getvalue(), "application/pdf")}
    r = client.post("/extract-text", files=files)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "no_text_layer"


def test_unsupported_type_is_415(client):
    r = client.post("/extract-text", files={"file": ("logo.png", b"\x89PNG\r\n", "image/png")})
    assert r.status_code == 415
    assert r.json()["error"]["code"] == "unsupported_file_type"


def test_empty_file_is_400(client):
    r = client.post("/extract-text", files={"file": ("empty.txt", b"", "text/plain")})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "empty_file"


def test_plaintext_passthrough(client):
    r = client.post("/extract-text", files={"file": ("c.txt", CONTRACT.encode(), "text/plain")})
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == CONTRACT
    assert body["truncated"] is False
    assert body["warning"] == ""


def test_oversized_upload_reports_truncation(client):
    from negotiation_agent.extract_text import _MAX_TEXT_CHARS

    big = ("x" * (_MAX_TEXT_CHARS + 100)).encode("utf-8")
    r = client.post("/extract-text", files={"file": ("big.txt", big, "text/plain")})
    assert r.status_code == 200
    body = r.json()
    assert body["truncated"] is True  # the cut is surfaced, not silent
    assert body["warning"]  # a human-readable message is present
    assert len(body["text"]) == _MAX_TEXT_CHARS
