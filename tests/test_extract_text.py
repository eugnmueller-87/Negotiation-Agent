"""File → text extraction — PDF, DOCX, plaintext, and the honest-failure cases.

Real files, not mocks: a genuine PDF is built with pypdf and a genuine .docx with
python-docx, so the tests exercise actual parsing. The failure cases (scanned PDF with
no text layer, corrupt file, unsupported type) are the point — they must RAISE a typed
error, never hand back a silent empty string.
"""

from __future__ import annotations

import io

import pytest

# The extraction libs live in the [web] extra; skip the whole module without them.
pytest.importorskip("pypdf")
pytest.importorskip("docx")

from negotiation_agent.extract_text import (  # noqa: E402
    CorruptFile,
    NoTextLayer,
    UnsupportedFileType,
    extract_docx,
    extract_file,
    extract_pdf,
)

CONTRACT_LINE = "SUPPLY AGREEMENT. Supplier: Acme GmbH. Unit price EUR 11.50 per unit. Net-30."


def _make_docx(paragraphs: list[str]) -> bytes:
    import docx

    d = docx.Document()
    for p in paragraphs:
        d.add_paragraph(p)
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


# ── PDF ────────────────────────────────────────────────────────────────────────
def test_extract_pdf_reads_the_text_layer(make_text_pdf):
    result = extract_pdf(make_text_pdf(CONTRACT_LINE))
    assert "Acme GmbH" in result.text
    assert "11.50" in result.text
    assert result.truncated is False


def test_scanned_pdf_raises_no_text_layer():
    # a blank page with no content stream = the scanned/image case: no text to extract
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=595, height=842)
    buf = io.BytesIO()
    writer.write(buf)
    with pytest.raises(NoTextLayer):
        extract_pdf(buf.getvalue())


def test_corrupt_pdf_raises_corrupt_file():
    with pytest.raises(CorruptFile):
        extract_pdf(b"%PDF-1.4 this is not a real pdf body")


# ── DOCX ───────────────────────────────────────────────────────────────────────
def test_extract_docx_reads_paragraphs():
    result = extract_docx(_make_docx(["SUPPLY AGREEMENT", CONTRACT_LINE]))
    assert "Acme GmbH" in result.text
    assert "SUPPLY AGREEMENT" in result.text


def test_corrupt_docx_raises_corrupt_file():
    with pytest.raises(CorruptFile):
        extract_docx(b"PK\x03\x04 not really a docx zip")


# ── routing / plaintext ──────────────────────────────────────────────────────────
def test_extract_file_routes_pdf_by_extension(make_text_pdf):
    assert "Acme GmbH" in extract_file("contract.pdf", make_text_pdf(CONTRACT_LINE)).text


def test_extract_file_routes_docx_by_extension():
    assert "Acme GmbH" in extract_file("contract.docx", _make_docx([CONTRACT_LINE])).text


def test_extract_file_decodes_plaintext():
    assert extract_file("c.txt", CONTRACT_LINE.encode("utf-8")).text == CONTRACT_LINE


def test_oversized_text_is_flagged_truncated():
    from negotiation_agent.extract_text import _MAX_TEXT_CHARS

    # a plain-text file past the cap must come back truncated=True — never a silent cut that
    # the /extract-text response would report as a complete read (audit issue #3).
    result = extract_file("big.txt", ("x" * (_MAX_TEXT_CHARS + 100)).encode("utf-8"))
    assert result.truncated is True
    assert len(result.text) == _MAX_TEXT_CHARS


def test_extract_file_rejects_unknown_type():
    with pytest.raises(UnsupportedFileType):
        extract_file("logo.png", b"\x89PNG\r\n\x1a\n")


def test_extract_file_plaintext_with_nul_is_corrupt():
    with pytest.raises(CorruptFile):
        extract_file("weird.txt", b"real text\x00\x00binary tail")
