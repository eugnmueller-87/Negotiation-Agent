"""Shared fixtures."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from negotiation_agent.envelope import Direction, Envelope, TermSpec, TermType


def _build_text_pdf(text: str) -> bytes:
    """A minimal but VALID one-page PDF with a Helvetica font resource and a real text layer.

    Hand-assembled (no reportlab, no pypdf internals) so it's stable across library versions;
    the /Font resource is what makes the text extractable. Shared by the extraction tests.
    """
    escaped = text.replace("\\", "\\\\").replace("(", r"\(").replace(")", r"\)")
    content = f"BT /F1 24 Tf 72 720 Td ({escaped}) Tj ET".encode("latin-1")
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(content), content),
    ]
    pdf = b"%PDF-1.4\n"
    offsets: list[int] = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(pdf))
        pdf += b"%d 0 obj\n%s\nendobj\n" % (i, body)
    xref_pos = len(pdf)
    pdf += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        pdf += b"%010d 00000 n \n" % off
    trailer = b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF"
    pdf += trailer % (len(objs) + 1, xref_pos)
    return pdf


@pytest.fixture
def make_text_pdf() -> Callable[[str], bytes]:
    """Return a builder that turns a string into a valid text-layer PDF (bytes)."""
    return _build_text_pdf


@pytest.fixture
def make_blocks_pdf() -> Callable[[list[list[str]]], bytes]:
    """Return a builder for a multi-page PDF from ``pages`` (a list of pages, each a list of
    block strings). Uses PyMuPDF so each string becomes its own geometric text block — the
    input the anchoring layer decomposes. Skips the whole test module if pymupdf isn't installed.
    """
    fitz = pytest.importorskip("fitz")

    def _build(pages: list[list[str]]) -> bytes:
        doc = fitz.open()
        for blocks in pages:
            page = doc.new_page()
            y = 72.0
            for text in blocks:
                # insert_textbox WRAPS within the rect (insert_text clips at the page edge,
                # truncating long clauses) — so a full paragraph lands intact in one block,
                # the way a real contract PDF stores it. Height grows with the wrapped length.
                lines = max(1, len(text) // 80 + 1)
                height = 16 * lines + 8
                rect = fitz.Rect(72, y, 540, y + height)
                page.insert_textbox(rect, text, fontsize=10)
                y += height + 12  # gap so fitz keeps consecutive paragraphs as distinct blocks
        data: bytes = doc.tobytes()
        doc.close()
        return data

    return _build


@pytest.fixture
def simple_envelope() -> Envelope:
    """Two continuous terms (price heavy, rebate light) — easy to reason about."""
    return Envelope(
        negotiation_id="test",
        version=1,
        signed_by="tester",
        target_utility=0.95,
        reservation_utility=0.55,
        terms=[
            TermSpec(
                name="price",
                term_type=TermType.PRICE,
                direction=Direction.MINIMIZE,
                best=9.0,
                worst=12.0,
                weight=0.7,
            ),
            TermSpec(
                name="rebate_pct",
                term_type=TermType.REBATE_PCT,
                direction=Direction.MAXIMIZE,
                best=8.0,
                worst=0.0,
                weight=0.3,
            ),
        ],
    )


@pytest.fixture
def mixed_envelope() -> Envelope:
    """Continuous + integer terms — exercises the integer snap."""
    return Envelope(
        negotiation_id="test",
        version=1,
        signed_by="tester",
        target_utility=0.95,
        reservation_utility=0.55,
        terms=[
            TermSpec(
                name="price",
                term_type=TermType.PRICE,
                direction=Direction.MINIMIZE,
                best=9.0,
                worst=12.0,
                weight=0.4,
            ),
            TermSpec(
                name="payment_days",
                term_type=TermType.PAYMENT_DAYS,
                direction=Direction.MAXIMIZE,
                best=90,
                worst=30,
                weight=0.3,
            ),
            TermSpec(
                name="volume_units",
                term_type=TermType.VOLUME_UNITS,
                direction=Direction.MINIMIZE,
                best=10000,
                worst=50000,
                weight=0.3,
            ),
        ],
    )
