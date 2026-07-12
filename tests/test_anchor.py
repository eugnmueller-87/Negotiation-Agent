"""The anchoring spine — the trust layer of the due-diligence cockpit.

The whole feature stands on one guarantee: a finding is shown as ``anchored`` iff its verbatim
quote actually appears in the block it cites. These tests prove that guarantee end-to-end against
a real multi-page PDF: a real quote anchors at the right id; a hallucinated quote, a real quote
under the wrong anchor, and a nonexistent anchor all quarantine. If any of these regress, the
tool would show hallucinated citations as trustworthy — the exact failure that kills day-one trust.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fitz")

from negotiation_agent import anchor  # noqa: E402

# A realistic two-page fragment: §12 liability on page 1, §18 indemnity on page 2.
_LIABILITY = (
    "The Supplier's aggregate liability arising out of or in connection with this Agreement, "
    "including any breach of data protection obligations, shall not exceed the fees paid by the "
    "Customer in the three (3) months preceding the claim."
)
_INDEMNITY = "The Customer shall indemnify the Supplier against all claims arising from misuse."
_PAGES = [["12. Limitation of liability", _LIABILITY], ["18. Indemnification", _INDEMNITY]]

# A verbatim fragment of the liability clause — a realistic finding quote.
_LIABILITY_QUOTE = (
    "shall not exceed the fees paid by the Customer "
    "in the three (3) months preceding the claim"
)


@pytest.fixture
def doc(make_blocks_pdf):
    return anchor.blocks_from_pdf(make_blocks_pdf(_PAGES), doc_id="MSA")


def _liability_anchor(doc: anchor.Document) -> str:
    return next(b.anchor_id for b in doc.blocks if "aggregate liability" in b.text)


# ── block extraction ────────────────────────────────────────────────────────────
def test_extracts_one_document_per_page(doc):
    assert doc.page_count == 2


def test_anchor_ids_are_page_and_block_scoped(doc):
    # every anchor id is p{page}-b{block}, page numbers are 1-indexed for humans
    assert all(b.anchor_id.startswith(("p1-", "p2-")) for b in doc.blocks)


def test_char_ranges_locate_the_block_in_the_flat_text(doc):
    # char_start/char_end must slice the block's own text out of the doc-level flat text
    b = next(b for b in doc.blocks if "aggregate liability" in b.text)
    assert doc.text[b.char_start : b.char_end] == b.text


# ── verification: the make-or-break gate ─────────────────────────────────────────
def test_real_quote_anchors_at_the_right_block(doc):
    v = anchor.verify_finding(doc, _liability_anchor(doc), _LIABILITY_QUOTE)
    assert v.status == "anchored" and v.anchor_id == _liability_anchor(doc)


def test_quote_with_linebreak_drift_still_anchors(doc):
    # a model returns the clause with its own line breaks; normalization must not defeat the match
    quote = (
        "The Supplier's aggregate liability\n arising out of "
        "or in connection with   this Agreement"
    )
    v = anchor.verify_finding(doc, _liability_anchor(doc), quote)
    assert v.status == "anchored"


def test_hallucinated_quote_is_quarantined(doc):
    quote = "The Supplier warrants 99.99% uptime with liquidated damages of EUR 10,000 per hour."
    v = anchor.verify_finding(doc, _liability_anchor(doc), quote)
    assert v.status == "quarantined"


def test_real_quote_under_the_wrong_anchor_is_quarantined(doc):
    # the liability quote cited against the indemnity block must NOT anchor — the model
    # can't launder a real quote through a mis-cited location
    indemnity = next(b.anchor_id for b in doc.blocks if "indemnify" in b.text)
    v = anchor.verify_finding(doc, indemnity, _LIABILITY_QUOTE)
    assert v.status == "quarantined"


def test_nonexistent_anchor_is_quarantined(doc):
    v = anchor.verify_finding(doc, "p99-b7", "shall not exceed the fees paid by the Customer")
    assert v.status == "quarantined" and v.reason == "anchor does not exist in document"


def test_too_short_a_quote_is_quarantined_not_fuzzy_matched(doc):
    # a few words appear all over a contract; too short to pin a finding to ONE location,
    # even though this substring is genuinely present in the cited block
    v = anchor.verify_finding(doc, _liability_anchor(doc), "the fees")
    assert v.status == "quarantined"


# ── scanned / empty PDF ──────────────────────────────────────────────────────────
def test_pdf_with_no_text_layer_raises(make_blocks_pdf):
    import fitz

    doc = fitz.open()
    doc.new_page()  # one blank page, no text
    data = doc.tobytes()
    doc.close()
    with pytest.raises(anchor.NoTextLayer):
        anchor.blocks_from_pdf(data)


def test_corrupt_bytes_raise_corrupt_file():
    with pytest.raises(anchor.CorruptFile):
        anchor.blocks_from_pdf(b"this is not a pdf at all")
