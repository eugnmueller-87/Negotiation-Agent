"""Contract PDF → stable, verifiable text anchors — the trust layer of the due-diligence cockpit.

The rule that makes the whole feature defensible: **the LLM says WHAT it found; deterministic
code proves WHERE.** A model that returns "page 14, ¶3" cannot be trusted — it hallucinates
citations. So instead every finding must carry a *verbatim quote*, and this module fuzzy-matches
that quote back against the actual text of the block it claims to come from:

  - match  → the finding is ``anchored`` and gets a working, deep-linkable reference.
  - no match → the finding is ``quarantined`` and is NEVER shown as a normal finding.

This single check is what kills hallucinated citations on day one. It is pure Python, no LLM,
no RNG — the "engine decides, LLM advises" principle doing real work.

Blocks come from PyMuPDF (``fitz``), which returns *geometric* text blocks per page with a
stable per-page ``block_no``. Unlike a flat text dump, that gives every block a stable id
(``p{page}-b{block}``) we can deep-link to. PDF is the canonical format on purpose: DOCX has no
stable page numbers (pagination is a rendering artifact), so anchoring a DOCX finding to a page
would be a lie. Convert to PDF upstream if you need DOCX support.

The file bytes are untrusted input. We cap pages and total characters (same bounds as
:mod:`negotiation_agent.extract_text`), decompress/evaluate nothing from the document, and every
failure is a typed :class:`AnchorError` the API maps to a 4xx.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Literal

from pydantic import BaseModel

# Bounds mirror extract_text.py so the two upload paths refuse the same pathological inputs.
_MAX_PDF_PAGES = 500
_MAX_TEXT_CHARS = 2_000_000

# A block below this many non-whitespace characters is layout noise (a lone bullet, a page
# number) — kept in the flat text but too short to anchor a finding to meaningfully.
_MIN_BLOCK_CHARS = 3

# A PDF yielding less than this much text overall is scanned/image-only — no text layer to anchor.
_MIN_DOC_CHARS = 20

# Verification threshold: the model's quote must match the block's text at least this well.
# 0.82 tolerates whitespace/line-break drift and a truncated "…" tail (the common case where the
# model quotes the first sentence of a long clause) while still rejecting an unrelated quote.
# Substring-after-normalization is checked FIRST and is an exact (ratio 1.0) hit regardless.
_MATCH_THRESHOLD = 0.82

# The shortest verbatim quote we accept. A handful of words ("the Customer") appears many times
# and matches too much to pin a finding to ONE location — a real clause citation is a sentence
# fragment. Below this we quarantine rather than pretend we verified a specific location.
_MIN_QUOTE_CHARS = 24


class Block(BaseModel):
    """One geometric text block of a contract PDF, with a stable, deep-linkable anchor.

    ``anchor_id`` (``p14-b3``) is stable across re-extraction of the same PDF because PyMuPDF's
    per-page ``block_no`` is stable. ``page`` is 0-indexed (internal); ``page_display`` is the
    1-indexed number a human sees ("Page 14 of 47"). ``char_start``/``char_end`` locate this
    block inside the doc-level flat text the LLM is also given, so a finding can be resolved
    either by anchor or by character range.
    """

    model_config = {"frozen": True}

    anchor_id: str
    page: int
    page_display: int
    block_index: int
    char_start: int
    char_end: int
    text: str


class Document(BaseModel):
    """A contract PDF decomposed into anchored blocks plus the flat text (blocks joined)."""

    model_config = {"frozen": True}

    page_count: int
    blocks: list[Block]
    text: str
    truncated: bool = False


class Verdict(BaseModel):
    """The result of verifying a model's quote against the block it claims to cite."""

    model_config = {"frozen": True}

    status: Literal["anchored", "quarantined"]
    anchor_id: str | None = None
    page_display: int | None = None
    score: float = 0.0
    reason: str = ""


class AnchorError(Exception):
    """Base for anchoring failures — carries a human-safe message and a code."""

    code = "anchor_failed"


class NoTextLayer(AnchorError):
    """A PDF with no extractable text — almost certainly scanned/image-only."""

    code = "no_text_layer"


class CorruptFile(AnchorError):
    """The bytes couldn't be parsed as a PDF (truncated, encrypted, or not a PDF)."""

    code = "corrupt_file"


_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Collapse all whitespace runs to single spaces and strip — so line-break and indentation
    differences between the model's quote and the block never cause a spurious mismatch."""
    return _WS_RE.sub(" ", text).strip()


def blocks_from_pdf(data: bytes, *, doc_id: str = "doc") -> Document:
    """Decompose a PDF into anchored text blocks. Raises :class:`NoTextLayer` if there is none.

    Bounded work: at most ``_MAX_PDF_PAGES`` pages, and block collection stops once the flat
    text passes ``_MAX_TEXT_CHARS`` (``truncated`` is then set — never a silent cut). Imports
    ``fitz`` lazily so the core package keeps its single runtime dependency.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise AnchorError("PDF anchoring needs the 'web' extra (pymupdf)") from e

    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as e:  # noqa: BLE001 - fitz raises varied errors on bad input
        raise CorruptFile("the PDF could not be opened — it may be corrupt or encrypted") from e

    try:
        if doc.page_count > _MAX_PDF_PAGES:
            raise CorruptFile(
                f"the PDF has more than {_MAX_PDF_PAGES} pages — split it or paste the key clauses."
            )

        blocks: list[Block] = []
        parts: list[str] = []
        cursor = 0
        truncated = False
        for page_no in range(doc.page_count):
            page = doc.load_page(page_no)
            # (x0, y0, x1, y1, text, block_no, block_type); block_type 0 == text, 1 == image.
            raw = page.get_text("blocks")
            # Sort by block_no so the index in anchor_id is the document's own stable numbering.
            raw.sort(key=lambda b: b[5])
            for block in raw:
                block_type = block[6]
                if block_type != 0:
                    continue  # skip image blocks — nothing to quote or anchor
                block_no = int(block[5])
                block_text = (block[4] or "").strip()
                if len(block_text.replace(" ", "")) < _MIN_BLOCK_CHARS:
                    continue
                start = cursor
                parts.append(block_text)
                cursor += len(block_text) + 2  # +2 for the "\n\n" join added below
                blocks.append(
                    Block(
                        anchor_id=f"p{page_no + 1}-b{block_no}",
                        page=page_no,
                        page_display=page_no + 1,
                        block_index=block_no,
                        char_start=start,
                        char_end=start + len(block_text),
                        text=block_text,
                    )
                )
                if cursor > _MAX_TEXT_CHARS:
                    truncated = True
                    break
            if truncated:
                break

        page_count = doc.page_count
    finally:
        doc.close()

    text = "\n\n".join(parts)
    if len(text.replace(" ", "").replace("\n", "")) < _MIN_DOC_CHARS:
        raise NoTextLayer(
            "no text layer found — this looks like a scanned image. "
            "Upload a text-based PDF, or paste the clauses."
        )
    return Document(page_count=page_count, blocks=blocks, text=text, truncated=truncated)


def _index(document: Document) -> dict[str, Block]:
    """Map anchor_id → Block for O(1) lookup during verification."""
    return {b.anchor_id: b for b in document.blocks}


def verify_quote(block: Block, quote: str) -> Verdict:
    """Verify a model's verbatim quote against ONE block's actual text.

    Returns an ``anchored`` verdict iff the normalized quote is a substring of the normalized
    block text (exact hit) OR fuzzy-matches it at ``>= _MATCH_THRESHOLD`` (tolerates whitespace/
    truncation drift). Otherwise ``quarantined`` — the finding is real content the model claims,
    but we could not prove it lives here, so it must not be shown as an anchored citation.
    """
    q = _normalize(quote)
    if len(q) < _MIN_QUOTE_CHARS:
        return Verdict(status="quarantined", score=0.0, reason="quote too short to verify")

    b = _normalize(block.text)
    if q in b:
        return Verdict(
            status="anchored",
            anchor_id=block.anchor_id,
            page_display=block.page_display,
            score=1.0,
            reason="exact match",
        )

    # Fuzzy: compare the quote against the best-aligned window of the block. SequenceMatcher's
    # ratio over the whole block under-scores a short quote inside a long block, so score the
    # quote against the block's longest matching span instead.
    matcher = SequenceMatcher(None, q, b, autojunk=False)
    match = matcher.find_longest_match(0, len(q), 0, len(b))
    covered = match.size / len(q) if q else 0.0
    if covered >= _MATCH_THRESHOLD:
        return Verdict(
            status="anchored",
            anchor_id=block.anchor_id,
            page_display=block.page_display,
            score=round(covered, 3),
            reason="fuzzy match",
        )
    return Verdict(
        status="quarantined",
        score=round(covered, 3),
        reason="quote not found in cited block",
    )


def verify_finding(document: Document, anchor_id: str, quote: str) -> Verdict:
    """Verify a finding's ``(anchor_id, quote)`` pair against the document.

    The model chose ``anchor_id``; we do NOT trust it. We look up that block and check the quote
    against it. If the anchor doesn't exist, or the quote isn't there, the finding is quarantined
    — a hallucinated anchor and a hallucinated quote both fail the same gate.
    """
    block = _index(document).get(anchor_id)
    if block is None:
        return Verdict(status="quarantined", score=0.0, reason="anchor does not exist in document")
    return verify_quote(block, quote)
