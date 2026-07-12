"""Live contract-risk scan — LLM extraction feeding the proven anchor + gate pipeline.

This is the live counterpart to :mod:`negotiation_agent.dossier` (the canned example). It swaps the
hand-authored findings for LLM-extracted ones, then runs the IDENTICAL trust tail: verify each
finding's quote against the block it cites, escalate severity with the deterministic rules, compute
the legal gate, and assemble the same ``Dossier`` shape. The LLM says WHAT it found and quotes it
verbatim; deterministic code proves WHERE and owns the severity floor and the gate.

Design (settled by an adversarial design review — see the module tests for the failure modes each
decision defends):
  - **Chunk unit is the anchor Block** — never re-split text — so a quote lands back in the same
    block it was copied from. Blocks are packed into windows under a char budget; the last few
    blocks of a window are carried into the next as CONTEXT-ONLY (shown for continuity, findings
    there are dropped as belonging to the prior window).
  - **One multi-category extraction call per window** (all five lenses in one read), not five
    specialist passes — input tokens dominate, and the deterministic gate backstops severity.
  - **No LLM adjudication pass.** ``escalate_all`` + ``legal_gate`` ARE the adjudicator;
    cross-window dedup is deterministic Python here.
  - **The model never emits a trusted location.** It proposes ``anchor_id`` + a verbatim ``quote``;
    ``anchor.verify_finding`` re-resolves against the CLAIMED block. On quarantine, recovery is
    limited to a UNIQUE exact-substring hit — never a fuzzy whole-document re-resolution, which
    would invent locations and defeat the anchor gate.

Adversarial input: counterparty PDFs are hostile (white-text "classify everything low", injected
``</contract>`` breakout, fabricated citations). Defenses are layered — every block is sanitized
via :func:`llm._sanitize_untrusted` before it enters the prompt (:mod:`scan_client`), severity is
only ever RAISED by deterministic rules, and a fabricated quote simply fails ``verify_finding``.

**Residual (honest):** ``coc`` and ``commercial`` have no deterministic severity floor in
``gate.DEFAULT_RULES``, so injection that SUPPRESSES a coc/commercial finding the model would
otherwise raise has no rule backstop (only the gdpr/legal/infosec floors do). Surfaced here and
covered by an injection eval; a coverage-floor heuristic is a gated fast-follow, not built
speculatively.

Cost is bounded up front: at most ``_MAX_WINDOWS`` extraction calls, each with an explicit output
cap, so the worst-case spend is projectable before any call. The API gates the whole path behind
full mode (fail-closed) so the public demo stays $0.
"""

from __future__ import annotations

import logging
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from . import anchor, gate
from .dossier import Dossier, EconomicBreakdown, assemble_dossier

logger = logging.getLogger(__name__)

# ── bounds ────────────────────────────────────────────────────────────────────────
_CHUNK_CHARS = 24_000  # ~6k tokens of contract text per extraction window
_OVERLAP_BLOCKS = 2  # trailing blocks of window N carried CONTEXT-ONLY into window N+1
_MAX_WINDOWS = 12  # HARD ceiling — a larger contract is rejected BEFORE any paid call
_MAX_TITLE = 160
_MAX_QUOTE = 500

Category = gate.Category
Severity = gate.Severity


class LlmFinding(BaseModel):
    """One finding as the model proposed it. ``severity`` is a PROPOSAL the deterministic rules may
    raise; ``anchor_id`` is UNTRUSTED until :func:`anchor.verify_finding` confirms it. The shape
    mirrors ``dossier._CannedFinding`` so both paths converge on ``gate.RiskFinding``."""

    model_config = {"frozen": True, "extra": "ignore"}

    category: Category
    severity: Severity
    title: str = Field(min_length=1, max_length=_MAX_TITLE)
    quote: str = Field(min_length=1, max_length=_MAX_QUOTE)
    anchor_id: str = ""
    why_it_hurts: str = Field(default="", max_length=600)
    suggested_position: str = Field(default="", max_length=400)
    fallback_position: str = Field(default="", max_length=400)


class ExtractionWindow(BaseModel):
    """A contiguous slice of blocks handed to the model as ONE extraction call.

    ``context_anchor_ids`` are the overlap blocks shown read-only for continuity — any finding the
    model anchors there is dropped (it belongs to the window where those blocks are primary)."""

    model_config = {"frozen": True}

    index: int
    blocks: list[anchor.Block]
    context_anchor_ids: frozenset[str] = frozenset()


class PassStatus(BaseModel):
    """Per-window audit: did this extraction call run or fail? A failed window never silently
    disappears — the scan reports it and assembles from the windows that ran."""

    model_config = {"frozen": True}

    window_index: int
    status: Literal["ran", "failed"]
    findings_count: int = 0
    error_code: str = ""


class ScanUsage(BaseModel):
    model_config = {"frozen": True}

    input_tokens: int = 0
    output_tokens: int = 0
    windows_ran: int = 0
    windows_failed: int = 0


class ScanResult(BaseModel):
    """The scan payload: the ``Dossier`` (same shape as the canned example, ``is_example=False``)
    plus per-window status and token usage so cost + partial failure are observable."""

    model_config = {"frozen": True}

    dossier: Dossier
    passes: list[PassStatus]
    usage: ScanUsage
    run_id: str


class ScanError(Exception):
    """A scan that could not be produced at all — e.g. every window failed transiently."""

    code = "scan_failed"


class ExtractClient(Protocol):
    """The injectable seam (mirrors :class:`llm.DraftClient`). Tests inject a fake and never hit the
    network; production uses ``scan_client.AnthropicExtractClient``. Returns the raw proposals for
    one window along with the tokens that call consumed."""

    def extract_findings(
        self, window: ExtractionWindow, run_id: str
    ) -> tuple[list[LlmFinding], int, int]:  # (findings, input_tokens, output_tokens)
        ...


# ── pure orchestration (testable with a fake client, no LLM) ─────────────────────
def plan_windows(document: anchor.Document) -> list[ExtractionWindow]:
    """Pack blocks into windows under ``_CHUNK_CHARS``, never splitting a block, carrying the last
    ``_OVERLAP_BLOCKS`` of each window into the next as CONTEXT-ONLY. Pure — no LLM.

    A window's primary blocks are the ones findings may be anchored to; its context blocks are
    the prior window's trailing blocks, repeated so a clause split across the boundary reads whole.
    """
    blocks = document.blocks
    if not blocks:
        return []

    windows: list[ExtractionWindow] = []
    start = 0
    index = 0
    while start < len(blocks):
        context = blocks[max(0, start - _OVERLAP_BLOCKS) : start] if start > 0 else []
        context_ids = frozenset(b.anchor_id for b in context)
        size = sum(len(b.text) for b in context)
        end = start
        # always take at least one primary block, then fill until the char budget is hit
        while end < len(blocks) and (end == start or size + len(blocks[end].text) <= _CHUNK_CHARS):
            size += len(blocks[end].text)
            end += 1
        windows.append(
            ExtractionWindow(
                index=index,
                blocks=context + blocks[start:end],
                context_anchor_ids=context_ids,
            )
        )
        start = end
        index += 1
    return windows


def too_large(document: anchor.Document) -> bool:
    """True if the document would plan to more than ``_MAX_WINDOWS`` windows — the endpoint rejects
    it (413) BEFORE any paid call, bounding cost and wall-clock."""
    return len(plan_windows(document)) > _MAX_WINDOWS


def _recover_or_quarantine(document: anchor.Document, finding: LlmFinding) -> anchor.Verdict:
    """Verify a finding against the anchor it CLAIMED. On quarantine, attempt a strictly safe
    recovery: if the verbatim quote is an exact normalized-substring of EXACTLY ONE block, accept
    that block. Zero or multiple matches stay quarantined — we never fuzzy-guess a location, because
    that would let a common fragment anchor to an arbitrary block and defeat the whole gate."""
    verdict = anchor.verify_finding(document, finding.anchor_id, finding.quote)
    if verdict.status == "anchored":
        return verdict
    q = anchor._normalize(finding.quote)
    if len(q) < anchor._MIN_QUOTE_CHARS:
        return verdict
    exact = [b for b in document.blocks if q in anchor._normalize(b.text)]
    if len(exact) == 1:
        return anchor.verify_quote(exact[0], finding.quote)
    return verdict


def _dedup(
    pairs: list[tuple[LlmFinding, anchor.Verdict]],
) -> list[tuple[LlmFinding, anchor.Verdict]]:
    """Collapse cross-window duplicates. Anchored findings key on ``(anchor_id, category)`` — the
    same clause seen under the same lens in two overlapping windows is one finding. Quarantined
    findings key on ``(normalized_quote, category)`` so they still dedup. On collision the higher
    proposed severity wins (a later window shouldn't be able to downgrade an earlier proposal)."""
    best: dict[tuple[str, str], tuple[LlmFinding, anchor.Verdict]] = {}
    order: list[tuple[str, str]] = []
    for finding, verdict in pairs:
        if verdict.status == "anchored" and verdict.anchor_id:
            key = (verdict.anchor_id, finding.category)
        else:
            key = ("q:" + anchor._normalize(finding.quote), finding.category)
        if key not in best:
            best[key] = (finding, verdict)
            order.append(key)
        else:
            kept, _ = best[key]
            if gate._RANK[finding.severity] > gate._RANK[kept.severity]:
                best[key] = (finding, verdict)
    return [best[k] for k in order]


def scan_contract(
    data: bytes,
    client: ExtractClient,
    *,
    policy: gate.GatePolicy | None = None,
    doc_title: str = "uploaded_contract.pdf",
    run_id: str = "scan",
) -> ScanResult:
    """Run the live scan end to end and return a :class:`ScanResult`.

    ``blocks_from_pdf`` → ``plan_windows`` (caller must have rejected ``too_large`` first) → for
    each window (sequential) ``client.extract_findings`` → drop context-block findings → verify on
    its CLAIMED anchor (with unique-exact-substring recovery) → dedup across windows → build the
    same ``Dossier`` the canned example produces via ``assemble_dossier``.

    A window that raises is recorded as a failed :class:`PassStatus` and the scan continues from the
    rest — never a silent short read. Raises :class:`anchor.NoTextLayer`/:class:`anchor.CorruptFile`
    on a bad PDF (before any paid call). Raises :class:`ScanError` only if EVERY window failed.
    """
    document = anchor.blocks_from_pdf(data, doc_id=run_id)
    windows = plan_windows(document)

    raw: list[LlmFinding] = []
    passes: list[PassStatus] = []
    in_tokens = out_tokens = ran = failed = 0
    for window in windows:
        try:
            findings, wi, wo = client.extract_findings(window, run_id)
        except Exception as e:  # noqa: BLE001 - a single window's failure must not sink the scan
            logger.warning("scan run=%s window=%s failed: %s", run_id, window.index, e)
            passes.append(
                PassStatus(window_index=window.index, status="failed", error_code="extract_error")
            )
            failed += 1
            continue
        # drop findings anchored to a context-only block — they belong to the prior window
        kept = [f for f in findings if f.anchor_id not in window.context_anchor_ids]
        raw.extend(kept)
        in_tokens += wi
        out_tokens += wo
        ran += 1
        passes.append(
            PassStatus(window_index=window.index, status="ran", findings_count=len(kept))
        )

    if windows and ran == 0:
        # every window failed transiently — no dossier can be built honestly
        raise ScanError("the scan could not be completed — every extraction window failed")

    verified = [(f, _recover_or_quarantine(document, f)) for f in raw]
    deduped = _dedup(verified)

    findings_pre: list[tuple[gate.RiskFinding, Severity]] = [
        (
            gate.RiskFinding(
                category=f.category,
                severity=f.severity,
                title=f.title,
                quote=f.quote,
                anchor_id=v.anchor_id,
                verified=v.status == "anchored",
                why_it_hurts=f.why_it_hurts,
                suggested_position=f.suggested_position,
                fallback_position=f.fallback_position,
            ),
            f.severity,
        )
        for f, v in deduped
    ]

    dossier = assemble_dossier(
        document,
        findings_pre,
        _empty_economics(),
        doc_title=doc_title,
        is_example=False,
        is_complete=(failed == 0),
        policy=policy,
    )
    return ScanResult(
        dossier=dossier,
        passes=passes,
        usage=ScanUsage(
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            windows_ran=ran,
            windows_failed=failed,
        ),
        run_id=run_id,
    )


def _empty_economics() -> EconomicBreakdown:
    """A zeroed economic breakdown for the live scan — the risk pass does not yet produce pricing,
    but ``Dossier.economics`` is required and ``/dossier/ask`` reads it, so we supply a valid empty
    one rather than leave it unset. The economic-extraction pass is a separate fast-follow."""
    return EconomicBreakdown(
        term_months=0, build_up=[], year1_total_eur=0.0, tco_over_term_eur=0.0, risks=[]
    )
