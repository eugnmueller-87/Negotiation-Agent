"""The HTTP surface — the only module that imports a web framework, and opt-in.

FastAPI + the Anthropic SDK are the ``[web]`` extra, not runtime dependencies;
importing this module without them raises a clear message. Three endpoint groups:

- ``POST /prepare`` — contract → extraction + supplier brief (the v1 pre-flight).
- ``POST /negotiate/open`` — sign the mandate, draft the buyer's opening anchor.
- ``POST /negotiate/step`` — fold the transcript, decide, draft, guard, redraft.

The server is stateless: it reconstructs the engine from the signed mandate and
re-derives state by folding ``decide`` over the transcript each call
(``docs/peitho-v2-architecture.md`` §3.2). Secrets — ``ANTHROPIC_API_KEY``,
``HADES_API_KEY``, ``PEITHO_MANDATE_SECRET`` — live only in the server environment.

Run locally (after ``pip install -e ".[web]"``)::

    uvicorn negotiation_agent.api:app --reload
"""

from __future__ import annotations

import hmac
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from negotiation_agent import scan

try:
    from fastapi import FastAPI, Request, UploadFile
    from fastapi.concurrency import run_in_threadpool
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse
    from pydantic import BaseModel, Field
except ImportError as e:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "The HTTP API needs the 'web' extra. Install it with: pip install -e '.[web]'"
    ) from e

from negotiation_agent.brief import build_move_brief
from negotiation_agent.engine import Outcome
from negotiation_agent.llm import AnthropicDraftClient, DraftClient
from negotiation_agent.mandate_factory import (
    CompiledMandate,
    ContractRow,
    compile_row,
    settled_savings,
)
from negotiation_agent.negotiate import (
    NegotiationClosed,
    build_engine,
    draft_and_guard,
    fold,
    offers_from_transcript,
    raw_offers_from_transcript,
    resolve_supplier_offer,
    turn_result,
)
from negotiation_agent.prepare import PreparedNegotiation, prepare_negotiation
from negotiation_agent.research import HadesClient, ResearchUnavailable
from negotiation_agent.signing import MandateError, sign_mandate, verify_mandate
from negotiation_agent.wire import (
    ApiError,
    GuardAudit,
    OpenRequest,
    OpenResponse,
    ResolveRequest,
    ResolveResponse,
    StepRequest,
    StepResponse,
    TranscriptView,
)

# One module logger. LLM/Hades degradations must leave a server-side trace — a silent
# fallback (200 OK with a canned template forever) hides a rotated key or retired model.
logger = logging.getLogger("negotiation_agent.api")

app = FastAPI(title="Negotiation Agent — negotiation API", version="2.0")

# CORS origins are env-configurable. Set PEITHO_ALLOWED_ORIGINS to a comma-separated
# allowlist (e.g. "https://peitho.example,https://www.peitho.example") to lock the API to
# known front-ends. Unset defaults to "*" so the demo still works opened as a file:// page;
# a public deployment should set the allowlist. The rate limits below are the real cost gate.
_origins_env = os.getenv("PEITHO_ALLOWED_ORIGINS", "").strip()
_allowed_origins = [o.strip() for o in _origins_env.split(",") if o.strip()] or ["*"]
if _allowed_origins == ["*"]:
    logger.warning(
        "CORS is open to all origins — set PEITHO_ALLOWED_ORIGINS to lock the API to your front-end"
    )
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Defense-in-depth security headers on every response (audit #10). The demo is served
# same-origin and renders untrusted contract/supplier text; a CSP contains any future
# escaping gap, nosniff blocks MIME-sniffing, frame-ancestors 'none' blocks clickjacking.
# The page uses one inline <script>, so script-src allows 'unsafe-inline'.
_SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
        "connect-src 'self' *; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
}


@app.middleware("http")
async def _security_headers(request: Request, call_next):  # type: ignore[no-untyped-def]
    response = await call_next(request)
    for name, value in _SECURITY_HEADERS.items():
        response.headers.setdefault(name, value)
    return response


# The v2 demo is served from this same app (same-origin, one deploy, no CORS needed
# in the common case). Path is resolved relative to the repo root at runtime.
_DEMO_HTML = Path(__file__).resolve().parents[2] / "demo" / "peitho-v2.html"

# Abuse gates (env-tunable). In-memory counters are correct for a single instance;
# a multi-replica deploy needs Redis (documented in the architecture doc, §9).
_MANDATE_TTL = int(os.getenv("PEITHO_MANDATE_TTL", "3600"))
# Per-IP requests/min. The default endpoints (draft one Opus message) get the standard cap;
# the research endpoints (a paid Hades call, up to 150s) get a stricter one.
_RATE_PER_MIN = int(os.getenv("PEITHO_RATE_PER_MIN", "20"))
_RESEARCH_RATE_PER_MIN = int(os.getenv("PEITHO_RESEARCH_RATE_PER_MIN", "4"))
# The live contract scan can fan out to several paid Haiku calls per upload — cap it hardest.
_SCAN_RATE_PER_MIN = int(os.getenv("PEITHO_SCAN_RATE_PER_MIN", "2"))
_rate_hits: dict[str, list[float]] = {}


def _drafter() -> DraftClient:
    """Construct the server-side drafter. Overridden in tests via dependency injection."""
    return AnthropicDraftClient()


# Tests replace this to inject a fake drafter (no network). Kept as a module hook so
# api.py stays importable and the fake is swapped without a DI framework.
draft_client_factory = _drafter


def _extractor() -> scan.ExtractClient:
    """Construct the live risk-scan extraction client. Overridden in tests (no network)."""
    from negotiation_agent.scan_client import AnthropicExtractClient

    return AnthropicExtractClient()


# Tests replace this to inject a fake extractor — the same seam pattern as draft_client_factory.
extraction_client_factory = _extractor


def _full_mode(request: Request) -> bool:
    """True only when the caller proves the secret ``PEITHO_FULL_TOKEN`` — unlocking the paid
    LLM/Hades path. DEMO MODE (default) uses the deterministic, network-free path so a public
    portfolio instance runs the real engine with ZERO spend.

    Fail-closed: if no ``PEITHO_FULL_TOKEN`` is configured, EVERY request is demo mode — the
    public URL cannot spend even though the API keys are present on the box. Constant-time
    compare, same pattern as the god-view gate."""
    token = os.getenv("PEITHO_FULL_TOKEN", "")
    if not token:
        return False
    return hmac.compare_digest(request.headers.get("X-Peitho-Full", ""), token)


def _drafter_for(request: Request) -> DraftClient:
    """The drafter for this request: the real (paid) client in full mode, else the deterministic
    templated drafter that makes no network call. This is the single seam that decides whether a
    request can spend on the buyer message."""
    from negotiation_agent.fallback import DeterministicDrafter

    return draft_client_factory() if _full_mode(request) else DeterministicDrafter()


def _secret() -> str:
    secret = os.getenv("PEITHO_MANDATE_SECRET", "")
    if not secret:
        raise MandateError("server is misconfigured (no PEITHO_MANDATE_SECRET)")
    return secret


# Number of trusted reverse proxies between the client and this app. X-Forwarded-For is
# append-only: each proxy APPENDS the peer it saw, so the real client is the entry
# PEITHO_TRUSTED_PROXIES-from-the-RIGHT. Everything to its left is attacker-supplied and must
# never be trusted. Railway terminates at one edge proxy → default 1.
_TRUSTED_PROXIES = max(0, int(os.getenv("PEITHO_TRUSTED_PROXIES", "1")))


def _client_ip(request: Request) -> str:
    """The caller's IP for rate-limit keying — the edge-resolved client, not a spoofable value.

    The old code took the LEFTMOST X-Forwarded-For entry, which is fully client-controlled
    (an attacker prepends a random IP per request → fresh rate bucket → cap bypassed, audit
    SEC-4). XFF is append-only, so the trustworthy client is ``_TRUSTED_PROXIES`` entries from
    the RIGHT: with one trusted proxy (Railway) that's the last entry the proxy itself wrote.
    Falls back to the socket peer when there is no XFF or it's too short to trust.
    """
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd and _TRUSTED_PROXIES > 0:
        parts = [p.strip() for p in fwd.split(",") if p.strip()]
        # Each trusted proxy appended the peer it saw, so the real client is _TRUSTED_PROXIES
        # entries from the right. With one proxy (Railway) that's the last entry — the IP the
        # edge itself wrote — and no attacker-supplied prefix can move it.
        idx = len(parts) - _TRUSTED_PROXIES
        if 0 <= idx < len(parts):
            return parts[idx]
    return request.client.host if request.client else "?"


def _rate_limited(bucket: str, now: float, *, limit: int) -> bool:
    """Sliding 60s window per key. Keys are IP-scoped (never a client-chosen session_id) so
    the cap can't be dodged by rotating ids. Evicts stale keys so the dict can't grow forever."""
    # Opportunistic eviction: drop any key whose entire window has aged out (audit issue #19).
    if len(_rate_hits) > 4096:
        for key in [k for k, ts in _rate_hits.items() if all(now - t >= 60.0 for t in ts)]:
            del _rate_hits[key]
    window = [t for t in _rate_hits.get(bucket, []) if now - t < 60.0]
    window.append(now)
    _rate_hits[bucket] = window
    return len(window) > limit


def _rate_gate(request: Request, route: str, *, limit: int) -> JSONResponse | None:
    """Apply the per-IP rate limit for ``route``; return a 429 response if exceeded, else None.
    Every cost-bearing endpoint calls this before doing any LLM/Hades/parse work."""
    if _rate_limited(f"{route}:{_client_ip(request)}", time.time(), limit=limit):
        return _err("rate_limited", "too many requests — slow down", 429)
    return None


def _err(code: str, message: str, status: int) -> JSONResponse:
    return JSONResponse(
        status_code=status, content={"error": ApiError(code=code, message=message).model_dump()}
    )


class PrepareRequest(BaseModel):
    contract_text: str = Field(max_length=2_000_000)  # matches the intake extractor's cap
    research: bool = True


@app.get("/", response_model=None)
def demo() -> FileResponse | JSONResponse:
    """Serve the v2 demo page at the root, so one URL is both API and UI."""
    if _DEMO_HTML.is_file():
        return FileResponse(_DEMO_HTML, media_type="text/html")
    return JSONResponse(
        content={"service": "Negotiation Agent v2 API", "health": "/health", "demo": "not bundled"}
    )


@app.post("/prepare", response_model=None)
def prepare(req: PrepareRequest, request: Request) -> PreparedNegotiation | JSONResponse:
    """Extract the opening position from a contract and brief the supplier.

    Research is best-effort: a misconfigured Hades (bad URL) must not 500 the whole
    endpoint — the extraction still returns, the brief is just absent (the designed
    ``brief=None`` degradation). So the client is built lazily and only when asked.
    """
    gate = _rate_gate(request, "prepare", limit=_RESEARCH_RATE_PER_MIN)
    if gate is not None:
        return gate
    # Demo mode never calls paid Hades — the extraction still returns, research is just off.
    do_research = req.research and _full_mode(request)
    researcher: HadesClient | None = None
    if do_research:
        try:
            researcher = HadesClient()
        except ResearchUnavailable as e:
            logger.warning("supplier research unavailable at /prepare: %s", e)
    return prepare_negotiation(req.contract_text, researcher=researcher, research=do_research)


@app.get("/dossier", response_model=None)
def dossier_example(request: Request) -> JSONResponse:
    """The due-diligence cockpit's worked example — findings clustered by risk, each with a
    verified page/¶ anchor, severities escalated, and the legal-review gate computed.

    This is the CANNED dossier: a bundled sample contract run through the REAL anchor + gate
    code (so the anchored badges, deep-links, and gate verdict are genuine, not hardcoded), with
    hand-authored findings standing in for the live LLM scan. No LLM, no network — always safe to
    serve in demo mode. The live pipeline will build the same shape from an uploaded PDF.
    """
    gate = _rate_gate(request, "dossier", limit=_RATE_PER_MIN)
    if gate is not None:
        return gate

    from negotiation_agent.dossier import build_dossier

    return JSONResponse(content=build_dossier().model_dump(mode="json"))


class AskRequest(BaseModel):
    question: str = Field(max_length=1000)


@app.post("/dossier/ask", response_model=None)
def dossier_ask(req: AskRequest, request: Request) -> JSONResponse:
    """Ask Opus 4.8 about the dossier's legal + economic risks, grounded in the anchored clauses.

    A PAID path — gated behind full mode (the demo shows a locked box, never calls this). The
    grounding clauses are rebuilt server-side from the canned dossier, NOT taken from the client,
    so the model can only be grounded in real, anchored text. Returns the answer + the anchor ids
    it cited (validated to exist), which the UI turns into deep-links.
    """
    gate = _rate_gate(request, "ask", limit=_RESEARCH_RATE_PER_MIN)
    if gate is not None:
        return gate
    if not _full_mode(request):
        return JSONResponse(
            status_code=403,
            content={"error": {"code": "full_mode_only", "message": "Ask-Opus is available in the "
                     "full version only."}},
        )

    from negotiation_agent.ask import ask_opus
    from negotiation_agent.dossier import build_dossier

    dossier = build_dossier()
    blocks = [b.model_dump(mode="json") for b in dossier.blocks]
    economics = dossier.economics.model_dump(mode="json")
    try:
        result = ask_opus(req.question, blocks, economics)
    except (ImportError, RuntimeError) as e:
        logger.warning("ask-opus unavailable: %s", e)
        return JSONResponse(
            status_code=503,
            content={"error": {"code": "ask_unavailable",
                     "message": "Ask-Opus is not configured."}},
        )
    return JSONResponse(content=result.model_dump(mode="json"))


# A PDF/DOCX upload is bulky (images) even though its text is small; cap the file at
# 20 MB, read in bounded chunks so a lying Content-Length can't exhaust memory.
_MAX_UPLOAD_BYTES = 20 * 1024 * 1024


@app.post("/dossier/scan", response_model=None)
async def dossier_scan(file: UploadFile, request: Request) -> JSONResponse:
    """Live risk scan of an uploaded contract — the paid counterpart to the /dossier example.

    PDF in → LLM extraction across the anchor blocks → every finding's quote verified back to its
    block → severities escalated + legal gate computed → the SAME Dossier shape the canned example
    returns (``is_example=False``). FULL-MODE gated, FAIL-CLOSED: the public demo cannot trigger a
    paid scan. The document is anchored BEFORE any LLM call, and a too-large contract is rejected
    (413) before spend, so worst-case cost is bounded up front.
    """
    gate = _rate_gate(request, "scan", limit=_SCAN_RATE_PER_MIN)
    if gate is not None:
        return gate
    if not _full_mode(request):
        return _err(
            "full_mode_only", "Live contract scan is available in the full version only.", 403
        )

    from starlette.concurrency import run_in_threadpool

    from negotiation_agent import anchor
    from negotiation_agent.scan import ScanError, scan_contract, too_large

    # Bounded read — count what actually arrives, don't trust Content-Length.
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > _MAX_UPLOAD_BYTES:
            return _err("file_too_large", "file exceeds the 20 MB limit", 413)
        chunks.append(chunk)
    data = b"".join(chunks)
    if not data:
        return _err("empty_file", "the uploaded file is empty", 400)

    # Anchor the document FIRST (off the event loop) — a bad/scanned PDF fails here, before spend.
    try:
        document = await run_in_threadpool(anchor.blocks_from_pdf, data)
    except anchor.AnchorError as e:
        return _err(e.code, str(e), 422)
    if too_large(document):
        return _err(
            "contract_too_large",
            "contract is too large to scan — split it or paste the key clauses.",
            413,
        )

    try:
        result = await run_in_threadpool(
            scan_contract,
            data,
            extraction_client_factory(),
            doc_title=file.filename or "contract.pdf",
        )
    except ScanError as e:
        logger.warning("scan failed: %s", e)
        return _err("scan_failed", "the scan could not be completed — try again shortly.", 503)
    return JSONResponse(content=result.model_dump(mode="json"))


@app.post("/extract-text", response_model=None)
async def extract_text_endpoint(file: UploadFile, request: Request) -> JSONResponse:
    """Extract the text from an uploaded contract file (PDF / Word .docx / plain text).

    The frontend sends the raw file; we return ``{"text": ...}`` which it then feeds to
    ``/intel`` — the same path a paste takes. Scanned PDFs (no text layer) and corrupt
    files come back as a typed 4xx with a human-safe message, never a silent empty string.
    """
    gate = _rate_gate(request, "extract-text", limit=_RATE_PER_MIN)
    if gate is not None:
        return gate

    from starlette.concurrency import run_in_threadpool

    from negotiation_agent.extract_text import ExtractError, extract_file

    # Read with a hard byte cap — don't trust Content-Length, count what actually arrives.
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > _MAX_UPLOAD_BYTES:
            return _err("file_too_large", "file exceeds the 20 MB limit", 413)
        chunks.append(chunk)
    data = b"".join(chunks)
    if not data:
        return _err("empty_file", "the uploaded file is empty", 400)

    try:
        # pypdf/python-docx are CPU-bound; run off the event loop so one big PDF can't
        # freeze every other request (including /health) on this single-worker instance.
        result = await run_in_threadpool(extract_file, file.filename or "", data)
    except ExtractError as e:
        # 415 for an unsupported type, 422 for a readable-but-unusable file (scanned/corrupt)
        status = 415 if e.code == "unsupported_file_type" else 422
        return _err(e.code, str(e), status)
    # truncated is surfaced so the human is never told a >2 MB-text read was complete
    warning = (
        "Only the first part of the document was read (it exceeded the analysis limit); "
        "split it or paste the key clauses to analyse the rest."
        if result.truncated
        else ""
    )
    return JSONResponse(
        content={
            "text": result.text,
            "filename": file.filename or "",
            "chars": len(result.text),
            "truncated": result.truncated,
            "warning": warning,
        }
    )


class IntelRequest(BaseModel):
    contract_text: str = Field(max_length=2_000_000)  # matches the intake extractor's cap
    research: bool = True


@app.post("/intel", response_model=None)
def intel(req: IntelRequest, request: Request) -> JSONResponse:
    """Deep-extract a contract, research the supplier, and propose mandate adjustments.

    The expensive, once-per-upload call: regex extraction (Zone A + B) + Hades brief +
    the deterministic rule engine. Returns the intelligence picture, the brief, and the
    proposed (human-reviewable) adjustments. Never shapes silently — the human approves.
    """
    gate = _rate_gate(request, "intel", limit=_RESEARCH_RATE_PER_MIN)
    if gate is not None:
        return gate

    import datetime as _dt

    from negotiation_agent.intelligence import prepare_intelligence

    brief = None
    note = None
    # Demo mode never calls the paid Hades API — the contract extraction (free regex) still
    # runs, only the supplier due-diligence is off. Full mode (token) enables live research.
    do_research = req.research and _full_mode(request)
    if req.research and not _full_mode(request):
        note = "Supplier research is available in the full version only."
    if do_research:
        try:
            brief = HadesClient().investigate(_supplier_from_text(req.contract_text))
        except ResearchUnavailable as e:
            note = str(e)
    result = prepare_intelligence(
        req.contract_text, brief, today=_dt.date.today(), research_note=note
    )
    return JSONResponse(content=result.model_dump(mode="json"))


class ReshapeRequest(BaseModel):
    base_envelope: dict[str, object]  # a serialized Envelope
    adjustments: list[dict[str, object]]  # serialized ProposedAdjustment[]
    accepted_rule_ids: list[str]
    supplier_appetite: dict[str, float] = {}


@app.post("/reshape", response_model=None)
def reshape(req: ReshapeRequest) -> JSONResponse:
    """Apply the accepted adjustments to a base envelope and return the shaped mandate.

    The cheap, pure call every toggle hits — no LLM, no Hades, just the deterministic
    ``apply_adjustments``. Returns the re-validated shaped envelope + appetite, or a
    409 conflict naming the offending rules (never silently drops one).
    """
    from negotiation_agent.envelope import Envelope
    from negotiation_agent.shaper import MandateConflict, ProposedAdjustment, apply_adjustments

    try:
        base = Envelope.model_validate(req.base_envelope)
        all_adj = [ProposedAdjustment.model_validate(a) for a in req.adjustments]
    except (ValueError, KeyError) as e:
        return _err("bad_reshape_input", f"invalid base or adjustments: {e}", 400)

    accepted = [a for a in all_adj if a.rule_id in set(req.accepted_rule_ids)]
    # gates don't touch the envelope; only envelope-affecting deltas go to the applier
    envelope_adj = [a for a in accepted if a.delta.kind != "add_gate"]
    try:
        shaped, appetite = apply_adjustments(base, envelope_adj, req.supplier_appetite)
    except MandateConflict as e:
        return JSONResponse(
            status_code=409,
            content={
                "error": {"code": "mandate_conflict", "message": str(e), "rule_ids": e.rule_ids}
            },
        )
    return JSONResponse(
        content={
            "shaped_envelope": shaped.model_dump(mode="json"),
            "supplier_appetite": appetite,
            "accepted_gates": [
                a.model_dump(mode="json") for a in accepted if a.delta.kind == "add_gate"
            ],
        }
    )


class TerminateRequest(BaseModel):
    contract_text: str = Field(max_length=2_000_000)  # matches the intake extractor's cap
    buyer_name: str = "[Buyer]"
    contract_reference: str | None = None
    intent: str = "non_renewal"  # "non_renewal" | "terminate"
    # Notice periods the regex extractor can't yet reach — human-supplied, optional.
    # Fields the extractor DID find (expiry) always win; these fill the gap only.
    termination_notice_days: int | None = None
    renewal_notice_days: int | None = None
    auto_renews: bool | None = None


@app.post("/terminate", response_model=None)
def terminate(req: TerminateRequest) -> JSONResponse:
    """Compute the termination notice clock from a contract and draft the notice.

    Pure and offline: regex Zone-B extraction → deterministic date math → a templated
    notice grounded only in the contract's own terms. No LLM, no legal ruling — the
    draft carries a mandatory "verify against local law" line. Human-supplied notice
    periods fill gaps the extractor can't reach; extracted facts always win.
    """
    import datetime as _dt

    from negotiation_agent.intelligence import DocumentGrounded, extract_intelligence
    from negotiation_agent.termination import compute_clock, draft_termination_notice

    if req.intent not in ("non_renewal", "terminate"):
        return _err("bad_intent", "intent must be 'non_renewal' or 'terminate'", 400)

    intel = extract_intelligence(req.contract_text)
    lifecycle = intel.lifecycle

    # Merge human-supplied notice terms the extractor couldn't reach. An extracted value
    # (e.g. expiry) is never overwritten; only a None field is filled.
    def _grounded(
        value: object | None, existing: DocumentGrounded | None
    ) -> DocumentGrounded | None:
        if existing is not None:
            return existing  # the contract's own value wins
        if value is None:
            return None
        return DocumentGrounded(value=str(value).lower(), source="regex", assurance="probable")

    if lifecycle is not None or any(
        v is not None
        for v in (req.termination_notice_days, req.renewal_notice_days, req.auto_renews)
    ):
        from negotiation_agent.intelligence import ContractLifecycle

        base = lifecycle or ContractLifecycle()
        lifecycle = ContractLifecycle(
            effective_date=base.effective_date,
            expiration_date=base.expiration_date,
            initial_term_months=base.initial_term_months,
            auto_renews=_grounded(req.auto_renews, base.auto_renews),
            renewal_notice_days=_grounded(req.renewal_notice_days, base.renewal_notice_days),
            termination_notice_days=_grounded(
                req.termination_notice_days, base.termination_notice_days
            ),
        )

    today = _dt.date.today()
    clock = compute_clock(lifecycle, intel.legal, today=today)
    notice = draft_termination_notice(
        clock,
        supplier_name=intel.supplier_name,
        buyer_name=req.buyer_name,
        contract_reference=req.contract_reference,
        today=today,
        intent=req.intent,  # type: ignore[arg-type]  # validated above
    )
    return JSONResponse(
        content={
            "clock": clock.model_dump(mode="json"),
            "notice_draft": notice,
            "supplier_name": intel.supplier_name,
        }
    )


def _detect_context(context: object, supplier_messages: list[str]) -> tuple[str, str]:
    """Auto-detect (procurement category, counterpart register) from all available signal.

    Category comes from the contract text if present, else the free-text hint, else the
    supplier's own words — so a MAVERICK purchase with no contract still gets a category and
    its mapped strategy. Register is read from the supplier's messages (formal by default)."""
    from negotiation_agent.knowledge.category import detect_category
    from negotiation_agent.knowledge.tone import detect_register

    contract_text = getattr(context, "contract_text", "") or ""
    hint = getattr(context, "category_hint", "") or ""
    # signal priority: contract body, else the hint, else what the supplier has said so far
    signal = contract_text or "\n".join(supplier_messages)
    category, _ = detect_category(signal, hint=hint)
    register = detect_register(supplier_messages)
    return category, register


def _correspondents_dict(correspondents: object, register: str) -> dict[str, str]:
    """The correspondents payload for the drafter, with the detected register folded in."""
    data = correspondents.model_dump() if hasattr(correspondents, "model_dump") else {}
    data["register"] = register
    return data


def _supplier_from_text(text: str) -> str:
    """Best-effort supplier name for the Hades lookup (regex; falls back to a label)."""
    from negotiation_agent.intake import extract_contract

    return extract_contract(text).supplier_name or "unknown supplier"


@app.post("/negotiate/open")
def negotiate_open(req: OpenRequest, request: Request) -> JSONResponse:
    """Sign the mandate and draft the buyer's opening anchor."""
    gate = _rate_gate(request, "open", limit=_RATE_PER_MIN)
    if gate is not None:
        return gate
    now = int(time.time())
    try:
        signed = sign_mandate(req.mandate, req.session_id, now, now + _MANDATE_TTL, _secret())
        engine, envelope = build_engine(req.mandate)
    except MandateError as e:
        return _err("misconfigured", str(e), 500)
    except (ValueError, KeyError) as e:
        return _err("bad_mandate", f"mandate is invalid: {e}", 400)

    decision, _, _ = fold(engine, [])  # anchor only
    brief = build_move_brief(
        decision, envelope, None, engine.config.max_rounds, priorities=engine.priorities
    )
    # Auto-detect the category (no contract yet at the anchor -> from the hint/description)
    # and start formal (no counterpart message to read).
    category, register = _detect_context(req.context, [])
    corr = _correspondents_dict(req.correspondents, register)
    try:
        message, audit = draft_and_guard(
            _drafter_for(request), brief, decision.approved_numbers, [], corr, category
        )
    except Exception:  # noqa: BLE001 - degrade to fallback, never leak the LLM error to the client
        # Log it: a persistent fallback (rotated key, retired model) must not be invisible.
        logger.exception("buyer draft failed at /negotiate/open; using deterministic fallback")
        from negotiation_agent.fallback import render_fallback, wrap_letter

        message = wrap_letter(render_fallback(brief), corr)
        audit = GuardAudit(released_by="fallback", attempts=[])

    turn = turn_result(
        decision,
        envelope,
        None,
        message,
        audit,
        "",
        engine.priorities,
        False,
        engine.config.max_rounds,
        category,
        register,
    )
    resp = OpenResponse(signed_mandate=signed, turn=turn)
    return JSONResponse(content=resp.model_dump(mode="json"))


@app.post("/negotiate/step")
def negotiate_step(req: StepRequest, request: Request) -> JSONResponse:
    """Advance one turn: gate, verify, fold, draft, guard, redraft, release."""
    # Key the limit on IP, not the client-chosen session_id (which was trivially rotatable
    # to dodge the cap — audit issue #4). One Opus draft per step, so the standard cap.
    gate = _rate_gate(request, "step", limit=_RATE_PER_MIN)
    if gate is not None:
        return gate
    now = time.time()

    # Resolve the secret OUTSIDE the verify try — a missing secret is a server
    # misconfiguration (500), not client tampering (400). Mirrors /negotiate/open.
    try:
        secret = _secret()
    except MandateError as e:
        return _err("misconfigured", str(e), 500)

    try:
        mandate = verify_mandate(req.signed_mandate, secret, int(now))
    except MandateError as e:
        code = "mandate_expired" if "expired" in str(e) else "mandate_tampered"
        return _err(
            code, "the mandate could not be verified", 400 if code == "mandate_tampered" else 410
        )

    try:
        engine, envelope = build_engine(mandate)
    except (ValueError, KeyError) as e:
        return _err("bad_mandate", f"mandate is invalid: {e}", 400)

    # The fold is bounded by the signed mandate: a transcript can't exceed the prior rounds
    # (max_rounds − 1) plus the current turn. Reject an over-long transcript rather than
    # replaying decide() over padding — the wire cap already bounds it, this makes it exact.
    if len(req.transcript.turns) > engine.config.max_rounds:
        return _err("transcript_too_long", "transcript exceeds the mandate's round budget", 400)

    # Re-extract EVERY prior supplier offer server-side from its raw_text (never trust the
    # client's terms cache — SEC-5), then the current turn the same way.
    prior_offers = offers_from_transcript(req.transcript.turns, envelope)
    supplier_text = req.supplier_input.raw_text
    prev_offer = prior_offers[-1] if prior_offers else None
    new_offer = resolve_supplier_offer(envelope, supplier_text, prev_offer)
    if new_offer is None:
        return _err("offer_unparseable", "couldn't read a price/terms in the supplier message", 422)

    all_offers = [*prior_offers, new_offer]
    try:
        decision, _, prev_counter = fold(engine, all_offers)
    except NegotiationClosed:
        return _err("negotiation_closed", "this negotiation has already ended", 409)

    # Auto-detect category + register from all supplier messages so far (the transcript plus
    # this turn). A maverick purchase with no contract still gets a category from these words.
    supplier_messages = [t.raw_text for t in req.transcript.turns] + [supplier_text]
    category, register = _detect_context(req.context, supplier_messages)
    corr = _correspondents_dict(req.correspondents, register)

    # Draft the buyer message, unless a human is playing the buyer (then guard their text).
    approved = dict(decision.approved_numbers)
    if req.buyer_input is not None:
        from negotiation_agent.guard import check

        violations = check(req.buyer_input.raw_text, approved)
        if violations:
            return _err(
                "buyer_text_off_mandate",
                f"your message states figures the engine didn't approve: {', '.join(violations)}",
                422,
            )
        message = req.buyer_input.raw_text
        # A human wrote this text, not the model — the audit trail must say so (was mislabeled
        # "model", which lied about authorship). resolved_by is unavailable on the step path.
        audit = GuardAudit(released_by="human", attempts=[])
    elif decision.outcome is Outcome.COUNTER or decision.outcome is Outcome.ACCEPT:
        brief = build_move_brief(
            decision, envelope, prev_counter, engine.config.max_rounds, priorities=engine.priorities
        )
        try:
            message, audit = draft_and_guard(
                _drafter_for(request), brief, approved, _thread(req), corr, category
            )
        except Exception:  # noqa: BLE001 - degrade to fallback, never a 500 from the LLM
            logger.exception("buyer draft failed at /negotiate/step; using deterministic fallback")
            from negotiation_agent.fallback import render_fallback, wrap_letter

            msg = wrap_letter(render_fallback(brief), corr)
            message, audit = msg, GuardAudit(released_by="fallback", attempts=[])
    else:  # ESCALATE — figure-free holding note
        brief = build_move_brief(
            decision, envelope, prev_counter, engine.config.max_rounds, priorities=engine.priorities
        )
        from negotiation_agent.fallback import render_fallback, wrap_letter

        msg = wrap_letter(render_fallback(brief), corr)
        message, audit = msg, GuardAudit(released_by="fallback", attempts=[])

    terminal = decision.outcome is not Outcome.COUNTER
    include_internal = _godview(request)
    turn = turn_result(
        decision,
        envelope,
        prev_counter,
        message,
        audit,
        supplier_text,
        engine.priorities,
        include_internal,
        engine.config.max_rounds,
        category,
        register,
    )
    resp = StepResponse(
        buyer_view=_buyer_view(req, message, supplier_text),
        supplier_view=_supplier_view(req, message, supplier_text),
        turn=turn,
        terminal=terminal,
    )
    return JSONResponse(content=resp.model_dump(mode="json"))


@app.post("/negotiate/resolve", response_model=None)
def negotiate_resolve(req: ResolveRequest, request: Request) -> JSONResponse:
    """Human resolution of a negotiation the engine handed off (ESCALATE / deadline).

    Two actions, both $0 (templated, no LLM in any mode):

    - ``approve``: close the deal at the supplier's LAST stated offer. The settlement figures are
      the supplier's OWN raw numbers (re-extracted un-clamped from their message), NOT the engine's
      clamped fold view — accepting must never claim a figure the supplier didn't state. The
      acceptance letter is templated from exactly those figures and passes the guard against them.
      If the raw offer scores below the mandate's reservation floor, ``override_below_floor`` + a
      named actor are REQUIRED: the engine never concedes past reservation, so a below-floor close
      is an explicit, named-human act (released_by="human", resolved_by recorded).

    - ``takeover``: hand composition to the human; the engine stops deciding. Returns the
      acceptance-free handover; the UI takes over. Recorded as a human act.

    Never writes to the OutcomeStore (a below-floor / human close would poison the PII-free priors —
    only engine ACCEPT closes feed learning).
    """
    gate = _rate_gate(request, "resolve", limit=_RATE_PER_MIN)
    if gate is not None:
        return gate
    now = time.time()

    try:
        secret = _secret()
    except MandateError as e:
        return _err("misconfigured", str(e), 500)

    try:
        mandate = verify_mandate(req.signed_mandate, secret, int(now))
    except MandateError as e:
        code = "mandate_expired" if "expired" in str(e) else "mandate_tampered"
        return _err(
            code, "the mandate could not be verified", 400 if code == "mandate_tampered" else 410
        )

    try:
        engine, envelope = build_engine(mandate)
    except (ValueError, KeyError) as e:
        return _err("bad_mandate", f"mandate is invalid: {e}", 400)

    if len(req.transcript.turns) > engine.config.max_rounds:
        return _err("transcript_too_long", "transcript exceeds the mandate's round budget", 400)

    # The transcript must be TERMINAL to resolve — you only hand off a negotiation the engine
    # closed (escalate or deadline-accept). Fold it and assert the last decision is not a live
    # COUNTER. (fold() returns normally on an escalate-terminated transcript; we do NOT catch
    # NegotiationClosed here — that's the "step past a closed deal" error, a different thing.)
    clamped_offers = offers_from_transcript(req.transcript.turns, envelope)
    if not clamped_offers:
        return _err("no_offer_to_resolve", "no supplier offer to resolve", 422)
    try:
        decision, _, _ = fold(engine, clamped_offers)
    except NegotiationClosed:
        return _err("negotiation_closed", "this negotiation has already ended", 409)
    if decision.outcome is Outcome.COUNTER:
        return _err(
            "not_terminal",
            "this negotiation is still live — step it or let it escalate before resolving",
            409,
        )

    category, register = _detect_context(
        req.context, [t.raw_text for t in req.transcript.turns]
    )
    corr = _correspondents_dict(req.correspondents, register)

    if req.action == "takeover":
        # Hand composition to the human — the engine is out. No acceptance figures; the UI's
        # free-compose mode keeps the mechanism-leak screen on but suspends the numeric allowlist.
        audit = GuardAudit(released_by="human", resolved_by=req.resolved_by, attempts=[])
        resp = ResolveResponse(
            action="takeover",
            resolved_by=req.resolved_by,
            message="",
            guard=audit,
        )
        return JSONResponse(content=resp.model_dump(mode="json"))

    # approve: close at the supplier's OWN raw stated offer (never the clamped fold view).
    raw_offers = raw_offers_from_transcript(req.transcript.turns, envelope)
    if not raw_offers:
        return _err("no_offer_to_resolve", "couldn't read the supplier's final offer", 422)
    settled = raw_offers[-1]
    accepted = {n: settled.terms[n] for n in envelope.term_map if n in settled.terms}

    try:
        settled_utility = round(envelope.utility(settled), 4)
    except (KeyError, ValueError):
        settled_utility = None
    below_floor = settled_utility is not None and settled_utility < envelope.reservation_utility

    # Below the floor, the ENGINE would never close — only a named human, explicitly. Require it.
    if below_floor and not req.override_below_floor:
        return _err(
            "below_floor",
            "the supplier's offer is below your reservation floor; approving it requires an "
            "explicit override + your name (the engine never concedes past the floor)",
            409,
        )

    # Templated acceptance from EXACTLY the accepted figures, wrapped as a letter. Because the prose
    # states only the supplier's own accepted numbers, the guard against that allowlist is clean.
    from negotiation_agent.fallback import _ACCEPT_TEMPLATES, _format_figures, wrap_letter

    body = _ACCEPT_TEMPLATES[0].format(figures=_format_figures(accepted))
    message = wrap_letter(body, corr)

    from negotiation_agent.guard import check

    violations = check(message, accepted)
    if violations:  # a defensive belt-and-suspenders; the templated body can't normally violate
        logger.error("resolve acceptance failed its own guard: %s", violations)
        return _err("acceptance_guard_failed", "could not draft a clean acceptance", 500)

    audit = GuardAudit(released_by="human", resolved_by=req.resolved_by, attempts=[])
    resp = ResolveResponse(
        action="approve",
        resolved_by=req.resolved_by,
        accepted_numbers=accepted,
        message=message,
        settled_utility=settled_utility,
        below_floor=below_floor,
        guard=audit,
    )
    return JSONResponse(content=resp.model_dump(mode="json"))


# --------------------------------------------------------------------------- #
# Phase 3 — POST /portfolio/simulate: $0 batch negotiation against a SIMULATED
# counterparty. Deterministic engine + ParametricSupplier only — no LLM call in
# ANY mode, so it is not gated by _full_mode. Cost control is CPU-shaped:
# an N-cap per request plus its own per-IP rate bucket.
# --------------------------------------------------------------------------- #

_PORTFOLIO_MAX_ROWS = int(os.getenv("PEITHO_PORTFOLIO_MAX_ROWS", "200"))
_SIMULATE_RATE_PER_MIN = int(os.getenv("PEITHO_SIMULATE_RATE_PER_MIN", "4"))
# Same schedule the tail-spend fleet demo runs (scripts/gen_tailspend_demo.py).
_SIM_MAX_ROUNDS = 8
_SIM_BETA = 4.0


def _mix(seed: int) -> float:
    """Deterministic float in [0,1) from an integer seed (splitmix64). Same
    scheme as scripts/gen_tailspend_demo.py — the engine forbids RNG, so all
    per-row variation (persona pick) is a pure function of the row index."""
    z = (seed * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    z = z ^ (z >> 31)
    return (z & 0xFFFFFFFF) / 0x100000000


class BatchRow(ContractRow):
    """A ContractRow plus the raw contract text (cancel rows only, for the
    termination clock). Capped per row so a batch can't smuggle a giant body."""

    contract_text: str | None = Field(default=None, max_length=200_000)


class SimulateRequest(BaseModel):
    signed_by: str = Field(min_length=1, max_length=200)
    buyer_name: str = Field(default="[Buyer]", max_length=200)
    # A STATIC max bounds the whole-body pydantic parse before the env-tuned runtime cap
    # runs (FIX 2) — otherwise a 10k-row body is fully parsed before the 413 fires.
    rows: list[BatchRow] = Field(min_length=1, max_length=1000)


class SimulatedRowResult(BaseModel):
    row_id: str
    supplier_name: str | None = None
    action: Literal["negotiated", "terminate_notice", "queued_human_confirm"]
    # A synthetic counterparty is not real money: EVERY row is method="simulated",
    # never "exact" — "exact" is reserved for Phase-1 human-resolved deals.
    method: Literal["simulated"] = "simulated"
    # negotiation fields (action == "negotiated")
    status: str | None = None  # closed_engine | closed_supplier | escalated | walked
    escalated: bool = False
    escalation_reason: str | None = None
    rounds: int | None = None
    persona: str | None = None
    buyer_utility: float | None = None
    settled_terms: dict[str, float] | None = None
    baseline_price: float | None = None
    settled_price: float | None = None
    savings_basis: Literal["exact_vs_baseline", "utility_only", "none"] = "none"
    saving_ratio: float | None = None  # (baseline - settled) / baseline; may be negative
    saved_eur: float | None = None  # annual_spend * saving_ratio; None without a baseline+spend
    # cancel fields (action == "terminate_notice") — cost avoidance is reported in
    # its OWN column and never summed into negotiated savings.
    cost_avoidance_eur: float | None = None
    clock: dict[str, object] | None = None
    notice_draft: str | None = None
    notes: list[str] = Field(default_factory=list)


class SimulateResponse(BaseModel):
    method: Literal["simulated"] = "simulated"
    n_rows: int
    n_negotiated: int
    n_closed: int
    n_escalated: int
    n_queued: int
    n_terminations: int
    total_saved_eur: float  # exact_vs_baseline closes only — honest money
    total_cost_avoidance_eur: float  # cancel rows only — separate headline
    rows: list[SimulatedRowResult]


def _terminate_row(
    row: BatchRow, buyer_name: str, reasons: list[str]
) -> SimulatedRowResult:
    """Cancel instruction: 100% reuse of the /terminate spine (termination.py).
    Cost avoidance = one year of the row's spend, booked ONLY when the notice
    window is not already MISSED — a missed window is not avoidable cost."""
    import datetime as _dt

    from negotiation_agent.intelligence import extract_intelligence
    from negotiation_agent.termination import compute_clock, draft_termination_notice

    notes = list(reasons)
    clock_payload: dict[str, object] | None = None
    notice: str | None = None
    # Cost avoidance is booked ONLY when the exit window is genuinely servable — i.e. the
    # clock says OPEN or CLOSING_SOON. A MISSED window means the next term is locked in; an
    # UNKNOWN / NO_DEADLINE / text-less row has NO evidence the exit is servable. Booking a
    # full year of avoidance in those cases would fabricate a saving (FIX 1, Fable verify).
    window_servable = False
    if row.contract_text:
        intel = extract_intelligence(row.contract_text)
        today = _dt.date.today()
        clock = compute_clock(intel.lifecycle, intel.legal, today=today)
        notice = draft_termination_notice(
            clock,
            supplier_name=intel.supplier_name or row.supplier_name,
            buyer_name=buyer_name,
            contract_reference=row.row_id,
            today=today,
            intent="non_renewal",
        )
        clock_payload = clock.model_dump(mode="json")
        window_servable = clock.window_status in ("OPEN", "CLOSING_SOON")
        if not window_servable:
            notes.append(
                f"notice window is {clock.window_status} — no evidence the exit is servable in "
                "time, so NO cost avoidance is booked (verify the clock manually)"
            )
    else:
        notes.append(
            "no contract text supplied — POST the contract to /terminate for the "
            "notice clock; no cost avoidance booked without a servable window"
        )

    if window_servable and row.annual_spend_eur:
        avoidance = float(row.annual_spend_eur)
        notes.append(
            "cost avoidance = ONE year of this row's spend, assuming the non-renewal is served "
            "in time — the avoided term may differ; verify the clock before relying on it"
        )
    else:
        avoidance = None
    return SimulatedRowResult(
        row_id=row.row_id,
        supplier_name=row.supplier_name,
        action="terminate_notice",
        cost_avoidance_eur=avoidance,
        clock=clock_payload,
        notice_draft=notice,
        notes=notes,
    )


def _negotiate_row(
    index: int, row: BatchRow, compiled: CompiledMandate
) -> SimulatedRowResult:
    """One real engine-vs-bot negotiation, savings derived from the settled price."""
    from negotiation_agent.engine import DealEngine, EngineConfig
    from negotiation_agent.simulator.loop import run_negotiation
    from negotiation_agent.simulator.personas import AGGRESSIVE, COOPERATIVE, EVASIVE
    from negotiation_agent.simulator.supplier import ParametricSupplier
    from negotiation_agent.supplier_model import SupplierModel

    buyer_env = compiled.buyer_envelope
    sup_env = compiled.supplier_envelope
    # Persona mix mirrors the tail-spend fleet (mostly cooperative), picked
    # deterministically per row index so the whole batch replays bit-identically.
    pool = [COOPERATIVE] * 6 + [EVASIVE] * 3 + [AGGRESSIVE] * 2
    persona = pool[int(_mix(index * 3 + 2) * len(pool))]
    cfg = EngineConfig(max_rounds=_SIM_MAX_ROUNDS, beta=_SIM_BETA)
    # Uniform belief only: production has no oracle into a real supplier's head,
    # so the simulation must not grant itself one (that would inflate capture).
    engine = DealEngine(buyer_env, SupplierModel.uniform(buyer_env), cfg)
    supplier = ParametricSupplier(sup_env, persona)
    result = run_negotiation(
        buyer_env,
        engine,
        supplier,
        supplier_envelope=sup_env,
        persona_name=persona.name,
        belief_source="uniform",
        config=cfg,
    )

    notes = list(compiled.reasons)
    closed = result.status in ("closed_engine", "closed_supplier")
    out: dict[str, object] = {
        "row_id": row.row_id,
        "supplier_name": row.supplier_name,
        "action": "negotiated",
        "status": result.status,
        "escalated": result.status == "escalated",
        "escalation_reason": result.escalation_reason,
        "rounds": result.rounds_used,
        "persona": persona.name,
        "baseline_price": row.baseline_price,
    }
    if closed and result.final_deal is not None:
        buyer_u = buyer_env.utility(result.final_deal)
        settled_price = result.final_deal.terms.get("price")
        out["buyer_utility"] = round(buyer_u, 4)
        out["settled_terms"] = {k: round(v, 4) for k, v in result.final_deal.terms.items()}
        out["settled_price"] = round(settled_price, 4) if settled_price is not None else None
        if compiled.price_scaled and row.baseline_price and settled_price is not None:
            ratio, eur = settled_savings(
                baseline_price=row.baseline_price,
                settled_price=settled_price,
                annual_spend_eur=row.annual_spend_eur,
            )
            out["savings_basis"] = "exact_vs_baseline"
            out["saving_ratio"] = ratio
            out["saved_eur"] = eur
            if eur is None:
                notes.append("no annual spend supplied — saving reported as a ratio only")
            if ratio < 0:
                notes.append(
                    "settled above baseline (a price increase inside the signed ±% band) — "
                    "negative saving reported honestly"
                )
        else:
            out["savings_basis"] = "utility_only"
            notes.append("no baseline price — utility-only result, no EUR derived")
    elif result.status == "escalated":
        notes.append("escalated to a human buyer — no savings booked")
    else:  # walked
        notes.append("synthetic supplier walked away — no deal, no savings booked")
    out["notes"] = notes
    return SimulatedRowResult(**out)


def _simulate_batch(req: SimulateRequest) -> dict[str, object]:
    """CPU-bound batch body — runs inside run_in_threadpool. Pure Python, $0."""
    rows: list[SimulatedRowResult] = []
    for i, row in enumerate(req.rows):
        compiled = compile_row(row, signed_by=req.signed_by)
        if compiled.route == "terminate":
            rows.append(_terminate_row(row, req.buyer_name, compiled.reasons))
        elif compiled.route == "human_confirm":
            rows.append(
                SimulatedRowResult(
                    row_id=row.row_id,
                    supplier_name=row.supplier_name,
                    action="queued_human_confirm",
                    notes=compiled.reasons,
                )
            )
        else:
            rows.append(_negotiate_row(i, row, compiled))

    negotiated = [r for r in rows if r.action == "negotiated"]
    response = SimulateResponse(
        n_rows=len(rows),
        n_negotiated=len(negotiated),
        n_closed=sum(1 for r in negotiated if r.status in ("closed_engine", "closed_supplier")),
        n_escalated=sum(1 for r in negotiated if r.escalated),
        n_queued=sum(1 for r in rows if r.action == "queued_human_confirm"),
        n_terminations=sum(1 for r in rows if r.action == "terminate_notice"),
        total_saved_eur=round(sum(r.saved_eur or 0.0 for r in negotiated), 2),
        total_cost_avoidance_eur=round(
            sum(r.cost_avoidance_eur or 0.0 for r in rows if r.action == "terminate_notice"), 2
        ),
        rows=rows,
    )
    return response.model_dump(mode="json")


@app.post("/portfolio/simulate", response_model=None)
async def portfolio_simulate(req: SimulateRequest, request: Request) -> JSONResponse:
    """Negotiate a whole contract portfolio against a SIMULATED counterparty. $0.

    Per row: compile the instruction into a baseline-scaled mandate
    (mandate_factory), run the real deterministic engine vs the parametric
    supplier bot, and derive savings from the settled price vs the row's own
    baseline. "cancel" rows reuse the /terminate spine and report cost
    avoidance separately. Every result is method="simulated" — a synthetic
    counterparty never earns "exact". No LLM call exists on this path in any
    mode, so it is not gated by _full_mode; it is N-capped and rate-limited
    because the cost here is CPU, not tokens.
    """
    gate = _rate_gate(request, "simulate", limit=_SIMULATE_RATE_PER_MIN)
    if gate is not None:
        return gate
    if len(req.rows) > _PORTFOLIO_MAX_ROWS:
        return _err(
            "too_many_rows",
            f"portfolio batch is capped at {_PORTFOLIO_MAX_ROWS} rows per request",
            413,
        )
    payload = await run_in_threadpool(_simulate_batch, req)
    return JSONResponse(content=payload)


@app.get("/health")
def health(request: Request) -> dict[str, object]:
    """Liveness + the effective mode for this caller (demo vs full). Anonymous callers get
    only ``{"status", "mode"}``; the diagnostic detail (model IDs, KB size — fingerprinting
    aids) is released only under the god-view token (audit #12). ``mode`` is not sensitive —
    it just tells the UI whether to show the honest "templated, no AI" banner."""
    mode = "full" if _full_mode(request) else "demo"
    if not _godview(request):
        return {"status": "ok", "mode": mode}

    from negotiation_agent.knowledge.retrieve import _load_index
    from negotiation_agent.llm import BUYER_MODEL, SUPPLIER_MODEL

    kb = _load_index()
    return {
        "status": "ok",
        "mode": mode,
        "buyer_model": BUYER_MODEL,
        "supplier_model": SUPPLIER_MODEL,
        "mandate_configured": bool(os.getenv("PEITHO_MANDATE_SECRET")),
        "knowledge_chunks": kb.n_docs if kb is not None else 0,
    }


def _thread(req: StepRequest) -> list[dict[str, str]]:
    """The recent conversation, role-labelled, for the drafter."""
    thread: list[dict[str, str]] = []
    for t in req.transcript.turns:
        thread.append({"role": "supplier", "text": t.raw_text})
    if req.supplier_input.raw_text:
        thread.append({"role": "supplier", "text": req.supplier_input.raw_text})
    return thread


def _buyer_view(req: StepRequest, buyer_message: str, supplier_text: str) -> TranscriptView:
    turns: list[dict[str, object]] = [
        {"role": "supplier", "text": t.raw_text} for t in req.transcript.turns
    ]
    turns.append({"role": "supplier", "text": supplier_text})
    turns.append({"role": "buyer", "text": buyer_message})
    return TranscriptView(turns=turns)


def _supplier_view(req: StepRequest, buyer_message: str, supplier_text: str) -> TranscriptView:
    # Same messages, no buyer-internal fields — nothing private to redact at the message level.
    return _buyer_view(req, buyer_message, supplier_text)


def _godview(request: Request) -> bool:
    """Release the buyer-internal block (threshold, reservation floor, utilities) only to a
    caller that proves a SERVER-SIDE SECRET — never on a client-controllable header alone.

    The old gate was ``header == "1" AND DEMO_GODVIEW == "1"``; the header is public (the demo
    hardcodes it), so it collapsed to the env flag and leaked ``reservation_utility`` — the
    buyer's walk-away floor — to any caller (audit SEC-3). Now the client must send the exact
    ``PEITHO_GODVIEW_TOKEN`` in the ``X-Peitho-Godview`` header, compared in constant time.
    Fail-closed: no token configured → god-view is OFF, so an internet-exposed instance with
    no token set can never release internals, whatever headers arrive.
    """
    token = os.getenv("PEITHO_GODVIEW_TOKEN", "")
    if not token:
        return False
    supplied = request.headers.get("X-Peitho-Godview", "")
    return hmac.compare_digest(supplied, token)
