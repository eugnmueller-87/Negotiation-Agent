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

import os
import time

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
except ImportError as e:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "The HTTP API needs the 'web' extra. Install it with: pip install -e '.[web]'"
    ) from e

from negotiation_agent.brief import build_move_brief
from negotiation_agent.engine import Outcome
from negotiation_agent.llm import AnthropicDraftClient, DraftClient
from negotiation_agent.negotiate import (
    NegotiationClosed,
    build_engine,
    draft_and_guard,
    fold,
    offers_from_transcript,
    resolve_supplier_offer,
    turn_result,
)
from negotiation_agent.prepare import PreparedNegotiation, prepare_negotiation
from negotiation_agent.research import HadesClient
from negotiation_agent.signing import MandateError, sign_mandate, verify_mandate
from negotiation_agent.wire import (
    ApiError,
    GuardAudit,
    OpenRequest,
    OpenResponse,
    StepRequest,
    StepResponse,
    TranscriptView,
)

app = FastAPI(title="Negotiation Agent — negotiation API", version="2.0")

# Abuse gates (env-tunable). In-memory counters are correct for a single instance;
# a multi-replica deploy needs Redis (documented in the architecture doc, §9).
_MANDATE_TTL = int(os.getenv("PEITHO_MANDATE_TTL", "3600"))
_RATE_PER_MIN = int(os.getenv("PEITHO_RATE_PER_MIN", "20"))
_rate_hits: dict[str, list[float]] = {}


def _drafter() -> DraftClient:
    """Construct the server-side drafter. Overridden in tests via dependency injection."""
    return AnthropicDraftClient()


# Tests replace this to inject a fake drafter (no network). Kept as a module hook so
# api.py stays importable and the fake is swapped without a DI framework.
draft_client_factory = _drafter


def _secret() -> str:
    secret = os.getenv("PEITHO_MANDATE_SECRET", "")
    if not secret:
        raise MandateError("server is misconfigured (no PEITHO_MANDATE_SECRET)")
    return secret


def _rate_limited(session_id: str, now: float) -> bool:
    window = [t for t in _rate_hits.get(session_id, []) if now - t < 60.0]
    window.append(now)
    _rate_hits[session_id] = window
    return len(window) > _RATE_PER_MIN


def _err(code: str, message: str, status: int) -> JSONResponse:
    return JSONResponse(
        status_code=status, content={"error": ApiError(code=code, message=message).model_dump()}
    )


class PrepareRequest(BaseModel):
    contract_text: str
    research: bool = True


@app.post("/prepare", response_model=PreparedNegotiation)
def prepare(req: PrepareRequest) -> PreparedNegotiation:
    """Extract the opening position from a contract and brief the supplier."""
    return prepare_negotiation(req.contract_text, researcher=HadesClient(), research=req.research)


@app.post("/negotiate/open")
def negotiate_open(req: OpenRequest) -> JSONResponse:
    """Sign the mandate and draft the buyer's opening anchor."""
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
    try:
        message, audit = draft_and_guard(
            draft_client_factory(), brief, decision.approved_numbers, []
        )
    except Exception:  # noqa: BLE001 - degrade to fallback, never leak the LLM error
        from negotiation_agent.fallback import render_fallback

        message = render_fallback(brief)
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
    )
    resp = OpenResponse(signed_mandate=signed, turn=turn)
    return JSONResponse(content=resp.model_dump(mode="json"))


@app.post("/negotiate/step")
def negotiate_step(req: StepRequest, request: Request) -> JSONResponse:
    """Advance one turn: gate, verify, fold, draft, guard, redraft, release."""
    now = time.time()
    ip = request.client.host if request.client else "?"
    if _rate_limited(f"{req.session_id}:{ip}", now):
        return _err("rate_limited", "too many requests — slow down", 429)

    try:
        mandate = verify_mandate(req.signed_mandate, _secret(), int(now))
    except MandateError as e:
        code = "mandate_expired" if "expired" in str(e) else "mandate_tampered"
        return _err(
            code, "the mandate could not be verified", 400 if code == "mandate_tampered" else 410
        )

    try:
        engine, envelope = build_engine(mandate)
    except (ValueError, KeyError) as e:
        return _err("bad_mandate", f"mandate is invalid: {e}", 400)

    # Re-extract the supplier's offer server-side (never trust the client's parse).
    prior_offers = offers_from_transcript(req.transcript.turns)
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
        audit = GuardAudit(released_by="model", attempts=[])
    elif decision.outcome is Outcome.COUNTER or decision.outcome is Outcome.ACCEPT:
        brief = build_move_brief(
            decision, envelope, prev_counter, engine.config.max_rounds, priorities=engine.priorities
        )
        try:
            message, audit = draft_and_guard(draft_client_factory(), brief, approved, _thread(req))
        except Exception:  # noqa: BLE001 - degrade to fallback, never a 500 from the LLM
            from negotiation_agent.fallback import render_fallback

            message, audit = render_fallback(brief), GuardAudit(released_by="fallback", attempts=[])
    else:  # ESCALATE — figure-free holding note
        brief = build_move_brief(
            decision, envelope, prev_counter, engine.config.max_rounds, priorities=engine.priorities
        )
        from negotiation_agent.fallback import render_fallback

        message, audit = render_fallback(brief), GuardAudit(released_by="fallback", attempts=[])

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
    )
    resp = StepResponse(
        buyer_view=_buyer_view(req, message, supplier_text),
        supplier_view=_supplier_view(req, message, supplier_text),
        turn=turn,
        terminal=terminal,
    )
    return JSONResponse(content=resp.model_dump(mode="json"))


@app.get("/health")
def health() -> dict[str, object]:
    from negotiation_agent.llm import BUYER_MODEL, SUPPLIER_MODEL

    return {
        "status": "ok",
        "buyer_model": BUYER_MODEL,
        "supplier_model": SUPPLIER_MODEL,
        "mandate_configured": bool(os.getenv("PEITHO_MANDATE_SECRET")),
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
    """The double gate: client header AND server env must both agree."""
    return request.headers.get("X-Peitho-Godview") == "1" and os.getenv("DEMO_GODVIEW") == "1"
