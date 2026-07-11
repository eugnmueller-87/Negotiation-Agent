"""Wire schemas for the negotiation HTTP API — pure pydantic, no web import.

Keeping these here (not in ``api.py``) preserves the invariant that ``api.py`` is
the *only* module that imports a web framework. The FastAPI routes consume and
produce these models; everything here is framework-agnostic and unit-testable
without a server. See ``docs/peitho-v2-architecture.md`` §3.

The negotiation server is **stateless**: the client round-trips the signed mandate
and the transcript, and the server re-derives engine state by folding
``DealEngine.decide`` over the transcript each request. So the wire types below are
the full contract — there is no server-side session store to consult.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from negotiation_agent.brief import MoveBrief
from negotiation_agent.research import SupplierBrief

# ---- The signed mandate (round-tripped verbatim every step) -------------------


class MandateConfig(BaseModel):
    """Engine config the mandate fixes for the whole negotiation."""

    model_config = {"frozen": True}

    max_rounds: int = Field(default=6, ge=1)
    beta: float = Field(default=2.5, ge=1.0)
    stall_rounds: int = Field(default=3, ge=1)
    on_unknown_terms: Literal["escalate", "ignore"] = "escalate"


class MandateEnvelope(BaseModel):
    """Everything the engine is constructed from, so one signature covers it all.

    ``envelope`` and ``supplier_appetite`` reconstruct the ``DealEngine``; the
    signature (in :class:`SignedMandate`) is taken over this whole object.
    """

    model_config = {"frozen": True}

    envelope: dict[str, object]  # a serialized Envelope (validated on load in api.py)
    supplier_appetite: dict[str, float] = Field(default_factory=dict)
    config: MandateConfig = Field(default_factory=MandateConfig)


class SignedMandate(BaseModel):
    """A mandate wrapped with a session-scoped, time-boxed HMAC tag.

    The signature binds ``session_id`` and the ``iat``/``exp`` window *inside* the
    signed payload, so it can't be replayed across sessions to dodge the per-session
    spend cap, and it expires. Verified server-side with ``hmac.compare_digest``.
    """

    model_config = {"frozen": True}

    mandate: MandateEnvelope
    session_id: str = Field(min_length=1)
    iat: int = Field(description="issued-at, server clock (unix seconds)")
    exp: int = Field(description="expiry, server clock (unix seconds)")
    sig: str = Field(min_length=1, description="hex HMAC-SHA256 over the canonical payload")


# ---- The transcript (the fold input) ------------------------------------------


class SupplierTurn(BaseModel):
    """One supplier offer in the transcript, as parsed term values.

    The server re-extracts these from ``raw_text`` server-side; the client copy is
    a display cache and is never trusted for the fold.
    """

    model_config = {"frozen": True}

    terms: dict[str, float]
    raw_text: str = ""


class Transcript(BaseModel):
    """Prior supplier offers, oldest first — the input the server folds over."""

    model_config = {"frozen": True}

    turns: list[SupplierTurn] = Field(default_factory=list)


# ---- Requests -----------------------------------------------------------------


class SupplierInput(BaseModel):
    model_config = {"frozen": True}

    mode: Literal["bot", "human"] = "bot"
    raw_text: str = ""
    persona: Literal["cooperative", "aggressive", "evasive"] = "aggressive"


class BuyerInput(BaseModel):
    model_config = {"frozen": True}

    mode: Literal["human"] = "human"
    raw_text: str


class Correspondents(BaseModel):
    """Who the letter is addressed to and signed by — for a proper salutation + sign-off.

    All optional: with a named ``supplier_contact`` the greeting is personal ("Dear Mr.
    Schmidt,"); with only ``supplier_name`` it's "Dear <company> team,"; with neither it
    falls back to a neutral greeting. None of these are figures, so the guard is unaffected.
    """

    model_config = {"frozen": True}

    supplier_name: str = ""
    supplier_contact: str = ""  # a named person, e.g. "Mr. Schmidt" / "Ms. Rossi"
    buyer_signature: str = "Procurement Team"


class OpenRequest(BaseModel):
    """Start a negotiation: the server signs the mandate and drafts the anchor."""

    model_config = {"frozen": True}

    mandate: MandateEnvelope
    session_id: str = Field(min_length=1)
    supplier_persona: Literal["cooperative", "aggressive", "evasive"] = "aggressive"
    correspondents: Correspondents = Field(default_factory=Correspondents)


class StepRequest(BaseModel):
    """Advance one turn. ``buyer_input`` is present only when a human plays buyer."""

    model_config = {"frozen": True}

    signed_mandate: SignedMandate
    transcript: Transcript = Field(default_factory=Transcript)
    supplier_input: SupplierInput = Field(default_factory=SupplierInput)
    buyer_input: BuyerInput | None = None
    session_id: str = Field(min_length=1)
    correspondents: Correspondents = Field(default_factory=Correspondents)


# ---- Responses ----------------------------------------------------------------


class GuardAttempt(BaseModel):
    model_config = {"frozen": True}

    draft: str
    ok: bool
    violations: list[str] = Field(default_factory=list)


class GuardAudit(BaseModel):
    """The record of the draft→guard→redraft loop for one buyer message."""

    model_config = {"frozen": True}

    released_by: Literal["model", "fallback"]
    attempts: list[GuardAttempt] = Field(default_factory=list)

    @property
    def redrafted(self) -> bool:
        return len(self.attempts) > 1


class InternalState(BaseModel):
    """Buyer-private numbers — released ONLY under the god-view double gate.

    Never serialized into a supplier-facing payload.
    """

    model_config = {"frozen": True}

    threshold: float
    incoming_utility: float | None = None
    counter_utility: float | None = None
    reservation_utility: float
    convergence: list[dict[str, float]] = Field(default_factory=list)


class ConsultedSource(BaseModel):
    """One knowledge-base passage the agent consulted this turn — shown, not secret.

    Carries the source + tag + a short label, never internal figures; it makes the KB
    legible the way give/get makes the trade legible."""

    model_config = {"frozen": True}

    source: str  # the manifest-relative path (e.g. "docs/kb/batna-and-reservation-value.md")
    tag: str
    label: str  # a human title derived from the source


class TurnResult(BaseModel):
    """The decision echo for one buyer turn. ``internal`` is god-view-gated."""

    model_config = {"frozen": True}

    outcome: Literal["accept", "counter", "escalate"]
    round_index: int
    reason_tag: str
    approved_numbers: dict[str, float] = Field(default_factory=dict)
    buyer_message: str
    supplier_message: str = ""
    move_brief: MoveBrief | None = None
    guard: GuardAudit | None = None
    bar_fills: dict[str, float] = Field(default_factory=dict)
    internal: InternalState | None = None
    consulted: list[ConsultedSource] = Field(default_factory=list)


class TranscriptView(BaseModel):
    """A role-scoped view of the thread. ``supplier_view`` is server-redacted."""

    model_config = {"frozen": True}

    turns: list[dict[str, object]] = Field(default_factory=list)


class StepResponse(BaseModel):
    model_config = {"frozen": True}

    buyer_view: TranscriptView
    supplier_view: TranscriptView
    turn: TurnResult
    terminal: bool


class OpenResponse(BaseModel):
    """The signed mandate to round-trip, plus the drafted opening anchor."""

    model_config = {"frozen": True}

    signed_mandate: SignedMandate
    turn: TurnResult
    supplier_brief: SupplierBrief | None = None


class ApiError(BaseModel):
    """The consistent error envelope: ``{"error": {"code", "message"}}``."""

    model_config = {"frozen": True}

    code: str
    message: str
