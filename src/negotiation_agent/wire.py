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

# Bounds on untrusted request strings. Every free-text field is an injection/DoS surface:
# it can reach the regex extractor, the ~233-signal category classifier, or the LLM prompt.
# A supplier message is a paragraph; a contract body is bounded by the same 2 MB the intake
# extractor caps at; the transcript can't exceed the mandate's max_rounds + the anchor.
_MAX_MESSAGE_CHARS = 20_000  # one supplier/buyer message
_MAX_CONTRACT_CHARS = 2_000_000  # a pasted contract body (matches intake's cap)
_MAX_LABEL_CHARS = 200  # a name, category hint, signature
_MAX_TRANSCRIPT_TURNS = 128  # > any sane max_rounds; the fold input is bounded by this

# ---- The signed mandate (round-tripped verbatim every step) -------------------


class MandateConfig(BaseModel):
    """Engine config the mandate fixes for the whole negotiation."""

    model_config = {"frozen": True}

    # Upper bounds: the mandate is client-authored, and the stateless fold replays decide()
    # over the whole transcript each step (O(n) per request). Without a ceiling a client
    # could sign max_rounds=10**7 and drive minutes of CPU per request — bound it here.
    max_rounds: int = Field(default=6, ge=1, le=64)
    beta: float = Field(default=2.5, ge=1.0, le=100.0)
    stall_rounds: int = Field(default=3, ge=1, le=64)
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
    raw_text: str = Field(default="", max_length=_MAX_MESSAGE_CHARS)


class Transcript(BaseModel):
    """Prior supplier offers, oldest first — the input the server folds over."""

    model_config = {"frozen": True}

    turns: list[SupplierTurn] = Field(default_factory=list, max_length=_MAX_TRANSCRIPT_TURNS)


# ---- Requests -----------------------------------------------------------------


class SupplierInput(BaseModel):
    model_config = {"frozen": True}

    mode: Literal["bot", "human"] = "bot"
    raw_text: str = Field(default="", max_length=_MAX_MESSAGE_CHARS)
    persona: Literal["cooperative", "aggressive", "evasive"] = "aggressive"


class BuyerInput(BaseModel):
    model_config = {"frozen": True}

    mode: Literal["human"] = "human"
    raw_text: str = Field(max_length=_MAX_MESSAGE_CHARS)


class Correspondents(BaseModel):
    """Who the letter is addressed to and signed by — for a proper salutation + sign-off.

    All optional: with a named ``supplier_contact`` the greeting is personal ("Dear Mr.
    Schmidt,"); with only ``supplier_name`` it's "Dear <company> team,"; with neither it
    falls back to a neutral greeting. None of these are figures, so the guard is unaffected.
    """

    model_config = {"frozen": True}

    supplier_name: str = Field(default="", max_length=_MAX_LABEL_CHARS)
    supplier_contact: str = Field(default="", max_length=_MAX_LABEL_CHARS)  # "Mr. Schmidt"
    buyer_signature: str = Field(default="Procurement Team", max_length=_MAX_LABEL_CHARS)


class NegotiationContext(BaseModel):
    """Free-text signal the server uses to auto-detect the procurement category — the
    contract body and/or the category label. Never a human-picked category (fully
    automatic); the server classifies it. All optional."""

    model_config = {"frozen": True}

    # capped: this text is re-scanned by the ~233-signal category classifier every step
    contract_text: str = Field(default="", max_length=_MAX_CONTRACT_CHARS)
    category_hint: str = Field(default="", max_length=_MAX_LABEL_CHARS)


class OpenRequest(BaseModel):
    """Start a negotiation: the server signs the mandate and drafts the anchor."""

    model_config = {"frozen": True}

    mandate: MandateEnvelope
    session_id: str = Field(min_length=1)
    supplier_persona: Literal["cooperative", "aggressive", "evasive"] = "aggressive"
    correspondents: Correspondents = Field(default_factory=Correspondents)
    context: NegotiationContext = Field(default_factory=NegotiationContext)


class StepRequest(BaseModel):
    """Advance one turn. ``buyer_input`` is present only when a human plays buyer."""

    model_config = {"frozen": True}

    signed_mandate: SignedMandate
    transcript: Transcript = Field(default_factory=Transcript)
    supplier_input: SupplierInput = Field(default_factory=SupplierInput)
    buyer_input: BuyerInput | None = None
    session_id: str = Field(min_length=1)
    correspondents: Correspondents = Field(default_factory=Correspondents)
    context: NegotiationContext = Field(default_factory=NegotiationContext)


# ---- Responses ----------------------------------------------------------------


class GuardAttempt(BaseModel):
    model_config = {"frozen": True}

    draft: str
    ok: bool
    violations: list[str] = Field(default_factory=list)


class GuardAudit(BaseModel):
    """The record of the draft→guard→redraft loop for one buyer message.

    ``released_by`` names who authored the released text: the LLM (``model``), the
    deterministic template (``fallback``), or a named human (``human`` — a human
    playing buyer, or a human resolving an escalation). ``resolved_by`` carries that
    human's actor label on a human-authored close, so the audit trail never lies
    about who wrote what.
    """

    model_config = {"frozen": True}

    released_by: Literal["model", "fallback", "human"]
    resolved_by: str = ""  # actor label when released_by == "human"; empty otherwise
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
    category: str = "unknown"  # detected procurement category
    category_label: str = ""  # human label for the category
    counterpart_tone: str = "formal"  # detected register (formal/informal)
    coverage_gap: str = ""  # non-empty when the KB has no playbook for this category


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


# ---- Human resolution of a terminal (escalated / deadline) negotiation --------


class ResolveRequest(BaseModel):
    """A human closing out a negotiation the engine handed off (ESCALATE or deadline).

    ``approve`` accepts the supplier's LAST stated offer as-is — the deal closes at the
    figures the supplier themselves stated (re-extracted raw from their message, never
    the engine's clamped view). ``takeover`` flips the project to human-led: the engine
    stops deciding and the human composes freely under their own mandate authority.

    ``override_below_floor`` is required when the supplier's raw offer scores below the
    mandate's reservation floor: the ENGINE never concedes past reservation, so a
    below-floor close is an explicit, named-human act. ``resolved_by`` records who.
    """

    model_config = {"frozen": True}

    signed_mandate: SignedMandate
    transcript: Transcript = Field(default_factory=Transcript)
    session_id: str = Field(min_length=1)
    action: Literal["approve", "takeover"]
    resolved_by: str = Field(min_length=1, max_length=_MAX_LABEL_CHARS)
    override_below_floor: bool = False
    correspondents: Correspondents = Field(default_factory=Correspondents)
    context: NegotiationContext = Field(default_factory=NegotiationContext)


class ResolveResponse(BaseModel):
    """The result of a human resolution.

    ``approve``: ``accepted_numbers`` are the supplier's own raw stated figures (the
    settlement), ``message`` a templated acceptance letter that passes the guard against
    exactly those figures, ``settled_utility`` where the deal landed on the buyer's scale
    (may be below reservation — then ``below_floor`` is true and the override was required).
    ``takeover``: ``message`` is empty and the UI takes over composition; ``below_floor``
    reflects the last offer for context.
    """

    model_config = {"frozen": True}

    action: Literal["approve", "takeover"]
    resolved_by: str
    accepted_numbers: dict[str, float] = Field(default_factory=dict)
    message: str = ""
    settled_utility: float | None = None
    below_floor: bool = False
    guard: GuardAudit | None = None


class ApiError(BaseModel):
    """The consistent error envelope: ``{"error": {"code", "message"}}``."""

    model_config = {"frozen": True}

    code: str
    message: str
