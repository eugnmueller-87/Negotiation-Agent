"""Stateless negotiation orchestration — the decide → draft → guard → redraft loop.

This is the product's core made real on the server: the engine decides, the model
drafts, and a violating draft **cannot** reach the caller — it is redrafted or
replaced by a deterministic template. Pure and framework-free (no web import), so it
runs under test with a fake :class:`~negotiation_agent.llm.DraftClient` and no network.

The server holds no state: it reconstructs the ``DealEngine`` from the signed mandate
and re-derives the negotiation by folding ``decide`` over the transcript each call
(``docs/peitho-v2-architecture.md`` §3.2). ``api.py`` wraps this in HTTP.
"""

from __future__ import annotations

from negotiation_agent.brief import MoveBrief, build_move_brief
from negotiation_agent.engine import (
    DealEngine,
    EngineConfig,
    EngineDecision,
    NegotiationState,
    Outcome,
)
from negotiation_agent.envelope import Envelope, Offer
from negotiation_agent.fallback import build_redraft_instruction, render_fallback
from negotiation_agent.guard import check
from negotiation_agent.intake import extract_contract
from negotiation_agent.knowledge.retrieve import advice_lines
from negotiation_agent.llm import DraftClient
from negotiation_agent.supplier_model import SupplierModel
from negotiation_agent.wire import (
    GuardAttempt,
    GuardAudit,
    InternalState,
    MandateEnvelope,
    SupplierTurn,
    TurnResult,
)

_MAX_REDRAFTS = 2


class NegotiationClosed(Exception):
    """The last decision was terminal — no further turns may be taken."""


def build_engine(mandate: MandateEnvelope) -> tuple[DealEngine, Envelope]:
    """Reconstruct the deterministic engine from a signed mandate."""
    envelope = Envelope.model_validate(mandate.envelope)
    if mandate.supplier_appetite:
        supplier_model = SupplierModel(appetite=dict(mandate.supplier_appetite))
    else:
        supplier_model = SupplierModel.uniform(envelope)
    config = EngineConfig(**mandate.config.model_dump())
    return DealEngine(envelope, supplier_model, config), envelope


def fold(
    engine: DealEngine, supplier_offers: list[Offer]
) -> tuple[EngineDecision, NegotiationState, Offer | None]:
    """Replay the negotiation: anchor, then each prior supplier offer.

    Returns ``(decision, state, prev_counter)`` for the LAST turn — the decision to
    act on, the state after it, and the counter that was on the table *before* it
    (the move-brief diff baseline). Raises :class:`NegotiationClosed` if a prior
    (non-final) decision was already terminal.
    """
    state = NegotiationState()
    decision, state = engine.decide(state, None)  # round-0 anchor
    prev_counter = None
    for i, offer in enumerate(supplier_offers):
        prev_counter = state.last_counter
        decision, state = engine.decide(state, offer)
        is_last = i == len(supplier_offers) - 1
        if not is_last and decision.outcome is not Outcome.COUNTER:
            raise NegotiationClosed(decision.reason.split(":", 1)[0])
    return decision, state, prev_counter


# The map from an engine move to a knowledge-base query. Pure — turns what the engine did
# (which terms moved/held, the round pressure) into retrieval text, so the advice matches
# the actual move rather than a generic prompt.
def _retrieve_advice(brief: MoveBrief) -> list[str]:
    """Retrieve negotiation guidance for this move. Always safe: [] if the index is absent."""
    if brief.outcome == "ESCALATE":
        return []  # no message move to advise on
    moved = " ".join(m.name.replace("_", " ") for m in brief.moved_terms)
    held = " ".join(h.name.replace("_", " ") for h in brief.held_terms)
    query = f"{brief.pressure} negotiation concede {moved} hold {held}".strip()
    # Blend general strategy (Fisher-Ury) with any move-specific lever ideas.
    strategy = advice_lines(query, tag="negotiation-strategy", top_k=2)
    levers = advice_lines(query, top_k=2)
    # de-dup by source line, preserve order (strategy first)
    seen: set[str] = set()
    out: list[str] = []
    for line in [*strategy, *levers]:
        if line not in seen:
            seen.add(line)
            out.append(line)
    return out[:3]


def draft_and_guard(
    drafter: DraftClient,
    brief: MoveBrief,
    approved: dict[str, float],
    thread: list[dict[str, str]],
) -> tuple[str, GuardAudit]:
    """Draft the buyer message, guarding each attempt; redraft or fall back.

    The returned message is guaranteed clean: it is either a model draft that passed
    the guard, or the deterministic fallback (which passes by construction). A
    violating draft is never returned.
    """
    attempts: list[GuardAttempt] = []
    work_thread = list(thread)
    advice = _retrieve_advice(brief)
    for _ in range(_MAX_REDRAFTS + 1):
        draft = drafter.draft_buyer(brief, work_thread, advice)
        violations = check(draft, approved)
        attempts.append(GuardAttempt(draft=draft, ok=not violations, violations=violations))
        if not violations:
            return draft, GuardAudit(released_by="model", attempts=attempts)
        # Self-contained repair: quote the offence + allowlist, don't echo the bad draft.
        work_thread = [
            *thread,
            {"role": "system", "text": build_redraft_instruction(violations, approved)},
        ]
    safe = render_fallback(brief, variant=len(attempts))
    attempts.append(GuardAttempt(draft=safe, ok=True, violations=[]))
    return safe, GuardAudit(released_by="fallback", attempts=attempts)


def resolve_supplier_offer(
    envelope: Envelope, raw_text: str, prev_offer: Offer | None
) -> Offer | None:
    """Re-extract the supplier's offer from their prose, server-side.

    Returns ``None`` when no envelope term could be parsed and there's no standing
    offer to inherit — the caller surfaces "couldn't read your position" rather than
    letting the engine's merge fabricate one (``docs`` §4.7).
    """
    extraction = extract_contract(raw_text)
    found = {t.name: t.value for t in extraction.terms if t.name in envelope.term_map}
    if not found and prev_offer is None:
        return None
    base = dict(prev_offer.terms) if prev_offer else {}
    base.update(found)
    # clamp every term into its scored span so a stray parse can't leave the envelope
    clamped = {n: envelope.term_map[n].clamp(base[n]) for n in envelope.term_map if n in base}
    if set(clamped) != set(envelope.term_map):
        return None
    return Offer(terms=clamped)


def turn_result(
    decision: EngineDecision,
    envelope: Envelope,
    prev_counter: Offer | None,
    buyer_message: str,
    guard: GuardAudit,
    supplier_message: str,
    priorities: dict[str, float] | None,
    include_internal: bool,
    max_rounds: int,
) -> TurnResult:
    """Assemble the decision echo. ``include_internal`` gates the buyer-private block."""
    brief = build_move_brief(decision, envelope, prev_counter, max_rounds, priorities=priorities)
    bar_fills = _bar_fills(decision, envelope)
    internal = None
    if include_internal:
        internal = InternalState(
            threshold=decision.threshold,
            incoming_utility=decision.incoming_utility,
            counter_utility=decision.counter_utility,
            reservation_utility=envelope.reservation_utility,
        )
    return TurnResult(
        outcome=decision.outcome.value,
        round_index=decision.round_index,
        reason_tag=decision.reason.split(":", 1)[0],
        approved_numbers=dict(decision.approved_numbers),
        buyer_message=buyer_message,
        supplier_message=supplier_message,
        move_brief=brief if decision.outcome is not Outcome.ESCALATE else None,
        guard=guard,
        bar_fills=bar_fills,
        internal=internal,
    )


def _bar_fills(decision: EngineDecision, envelope: Envelope) -> dict[str, float]:
    """Per-term buyer-utility fills for the drawer micro-bars, pre-computed here."""
    offer = decision.counter or (
        Offer(terms=dict(decision.approved_numbers)) if decision.approved_numbers else None
    )
    if offer is None:
        return {}
    return {
        t.name: round(t.value(offer.terms[t.name]), 4)
        for t in envelope.terms
        if t.name in offer.terms
    }


def offers_from_transcript(turns: list[SupplierTurn]) -> list[Offer]:
    return [Offer(terms=dict(t.terms)) for t in turns]
