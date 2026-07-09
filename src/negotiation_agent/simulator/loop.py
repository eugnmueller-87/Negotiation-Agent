"""Negotiation loop + audit records.

The engine moves first (a logrolled anchor at target). Then supplier and engine
alternate until someone accepts, the supplier walks, or the engine escalates.
The loop is bounded by ``2 * T + 1`` turns.

Every turn is recorded as a :class:`Turn` — the audit line. ``supplier_utility``
is filled by the harness from the hidden supplier envelope for evaluation only;
the engine never has access to it.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from negotiation_agent.engine import DealEngine, EngineConfig, NegotiationState, Outcome
from negotiation_agent.envelope import Envelope, Offer
from negotiation_agent.simulator.supplier import SupplierAgent

# The four terminal states of a negotiation. One alias, used everywhere, so the
# status never widens to plain `str` and mypy proves each call site passes a
# valid literal.
Status = Literal["closed_engine", "closed_supplier", "escalated", "walked"]


class Turn(BaseModel):
    model_config = {"frozen": True}

    seq: int
    round_index: int
    actor: Literal["engine", "supplier"]
    kind: Literal["counter", "accept", "reject", "escalate", "offer"]
    offer: Offer | None = None
    buyer_utility: float | None = None
    threshold: float | None = None
    supplier_utility: float | None = None  # eval-only, hidden from the engine
    reason: str | None = None


class Transcript(BaseModel):
    model_config = {"frozen": True}

    negotiation_id: str
    envelope_version: int
    persona: str
    belief_source: str
    turns: list[Turn] = Field(default_factory=list)


class NegotiationResult(BaseModel):
    model_config = {"frozen": True}

    transcript: Transcript
    status: Status
    final_deal: Offer | None = None
    escalation_reason: str | None = None
    rounds_used: int = 0


def run_negotiation(
    buyer_envelope: Envelope,
    engine: DealEngine,
    supplier: SupplierAgent,
    *,
    supplier_envelope: Envelope | None = None,
    persona_name: str = "unknown",
    belief_source: str = "unknown",
    config: EngineConfig | None = None,
) -> NegotiationResult:
    """Drive one engine-vs-supplier negotiation to termination.

    ``supplier_envelope`` (the hidden truth) is used only to annotate turns with
    supplier utility for evaluation; pass None to omit that annotation.
    """
    cfg = config or engine.config
    state = NegotiationState()
    turns: list[Turn] = []
    seq = 0

    def sup_util(offer: Offer | None) -> float | None:
        if supplier_envelope is None or offer is None:
            return None
        projected = Offer(
            terms={
                n: offer.terms.get(n, supplier_envelope.term_map[n].worst)
                for n in supplier_envelope.term_map
            }
        )
        return supplier_envelope.utility(projected)

    # --- Engine opens with the anchor. ---
    decision, state = engine.decide(state, incoming=None)
    turns.append(
        Turn(
            seq=seq,
            round_index=decision.round_index,
            actor="engine",
            kind="counter",
            offer=decision.counter,
            buyer_utility=decision.counter_utility,
            threshold=decision.threshold,
            supplier_utility=sup_util(decision.counter),
            reason=decision.reason,
        )
    )
    seq += 1
    standing_counter = decision.counter

    # --- Alternation. Hard bound: 2*T + 2 iterations. ---
    for _ in range(2 * cfg.max_rounds + 2):
        move = supplier.respond(state.round_index, standing_counter, max_rounds=cfg.max_rounds)
        turns.append(
            Turn(
                seq=seq,
                round_index=state.round_index,
                actor="supplier",
                kind=move.kind if move.kind != "offer" else "offer",
                offer=move.offer,
                buyer_utility=(
                    buyer_envelope.utility(
                        _merge_for_score(buyer_envelope, move.offer, standing_counter)
                    )
                    if move.offer is not None
                    else None
                ),
                supplier_utility=sup_util(move.offer or standing_counter),
                reason=None,
            )
        )
        seq += 1

        if move.kind == "accept":
            return _result(
                turns,
                "closed_supplier",
                standing_counter,
                None,
                state.round_index,
                buyer_envelope,
                persona_name,
                belief_source,
            )
        if move.kind == "reject":
            return _result(
                turns,
                "walked",
                None,
                None,
                state.round_index,
                buyer_envelope,
                persona_name,
                belief_source,
            )

        # Supplier countered -> engine responds.
        decision, state = engine.decide(state, move.offer)
        merged = _merge_for_score(buyer_envelope, move.offer, standing_counter)
        turns.append(
            Turn(
                seq=seq,
                round_index=decision.round_index,
                actor="engine",
                kind=(
                    "accept"
                    if decision.outcome is Outcome.ACCEPT
                    else "escalate"
                    if decision.outcome is Outcome.ESCALATE
                    else "counter"
                ),
                offer=decision.counter,
                buyer_utility=(
                    decision.counter_utility
                    if decision.outcome is Outcome.COUNTER
                    else decision.incoming_utility
                ),
                threshold=decision.threshold,
                supplier_utility=sup_util(
                    decision.counter if decision.outcome is Outcome.COUNTER else merged
                ),
                reason=decision.reason,
            )
        )
        seq += 1

        if decision.outcome is Outcome.ACCEPT:
            return _result(
                turns,
                "closed_engine",
                merged,
                None,
                decision.round_index,
                buyer_envelope,
                persona_name,
                belief_source,
            )
        if decision.outcome is Outcome.ESCALATE:
            return _result(
                turns,
                "escalated",
                None,
                decision.reason,
                decision.round_index,
                buyer_envelope,
                persona_name,
                belief_source,
            )
        standing_counter = decision.counter

    # Structurally unreachable: the engine escalates at t == max_rounds, which
    # the loop's 2*T+2 iteration bound always reaches first. Kept as a backstop.
    return _result(  # pragma: no cover - defensive
        turns,
        "escalated",
        None,
        "loop_bound_exceeded",
        state.round_index,
        buyer_envelope,
        persona_name,
        belief_source,
    )


def _merge_for_score(envelope: Envelope, offer: Offer | None, standing: Offer | None) -> Offer:
    """Fill any terms the supplier didn't address from the standing counter so
    the offer scores under the full envelope."""
    base = standing.terms if standing is not None else {}
    src = offer.terms if offer is not None else {}
    return Offer(
        terms={n: src.get(n, base.get(n, envelope.term_map[n].worst)) for n in envelope.term_map}
    )


def _result(
    turns: list[Turn],
    status: Status,
    deal: Offer | None,
    reason: str | None,
    rounds: int,
    envelope: Envelope,
    persona_name: str,
    belief_source: str,
) -> NegotiationResult:
    transcript = Transcript(
        negotiation_id=envelope.negotiation_id,
        envelope_version=envelope.version,
        persona=persona_name,
        belief_source=belief_source,
        turns=turns,
    )
    return NegotiationResult(
        transcript=transcript,
        status=status,
        final_deal=deal,
        escalation_reason=reason,
        rounds_used=rounds,
    )
