"""Deal engine — deterministic core. No LLM, no authority delegated.

The engine scores incoming supplier offers against the buyer's envelope, decides
whether to accept / counter / escalate on a Boulware concession schedule, and —
when it counters — generates a logrolled package via :mod:`packages`.

## Boulware concession curve

The per-round acceptance threshold decays from ``target_utility`` toward
``reservation_utility``:

    threshold(t) = reservation + (target - reservation) * (1 - (t / T) ** beta)

with ``beta > 1`` (Boulware: concede late and steeply) and ``T = max_rounds``.
Round index ``t`` = engine decisions already made, so ``threshold(0) = target``
and ``threshold(T) = reservation`` exactly. The engine counters at t = 0..T-1;
at t = T it never counters — it accepts (if U >= reservation) or escalates.

## Numeric-guard contract

Every decision carries ``approved_numbers`` — the ONLY numeric values a
downstream LLM-composed reply may contain. Thresholds, utilities, reservation,
and beta are internal and must never reach supplier-facing prose. This is the
confidentiality line of the whole system.
"""

from __future__ import annotations

import enum
from typing import Literal

from pydantic import BaseModel, Field

from negotiation_agent.envelope import Envelope, Offer
from negotiation_agent.packages import InfeasiblePackage, fill_package
from negotiation_agent.strategy import adapt_threshold, choose_tactic
from negotiation_agent.supplier_model import SupplierModel

_OPENING_RATIONALE = (
    "Opening with a logrolled package at target — establish the reference point."
)


class Outcome(enum.StrEnum):
    ACCEPT = "accept"
    COUNTER = "counter"
    ESCALATE = "escalate"


class EngineConfig(BaseModel):
    model_config = {"frozen": True}

    max_rounds: int = Field(default=8, ge=1, description="T; engine may counter t=0..T-1")
    beta: float = Field(default=4.0, ge=1.0, description="Boulware exponent; >1 concedes late")
    stall_rounds: int = Field(default=3, ge=1, description="identical supplier offers -> escalate")
    on_unknown_terms: Literal["escalate", "ignore"] = "escalate"
    # Opt-in "senior negotiator" layer: re-infer the supplier's priorities from their observed moves
    # and adapt the concession PACE to their behaviour. Default OFF preserves the fixed-schedule
    # engine exactly. The adaptation is strictly bounded and can never cross the reservation floor.
    adaptive: bool = False


class NegotiationState(BaseModel):
    """Immutable snapshot threaded through :meth:`DealEngine.decide`.

    A fresh instance is returned per decision so the whole negotiation replays as
    a fold over the transcript.
    """

    model_config = {"frozen": True}

    round_index: int = 0
    last_counter: Offer | None = None
    last_incoming: Offer | None = None
    best_incoming_utility: float = -1.0
    stall_count: int = 0
    # Per-term ceiling on offered v, ratcheted down each round so concessions
    # never retract even if the belief is updated mid-negotiation (v1).
    concession_caps: dict[str, float] = Field(default_factory=dict)
    # The supplier's offers in order (adaptive mode only). Threaded through the fold so opponent
    # modelling reads the full move history without breaking decide's pure (state, incoming) shape.
    supplier_history: list[Offer] = Field(default_factory=list)


class EngineDecision(BaseModel):
    model_config = {"frozen": True}

    outcome: Outcome
    round_index: int
    threshold: float
    incoming_utility: float | None = None
    counter: Offer | None = None
    counter_utility: float | None = None
    approved_numbers: dict[str, float] = Field(default_factory=dict)
    reason: str = ""
    # Adaptive-mode audit trail: the tactic the engine played and why, and the base (pre-adaptation)
    # threshold so a reviewer sees exactly how far the pace was nudged. Empty in non-adaptive mode.
    tactic: str = ""
    tactic_rationale: str = ""
    base_threshold: float | None = None


class DealEngine:
    """Deterministic negotiator. One instance can serve many negotiations."""

    def __init__(
        self,
        envelope: Envelope,
        supplier_model: SupplierModel,
        config: EngineConfig | None = None,
    ) -> None:
        self.envelope = envelope
        self.config = config or EngineConfig()
        self._priorities = supplier_model.priorities(envelope)

    @property
    def priorities(self) -> dict[str, float]:
        """The supplier-appetite belief the logrolling search routes concessions by.

        Exposed read-only so the move-brief builder can honesty-gate its rationale on
        the belief (see ``brief.build_move_brief``) without reaching into internals.
        """
        return dict(self._priorities)

    # ---- Boulware schedule -------------------------------------------------

    def threshold(self, round_index: int) -> float:
        """Acceptance threshold at ``round_index`` (clamped to [0, T])."""
        env, cfg = self.envelope, self.config
        t = min(max(round_index, 0), cfg.max_rounds)
        frac = t / cfg.max_rounds
        span = env.target_utility - env.reservation_utility
        return float(env.reservation_utility + span * (1.0 - frac**cfg.beta))

    # ---- Decision ----------------------------------------------------------

    def decide(
        self, state: NegotiationState, incoming: Offer | None
    ) -> tuple[EngineDecision, NegotiationState]:
        """Return ``(decision, next_state)`` for one engine move.

        ``incoming=None`` is the opening anchor. Pure: the same
        ``(state, incoming)`` always yields the same result.

        In adaptive mode the supplier's priorities are re-inferred from their move history and the
        concession pace is nudged to their behaviour — but the adaptation is strictly bounded to the
        mandate band and the fixed-schedule engine is recovered exactly when ``adaptive=False``.
        """
        base_theta = self.threshold(state.round_index)
        theta, priorities, recip = self._effective_schedule(state, incoming, base_theta)

        decision, nxt = self._decide_core(state, incoming, theta, priorities)

        # append the supplier's offer to the history (adaptive opponent modelling reads it next
        # turn)
        if self.config.adaptive and incoming is not None:
            nxt = nxt.model_copy(
                update={"supplier_history": [*state.supplier_history, incoming]}
            )
        # attach the strategy read (tactic + rationale + base threshold) for the human audit trail
        if self.config.adaptive:
            decision = self._annotate(decision, base_theta, recip, incoming, nxt.supplier_history)
        return decision, nxt

    def _effective_schedule(
        self, state: NegotiationState, incoming: Offer | None, base_theta: float
    ) -> tuple[float, dict[str, float], float]:
        """The (theta, priorities, reciprocity) this decision uses. In non-adaptive mode it is the
        fixed base. In adaptive mode, priorities are re-inferred from the supplier's move history
        and theta is nudged by reciprocity — always clamped to [reservation, target] by
        adapt_threshold, so the floor cannot be crossed."""
        if not self.config.adaptive or incoming is None:
            return base_theta, self._priorities, 0.0
        history = [*state.supplier_history, incoming]
        if len(history) < 2:
            return base_theta, self._priorities, 0.0
        from negotiation_agent.opponent_model import infer_appetite, reciprocity

        priorities = infer_appetite(self.envelope, history).priorities(self.envelope)
        recip = reciprocity(self.envelope, history)
        theta = adapt_threshold(
            base_theta,
            reciprocity=recip,
            reservation=self.envelope.reservation_utility,
            target=self.envelope.target_utility,
        )
        return theta, priorities, recip

    def _decide_core(
        self,
        state: NegotiationState,
        incoming: Offer | None,
        theta: float,
        priorities: dict[str, float],
    ) -> tuple[EngineDecision, NegotiationState]:
        """The engine's accept/counter/escalate decision at the (possibly adapted) ``theta`` using
        ``priorities`` for the logrolling search. Pure; identical to the fixed engine when theta and
        priorities are the fixed base."""
        t = state.round_index

        # --- Opening anchor: logrolled package at target, not a naive ideal. ---
        if incoming is None:
            return self._counter(state, theta, reason="opening_anchor", priorities=priorities)

        # --- Unknown-term guard: accepting unmodeled obligations is a trap. ---
        unknown = set(incoming.terms) - set(self.envelope.term_map)
        if unknown and self.config.on_unknown_terms == "escalate":
            return (
                self._escalate(t, theta, reason=f"unmodeled_terms:{sorted(unknown)}"),
                state,
            )

        # --- Merge partial offers: unaddressed terms inherit the standing
        #     counter. Envelope.utility raises on missing terms, so this must
        #     happen before scoring. ---
        merged = self._merge(incoming, state.last_counter)
        if merged is None:
            return self._escalate(t, theta, reason="malformed_offer"), state
        u_in = self.envelope.utility(merged)
        best_u = max(state.best_incoming_utility, u_in)

        # --- Stall tracking: deterministic suppliers repeat exactly. ---
        # Only escalate a stall for an offer that does NOT clear the current threshold. A
        # supplier who holds firm at an acceptable offer while the Boulware curve decays to
        # meet it should be ACCEPTED, not handed to a human — the accept clause below does
        # exactly that. Gating on u_in < theta keeps the stall guard from stealing a deal
        # the engine's own schedule says to take.
        stalled = state.last_incoming is not None and merged.terms == state.last_incoming.terms
        stall_count = state.stall_count + 1 if stalled else 0
        if stall_count >= self.config.stall_rounds and u_in < theta:
            nxt = state.model_copy(
                update={
                    "last_incoming": merged,
                    "best_incoming_utility": best_u,
                    "stall_count": stall_count,
                }
            )
            return (
                self._escalate(t, theta, reason="supplier_stalled", incoming_utility=u_in),
                nxt,
            )

        # --- Accept rule: single clause. Counter utility is always >= theta by
        #     construction, so an "accept if incoming >= my counter" clause is
        #     provably redundant. ---
        if u_in >= theta:
            approved = {n: merged.terms[n] for n in self.envelope.term_map}
            decision = EngineDecision(
                outcome=Outcome.ACCEPT,
                round_index=t,
                threshold=theta,
                incoming_utility=u_in,
                approved_numbers=approved,
                reason="accept_threshold",
            )
            nxt = state.model_copy(
                update={
                    "last_incoming": merged,
                    "best_incoming_utility": best_u,
                    "stall_count": stall_count,
                }
            )
            return decision, nxt

        # --- Deadline: at t == T we never counter. ---
        if t >= self.config.max_rounds:
            nxt = state.model_copy(
                update={
                    "last_incoming": merged,
                    "best_incoming_utility": best_u,
                    "stall_count": stall_count,
                }
            )
            return (
                self._escalate(
                    t,
                    theta,
                    reason=f"deadline_no_deal:best_u={best_u:.4f}",
                    incoming_utility=u_in,
                ),
                nxt,
            )

        # --- Counter with a fresh logrolled package at this round's threshold. ---
        return self._counter(
            state,
            theta,
            reason="counter",
            incoming=merged,
            u_in=u_in,
            best_u=best_u,
            stall_count=stall_count,
            priorities=priorities,
        )

    # ---- Helpers -----------------------------------------------------------

    def _annotate(
        self,
        decision: EngineDecision,
        base_theta: float,
        reciprocity: float,
        incoming: Offer | None,
        history: list[Offer],
    ) -> EngineDecision:
        """Attach the strategy read (tactic + rationale + base threshold) to a decision for the
        human audit trail. Advisory only — it labels the decision the engine already made."""
        from negotiation_agent.opponent_model import held_firm

        held = [
            t.name for t in self.envelope.terms if held_firm(self.envelope, history, term=t.name)
        ]
        tactic, rationale = choose_tactic(
            is_opening=incoming is None,
            incoming_utility=decision.incoming_utility,
            adjusted_threshold=decision.threshold,
            reservation=self.envelope.reservation_utility,
            reciprocity=reciprocity,
            supplier_held_terms=held,
            at_deadline=decision.round_index >= self.config.max_rounds,
        )
        return decision.model_copy(
            update={
                "tactic": tactic,
                "tactic_rationale": rationale,
                "base_threshold": base_theta,
            }
        )

    def _counter(
        self,
        state: NegotiationState,
        theta: float,
        *,
        reason: str,
        incoming: Offer | None = None,
        u_in: float | None = None,
        best_u: float | None = None,
        stall_count: int = 0,
        priorities: dict[str, float] | None = None,
    ) -> tuple[EngineDecision, NegotiationState]:
        try:
            offer, realized, planned_v = fill_package(
                self.envelope,
                theta,
                priorities if priorities is not None else self._priorities,
                caps=state.concession_caps or None,
            )
        except InfeasiblePackage:
            # Caps have ratcheted below what theta needs — nothing legal left to offer.
            return (
                self._escalate(
                    state.round_index, theta, reason="no_feasible_counter", incoming_utility=u_in
                ),
                state,
            )

        new_caps = {
            name: min(state.concession_caps.get(name, 1.0), planned_v[name]) for name in planned_v
        }
        decision = EngineDecision(
            outcome=Outcome.COUNTER,
            round_index=state.round_index,
            threshold=theta,
            incoming_utility=u_in,
            counter=offer,
            counter_utility=realized,
            approved_numbers=dict(offer.terms),
            reason=reason,
        )
        nxt = state.model_copy(
            update={
                "round_index": state.round_index + 1,
                "last_counter": offer,
                "last_incoming": incoming if incoming is not None else state.last_incoming,
                "best_incoming_utility": (
                    best_u if best_u is not None else state.best_incoming_utility
                ),
                "stall_count": stall_count,
                "concession_caps": new_caps,
            }
        )
        return decision, nxt

    def _escalate(
        self,
        round_index: int,
        theta: float,
        *,
        reason: str,
        incoming_utility: float | None = None,
    ) -> EngineDecision:
        return EngineDecision(
            outcome=Outcome.ESCALATE,
            round_index=round_index,
            threshold=theta,
            incoming_utility=incoming_utility,
            reason=reason,
        )

    def _merge(self, incoming: Offer, last_counter: Offer | None) -> Offer | None:
        """Fill terms the supplier didn't address from the standing counter.

        Returns None if a term is unaddressed and there is no standing counter
        to inherit from (malformed opening from the supplier).
        """
        merged: dict[str, float] = {}
        for name in self.envelope.term_map:
            if name in incoming.terms:
                merged[name] = incoming.terms[name]
            elif last_counter is not None and name in last_counter.terms:
                merged[name] = last_counter.terms[name]
            else:
                return None
        return Offer(terms=merged)
