"""Supplier preference belief — pure data, no LLM.

The engine must never see the supplier's true preferences. What it consumes is a
*belief*: for each negotiable term, how strongly the engine thinks the supplier
wants the buyer to move off its ideal on that term. That belief drives which
terms the engine concedes on (logrolling), but it is only ever numbers.

In v0 the simulator/scenario harness supplies the belief directly. In v1 the LLM
classifies the supplier's free-text messages into a small set of *intent labels*
(cash flow, volume certainty, term length, margin) and those labels map to terms
via the static table below. The mapping lives here, as data, so the LLM boundary
stays entirely outside the engine.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from negotiation_agent.envelope import Envelope, TermType

_EPS = 1e-6

# LLM intent label -> the term types that label bears on. Kept as data so the v1
# classifier only has to emit labels; it never needs to know envelope internals.
INTENT_TO_TERM_TYPES: dict[str, tuple[TermType, ...]] = {
    "cash_flow": (TermType.PAYMENT_DAYS,),
    "volume_certainty": (TermType.VOLUME_UNITS,),
    "term_length": (TermType.CONTRACT_MONTHS,),
    "margin": (TermType.PRICE, TermType.REBATE_PCT),
}


class SupplierModel(BaseModel):
    """Belief about supplier concession appetite, per term name, in [0, 1].

    ``appetite[name]`` = believed intensity with which the supplier wants the
    buyer to move off ``best`` on that term. Higher appetite => the engine
    prefers to spend its concession budget there.
    """

    model_config = {"frozen": True}

    appetite: dict[str, float] = Field(default_factory=dict)
    source: Literal["default", "simulator", "llm"] = "default"

    @classmethod
    def uniform(cls, envelope: Envelope) -> SupplierModel:
        """Flat belief — no information. Every term equally likely to be traded."""
        return cls(appetite={t.name: 1.0 for t in envelope.terms}, source="default")

    @classmethod
    def from_intents(
        cls,
        envelope: Envelope,
        intent_scores: dict[str, float],
        *,
        source: Literal["default", "simulator", "llm"] = "llm",
    ) -> SupplierModel:
        """Build a belief from LLM intent-label scores via ``INTENT_TO_TERM_TYPES``.

        Unmapped labels are ignored; terms no label touches get zero appetite
        (they will be held longest). This is the seam the v1 classifier plugs into.
        """
        appetite: dict[str, float] = {t.name: 0.0 for t in envelope.terms}
        for label, score in intent_scores.items():
            for term_type in INTENT_TO_TERM_TYPES.get(label, ()):
                for term in envelope.terms:
                    if term.term_type is term_type:
                        appetite[term.name] = max(appetite[term.name], score)
        return cls(appetite=appetite, source=source)

    def priorities(self, envelope: Envelope) -> dict[str, float]:
        """Appetites restricted to envelope terms, floored and normalized to sum 1.

        The floor keeps the downstream cost-ratio sort total and deterministic
        even when several appetites are zero. An all-zero belief degenerates to
        uniform rather than dividing by zero.
        """
        raw = {t.name: max(_EPS, self.appetite.get(t.name, 0.0)) for t in envelope.terms}
        total = sum(raw.values())
        if total <= 0.0:  # unreachable given the floor, but keep it total
            return {name: 1.0 / len(raw) for name in raw}
        return {name: v / total for name, v in raw.items()}
