"""Deal envelope schema.

The *envelope* is the category manager's mandate, expressed as data: which
terms are negotiable, how each maps to buyer utility, how much each matters
(weights), and the reservation point below which the engine must walk away or
escalate. It is deterministic and human-owned — the LLM never edits it.

Envelopes are versioned and carry ``signed_by`` so the audit trail can prove
which human authorized the mandate an agent negotiated under. Bumping any term,
weight, or threshold is a new version, not a mutation.
"""

from __future__ import annotations

import enum
from functools import cached_property

from pydantic import BaseModel, Field, model_validator

from negotiation_agent.value import linear_inverse, linear_value


class Direction(enum.StrEnum):
    """Which end of a term's range the buyer prefers."""

    MINIMIZE = "minimize"  # lower is better for the buyer (e.g. unit price)
    MAXIMIZE = "maximize"  # higher is better for the buyer (e.g. payment days)


class TermType(enum.StrEnum):
    """Semantic type of a negotiable term. Drives units, rounding, and the
    intent labels the LLM classifier can map supplier priorities onto."""

    PRICE = "price"
    PAYMENT_DAYS = "payment_days"
    CONTRACT_MONTHS = "contract_months"
    VOLUME_UNITS = "volume_units"
    REBATE_PCT = "rebate_pct"


# Whole-number terms are rounded to integers in generated counteroffers so we
# never propose "37.4 payment days". Price and rebate stay continuous.
_INTEGER_TERMS = {TermType.PAYMENT_DAYS, TermType.CONTRACT_MONTHS, TermType.VOLUME_UNITS}


class TermSpec(BaseModel):
    """One negotiable term: its bounds, preferred direction, and weight.

    ``best`` / ``worst`` are the term values scoring 1.0 / 0.0. They encode the
    direction implicitly (for a MINIMIZE term ``best < worst``), but we also
    carry ``direction`` explicitly for validation and readability.
    """

    model_config = {"frozen": True}

    name: str = Field(min_length=1)
    term_type: TermType
    direction: Direction
    best: float = Field(description="term value scoring utility 1.0 (ideal for buyer)")
    worst: float = Field(description="term value scoring utility 0.0 (reservation edge)")
    weight: float = Field(
        gt=0.0, le=1.0, description="relative importance; envelope weights sum to 1"
    )

    @model_validator(mode="after")
    def _check_direction(self) -> TermSpec:
        if self.best == self.worst:
            raise ValueError(f"term {self.name!r}: best and worst must differ")
        if self.direction is Direction.MINIMIZE and self.best >= self.worst:
            raise ValueError(f"term {self.name!r}: MINIMIZE requires best < worst")
        if self.direction is Direction.MAXIMIZE and self.best <= self.worst:
            raise ValueError(f"term {self.name!r}: MAXIMIZE requires best > worst")
        return self

    @property
    def is_integer(self) -> bool:
        return self.term_type in _INTEGER_TERMS

    def value(self, x: float) -> float:
        """Utility contribution of term value ``x``, in [0, 1] (weight excluded)."""
        return linear_value(x, best=self.best, worst=self.worst)

    def value_to_x(self, v: float) -> float:
        """Term value achieving utility contribution ``v`` (weight excluded).

        Rounded to an integer for whole-number term types. The result is
        clamped to the [worst, best] span by :func:`linear_inverse`.
        """
        x = linear_inverse(v, best=self.best, worst=self.worst)
        return float(round(x)) if self.is_integer else x

    def clamp(self, x: float) -> float:
        """Clamp a term value into the scored [worst, best] span."""
        lo, hi = sorted((self.best, self.worst))
        return min(hi, max(lo, x))


class Offer(BaseModel):
    """A concrete package: one value per envelope term, keyed by term name.

    This is what the LLM extractor produces from a supplier's free-text reply
    and what the engine emits as a counteroffer. It carries no utility of its
    own — utility is always computed against a specific envelope, so the same
    offer can be scored under different mandates in the audit replay.
    """

    model_config = {"frozen": True}

    terms: dict[str, float]

    def with_updates(self, **changes: float) -> Offer:
        merged = dict(self.terms)
        merged.update(changes)
        return Offer(terms=merged)


class Envelope(BaseModel):
    """Versioned negotiation mandate.

    ``target_utility`` is where the concession curve starts (the aspiration);
    ``reservation_utility`` is the walk-away floor — the utility of the buyer's
    BATNA (best alternative to a negotiated agreement) expressed on the [0,1]
    scale. An offer scoring below the reservation is never acceptable and, once
    the concession curve has fully decayed to it without a deal, the negotiation
    escalates to a human buyer. The span between them, scored against the
    supplier's own reservation, is the ZOPA (see ``simulator.scenarios.zopa_check``).
    """

    model_config = {"frozen": True}

    negotiation_id: str = Field(min_length=1)
    version: int = Field(ge=1)
    signed_by: str = Field(min_length=1, description="human who authorized this mandate")
    terms: list[TermSpec] = Field(min_length=1)
    target_utility: float = Field(ge=0.0, le=1.0, default=0.95)
    reservation_utility: float = Field(ge=0.0, le=1.0, default=0.55)

    @model_validator(mode="after")
    def _check(self) -> Envelope:
        names = [t.name for t in self.terms]
        if len(names) != len(set(names)):
            raise ValueError("duplicate term names in envelope")
        total = sum(t.weight for t in self.terms)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"term weights must sum to 1.0, got {total:.6f}")
        if self.reservation_utility >= self.target_utility:
            raise ValueError("reservation_utility must be below target_utility")
        return self

    @cached_property
    def term_map(self) -> dict[str, TermSpec]:
        return {t.name: t for t in self.terms}

    def utility(self, offer: Offer) -> float:
        """Buyer utility U = sum_i w_i * v_i(x_i) for ``offer`` under this envelope.

        Raises ``KeyError`` if the offer is missing a term the envelope scores;
        extra keys in the offer are ignored (forward-compatible with richer
        supplier packages).
        """
        u = 0.0
        for term in self.terms:
            if term.name not in offer.terms:
                raise KeyError(f"offer missing term {term.name!r}")
            u += term.weight * term.value(offer.terms[term.name])
        return u

    def ideal_offer(self) -> Offer:
        """The all-``best`` package (utility 1.0). Used as the buyer's anchor."""
        return Offer(terms={t.name: t.best for t in self.terms})

    def reservation_offer(self) -> Offer:
        """The all-``worst`` package (utility 0.0). The absolute floor per term."""
        return Offer(terms={t.name: t.worst for t in self.terms})
