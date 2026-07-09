"""Scenarios — buyer envelope + hidden supplier envelope + belief condition.

A scenario pairs a buyer mandate with the supplier's hidden truth and the belief
the engine is told about supplier priorities. ``zopa_check`` gates scenarios so
that closure metrics measure the *engine*, not the fixture: if no zone of
possible agreement exists, a failure to close is the scenario's fault.

The belief conditions — oracle / uniform / inverted — are what let the eval plot
belief quality vs captured utility, which is the proof of the logrolling pitch.
"""

from __future__ import annotations

from pydantic import BaseModel

from negotiation_agent.envelope import Direction, Envelope, TermSpec, TermType
from negotiation_agent.packages import fill_package
from negotiation_agent.simulator.personas import (
    AGGRESSIVE,
    COOPERATIVE,
    EVASIVE,
    PersonaConfig,
)
from negotiation_agent.supplier_model import SupplierModel


class Scenario(BaseModel):
    model_config = {"frozen": True}

    name: str
    buyer_envelope: Envelope
    supplier_envelope: Envelope  # hidden truth
    persona: PersonaConfig
    belief: SupplierModel  # what the engine is told
    belief_source: str

    def model_post_init(self, _ctx) -> None:
        b = set(self.buyer_envelope.term_map)
        s = set(self.supplier_envelope.term_map)
        if b != s:
            raise ValueError(f"scenario {self.name!r}: term-name mismatch {b ^ s}")


def zopa_check(buyer_env: Envelope, supplier_env: Envelope) -> float:
    """Supplier utility of the buyer's *reservation* package, logrolled toward
    the supplier's true priorities.

    A value >= the supplier's reservation utility means a ZOPA exists: the engine
    can, in principle, reach a package both sides accept. Build scenarios so this
    holds, or closure rate measures scenario authoring rather than the engine.
    """
    true_priorities = _true_priorities(supplier_env)
    offer, _, _ = fill_package(buyer_env, buyer_env.reservation_utility, true_priorities)
    projected = offer  # same term names by scenario construction
    return supplier_env.utility(projected)


def _true_priorities(supplier_env: Envelope) -> dict[str, float]:
    """The supplier's real appetite: heavier own weight => wants more movement."""
    raw = {t.name: t.weight for t in supplier_env.terms}
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


# --------------------------------------------------------------------------- #
# A concrete reference envelope pair. Buyer and supplier oppose on price/rebate
# (margin) but the buyer weights price heavily while the supplier's real pain is
# cash flow (payment days) and volume certainty — exactly the logrolling setup.
# --------------------------------------------------------------------------- #

def _buyer_envelope() -> Envelope:
    return Envelope(
        negotiation_id="ref-widget-supply",
        version=1,
        signed_by="category.manager@buyer.example",
        target_utility=0.95,
        reservation_utility=0.55,
        terms=[
            # Buyer cares most about price, barely about term length.
            TermSpec(name="price", term_type=TermType.PRICE, direction=Direction.MINIMIZE,
                     best=9.0, worst=12.0, weight=0.45),
            TermSpec(name="payment_days", term_type=TermType.PAYMENT_DAYS,
                     direction=Direction.MAXIMIZE, best=90, worst=30, weight=0.15),
            TermSpec(name="contract_months", term_type=TermType.CONTRACT_MONTHS,
                     direction=Direction.MINIMIZE, best=12, worst=36, weight=0.10),
            TermSpec(name="volume_units", term_type=TermType.VOLUME_UNITS,
                     direction=Direction.MINIMIZE, best=10000, worst=50000, weight=0.10),
            TermSpec(name="rebate_pct", term_type=TermType.REBATE_PCT,
                     direction=Direction.MAXIMIZE, best=8.0, worst=0.0, weight=0.20),
        ],
    )


def _supplier_envelope() -> Envelope:
    # Supplier's real pain: cash flow (wants SHORT payment days) and volume
    # certainty (wants HIGH volume). Price matters but less than the buyer thinks.
    return Envelope(
        negotiation_id="ref-widget-supply",
        version=1,
        signed_by="sales.lead@supplier.example",
        target_utility=0.92,
        reservation_utility=0.50,
        terms=[
            TermSpec(name="price", term_type=TermType.PRICE, direction=Direction.MAXIMIZE,
                     best=12.0, worst=9.0, weight=0.20),
            TermSpec(name="payment_days", term_type=TermType.PAYMENT_DAYS,
                     direction=Direction.MINIMIZE, best=30, worst=90, weight=0.30),
            TermSpec(name="contract_months", term_type=TermType.CONTRACT_MONTHS,
                     direction=Direction.MAXIMIZE, best=36, worst=12, weight=0.10),
            TermSpec(name="volume_units", term_type=TermType.VOLUME_UNITS,
                     direction=Direction.MAXIMIZE, best=50000, worst=10000, weight=0.30),
            TermSpec(name="rebate_pct", term_type=TermType.REBATE_PCT,
                     direction=Direction.MINIMIZE, best=0.0, worst=8.0, weight=0.10),
        ],
    )


def _belief(condition: str, supplier_env: Envelope, buyer_env: Envelope) -> SupplierModel:
    """Three belief conditions for the eval sweep."""
    if condition == "oracle":  # engine told the truth
        return SupplierModel(appetite=_true_priorities(supplier_env), source="simulator")
    if condition == "uniform":  # no information
        return SupplierModel.uniform(buyer_env)
    if condition == "inverted":  # worst-case misclassification
        true = _true_priorities(supplier_env)
        m = max(true.values()) + min(true.values())
        return SupplierModel(appetite={k: m - v for k, v in true.items()}, source="simulator")
    raise ValueError(condition)


def reference_matrix() -> list[Scenario]:
    """3 personas x 3 belief conditions = 9 reference scenarios."""
    buyer, supplier = _buyer_envelope(), _supplier_envelope()
    scenarios: list[Scenario] = []
    for persona in (AGGRESSIVE, COOPERATIVE, EVASIVE):
        for condition in ("oracle", "uniform", "inverted"):
            scenarios.append(
                Scenario(
                    name=f"ref/{persona.name}/{condition}",
                    buyer_envelope=buyer,
                    supplier_envelope=supplier,
                    persona=persona,
                    belief=_belief(condition, supplier, buyer),
                    belief_source=condition,
                )
            )
    return scenarios
