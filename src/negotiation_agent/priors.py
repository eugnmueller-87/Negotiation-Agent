"""Learned priors from negotiation history — the engine starts smarter, not deciding differently.

This is the "gets smarter over time" layer, built the OFF-HANDS way: it turns the outcome log
(:mod:`negotiation_agent.outcomes`) into priors that IMPROVE the deterministic engine's starting
point — a warm-start supplier belief and advisory settlement expectations — for the next negotiation
in the same category. It is pure statistics over the buyer's own history. It NEVER decides a move,
NEVER changes the human's mandate (target / reservation), and NEVER touches the reservation floor.

Cold start (no history) → the priors are empty and the engine behaves exactly as it does today (a
uniform supplier belief, the mandate as authored). Every negotiation logged makes the *next* one's
starting point a little better; the engine's decision rules are unchanged.

Why this fits the architecture (and matches how the industry does it): the model/statistics ADVISE,
the deterministic engine DECIDES. History informs the anchor and the belief; it can never make the
engine accept a worse deal or cross the floor.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .envelope import Envelope
from .outcomes import NegotiationOutcome
from .supplier_model import SupplierModel

# Below this many prior outcomes we don't trust the statistics enough to warm-start — the prior is
# reported as low-confidence and the caller should fall back to the uniform belief. Small samples
# overfit; a handful of deals is a hint, not a law.
_MIN_SAMPLES_FOR_CONFIDENCE = 5


class CategoryPrior(BaseModel):
    """What the buyer's history in one category suggests for the next negotiation. Advisory only —
    it seeds the engine's starting point; the human's mandate and the engine's rules stay intact."""

    model_config = {"frozen": True}

    category: str
    samples: int = Field(ge=0)  # how many past outcomes this prior is built from
    confident: bool = False  # samples >= threshold — below it, treat as a weak hint
    # believed supplier concession appetite per term (the warm-start for SupplierModel): HIGH for a
    # term suppliers here historically DEFEND, LOW for one they concede — the engine routes its
    # concessions to the low-appetite terms. Empty on cold start.
    appetite_prior: dict[str, float] = Field(default_factory=dict)
    # where accepted deals in this category historically settled, as buyer utility in [0, 1] — an
    # ADVISORY expectation shown to the human, NOT a change to target/reservation. None if no deal
    # ever closed here.
    typical_settled_utility: float | None = None
    # fraction of past negotiations here that escalated without a deal — a category risk signal.
    escalation_rate: float = Field(ge=0.0, le=1.0, default=0.0)


def learn_category_prior(category: str, history: list[NegotiationOutcome]) -> CategoryPrior:
    """Compute the prior for ``category`` from its past outcomes. Pure statistics; no side effects.

    ``appetite_prior[term]`` is HIGH for a term suppliers here historically DEFENDED and LOW for one
    they conceded — the inverse of the observed concession frequency. This warm-starts the engine's
    supplier belief so it routes concessions to the terms suppliers give cheaply, from turn one,
    instead of learning it from scratch each negotiation.
    """
    relevant = [o for o in history if o.category == category]
    n = len(relevant)
    if n == 0:
        return CategoryPrior(category=category, samples=0, confident=False)

    # concession frequency per term → appetite prior (defended = high appetite = 1 - concede_freq)
    concede_count: dict[str, int] = {}
    for o in relevant:
        for term in o.conceded_terms:
            concede_count[term] = concede_count.get(term, 0) + 1
    appetite_prior = {term: round(1.0 - count / n, 4) for term, count in concede_count.items()}

    accepted = [o for o in relevant if o.outcome == "accepted" and o.settled_utility is not None]
    typical = (
        round(sum(o.settled_utility for o in accepted) / len(accepted), 4)  # type: ignore[misc]
        if accepted
        else None
    )
    escalated = sum(1 for o in relevant if o.outcome == "escalated")

    return CategoryPrior(
        category=category,
        samples=n,
        confident=n >= _MIN_SAMPLES_FOR_CONFIDENCE,
        appetite_prior=appetite_prior,
        typical_settled_utility=typical,
        escalation_rate=round(escalated / n, 4),
    )


def seed_supplier_model(envelope: Envelope, prior: CategoryPrior) -> SupplierModel:
    """Warm-start a :class:`SupplierModel` from a category prior — the belief starts informed by
    history instead of uniform. Terms with no prior signal fall back to neutral (0.5). If the prior
    is not confident (too few samples) OR empty, returns the uniform belief — i.e. exactly today's
    cold-start behaviour, so a thin history can never mislead the engine.

    This ONLY sets the initial belief the logrolling routes by. It does not change target,
    reservation, the concession schedule, or any decision — the human's mandate and the engine's
    rules are fully intact.
    """
    if not prior.confident or not prior.appetite_prior:
        return SupplierModel.uniform(envelope)
    appetite = {
        t.name: prior.appetite_prior.get(t.name, 0.5) for t in envelope.terms
    }
    return SupplierModel(appetite=appetite, source="default")
