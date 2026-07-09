"""Price-split baseline — the naive negotiator logrolling is measured against.

The headline claim is *logrolling, not price-splitting*. A claim like that is only
proven if you run it against the very thing it says it beats. This module is that
control: a package builder that reaches the same buyer utility ``threshold`` as
the real engine but distributes its concessions **uniformly across every term**,
blind to what the supplier actually values.

Both builders hit the identical buyer utility, so buyer cost is matched. The only
difference is *which* terms the concessions land on. If logrolling is real, the
appetite-aware package leaves the supplier materially better off at the same
buyer cost — a strictly better trade, not a giveaway.

``compare_at_threshold`` returns that head-to-head delta; it is the honest,
falsifiable proof behind the pitch.
"""

from __future__ import annotations

from pydantic import BaseModel

from negotiation_agent.envelope import Envelope, Offer
from negotiation_agent.packages import fill_package, snap_toward_best
from negotiation_agent.supplier_model import SupplierModel
from negotiation_agent.value import linear_inverse


def uniform_split_package(envelope: Envelope, threshold: float) -> Offer:
    """Package at buyer utility ``threshold`` that concedes evenly on every term.

    Each term is set to the same value fraction ``v = threshold`` (weights sum to
    1, so ``sum_i w_i * v = threshold``). That is the "split the difference on
    everything" move: no term is prioritized, supplier preferences are ignored
    entirely. Integer terms snap toward the buyer's ideal via the same directional
    rule the real engine uses (:func:`snap_toward_best`), so realized buyer
    utility is always ``>= threshold`` — a fair, buyer-favorable control.
    """
    v = min(1.0, max(0.0, threshold))
    terms: dict[str, float] = {}
    for term in envelope.terms:
        x = linear_inverse(v, best=term.best, worst=term.worst)
        if term.is_integer:
            x = snap_toward_best(x, term)
        terms[term.name] = x
    return Offer(terms=terms)


class LogrollComparison(BaseModel):
    """Head-to-head result at one buyer threshold."""

    model_config = {"frozen": True}

    threshold: float
    buyer_utility_logroll: float
    buyer_utility_split: float
    supplier_utility_logroll: float
    supplier_utility_split: float

    @property
    def supplier_gain(self) -> float:
        """Extra supplier utility logrolling captures at matched buyer cost."""
        return self.supplier_utility_logroll - self.supplier_utility_split


def compare_at_threshold(
    buyer_env: Envelope,
    supplier_env: Envelope,
    threshold: float,
    supplier_model: SupplierModel,
) -> LogrollComparison:
    """Build both packages at ``threshold`` and score each under both envelopes.

    ``supplier_model`` is the belief the logrolling builder uses (in the eval,
    the oracle belief = the supplier's true priorities). The uniform-split
    builder ignores it. Both packages are scored under ``supplier_env`` (the
    hidden truth) to get the real supplier utility.
    """
    priorities = supplier_model.priorities(buyer_env)
    logroll_offer, _, _ = fill_package(buyer_env, threshold, priorities)
    split_offer = uniform_split_package(buyer_env, threshold)

    return LogrollComparison(
        threshold=threshold,
        buyer_utility_logroll=buyer_env.utility(logroll_offer),
        buyer_utility_split=buyer_env.utility(split_offer),
        supplier_utility_logroll=_score_under(supplier_env, logroll_offer),
        supplier_utility_split=_score_under(supplier_env, split_offer),
    )


def _score_under(envelope: Envelope, offer: Offer) -> float:
    """Score ``offer`` under ``envelope``, filling any absent term from its floor."""
    projected = Offer(
        terms={n: offer.terms.get(n, envelope.term_map[n].worst) for n in envelope.term_map}
    )
    return envelope.utility(projected)
