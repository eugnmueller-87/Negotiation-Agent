"""Supplier agents.

``SupplierAgent`` is the seam: a two-method interface the engine loop drives.
``ParametricSupplier`` is the v0 deterministic bot; a Claude-backed supplier
implements the same ``respond`` in v1 with no upstream change.

The supplier is parameterized by *its own* :class:`Envelope` — the schema is
side-agnostic (same term names, ``best``/``worst`` reflecting the supplier's
preferences, its own ``reservation_utility``), so the supplier reuses
:func:`fill_package` against its own mandate. That envelope and the persona are
the hidden truth; the engine never sees them.
"""

from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel

from negotiation_agent.envelope import Envelope, Offer
from negotiation_agent.packages import InfeasiblePackage, fill_package
from negotiation_agent.simulator.personas import PersonaConfig
from negotiation_agent.supplier_model import SupplierModel


class SupplierMove(BaseModel):
    model_config = {"frozen": True}

    kind: Literal["offer", "accept", "reject"]
    offer: Offer | None = None


class SupplierAgent(Protocol):
    def respond(
        self, round_index: int, buyer_offer: Offer | None, *, max_rounds: int = 8
    ) -> SupplierMove:
        """React to the buyer's standing package.

        ``buyer_offer`` is the engine's last counter (``None`` only if the engine
        somehow opened with no offer, which the loop never does). The v1
        Claude-backed supplier implements exactly this signature.
        """
        ...


class ParametricSupplier:
    """Deterministic supplier bot driven by a hidden envelope + persona.

    ``buyer_priority_guess`` is the supplier's belief about what the *buyer*
    cares about, used to shape its own counteroffers (the mirror of the engine's
    SupplierModel). Defaults to uniform.
    """

    def __init__(
        self,
        envelope: Envelope,
        persona: PersonaConfig,
        buyer_priority_guess: dict[str, float] | None = None,
    ) -> None:
        self.envelope = envelope
        self.persona = persona
        self._priorities = buyer_priority_guess or SupplierModel.uniform(envelope).priorities(
            envelope
        )
        self._caps: dict[str, float] = {}
        self._last_offer: Offer | None = None

    def _threshold(self, round_index: int, max_rounds: int) -> float:
        env, p = self.envelope, self.persona
        t = min(max(round_index, 0), max_rounds)
        frac = t / max_rounds
        span = env.target_utility - env.reservation_utility
        return float(env.reservation_utility + span * (1.0 - frac**p.beta_s))

    def respond(
        self, round_index: int, buyer_offer: Offer | None, *, max_rounds: int = 8
    ) -> SupplierMove:
        # With no buyer offer on the table there is nothing to react to; hold.
        if buyer_offer is None:
            return SupplierMove(kind="offer", offer=self._last_offer)
        # Supplier utility of the buyer's package under the hidden envelope.
        u_s = self.envelope.utility(self._project(buyer_offer))
        theta_s = self._threshold(round_index, max_rounds)

        # Accept if the buyer's standing package clears the (margin-relaxed) schedule.
        if u_s >= theta_s - self.persona.accept_margin:
            return SupplierMove(kind="accept")

        # Evasive stall: on non-multiple rounds, repeat the previous offer verbatim.
        if (
            self.persona.stall_period > 1
            and round_index % self.persona.stall_period != 0
            and self._last_offer is not None
        ):
            return SupplierMove(kind="offer", offer=self._last_offer)

        # Otherwise counter on its own Boulware schedule via the shared search.
        target = max(theta_s, self.envelope.reservation_utility)
        try:
            offer, _, planned_v = fill_package(
                self.envelope, target, self._priorities, caps=self._caps or None
            )
        except InfeasiblePackage:  # pragma: no cover - defensive
            # Unreachable for the v0 personas: a supplier targets at least its own
            # reservation, and its caps never ratchet below that, so fill_package
            # always finds a package. Kept so a v1 belief-updating supplier degrades
            # gracefully (hold last offer, or walk at the deadline) instead of
            # crashing mid-negotiation.
            if self._last_offer is not None:
                return SupplierMove(kind="offer", offer=self._last_offer)
            if self.persona.walkaway_at_deadline and round_index >= max_rounds:
                return SupplierMove(kind="reject")
            return SupplierMove(kind="offer", offer=self.envelope.reservation_offer())

        self._caps = {n: min(self._caps.get(n, 1.0), planned_v[n]) for n in planned_v}
        self._last_offer = offer
        return SupplierMove(kind="offer", offer=offer)

    def _project(self, buyer_offer: Offer) -> Offer:
        """Restrict a buyer offer to this supplier's envelope terms (drop extras,
        fill any gap from the supplier's reservation as a conservative floor)."""
        res = self.envelope.reservation_offer()
        return Offer(
            terms={
                name: buyer_offer.terms.get(name, res.terms[name])
                for name in self.envelope.term_map
            }
        )
