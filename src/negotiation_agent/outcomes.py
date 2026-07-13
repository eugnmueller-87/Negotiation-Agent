"""Negotiation outcome logging — the memory that lets the engine start smarter over time.

The engine adapts *within* a negotiation (opponent modelling); this module is how it learns
*across* negotiations. It records what happened when a deal closed — coarsely and pseudonymously —
so :mod:`negotiation_agent.priors` can compute better starting points (opening anchors, supplier-
appetite priors, realistic settlement expectations) for the NEXT negotiation in the same category.

The learning is strictly OFF-HANDS: an outcome record NEVER decides anything. It feeds priors that
*inform* the deterministic engine's starting point; the human still owns the mandate (target /
reservation) and the engine still owns every decision. Nothing here is a model that negotiates.

PRIVACY (per data-privacy-procurement.md — supplier/spend data is GDPR-relevant):
  - We DO NOT store raw supplier names, contact PII, or euro amounts. The learnable signal is
    dimensionless: a coarse ``category`` label, per-term settled UTILITY fractions in [0, 1], which
    terms the supplier conceded, round count, and outcome. A settled utility of 0.72 on "price" says
    where the deal landed on the buyer's own [worst, best] scale — it reveals nothing about the
    actual price or the counterparty.
  - The store is a local, append-only JSONL file the buyer controls — no external transmission.
  - This keeps the derived artifact free of PII while still capturing everything the priors need.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .engine import EngineDecision, Outcome
from .envelope import Envelope, Offer

# The learnable outcome of a closed negotiation. Every field is coarse/dimensionless on purpose —
# see the privacy note in the module docstring.
OutcomeKind = Literal["accepted", "escalated"]


class NegotiationOutcome(BaseModel):
    """One closed negotiation, recorded coarsely for cross-deal learning. No PII, no prices."""

    model_config = {"frozen": True}

    category: str = "unknown"  # coarse procurement category (e.g. "corrugated_packaging")
    outcome: OutcomeKind
    rounds: int = Field(ge=0)
    # the total buyer utility the deal settled at, in [0, 1] — where it landed on the buyer's own
    # scale, NOT a price. None if the negotiation escalated without a deal.
    settled_utility: float | None = None
    # per-term settled UTILITY fraction in [0, 1] — how good each term ended up for the buyer.
    settled_term_utilities: dict[str, float] = Field(default_factory=dict)
    # which terms the supplier conceded on (buyer's view) — the observed appetite signal, as a list
    # of term names. Learned into a prior over which terms suppliers in this category give cheaply.
    conceded_terms: list[str] = Field(default_factory=list)
    target_utility: float = Field(ge=0.0, le=1.0)  # the mandate's aspiration (prior context)
    reservation_utility: float = Field(ge=0.0, le=1.0)  # the mandate's floor (context)


def outcome_from_close(
    envelope: Envelope,
    final_decision: EngineDecision,
    supplier_offers: list[Offer],
    *,
    category: str = "unknown",
) -> NegotiationOutcome:
    """Build a privacy-minimized outcome record from a closed negotiation.

    Reads only the buyer-side utility of where the deal landed and which terms the supplier moved on
    — never a raw price or supplier identity. ``final_decision`` is the ACCEPT or ESCALATE that
    closed it; ``supplier_offers`` is the observed offer sequence (for the concession signal).
    """
    kind: OutcomeKind = "accepted" if final_decision.outcome is Outcome.ACCEPT else "escalated"

    settled_utility: float | None = None
    term_utils: dict[str, float] = {}
    if final_decision.outcome is Outcome.ACCEPT and final_decision.approved_numbers:
        accepted = Offer(terms=dict(final_decision.approved_numbers))
        # per-term utility fraction (dimensionless), and the weighted total
        for term in envelope.terms:
            if term.name in accepted.terms:
                term_utils[term.name] = round(term.value(accepted.terms[term.name]), 4)
        try:
            settled_utility = round(envelope.utility(accepted), 4)
        except KeyError:
            settled_utility = None

    conceded = _conceded_terms(envelope, supplier_offers)

    return NegotiationOutcome(
        category=category,
        outcome=kind,
        rounds=final_decision.round_index,
        settled_utility=settled_utility,
        settled_term_utilities=term_utils,
        conceded_terms=conceded,
        target_utility=envelope.target_utility,
        reservation_utility=envelope.reservation_utility,
    )


def _conceded_terms(envelope: Envelope, supplier_offers: list[Offer]) -> list[str]:
    """Which terms the supplier moved toward the buyer on, across the observed offers. Reuses the
    opponent-model movement math so the concession signal is consistent with in-negotiation
    inference. Returns term names, no values."""
    if len(supplier_offers) < 2:
        return []
    from .opponent_model import movements_between

    moved: set[str] = set()
    for prev, curr in zip(supplier_offers, supplier_offers[1:], strict=False):
        for mv in movements_between(envelope, prev, curr):
            if mv.toward_buyer > 0.01:  # a real concession, not rounding drift
                moved.add(mv.term)
    return sorted(moved)


# ── the local append-only store ──────────────────────────────────────────────────
class OutcomeStore:
    """A local, append-only JSONL log of negotiation outcomes the buyer controls.

    No external transmission, no PII — see the module privacy note. Append is atomic-per-line
    (one JSON object per line) so a crash mid-write can at worst drop the last record, never
    corrupt earlier ones. Reading tolerates a malformed trailing line."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def append(self, outcome: NegotiationOutcome) -> None:
        """Append one outcome record. Creates the file/parents if absent."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(outcome.model_dump(mode="json"), ensure_ascii=False)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def load(self) -> list[NegotiationOutcome]:
        """Load all outcome records. A malformed line is skipped (logged by the caller if needed),
        never raised — a corrupt tail must not break learning from the good history before it."""
        if not self._path.is_file():
            return []
        out: list[NegotiationOutcome] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(NegotiationOutcome.model_validate_json(line))
            except ValueError:
                continue  # skip a malformed/partial record, keep the rest
        return out

    def load_category(self, category: str) -> list[NegotiationOutcome]:
        """Outcomes for one category — the slice the priors for a new negotiation learn from."""
        return [o for o in self.load() if o.category == category]
