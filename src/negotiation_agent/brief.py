"""The move brief — a DETERMINISTIC, Python-built account of the engine's move.

The buyer's message drafter (the LLM) needs something to *say*: which term moved,
in which direction, what was held, and the honest trade rationale. That narrative
is a **causal interpretation of the deterministic engine's output** — so it must be
built in Python, never by the LLM. If the model authored it, the model would be
explaining what the deterministic engine decided, under a header that reads
"DETERMINISTIC ENGINE" — the thesis inverted.

``build_move_brief`` consumes an existing :class:`EngineDecision` plus the offer
that was on the table *before* this decision (the inbound ``last_counter``). It
adds no field to the engine and calls no model. See
``docs/peitho-v2-architecture.md`` §4.1–4.3.

Four corrections proved against ``engine.py`` / ``envelope.py`` are baked in:
  1. Diff against the INBOUND baseline (the ``last_counter`` passed *into* ``decide``),
     not the returned state — the engine sets ``next_state.last_counter`` to the
     package it just built, so diffing against it yields zero moved terms every turn.
  2. Threshold moves on a MATERIAL delta in display units and report at most the
     top 1–2 moved terms; the Phase-C re-solve nudges nearly every term every round.
  3. ``direction_word`` comes from ``sign(new − old)``, never from the term type —
     direction is per-envelope data (``TermSpec.direction``), a mandate can invert it.
  4. Use BUYER satisfaction (``TermSpec.value`` is buyer utility), never a "supplier
     gap" — the engine has no supplier ideal, only a scalar appetite belief.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from negotiation_agent.engine import EngineDecision, Outcome
from negotiation_agent.envelope import Envelope, Offer, TermSpec

# A term counts as "moved" only if its value changed by more than this fraction of
# its scored span — above the integer-snap / Phase-C rounding jitter, so the brief
# reports strategy, not arithmetic noise.
_MATERIAL_DELTA_FRAC = 0.02
# A supplier appetite is "high" only if it stands materially above the belief mean;
# below this the rationale falls back to the buyer-side story (never asserts a
# supplier preference the engine didn't infer).
_APPETITE_ABOVE_MEAN = 0.05

RoundBand = Literal["opening", "early", "mid", "late"]
Pressure = Literal["anchor", "hold_firm", "reciprocity", "deadline", "handoff"]


class MovedTerm(BaseModel):
    model_config = {"frozen": True}

    name: str
    from_display: str
    to_display: str
    direction_word: str
    role: Literal["concession", "ask"]


class HeldTerm(BaseModel):
    model_config = {"frozen": True}

    name: str
    display: str


class TradeAxis(BaseModel):
    model_config = {"frozen": True}

    conceded_on: list[str] = Field(default_factory=list)
    held_on: list[str] = Field(default_factory=list)
    rationale: str = ""


class SatisfactionNote(BaseModel):
    model_config = {"frozen": True}

    name: str
    status: Literal["close", "apart"]


class MoveBrief(BaseModel):
    """What the drafter is told about the engine's move. No internal figures."""

    model_config = {"frozen": True}

    outcome: Literal["COUNTER", "ACCEPT", "ESCALATE"]
    is_opening: bool
    round_band: RoundBand
    pressure: Pressure
    approved_numbers: dict[str, float] = Field(default_factory=dict)
    moved_terms: list[MovedTerm] = Field(default_factory=list)
    held_terms: list[HeldTerm] = Field(default_factory=list)
    trade_axis: TradeAxis = Field(default_factory=TradeAxis)
    buyer_satisfaction: list[SatisfactionNote] = Field(default_factory=list)
    reason_tag: str = ""

    def sentence(self) -> str:
        """A one-line human summary for the reasoning drawer (deterministic)."""
        if self.outcome == "ESCALATE":
            return "Escalating to a human buyer — no move made."
        if self.is_opening:
            return "Opening the negotiation with our target package."
        if self.outcome == "ACCEPT":
            return "Accepting — the supplier's offer clears our line."
        if self.moved_terms:
            m = self.moved_terms[0]
            held = (
                self.held_terms[0].name.replace("_", " ")
                if self.held_terms
                else "our priorities"
            )
            return (
                f"Conceding {m.name.replace('_', ' ')} ({m.from_display} → {m.to_display}) "
                f"while holding {held}."
            )
        return "Holding position; only minor adjustments this round."


def _display(term: TermSpec, x: float) -> str:
    if term.name == "price":
        return f"€{x:.2f}"
    if term.name == "payment_days":
        return f"net {int(round(x))}"
    if term.name == "contract_months":
        return f"{int(round(x))} months"
    return f"{int(round(x))}" if term.is_integer else f"{x:.2f}"


def _direction_word(name: str, delta: float) -> str:
    later = delta > 0
    if name == "payment_days":
        return "later" if later else "sooner"
    if name == "price":
        return "higher" if later else "lower"
    if name == "contract_months":
        return "longer" if later else "shorter"
    return "up" if later else "down"


def _round_band(round_index: int, max_rounds: int) -> RoundBand:
    if round_index == 0:
        return "opening"
    frac = round_index / max_rounds
    if frac < 0.34:
        return "early"
    if frac < 0.75:
        return "mid"
    return "late"


_BAND_PRESSURE: dict[RoundBand, Pressure] = {
    "opening": "anchor",
    "early": "hold_firm",
    "mid": "reciprocity",
    "late": "deadline",
}


def _pressure(outcome: Outcome, band: RoundBand) -> Pressure:
    if outcome is Outcome.ESCALATE:
        return "handoff"
    return _BAND_PRESSURE[band]


def _appetite_high(priorities: dict[str, float], name: str) -> bool:
    """True when the belief's appetite for ``name`` is materially above the mean."""
    if not priorities:
        return False
    mean = sum(priorities.values()) / len(priorities)
    return priorities.get(name, 0.0) - mean > _APPETITE_ABOVE_MEAN


def build_move_brief(
    decision: EngineDecision,
    envelope: Envelope,
    prev_counter: Offer | None,
    max_rounds: int,
    *,
    priorities: dict[str, float] | None = None,
) -> MoveBrief:
    """Build the deterministic move brief for a decision.

    ``prev_counter`` is the offer that was on the table *before* this decision —
    the ``last_counter`` the caller passed *into* ``decide`` (NOT the returned
    ``next_state.last_counter``). ``priorities`` is the engine's supplier-appetite
    belief; when omitted, the rationale uses only the buyer-side story.
    """
    band = _round_band(decision.round_index, max_rounds)
    pressure = _pressure(decision.outcome, band)
    reason_tag = decision.reason.split(":", 1)[0]

    if decision.outcome is Outcome.ESCALATE:
        return MoveBrief(
            outcome="ESCALATE", is_opening=False, round_band=band, pressure="handoff",
            reason_tag=reason_tag,
        )

    if decision.outcome is Outcome.ACCEPT:
        return MoveBrief(
            outcome="ACCEPT", is_opening=False, round_band=band, pressure=pressure,
            approved_numbers=dict(decision.approved_numbers), reason_tag=reason_tag,
        )

    # COUNTER — diff the new package against the INBOUND baseline.
    counter = decision.counter
    assert counter is not None  # COUNTER always carries a package
    is_opening = decision.reason == "opening_anchor"

    moved: list[tuple[float, MovedTerm]] = []
    held: list[HeldTerm] = []
    for term in envelope.terms:
        new_x = counter.terms[term.name]
        if is_opening or prev_counter is None or term.name not in prev_counter.terms:
            held.append(HeldTerm(name=term.name, display=_display(term, new_x)))
            continue
        old_x = prev_counter.terms[term.name]
        span = abs(term.best - term.worst)
        delta = new_x - old_x
        if span == 0 or abs(delta) / span <= _MATERIAL_DELTA_FRAC:
            held.append(HeldTerm(name=term.name, display=_display(term, new_x)))
            continue
        is_concession = term.value(new_x) < term.value(old_x)
        moved.append(
            (
                abs(delta) / span,
                MovedTerm(
                    name=term.name,
                    from_display=_display(term, old_x),
                    to_display=_display(term, new_x),
                    direction_word=_direction_word(term.name, delta),
                    role="concession" if is_concession else "ask",
                ),
            )
        )

    moved.sort(key=lambda pair: pair[0], reverse=True)
    top_moved = [mt for _, mt in moved[:2]]
    conceded = [mt.name for mt in top_moved if mt.role == "concession"]
    held_names = [h.name for h in held]

    rationale = _trade_rationale(conceded, priorities)
    satisfaction = _buyer_satisfaction(envelope, counter)

    return MoveBrief(
        outcome="COUNTER",
        is_opening=is_opening,
        round_band=band,
        pressure=pressure,
        approved_numbers=dict(decision.approved_numbers),
        moved_terms=top_moved,
        held_terms=held,
        trade_axis=TradeAxis(conceded_on=conceded, held_on=held_names, rationale=rationale),
        buyer_satisfaction=satisfaction,
        reason_tag=reason_tag,
    )


def _trade_rationale(conceded: list[str], priorities: dict[str, float] | None) -> str:
    if not conceded:
        return "Holding our position this round."
    term = conceded[0].replace("_", " ")
    # Assert "the supplier values X" only when the belief actually says so; else
    # fall back to the buyer-side story the engine can always stand behind.
    if priorities is not None and _appetite_high(priorities, conceded[0]):
        return (
            f"{term} is lower-priority for us and something the supplier weights more, "
            "so it's where we have room to move — our priority terms stay put."
        )
    return (
        f"{term} is lower-priority for us, so it's where we have room — "
        "our priority terms stay put."
    )


def _buyer_satisfaction(envelope: Envelope, offer: Offer) -> list[SatisfactionNote]:
    notes: list[SatisfactionNote] = []
    for term in envelope.terms:
        v = term.value(offer.terms[term.name])
        notes.append(
            SatisfactionNote(name=term.name, status="close" if v >= 0.75 else "apart")
        )
    return notes
