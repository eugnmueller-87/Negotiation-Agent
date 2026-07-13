"""Negotiation strategy — adaptive concession + tactic selection, as deterministic code.

The deterministic engine already concedes on a fixed Boulware schedule. A *senior* negotiator does
more: they read whether the counterpart is engaging or stalling and adjust their pace, and they name
the move they're making (anchor / hold / concede / trade / walk) with a reason. This module adds
that
judgment — pure Python, instant, auditable, and STRICTLY bounded so it can never cross the mandate's
reservation floor. No LLM: negotiation strategy is logic, and encoding it as logic makes it
reproducible and impossible to talk past the floor.

Two pieces:
  - :func:`adapt_threshold` — nudge the base Boulware acceptance threshold by how much the supplier
    is reciprocating. If they move toward the buyer, the engine may ease slightly to close; if they
    are stalling, it holds firmer. The result is ALWAYS clamped to ``[reservation, target]`` — the
    adaptation only ever moves *within* the mandate the engine already owns, never below the floor.
  - :func:`choose_tactic` — label the move the engine is making and why, from the observable state.
    Advisory: it explains the decision the engine's own rules produced; it does not override them.

Nothing here sees or sets a number the engine didn't compute. It reshapes the *pace* of concession
inside the mandate; the engine still owns accept/counter/escalate and every value in the counter.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

# How far reciprocity may nudge the threshold, as a fraction of the target–reservation span. Small
# and bounded on purpose: adaptation tunes the pace, it does not rewrite the mandate. At the extreme
# a fully-reciprocating supplier eases the threshold by this fraction of the span (never below the
# floor); a fully-stalling one firms it up by the same fraction (never above target).
_MAX_NUDGE_FRAC = 0.15

# reciprocity above this reads as "the supplier is genuinely engaging"; below _STALL_BELOW as
# "stalling". Between them is neutral — no nudge.
_ENGAGING_ABOVE = 0.15
_STALL_BELOW = 0.03

Tactic = Literal["anchor", "hold", "concede", "trade", "walk"]


class StrategyRead(BaseModel):
    """The strategy layer's read of a turn: the adjusted threshold, the tactic, and WHY — the
    audit trail a human reviewer sees. ``base_threshold``/``adjusted_threshold`` show exactly how
    far (and never past the floor) the pace was nudged."""

    model_config = {"frozen": True}

    tactic: Tactic
    base_threshold: float
    adjusted_threshold: float
    reciprocity: float
    rationale: str


def adapt_threshold(
    base_threshold: float,
    *,
    reciprocity: float,
    reservation: float,
    target: float,
) -> float:
    """Nudge the Boulware acceptance threshold by supplier reciprocity, clamped to the mandate.

    Engaging supplier (moving toward the buyer) → ease the threshold slightly toward reservation so
    a close is reachable sooner. Stalling supplier → firm it up toward target so the engine doesn't
    give ground to someone who isn't. The nudge is at most ``_MAX_NUDGE_FRAC`` of the span, and the
    return is ALWAYS clamped to ``[reservation, target]`` — the strategy can NEVER push the engine
    to accept below its floor. Pure and monotone in ``reciprocity``.
    """
    span = target - reservation
    if span <= 0:
        return max(reservation, min(target, base_threshold))

    if reciprocity >= _ENGAGING_ABOVE:
        # ease down toward reservation, proportional to how much they're engaging (capped)
        strength = min(1.0, (reciprocity - _ENGAGING_ABOVE) / (1.0 - _ENGAGING_ABOVE))
        adjusted = base_threshold - _MAX_NUDGE_FRAC * span * strength
    elif reciprocity <= _STALL_BELOW:
        # firm up toward target — hold ground against a staller
        strength = min(1.0, (_STALL_BELOW - reciprocity) / max(_STALL_BELOW, 1e-9))
        adjusted = base_threshold + _MAX_NUDGE_FRAC * span * strength
    else:
        adjusted = base_threshold

    # HARD invariant: the adjusted threshold never leaves the mandate band. The engine's floor
    # holds. NOTE: load-bearing for floor safety, including the NaN case — max(reservation,
    # min(target, NaN)) returns `target` (CPython min/max return the first arg on a NaN compare),
    # so a poisoned `adjusted` firms UP to target, never below reservation. Keep this exact form.
    return max(reservation, min(target, adjusted))


def choose_tactic(
    *,
    is_opening: bool,
    incoming_utility: float | None,
    adjusted_threshold: float,
    reservation: float,
    reciprocity: float,
    supplier_held_terms: list[str],
    at_deadline: bool,
) -> tuple[Tactic, str]:
    """Name the move the engine is making this turn and give a one-line rationale.

    Advisory / explanatory — it reads the same state the engine decides on and labels it, so a human
    sees *why* the engine anchored / held / conceded / traded / walked. It never changes the
    outcome.
    """
    if is_opening:
        return "anchor", (
            "Opening with a logrolled package at target — establish the reference point."
        )

    u = incoming_utility if incoming_utility is not None else -1.0

    if at_deadline and u < reservation:
        return "walk", (
            f"At the deadline the best offer (u={u:.2f}) is below the walk-away floor "
            f"({reservation:.2f}) — escalate to a human rather than accept a bad deal."
        )

    if u >= adjusted_threshold:
        return "concede", (
            f"The offer (u={u:.2f}) clears the current bar ({adjusted_threshold:.2f}) — accept; "
            "holding out for more risks the relationship for little gain."
        )

    if supplier_held_terms:
        held = ", ".join(t.replace("_", " ") for t in supplier_held_terms)
        return "trade", (
            f"The supplier is defending {held}; concede on a term they give cheaply and hold "
            "there — logroll to keep total value up."
        )

    if reciprocity <= _STALL_BELOW:
        return "hold", (
            "The supplier is not moving — hold firm; conceding now rewards a stall and shifts the "
            "midpoint against us."
        )

    return "concede", (
        f"Countering down the Boulware curve toward {adjusted_threshold:.2f} as the round advances."
    )
