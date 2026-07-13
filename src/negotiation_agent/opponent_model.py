"""Opponent modelling — infer the supplier's priorities from how they actually move.

A senior negotiator reads the room: they watch which terms the counterpart gives ground on and
which they defend, and they route their own concessions accordingly. This module does exactly that,
as pure deterministic math — no LLM, instant, auditable, reproducible.

The insight: **a supplier reveals its priorities by what it refuses to move on.** Across a sequence
of supplier offers, a term the supplier concedes freely (its value to the *buyer* keeps improving)
is one the supplier does not care much about — the buyer should spend its concession budget
elsewhere and take the free ground here. A term the supplier holds firm on is one it values highly
— trading a buyer-light term for movement there is the logrolling win.

The output is a :class:`SupplierModel` belief (the same structure the engine already consumes for
its logrolling search), so this drops into the existing seam: the engine's concession routing gets
smarter round by round without any change to how it *decides* — the mandate, the numbers, and the
reservation floor stay entirely owned by the deterministic engine.

Everything here is a pure function of the observed offer history and the envelope. No mutation, no
network, no randomness. It NEVER sees or infers the buyer's floor — it only reads supplier
behaviour.
"""

from __future__ import annotations

from pydantic import BaseModel

from .envelope import Direction, Envelope, Offer
from .supplier_model import SupplierModel

# Below this fractional movement on a term, we treat the supplier as having HELD (not conceded).
# Used two ways: _MOVED_EPS for a single-step "did they move at all" check, and _HELD_FIRM_FRAC
# for the cumulative "have they defended this term across the whole negotiation" read.
_MOVED_EPS = 0.01
_HELD_FIRM_FRAC = 0.05


class TermMovement(BaseModel):
    """How the supplier moved on one term between two consecutive offers, from the BUYER's view.

    ``toward_buyer`` in [0, 1] is the fraction of the term's full range the supplier gave up in the
    buyer's favour this step (0 = held firm, 1 = jumped to the buyer's ideal). It's direction-aware:
    a supplier lowering a MINIMIZE price and a supplier raising a MAXIMIZE payment-days both
    register
    as positive movement toward the buyer."""

    model_config = {"frozen": True}

    term: str
    toward_buyer: float


def _fraction_toward_buyer(
    env_term_best: float, env_term_worst: float, old: float, new: float
) -> float:
    """Signed fraction of the term's range the move covered toward the buyer's ``best`` end.

    Positive = the supplier moved in the buyer's favour; negative = they walked it back (rare, but
    we clamp its effect). Normalised by the term's own span so terms on different scales compare."""
    span = env_term_best - env_term_worst
    if span == 0:
        return 0.0
    # (new - old) / span is positive when new is closer to best than old, for both directions,
    # because best-worst carries the sign (best<worst for MINIMIZE, best>worst for MAXIMIZE).
    return (new - old) / span


def movements_between(env: Envelope, prev: Offer, curr: Offer) -> list[TermMovement]:
    """The per-term movement from ``prev`` to ``curr``, for every term both offers state."""
    out: list[TermMovement] = []
    for term in env.terms:
        name = term.name
        if name not in prev.terms or name not in curr.terms:
            continue
        frac = _fraction_toward_buyer(term.best, term.worst, prev.terms[name], curr.terms[name])
        out.append(TermMovement(term=name, toward_buyer=frac))
    return out


def infer_appetite(env: Envelope, supplier_offers: list[Offer]) -> SupplierModel:
    """Infer supplier concession appetite per term from the sequence of supplier offers.

    ``appetite[name]`` (the belief the engine's logrolling routes concessions by) is HIGH for a term
    the supplier defended (moved little) and LOW for one it conceded freely — because the engine
    should spend its own concessions where the supplier gives ground cheaply, and hold where the
    supplier holds. Fewer than two offers → no movement observed → a uniform (no-information)
    belief, exactly what the engine starts from today.

    Pure: same offers + envelope always yield the same belief. Never inspects the buyer's mandate
    beyond the term ranges needed to normalise movement.
    """
    if len(supplier_offers) < 2:
        return SupplierModel.uniform(env)

    # Accumulate total observed movement-toward-buyer per term across all consecutive pairs.
    # CUMULATIVE (not per-step average) is the right signal: a supplier that conceded a term's
    # WHOLE range over the negotiation values it little, regardless of how many steps it took.
    total_move: dict[str, float] = {t.name: 0.0 for t in env.terms}
    seen: dict[str, int] = {t.name: 0 for t in env.terms}
    for prev, curr in zip(supplier_offers, supplier_offers[1:], strict=False):
        for mv in movements_between(env, prev, curr):
            # only positive movement (toward buyer) counts as a concession; walking back is
            # clamped to 0 so a supplier can't lower our inferred appetite by feinting backward
            total_move[mv.term] += max(0.0, mv.toward_buyer)
            seen[mv.term] += 1

    # Convert cumulative movement → appetite. A term the supplier conceded fully (cumulative
    # movement → its whole range) gets appetite → 0 (they don't value it); a term they held (near
    # zero movement) gets appetite → 1 (they defend it). A term never observed keeps the neutral
    # prior. This is a direct read, NOT blended toward neutral, so a clear signal isn't muted —
    # a supplier that gives up all of payment while defending price must show a sharp split.
    appetite: dict[str, float] = {}
    for term in env.terms:
        name = term.name
        if seen[name] == 0:
            appetite[name] = 0.5
            continue
        conceded = min(1.0, total_move[name])  # fraction of the range given up in total
        appetite[name] = 1.0 - conceded
    return SupplierModel(appetite=appetite, source="simulator")


def held_firm(env: Envelope, supplier_offers: list[Offer], *, term: str) -> bool:
    """True if the supplier has effectively not moved on ``term`` across the observed offers —
    the signal that ``term`` is one they value and the buyer should trade *for*, not spend on."""
    if len(supplier_offers) < 2:
        return False
    total = 0.0
    for prev, curr in zip(supplier_offers, supplier_offers[1:], strict=False):
        for mv in movements_between(env, prev, curr):
            if mv.term == term:
                total += max(0.0, mv.toward_buyer)
    return total < _HELD_FIRM_FRAC


def reciprocity(env: Envelope, supplier_offers: list[Offer]) -> float:
    """A [0, 1] read of how much the supplier is *moving overall* in the buyer's favour recently —
    the signal an adaptive concession schedule uses to decide whether to reciprocate (they're
    engaging → we can give a little) or hold firm (they're stalling → we hold).

    0 = the supplier is holding firm / not conceding; 1 = large recent movement toward the buyer.
    Uses only the most recent step so it tracks current behaviour, not the whole history.
    """
    if len(supplier_offers) < 2:
        return 0.0
    prev, curr = supplier_offers[-2], supplier_offers[-1]
    moves = movements_between(env, prev, curr)
    if not moves:
        return 0.0
    forward = sum(max(0.0, m.toward_buyer) for m in moves) / len(moves)
    return min(1.0, forward)


# Keep Direction imported for callers that reason about term direction alongside movement.
__all__ = [
    "TermMovement",
    "movements_between",
    "infer_appetite",
    "held_firm",
    "reciprocity",
    "Direction",
]
