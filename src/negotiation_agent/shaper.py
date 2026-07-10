"""The deterministic finding → mandate transform — the thesis, applied to mandate
construction.

Contract intelligence and supplier risk can *shape* the negotiation mandate, but the
shaping must never break the guarantee the product rests on: the LLM never edits the
envelope, never invents a number, never moves the floor. So the transform here is a
**pure function** — an LLM extracts facts, a table-driven rule set (all constants) maps
them to **bounded, reversible** envelope deltas, and a human approves the diff before the
mandate is signed. See ``docs/contract-intelligence-architecture.md`` §3.

Three properties make "code decides the mandate" true, not aspirational:

1. ``apply_adjustments`` has no I/O and no LLM call — same inputs give byte-identical
   output. That reproducibility is what makes the human's signature meaningful.
2. Rules read only a discrete ``assurance`` (``confirmed``/``probable``/``unknown``),
   never the LLM's continuous ``confidence`` — so sampling drift can't change the
   envelope. A delta is a fixed constant from the table, never ``k × confidence``.
3. Every delta is re-validated by the real :class:`Envelope` constructor. The transform
   *physically cannot* emit an invalid mandate — pydantic rejects it first, loudly.
"""

from __future__ import annotations

import datetime as _dt
from typing import Literal

from pydantic import BaseModel, Field

from negotiation_agent.envelope import Direction, Envelope, Offer, TermSpec, TermType

# ── tunable constants — the transform's only numbers. Never LLM-authored. ──────
W_COMPLIANCE = 0.10  # weight bump for a compliance-critical term
W_HEDGE = 0.08  # weight for an added risk-hedge term
W_GIVE = 0.06  # weight for an added trade-bait ("give") term
_EXPIRING_SOON_DAYS = 30
_EXPIRING_FAR_DAYS = 180
_MIN_WEIGHT = 1e-6  # TermSpec weight is gt=0.0; clamp to this floor before renormalising
_TARGET_RESERVATION_GAP = 1e-3  # keep reservation strictly below target

Assurance = Literal["confirmed", "probable", "unknown"]
Role = Literal["give", "hold", "hedge"]
Severity = Literal["low", "medium", "high", "critical"]


# ── delta types — atomic, bounded, reversible ─────────────────────────────────
class WeightBump(BaseModel):
    model_config = {"frozen": True}
    kind: Literal["weight_bump"] = "weight_bump"
    term_name: str
    delta: float


class AddTerm(BaseModel):
    model_config = {"frozen": True}
    kind: Literal["add_term"] = "add_term"
    spec: TermSpec
    appetite: float = Field(ge=0.0, le=1.0, default=0.5)  # supplier-appetite hint


class TightenBounds(BaseModel):
    model_config = {"frozen": True}
    kind: Literal["tighten_bounds"] = "tighten_bounds"
    term_name: str
    new_worst: float  # shrinks the span toward `best` (direction-aware in the applier)


class ShiftTarget(BaseModel):
    model_config = {"frozen": True}
    kind: Literal["shift_target"] = "shift_target"
    target_delta: float = 0.0
    reservation_delta: float = 0.0


class AddGate(BaseModel):
    """A binary must-have that is NOT logrolled — surfaced to the human + drafter as a
    required clause. Does not change the envelope (v1); see architecture doc §3.1."""

    model_config = {"frozen": True}
    kind: Literal["add_gate"] = "add_gate"
    gate_id: str
    label: str
    severity: Literal["required", "preferred"] = "required"


Delta = WeightBump | AddTerm | TightenBounds | ShiftTarget | AddGate


class ProposedAdjustment(BaseModel):
    """One rule's proposal: what the model found, which rule moved it, the bounded
    delta, and the rationale the human reviews before accepting."""

    model_config = {"frozen": True}

    rule_id: str
    severity: Severity
    role: Role
    delta: Delta
    rationale: str
    default_accepted: bool = True


class MandateConflict(Exception):
    """The accepted subset can't produce a valid envelope — names the offending rules
    so the human deselects one, rather than a rule being silently dropped."""

    def __init__(self, rule_ids: list[str], detail: str) -> None:
        self.rule_ids = rule_ids
        super().__init__(f"{detail} (rules: {', '.join(rule_ids)})")


# ── the date parser — a bounded, tested helper (never a bare parse in a predicate) ──
def days_until(date_text: str | None, *, today: _dt.date) -> int | None:
    """Days from ``today`` to a free-text date, or None if unparseable.

    Deliberately conservative: tries a small set of unambiguous ISO/EU formats and
    returns None on anything it can't be sure of, so a rule simply does not fire on an
    ambiguous date rather than throwing on a real contract.
    """
    if not date_text:
        return None
    text = date_text.strip()
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%d %B %Y", "%B %d, %Y", "%d %b %Y"):
        try:
            parsed = _dt.datetime.strptime(text, fmt).date()
            return (parsed - today).days
        except ValueError:
            continue
    return None


# ── the applier — deterministic, invariant-safe, reject-on-conflict ───────────
def apply_adjustments(
    base: Envelope,
    accepted: list[ProposedAdjustment],
    supplier_appetite: dict[str, float] | None = None,
) -> tuple[Envelope, dict[str, float]]:
    """Apply the accepted deltas to ``base`` in a fixed phase order and return the new
    envelope + merged supplier-appetite. Pure. Raises :class:`MandateConflict` if the
    result can't validate — never emits an invalid mandate, never silently drops a rule.

    Phase order (deterministic, not order-independent — same subset gives the same
    envelope): 1 add/tighten terms, 2 weight bumps + renormalise, 3 shift target/floor,
    4 construct + validate.
    """
    terms: dict[str, TermSpec] = {t.name: t for t in base.terms}
    appetite: dict[str, float] = dict(supplier_appetite or {})
    target, reservation = base.target_utility, base.reservation_utility
    raw_weights: dict[str, float] = {t.name: t.weight for t in base.terms}
    fired = [a.rule_id for a in accepted]

    # phase 1 — new terms and bound tightening (each re-validates on construction)
    for adj in accepted:
        d = adj.delta
        if isinstance(d, AddTerm):
            spec = d.spec
            terms[spec.name] = spec
            raw_weights[spec.name] = spec.weight
            appetite[spec.name] = d.appetite
        elif isinstance(d, TightenBounds):
            t = terms.get(d.term_name)
            if t is None:
                continue
            terms[d.term_name] = _tighten(t, d.new_worst)

    # phase 2 — weight bumps, then renormalise all weights to sum to 1.0
    for adj in accepted:
        if isinstance(adj.delta, WeightBump) and adj.delta.term_name in raw_weights:
            raw_weights[adj.delta.term_name] += adj.delta.delta
    clamped = {n: max(_MIN_WEIGHT, w) for n, w in raw_weights.items()}
    total = sum(clamped.values())
    norm_weights = {n: w / total for n, w in clamped.items()}

    # phase 3 — target / reservation shifts, clamped so reservation stays below target
    for adj in accepted:
        if isinstance(adj.delta, ShiftTarget):
            target += adj.delta.target_delta
            reservation += adj.delta.reservation_delta
    target = min(1.0, max(0.0, target))
    reservation = max(0.0, min(reservation, target - _TARGET_RESERVATION_GAP))
    if reservation >= target:  # degenerate base spread + opposing shifts
        raise MandateConflict(fired, "reservation would meet or exceed target after shaping")

    # phase 4 — construct the real Envelope (runs _check: weights=1.0, reservation<target)
    new_terms = [terms[n].model_copy(update={"weight": norm_weights[n]}) for n in terms]
    try:
        shaped = Envelope(
            negotiation_id=base.negotiation_id,
            version=base.version + 1,
            signed_by=base.signed_by,
            terms=new_terms,
            target_utility=target,
            reservation_utility=reservation,
        )
    except ValueError as e:
        raise MandateConflict(fired, f"shaped mandate failed validation: {e}") from e
    return shaped, appetite


def _tighten(term: TermSpec, new_worst: float) -> TermSpec:
    """Move a term's ``worst`` toward ``best`` (shrinking the span), direction-safe.

    MINIMIZE has best<worst → new_worst must stay > best; MAXIMIZE has best>worst →
    new_worst must stay < best. An out-of-range value is clamped just inside `best` so
    the TermSpec direction validator (envelope.py) still passes.
    """
    if term.direction is Direction.MINIMIZE:
        safe = max(term.best + _TARGET_RESERVATION_GAP, min(new_worst, term.worst))
    else:
        safe = min(term.best - _TARGET_RESERVATION_GAP, max(new_worst, term.worst))
    return term.model_copy(update={"worst": safe})


def incumbent_scores_below_floor(shaped: Envelope, incumbent: Offer | None) -> bool:
    """True if the current contract's own terms now score below the shaped reservation —
    i.e. the agent would escalate rather than accept the incumbent position. One
    ``utility()`` call turns an invisible footgun into a visible review row.
    """
    if incumbent is None:
        return False
    try:
        return shaped.utility(incumbent) < shaped.reservation_utility
    except KeyError:
        return False  # incumbent doesn't cover every term — not a floor breach to flag


def _add_term_spec(
    name: str, ttype: TermType, direction: Direction, best: float, worst: float, weight: float
) -> TermSpec:
    return TermSpec(
        name=name, term_type=ttype, direction=direction, best=best, worst=worst, weight=weight
    )
