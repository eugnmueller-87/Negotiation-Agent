"""Logrolling package search — the core IP.

Given a buyer utility ``threshold`` the package must clear and a belief about
supplier appetite, find the package the supplier is likeliest to accept while
still clearing the threshold. Instead of splitting the difference on price, we
route concessions to the terms the buyer weights lightly and the supplier
values highly, and hold the terms the buyer weights heavily. That is the
win-win trade humans skip when they anchor on price alone.

## The optimization

Choose per-term utility contributions ``v_i in [0, cap_i]`` to

    minimize    sum_i  p_i * v_i          (spend held buyer-utility where the
    subject to  sum_i  w_i * v_i = theta    supplier cares least; equivalently,
                0 <= v_i <= cap_i           route concessions 1 - v_i to the
                                            terms with highest supplier appetite)

with ``w`` the buyer weights and ``p`` the normalized supplier priorities. This
is a fractional-knapsack LP; its exact optimum is a greedy fill ordered by the
cost ratio ``r_i = p_i / w_i``. Low ``r_i`` (heavy buyer weight, low supplier
appetite) is held at buyer-best; high ``r_i`` (light buyer weight, high supplier
appetite) is conceded first. Because ``p`` and ``w`` are static within a
negotiation, the fill order is fixed and concessions nest as ``theta`` decays —
they never retract.

This is a pure function of its arguments: no RNG, no globals. It is shared by
both sides of the simulator (buyer engine and supplier bot both search their own
envelope with it), which is why it lives outside ``engine.py``.
"""

from __future__ import annotations

from negotiation_agent.envelope import Envelope, Offer, TermSpec
from negotiation_agent.value import linear_inverse

_EPS = 1e-9


class InfeasiblePackage(Exception):
    """No package under the given caps reaches ``threshold`` (sum w*cap < theta).

    The engine translates this to an ESCALATE decision.
    """


def _fill_order(envelope: Envelope, priorities: dict[str, float]) -> list[TermSpec]:
    """Deterministic concede order: cheapest-to-hold first.

    Sort key ``(p_i/w_i asc, w_i desc, name asc)``:
      * primary: low cost ratio => cheap to hold => filled (held) first;
      * on a ratio tie, prefer holding the heavier buyer term (w desc);
      * final lexicographic tie-break makes runs platform-stable.
    """
    return sorted(
        envelope.terms,
        key=lambda t: (priorities[t.name] / t.weight, -t.weight, t.name),
    )


def _greedy_fill(
    terms: list[TermSpec],
    threshold: float,
    caps: dict[str, float],
) -> dict[str, float]:
    """Phase A: exact LP optimum in v-space. Fill held utility along ``terms``
    (already in concede order) until the weighted budget ``threshold`` is met.

    Raises :class:`InfeasiblePackage` if the caps cannot reach ``threshold``.
    """
    v: dict[str, float] = {t.name: 0.0 for t in terms}
    budget = threshold
    for term in terms:
        if budget <= _EPS:
            break
        cap = caps.get(term.name, 1.0)
        take = min(cap, budget / term.weight)
        take = max(0.0, take)
        v[term.name] = take
        budget -= term.weight * take
    if budget > 1e-7:
        raise InfeasiblePackage(
            f"caps cannot reach threshold {threshold:.4f}; short by {budget:.4f}"
        )
    return v


def fill_package(
    envelope: Envelope,
    threshold: float,
    priorities: dict[str, float],
    caps: dict[str, float] | None = None,
) -> tuple[Offer, float, dict[str, float]]:
    """Best logrolled package clearing ``threshold``.

    Returns ``(offer, realized_utility, planned_v)`` where ``realized_utility``
    is always ``>= threshold`` (equal, when at least one continuous term has
    slack to absorb integer-snap surplus).

    ``priorities`` must be the normalized belief from
    :meth:`SupplierModel.priorities`. ``caps`` (per-term max ``v_i``) enforces
    monotone concessions across rounds; omit for the opening anchor.
    """
    caps = dict(caps or {})
    order = _fill_order(envelope, priorities)

    # Phase A — exact LP optimum in v-space.
    v = _greedy_fill(order, threshold, caps)

    # Phase B — integer snap, buyer-favorable. Rounding toward `best` can only
    # ADD buyer utility, never drop the package below threshold. (We deliberately
    # do not route through TermSpec.value_to_x: its round-half-to-even is
    # direction-agnostic and would fight this guarantee.)
    x_final: dict[str, float] = {}
    integer_utility = 0.0
    for term in order:
        if not term.is_integer:
            continue
        x = linear_inverse(v[term.name], best=term.best, worst=term.worst)
        x_snapped = snap_toward_best(x, term)
        v[term.name] = term.value(x_snapped)
        x_final[term.name] = x_snapped
        integer_utility += term.weight * v[term.name]

    # Phase C — exact re-solve on continuous terms with the leftover budget, so
    # the surplus that snapping added is handed back on the terms the supplier
    # values most and the package lands on ``threshold`` exactly.
    continuous = [t for t in order if not t.is_integer]
    if continuous:
        leftover = max(0.0, threshold - integer_utility)
        v_cont = _greedy_fill(continuous, leftover, caps)
        for term in continuous:
            v[term.name] = v_cont[term.name]
            x_final[term.name] = linear_inverse(v_cont[term.name], best=term.best, worst=term.worst)

    offer = Offer(terms={t.name: x_final[t.name] for t in envelope.terms})
    realized = envelope.utility(offer)
    return offer, realized, v


def snap_toward_best(x: float, term: TermSpec) -> float:
    """Round ``x`` to the integer on the ``best`` side, then clamp to span.

    Snapping toward ``best`` guarantees the realized utility contribution is
    >= the planned one, so integer rounding can never push a package below the
    threshold it was built to clear.
    """
    import math

    if term.best > term.worst:  # MAXIMIZE: best is the larger value -> round up
        x_snapped = math.ceil(x - _EPS)
    else:  # MINIMIZE: best is the smaller value -> round down
        x_snapped = math.floor(x + _EPS)
    return term.clamp(float(x_snapped))
