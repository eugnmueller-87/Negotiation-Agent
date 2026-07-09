"""Normalized value functions.

Every negotiable term is mapped to a utility contribution in the unit interval
[0, 1] via a per-term value function v_i(x_i). The buyer's total utility for an
offer is U = sum_i w_i * v_i(x_i) with sum_i w_i == 1, so U is itself in [0, 1].

Design choices worth stating explicitly, since the engine's whole credibility
rests on this being deterministic and monotone:

* A term has a *good* end and a *bad* end for the buyer. For price the good end
  is low; for payment days the good end is high (the buyer wants to pay later);
  the caller declares this with ``Direction``.
* We normalize linearly between a ``best`` value (v = 1.0) and a ``worst`` value
  (v = 0.0). Values beyond ``best`` clamp to 1.0; beyond ``worst`` clamp to 0.0.
  This makes the utility bounded and prevents an absurd supplier concession on
  one term from dominating the whole package.

Non-linear shapes (e.g. diminishing returns on payment terms) can be layered in
later by swapping the interpolation; the linear form is the honest v0 default.
"""

from __future__ import annotations


def linear_value(x: float, *, best: float, worst: float) -> float:
    """Map ``x`` to [0, 1] by linear interpolation between ``worst`` and ``best``.

    ``best`` is the value scoring 1.0 (most desirable for the buyer) and
    ``worst`` the value scoring 0.0. ``best`` and ``worst`` may be given in
    either numeric order — the "good" direction is defined purely by which
    endpoint is labelled ``best``. Values outside the [worst, best] span clamp.

    Raises ``ValueError`` if ``best == worst`` (a degenerate, unscoreable term).
    """
    span = best - worst
    if span == 0:
        raise ValueError("value function needs best != worst")
    t = (x - worst) / span
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    return t


def linear_inverse(v: float, *, best: float, worst: float) -> float:
    """Inverse of :func:`linear_value`: the ``x`` whose value is ``v``.

    Used by the counteroffer search to ask "what term value yields this utility
    contribution?". ``v`` is clamped to [0, 1] first.
    """
    v = min(1.0, max(0.0, v))
    return worst + v * (best - worst)
