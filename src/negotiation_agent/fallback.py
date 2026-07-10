"""The deterministic floor under the LLM layer, and the redraft prompt builder.

Two pieces, both pure:
  - ``render_fallback`` — a template message keyed by (outcome, pressure) whose every
    figure slot is an ``approved_numbers`` value, so it passes the guard *by
    construction* (and is run through the guard anyway, defense in depth). This is
    what ships if the model can't comply within the redraft budget, or if the model
    is down. A stiff-but-correct reply beats a 500; mirrors ``prepare.py``'s "failure
    is never fatal."
  - ``build_redraft_instruction`` — the self-contained repair turn. It does NOT
    re-send the model's rejected draft as an assistant turn (that anchors the model
    to its mistake); it quotes the offending spans as data plus the allowlist plus the
    framing to keep. See ``docs/peitho-v2-architecture.md`` §4.5-4.6.
"""

from __future__ import annotations

from negotiation_agent.brief import MoveBrief

# Three variants per (outcome, pressure) key; the caller rotates to avoid repeating a
# skeleton back-to-back. ESCALATE variants are all figure-free (empty allowlist).
_COUNTER_TEMPLATES = [
    "Thanks for the detail. Here's where we can land: {figures}. There's room to work "
    "together if the commercial terms line up — happy to talk it through.",
    "Appreciate the movement. Our position: {figures}. We can be flexible where it helps "
    "you, provided the priorities hold.",
    "Understood. To keep this moving, we're proposing {figures}. Let's find the shape that "
    "works for both sides.",
]

_ACCEPT_TEMPLATES = [
    "That works for us — we're happy to proceed at {figures}. I'll have the paperwork "
    "drawn up.",
    "Agreed. Let's lock it in at {figures} and move to signature.",
    "We have a deal at {figures}. I'll get the contract started.",
]

_ESCALATE_TEMPLATES = [
    "Thank you for the discussion. I'm not able to move further on my side right now — let "
    "me take this back internally and come back to you.",
    "I appreciate the exchange. This needs an internal review before I can go further; I'll "
    "follow up shortly.",
    "Thanks for your patience. I'd like to bring a colleague in before we continue — I'll be "
    "in touch.",
]

# A distinct handoff when the supplier introduced terms outside the mandate — this
# showcases the unknown-term guard rather than defaulting it away.
_UNMODELED_HANDOFF = (
    "Your proposal introduces terms that sit outside my current mandate, so I'm bringing in "
    "a colleague who can consider them properly. Thank you for your patience."
)

# A no-parsed-offer nudge — does NOT advance the engine round (see api.py).
NO_OFFER_NUDGE = (
    "I want to make sure I capture your position correctly — could you restate the unit "
    "price, payment terms, and contract length you have in mind?"
)


def _format_figures(approved: dict[str, float]) -> str:
    parts: list[str] = []
    if "price" in approved:
        parts.append(f"€{approved['price']:.2f} per unit")
    if "payment_days" in approved:
        parts.append(f"net-{int(round(approved['payment_days']))} payment terms")
    if "contract_months" in approved:
        parts.append(f"a {int(round(approved['contract_months']))}-month contract")
    for name, value in approved.items():
        if name in ("price", "payment_days", "contract_months"):
            continue
        parts.append(f"{name} {value:g}")
    return ", ".join(parts) if parts else "the terms on the table"


def render_fallback(brief: MoveBrief, *, variant: int = 0) -> str:
    """A deterministic, guard-passing message for this move. Figures from the brief."""
    figures = _format_figures(brief.approved_numbers)
    if brief.outcome == "ESCALATE":
        if brief.reason_tag == "unmodeled_terms":
            return _UNMODELED_HANDOFF
        return _ESCALATE_TEMPLATES[variant % len(_ESCALATE_TEMPLATES)]
    if brief.outcome == "ACCEPT":
        return _ACCEPT_TEMPLATES[variant % len(_ACCEPT_TEMPLATES)].format(figures=figures)
    return _COUNTER_TEMPLATES[variant % len(_COUNTER_TEMPLATES)].format(figures=figures)


def build_redraft_instruction(violations: list[str], approved: dict[str, float]) -> str:
    """A self-contained repair instruction (no rejected draft echoed back)."""
    listed = ", ".join(f"{v:g}" for v in approved.values()) or "(no figures — escalation)"
    offending = ", ".join(violations)
    return (
        "Your previous draft contained figures or phrasing that are not permitted: "
        f"{offending}. You may state ONLY these exact figures and no others: {listed}. "
        "Do not mention any internal terms (thresholds, targets, walk-away, utility). "
        "Rewrite the message keeping the same intent and trade framing, using only the "
        "permitted figures."
    )
