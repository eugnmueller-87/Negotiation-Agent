"""Contract termination — the deterministic notice clock + a grounded notice draft.

This is NOT walking away from a negotiation (that is ``Outcome.ESCALATE`` in the
engine). This terminates a *running* contract correctly: from the contract's own
extracted lifecycle terms, compute when notice must be served, whether the
auto-renewal window is still open, and draft the notice letter.

Honesty line (``docs/contract-intelligence-architecture.md`` and the compliance rule
in ``data-privacy-procurement.md``): this does deterministic **date math** and emits a
**templated** notice grounded only in the contract's own extracted terms. It does NOT
fabricate a legal ruling. "Local legal requirements" vary by jurisdiction; the draft
surfaces the form the *contract itself* states (governing law, written-notice clause)
and carries a mandatory "verify against local law" line. Not a legal engine.

Everything here is pure: lifecycle facts in, a :class:`TerminationClock` and a text
draft out. No LLM, no I/O, no clock read — ``today`` is passed in.
"""

from __future__ import annotations

import datetime as _dt
from typing import Literal

from pydantic import BaseModel

from negotiation_agent.intelligence import ContractLifecycle, DocumentGrounded, LegalFlags

# Window status is a discrete, human-legible state — never a raw day count in a UI badge.
WindowStatus = Literal["OPEN", "CLOSING_SOON", "MISSED", "NO_DEADLINE", "UNKNOWN"]

# A deadline this many days out or fewer is CLOSING_SOON — act now, not "eventually".
_CLOSING_SOON_DAYS = 30

# The formats ``parse_date`` accepts. Deliberately the same unambiguous set as
# ``shaper.days_until`` — anything outside it yields None so a rule declines to fire
# rather than guessing a date on a real contract.
_DATE_FORMATS = ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%d %B %Y", "%B %d, %Y", "%d %b %Y")


def parse_date(date_text: str | None) -> _dt.date | None:
    """Parse a free-text date to a ``date``, or None if not unambiguously parseable.

    Conservative by design (see module docstring): only the fixed ``_DATE_FORMATS``.
    Returns the ``date`` itself — the clock needs the value to print the deadline, not
    just the delta ``shaper.days_until`` returns.
    """
    if not date_text:
        return None
    text = date_text.strip()
    for fmt in _DATE_FORMATS:
        try:
            return _dt.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _as_int(grounded: DocumentGrounded | None) -> int | None:
    """A non-negative integer from a DocumentGrounded value, or None. Notice periods
    are day counts; a negative or non-numeric value is treated as absent, not an error."""
    if grounded is None or grounded.value is None:
        return None
    try:
        n = int(str(grounded.value).strip())
    except ValueError:
        return None
    return n if n >= 0 else None


def _as_bool(grounded: DocumentGrounded | None) -> bool | None:
    """Tri-state bool from a DocumentGrounded ``"true"``/``"false"`` value, else None."""
    if grounded is None or grounded.value is None:
        return None
    v = str(grounded.value).strip().lower()
    if v in ("true", "yes", "1"):
        return True
    if v in ("false", "no", "0"):
        return False
    return None


class TerminationClock(BaseModel):
    """The computed notice timeline. Every field is derived by pure date math from the
    contract's own extracted lifecycle terms — nothing here is invented."""

    model_config = {"frozen": True}

    window_status: WindowStatus
    expiration_date: _dt.date | None = None
    notice_period_days: int | None = None  # the effective period the deadline used
    notice_deadline: _dt.date | None = None  # expiry − notice_period_days
    days_to_deadline: int | None = None  # today → deadline (negative = already missed)
    auto_renews: bool | None = None
    auto_renewal_trap: bool = False  # auto-renews AND deadline is soon/missed
    governing_law: str | None = None
    notes: list[str] = []  # human-facing, deterministic explanations of what was found


def compute_clock(
    lifecycle: ContractLifecycle | None,
    legal: LegalFlags | None = None,
    *,
    today: _dt.date,
) -> TerminationClock:
    """Compute the termination notice clock from the contract's lifecycle terms.

    Notice deadline = expiry − effective notice period, where the effective period is
    the LARGER of ``termination_notice_days`` and ``renewal_notice_days`` when both are
    present (the earlier deadline binds — miss it and either the exit or the
    non-renewal is foreclosed). ``UNKNOWN`` when the expiry is unparseable;
    ``NO_DEADLINE`` when an expiry exists but no notice period does.
    """
    notes: list[str] = []
    governing_law = _text(legal.governing_law) if legal else None

    if lifecycle is None:
        return TerminationClock(
            window_status="UNKNOWN",
            governing_law=governing_law,
            notes=["No lifecycle terms extracted — cannot compute a notice deadline."],
        )

    expiry = parse_date(_text(lifecycle.expiration_date))
    auto_renews = _as_bool(lifecycle.auto_renews)
    term_notice = _as_int(lifecycle.termination_notice_days)
    renewal_notice = _as_int(lifecycle.renewal_notice_days)

    # The binding period is the larger of the two — the earlier deadline governs.
    periods = [p for p in (term_notice, renewal_notice) if p is not None]
    notice_period = max(periods) if periods else None

    if expiry is None:
        notes.append("Expiration date missing or unparseable — no deadline computed.")
        return TerminationClock(
            window_status="UNKNOWN",
            notice_period_days=notice_period,
            auto_renews=auto_renews,
            governing_law=governing_law,
            notes=notes,
        )

    if notice_period is None:
        notes.append(
            "Expiry known but no notice period stated — verify the notice clause in the "
            "contract before relying on the expiry date alone."
        )
        return TerminationClock(
            window_status="NO_DEADLINE",
            expiration_date=expiry,
            auto_renews=auto_renews,
            governing_law=governing_law,
            notes=notes,
        )

    deadline = expiry - _dt.timedelta(days=notice_period)
    days_to_deadline = (deadline - today).days

    if days_to_deadline < 0:
        status: WindowStatus = "MISSED"
        notes.append(
            f"Notice deadline was {deadline.isoformat()} "
            f"({-days_to_deadline} day(s) ago) — the window to serve notice has passed."
        )
    elif days_to_deadline <= _CLOSING_SOON_DAYS:
        status = "CLOSING_SOON"
        notes.append(
            f"Notice deadline {deadline.isoformat()} is in {days_to_deadline} day(s) — "
            "serve notice now."
        )
    else:
        status = "OPEN"
        notes.append(
            f"Notice may be served any time up to {deadline.isoformat()} "
            f"({days_to_deadline} day(s) from now)."
        )

    trap = auto_renews is True and status in ("CLOSING_SOON", "MISSED")
    if trap:
        notes.append(
            "AUTO-RENEWAL TRAP: this contract auto-renews and the notice window is "
            "closing or closed — missing it locks in another term."
        )

    return TerminationClock(
        window_status=status,
        expiration_date=expiry,
        notice_period_days=notice_period,
        notice_deadline=deadline,
        days_to_deadline=days_to_deadline,
        auto_renews=auto_renews,
        auto_renewal_trap=trap,
        governing_law=governing_law,
        notes=notes,
    )


# The disclaimer is NON-NEGOTIABLE — it appears on every draft. This is the honesty
# line as code: the tool does date math and templating, it does not rule on law.
_LEGAL_DISCLAIMER = (
    "This notice is a draft generated from the contract's own stated terms. It is not "
    "legal advice. Verify the notice period, form (written / registered post / e-mail), "
    "and any jurisdiction-specific requirements against local law and the full contract "
    "before serving."
)


def draft_termination_notice(
    clock: TerminationClock,
    *,
    supplier_name: str | None,
    buyer_name: str,
    contract_reference: str | None = None,
    today: _dt.date,
    intent: Literal["terminate", "non_renewal"] = "non_renewal",
) -> str:
    """Draft a termination / non-renewal notice grounded only in the clock's facts.

    ``intent="non_renewal"`` gives notice not to renew at expiry (the common case for
    an auto-renewing contract); ``"terminate"`` gives notice to end the contract. The
    draft states the contract's own governing law and always ends with the disclaimer.
    Placeholders (``[…]``) are left where the contract did not supply a fact — the tool
    does not invent a date, a party, or a clause it never read.
    """
    supplier = supplier_name or "[Supplier name]"
    reference = contract_reference or "[Contract reference]"
    expiry = clock.expiration_date.isoformat() if clock.expiration_date else "[expiry date]"

    if intent == "non_renewal":
        subject = "Notice of Non-Renewal"
        body_action = (
            f"we give notice that we will not renew the above agreement and that it will "
            f"terminate on its current expiry date of {expiry}."
        )
    else:
        subject = "Notice of Termination"
        body_action = (
            f"we give notice to terminate the above agreement in accordance with its "
            f"termination provisions, effective {expiry}."
        )

    lines = [
        f"Date: {today.isoformat()}",
        f"To: {supplier}",
        f"From: {buyer_name}",
        f"Re: {subject} — {reference}",
        "",
        f"Dear {supplier},",
        "",
        f"Pursuant to the terms of the agreement referenced above, {body_action}",
    ]

    if clock.notice_period_days is not None and clock.notice_deadline is not None:
        if clock.window_status == "MISSED":
            # Never claim the notice is timely when the deadline has passed — that would
            # put a false statement in a legal notice. State the miss plainly instead.
            lines.append(
                f"\nNote: the {clock.notice_period_days}-day notice deadline "
                f"({clock.notice_deadline.isoformat()}) has passed. This notice may be "
                "late under the agreement — confirm the effect with counsel before serving."
            )
        else:
            lines.append(
                f"\nThis notice is served within the {clock.notice_period_days}-day notice "
                f"period the agreement requires (notice deadline "
                f"{clock.notice_deadline.isoformat()})."
            )

    if clock.governing_law:
        lines.append(
            f"\nThis notice is given under the governing law of the agreement "
            f"({clock.governing_law})."
        )

    lines += [
        "",
        "Please confirm receipt and the effective end date in writing.",
        "",
        "Regards,",
        buyer_name,
        "",
        "---",
        _LEGAL_DISCLAIMER,
    ]
    return "\n".join(lines)


def _text(grounded: DocumentGrounded | None) -> str | None:
    """The string value of a DocumentGrounded field, or None."""
    if grounded is None:
        return None
    return grounded.value
