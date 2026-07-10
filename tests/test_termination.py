"""Contract termination — the deterministic notice clock + the grounded notice draft.

The clock is pure date math over the contract's own lifecycle terms; the draft must
ground every fact in those terms and never fabricate a legal ruling. These verify the
window state machine, the auto-renewal trap, and that the draft only ever states facts
the contract supplied (placeholders elsewhere) + always carries the disclaimer.
"""

from __future__ import annotations

import datetime as dt

import pytest

from negotiation_agent.intelligence import (
    ContractLifecycle,
    DocumentGrounded,
    LegalFlags,
)
from negotiation_agent.termination import (
    _LEGAL_DISCLAIMER,
    compute_clock,
    draft_termination_notice,
    parse_date,
)

TODAY = dt.date(2026, 7, 10)


def _life(**kw):
    """A ContractLifecycle from DocumentGrounded string values, e.g. expiry='2026-12-31'."""
    fields = {}
    for key in (
        "expiration_date",
        "auto_renews",
        "renewal_notice_days",
        "termination_notice_days",
    ):
        if key in kw:
            fields[key] = DocumentGrounded(value=str(kw[key]), assurance="confirmed")
    return ContractLifecycle(**fields)


# ── parse_date ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2026-12-31", dt.date(2026, 12, 31)),
        ("31.12.2026", dt.date(2026, 12, 31)),
        ("December 31, 2026", dt.date(2026, 12, 31)),
        ("not a date", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_date_conservative(text, expected):
    assert parse_date(text) == expected


# ── window state machine ────────────────────────────────────────────────────────
def test_open_when_deadline_is_far_out():
    # expiry 2026-12-31, 30-day notice -> deadline 2026-12-01, ~144 days out
    clock = compute_clock(
        _life(expiration_date="2026-12-31", termination_notice_days=30), today=TODAY
    )
    assert clock.window_status == "OPEN"
    assert clock.notice_deadline == dt.date(2026, 12, 1)
    assert clock.days_to_deadline == (dt.date(2026, 12, 1) - TODAY).days


def test_closing_soon_within_the_threshold():
    # deadline 2026-07-25 is 15 days out -> within the 30-day CLOSING_SOON band
    clock = compute_clock(
        _life(expiration_date="2026-08-24", termination_notice_days=30), today=TODAY
    )
    assert clock.window_status == "CLOSING_SOON"
    assert clock.notice_deadline == dt.date(2026, 7, 25)


def test_missed_when_deadline_is_past():
    # expiry 2026-07-15, 30-day notice -> deadline 2026-06-15, before today (2026-07-10)
    clock = compute_clock(
        _life(expiration_date="2026-07-15", termination_notice_days=30), today=TODAY
    )
    assert clock.window_status == "MISSED"
    assert clock.days_to_deadline < 0


def test_no_deadline_when_period_missing():
    clock = compute_clock(_life(expiration_date="2026-12-31"), today=TODAY)
    assert clock.window_status == "NO_DEADLINE"
    assert clock.notice_deadline is None


def test_unknown_when_expiry_unparseable():
    clock = compute_clock(
        _life(expiration_date="whenever", termination_notice_days=30), today=TODAY
    )
    assert clock.window_status == "UNKNOWN"


def test_unknown_when_no_lifecycle():
    clock = compute_clock(None, today=TODAY)
    assert clock.window_status == "UNKNOWN"


# ── the effective period is the larger of the two notice fields ──────────────────
def test_larger_notice_period_binds():
    # termination 30 vs renewal 90 -> the 90-day (earlier) deadline governs
    clock = compute_clock(
        _life(expiration_date="2026-12-31", termination_notice_days=30, renewal_notice_days=90),
        today=TODAY,
    )
    assert clock.notice_period_days == 90
    assert clock.notice_deadline == dt.date(2026, 12, 31) - dt.timedelta(days=90)


# ── the auto-renewal trap ────────────────────────────────────────────────────────
def test_auto_renewal_trap_fires_when_window_closing():
    clock = compute_clock(
        _life(expiration_date="2026-08-24", termination_notice_days=30, auto_renews="true"),
        today=TODAY,
    )
    assert clock.window_status == "CLOSING_SOON"
    assert clock.auto_renewal_trap is True


def test_no_trap_when_window_open():
    clock = compute_clock(
        _life(expiration_date="2026-12-31", termination_notice_days=30, auto_renews="true"),
        today=TODAY,
    )
    assert clock.window_status == "OPEN"
    assert clock.auto_renewal_trap is False


def test_no_trap_when_not_auto_renewing():
    clock = compute_clock(
        _life(expiration_date="2026-07-15", termination_notice_days=30, auto_renews="false"),
        today=TODAY,
    )
    assert clock.window_status == "MISSED"
    assert clock.auto_renewal_trap is False


# ── the draft is grounded: only contract facts, placeholders elsewhere, disclaimer ──
def test_draft_states_only_supplied_facts():
    clock = compute_clock(
        _life(expiration_date="2026-12-31", termination_notice_days=30), today=TODAY
    )
    notice = draft_termination_notice(
        clock, supplier_name="Nordwerk GmbH", buyer_name="ACME Buyer", today=TODAY
    )
    assert "Nordwerk GmbH" in notice
    assert "2026-12-31" in notice  # the contract's own expiry
    assert "30-day notice period" in notice
    assert _LEGAL_DISCLAIMER in notice


def test_draft_uses_placeholder_for_missing_supplier():
    clock = compute_clock(
        _life(expiration_date="2026-12-31", termination_notice_days=30), today=TODAY
    )
    notice = draft_termination_notice(clock, supplier_name=None, buyer_name="ACME", today=TODAY)
    assert "[Supplier name]" in notice


def test_draft_includes_governing_law_when_present():
    clock = compute_clock(
        _life(expiration_date="2026-12-31", termination_notice_days=30),
        LegalFlags(governing_law=DocumentGrounded(value="German law", assurance="confirmed")),
        today=TODAY,
    )
    notice = draft_termination_notice(
        clock, supplier_name="Nordwerk", buyer_name="ACME", today=TODAY
    )
    assert "German law" in notice


def test_non_renewal_and_terminate_intents_differ():
    clock = compute_clock(
        _life(expiration_date="2026-12-31", termination_notice_days=30), today=TODAY
    )
    nr = draft_termination_notice(
        clock, supplier_name="X", buyer_name="Y", today=TODAY, intent="non_renewal"
    )
    tm = draft_termination_notice(
        clock, supplier_name="X", buyer_name="Y", today=TODAY, intent="terminate"
    )
    assert "Non-Renewal" in nr
    assert "Termination" in tm
    assert nr != tm
