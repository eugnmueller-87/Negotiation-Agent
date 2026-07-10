"""The numeric guard — the mechanism that makes the product's claim true.

Adversarial: every string a hostile or sloppy model might produce must be caught,
and the deterministic fallback must pass its own guard by construction.
"""

from __future__ import annotations

import pytest

from negotiation_agent.brief import MoveBrief
from negotiation_agent.fallback import build_redraft_instruction, render_fallback
from negotiation_agent.guard import check, is_clean
from negotiation_agent.numbers import spelled_numbers

APPROVED = {"price": 9.00, "payment_days": 56, "contract_months": 12}


def test_clean_message_passes():
    msg = "We can land at €9.00 per unit, net-56, on a 12-month contract."
    assert is_clean(msg, APPROVED)


@pytest.mark.parametrize(
    "bad",
    [
        "We'll pay €10.20 per unit.",  # unapproved price
        "Let's do net-45.",  # unapproved payment
        "How about a 24-month term?",  # unapproved months
        "We can offer €9.00 and a 2% rebate on top.",  # 2 is not approved
    ],
)
def test_unapproved_numeric_is_caught(bad):
    violations = check(bad, APPROVED)
    assert violations, f"guard missed an unapproved number in: {bad!r}"


def test_spelled_number_is_caught():
    # "net sixty" -> 60, not approved (56 is)
    violations = check("We could stretch to net sixty days.", APPROVED)
    assert any("60" in v for v in violations)


def test_spelled_approved_number_passes():
    # "twelve months" -> 12, which IS approved
    assert is_clean("Happy with a twelve month contract at €9.00, net-56.", APPROVED)


def test_mechanism_leak_is_caught():
    violations = check("This is above our reservation utility.", APPROVED)
    assert any(v.startswith("leak:") for v in violations)


def test_ordinary_target_word_is_not_a_leak():
    # bare "target"/"floor" must NOT false-positive on normal buyer prose (no stray
    # digits — the guard rightly catches those separately; this isolates the leak layer)
    msg = (
        "Our target go-live is next quarter and we need a floor on volume — "
        "€9.00, net-56, 12 months."
    )
    assert is_clean(msg, APPROVED)


def test_concession_frame_is_caught():
    violations = check("And we'll waive the setup fee at €9.00, net-56, 12 months.", APPROVED)
    assert any(v.startswith("concession:") for v in violations)


def test_ask_about_rebate_is_not_a_concession():
    # asking is fine; only first-person GIVING frames are flagged
    assert is_clean("Can you improve the rebate? We're at €9.00, net-56, 12 months.", APPROVED)


def test_escalate_empty_allowlist_rejects_any_number():
    # ESCALATE -> approved_numbers == {} -> any figure is a violation
    assert check("I'll come back to you within 5 days.", {}) == ["5"]
    assert is_clean("Let me take this back internally and follow up.", {})


def test_integer_approved_matches_exactly_not_within_tolerance():
    approved = {"payment_days": 56}
    assert is_clean("net-56 works.", approved)
    assert check("net-57 works.", approved)  # 57 must NOT match 56


def test_thousands_separator_parsed():
    approved = {"volume_units": 40000}
    assert is_clean("A minimum of 40,000 units.", approved)
    assert check("A minimum of 50,000 units.", approved)


@pytest.mark.parametrize("outcome", ["COUNTER", "ACCEPT", "ESCALATE"])
def test_fallback_always_passes_its_own_guard(outcome):
    # the deterministic floor must be guard-clean by construction
    approved = {} if outcome == "ESCALATE" else APPROVED
    brief = MoveBrief(
        outcome=outcome, is_opening=False, round_band="mid", pressure="reciprocity",
        approved_numbers=approved, reason_tag="counter",
    )
    for variant in range(3):
        msg = render_fallback(brief, variant=variant)
        assert is_clean(msg, approved), f"fallback leaked a figure: {msg!r}"


def test_redraft_instruction_lists_only_approved_figures():
    instr = build_redraft_instruction(["10.20", "45"], APPROVED)
    assert "10.20" in instr and "45" in instr  # names the violations
    assert "9" in instr and "56" in instr  # lists the allowlist
    assert "threshold" in instr.lower()  # warns against internal leak


def test_spelled_numbers_parser_handles_hyphen_and_compound():
    assert spelled_numbers("forty-five days") == [45]
    assert spelled_numbers("one hundred and twenty units") == [120]
    assert spelled_numbers("no numbers here") == []
