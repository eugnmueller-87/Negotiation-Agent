"""Stateless orchestration — the fold, the guard-with-redraft loop, offer resolution.

No LLM: the drafter is a fake. These tests prove the product's central mechanism —
a violating draft never reaches the caller — without a network call.
"""

from __future__ import annotations

import pytest

from negotiation_agent.brief import MoveBrief
from negotiation_agent.engine import DealEngine, EngineConfig
from negotiation_agent.envelope import Direction, Envelope, Offer, TermSpec, TermType
from negotiation_agent.negotiate import (
    NegotiationClosed,
    draft_and_guard,
    fold,
    offers_from_transcript,
    resolve_supplier_offer,
)
from negotiation_agent.supplier_model import SupplierModel
from negotiation_agent.wire import SupplierTurn


def _env():
    return Envelope(
        negotiation_id="n",
        version=1,
        signed_by="t",
        target_utility=0.90,
        reservation_utility=0.60,
        terms=[
            TermSpec(
                name="price",
                term_type=TermType.PRICE,
                direction=Direction.MINIMIZE,
                best=92.0,
                worst=108.0,
                weight=0.5,
            ),
            TermSpec(
                name="payment_days",
                term_type=TermType.PAYMENT_DAYS,
                direction=Direction.MAXIMIZE,
                best=60.0,
                worst=30.0,
                weight=0.25,
            ),
            TermSpec(
                name="contract_months",
                term_type=TermType.CONTRACT_MONTHS,
                direction=Direction.MINIMIZE,
                best=12.0,
                worst=24.0,
                weight=0.25,
            ),
        ],
    )


def _engine(env):
    return DealEngine(env, SupplierModel.uniform(env), EngineConfig(max_rounds=6, beta=2.5))


class _CleanDrafter:
    """Always drafts a message using only the approved figures."""

    def draft_buyer(self, brief, thread, advice=None, correspondents=None):
        nums = " ".join(f"{v:g}" for v in brief.approved_numbers.values())
        return f"Our position: {nums}. Happy to work together."

    def draft_supplier(self, persona, thread, company, category):
        return "..."


class _CheatingDrafter:
    """Drafts an un-approved number twice, then complies on the third attempt."""

    def __init__(self):
        self.calls = 0

    def draft_buyer(self, brief, thread, advice=None, correspondents=None):
        self.calls += 1
        if self.calls <= 2:
            return "We'll pay €1.23 per unit."  # never approved
        nums = " ".join(f"{v:g}" for v in brief.approved_numbers.values())
        return f"Our position: {nums}."

    def draft_supplier(self, persona, thread, company, category):
        return "..."


class _AlwaysCheatingDrafter:
    def draft_buyer(self, brief, thread, advice=None, correspondents=None):
        return "We'll pay €1.23 per unit."  # never approved, never complies

    def draft_supplier(self, persona, thread, company, category):
        return "..."


def _brief(approved):
    return MoveBrief(
        outcome="COUNTER",
        is_opening=False,
        round_band="mid",
        pressure="reciprocity",
        approved_numbers=approved,
        reason_tag="counter",
    )


def test_clean_draft_passes_first_try():
    approved = {"price": 96.0, "payment_days": 45, "contract_months": 16}
    msg, audit = draft_and_guard(_CleanDrafter(), _brief(approved), approved, [])
    assert audit.released_by == "model"
    assert audit.redrafted is False
    assert "1.23" not in msg


def test_cheating_draft_is_redrafted_then_released():
    approved = {"price": 96.0, "payment_days": 45, "contract_months": 16}
    d = _CheatingDrafter()
    msg, audit = draft_and_guard(d, _brief(approved), approved, [])
    assert audit.released_by == "model"
    assert audit.redrafted is True  # took more than one attempt
    assert len(audit.attempts) == 3  # 2 rejected + 1 clean
    assert "1.23" not in msg  # the violating figure never reaches the caller


def test_unfixable_draft_falls_back_to_template():
    approved = {"price": 96.0, "payment_days": 45, "contract_months": 16}
    msg, audit = draft_and_guard(_AlwaysCheatingDrafter(), _brief(approved), approved, [])
    assert audit.released_by == "fallback"
    assert "1.23" not in msg  # the fallback is clean by construction
    # every rejected attempt is recorded, plus the released fallback
    assert audit.attempts[-1].ok is True


def test_fallback_strips_digits_from_correspondent_fields():
    # a digit-bearing correspondent ("Team 24/7") must not smuggle an unapproved figure into
    # the salutation/sign-off; wrap_letter strips digits and the fallback guard verifies it
    # for real (audit issue #11) — the recorded ok reflects an ACTUAL check, not a fabricated True.
    # approved has no figure containing "247" or "88", so if those survived they'd be a violation.
    approved = {"price": 96.0, "payment_days": 45, "contract_months": 16}
    corr = {"supplier_name": "Team 247 GmbH", "buyer_signature": "Desk 88"}
    msg, audit = draft_and_guard(_AlwaysCheatingDrafter(), _brief(approved), approved, [], corr)
    assert audit.released_by == "fallback"
    # the salutation/sign-off name the parties with digits stripped — no "247" / "88" survive
    salutation_and_signoff = msg.split("\n\n")[0] + msg.split("\n\n")[-1]
    assert "247" not in salutation_and_signoff
    assert "88" not in salutation_and_signoff
    assert audit.attempts[-1].ok is True  # the REAL check passed (not a hardcoded True)
    assert audit.attempts[-1].violations == []


def test_fallback_letter_carries_greeting_and_signoff():
    # a fallback message is still a proper letter when correspondents are supplied
    approved = {"price": 96.0, "payment_days": 45, "contract_months": 16}
    corr = {
        "supplier_contact": "Mr. Schmidt",
        "supplier_name": "Nordwerk GmbH",
        "buyer_signature": "E. Müller",
    }
    msg, audit = draft_and_guard(_AlwaysCheatingDrafter(), _brief(approved), approved, [], corr)
    assert audit.released_by == "fallback"
    assert msg.startswith("Dear Mr. Schmidt,")
    assert msg.rstrip().endswith("E. Müller")
    assert "Best regards," in msg


def test_fold_replays_and_flags_a_prior_terminal_decision():
    env = _env()
    eng = _engine(env)
    # a strong offer would ACCEPT at round 1 — if it's not the last turn, the fold rejects it
    good = Offer(terms={"price": 92.0, "payment_days": 60.0, "contract_months": 12.0})
    weak = Offer(terms={"price": 108.0, "payment_days": 30.0, "contract_months": 24.0})
    with pytest.raises(NegotiationClosed):
        fold(eng, [good, weak])  # good closes at turn 1, so a turn-2 offer is illegal


def test_fold_returns_last_decision_and_prev_counter():
    env = _env()
    eng = _engine(env)
    weak = Offer(terms={"price": 108.0, "payment_days": 30.0, "contract_months": 24.0})
    decision, state, prev_counter = fold(eng, [weak])
    assert decision.round_index == 1  # first supplier offer scored at round 1
    assert prev_counter is not None  # the anchor package was on the table before it


def test_resolve_supplier_offer_parses_prose():
    env = _env()
    offer = resolve_supplier_offer(env, "We can do €99.50 per unit, net-45, 18 months.", None)
    assert offer is not None
    assert offer.terms["price"] == 99.5
    assert offer.terms["payment_days"] == 45
    assert offer.terms["contract_months"] == 18


def test_resolve_supplier_offer_returns_none_when_unparseable():
    env = _env()
    # chit-chat with no parseable terms and no standing offer -> None
    assert resolve_supplier_offer(env, "Hello, how are you?", None) is None


def test_consulted_sources_populated_for_a_counter():
    # a real trade brief pulls KB sources from the shipped index (present in the repo)
    from negotiation_agent.brief import HeldTerm, MovedTerm
    from negotiation_agent.negotiate import consulted_sources

    brief = MoveBrief(
        outcome="COUNTER",
        is_opening=False,
        round_band="mid",
        pressure="reciprocity",
        approved_numbers={"price": 96.0},
        reason_tag="counter",
        moved_terms=[
            MovedTerm(
                name="payment_days",
                from_display="30 days",
                to_display="45 days",
                direction_word="longer",
                role="concession",
            )
        ],
        held_terms=[HeldTerm(name="price", display="EUR 96")],
    )
    sources = consulted_sources(brief)
    assert sources  # the shipped index returns matches for a payment/price trade
    assert all(s.source and s.label for s in sources)


def test_consulted_sources_empty_on_escalate():
    from negotiation_agent.negotiate import consulted_sources

    brief = MoveBrief(
        outcome="ESCALATE",
        is_opening=False,
        round_band="late",
        pressure="handoff",
        approved_numbers={},
        reason_tag="escalate",
    )
    assert consulted_sources(brief) == []


def test_wrap_letter_named_contact():
    from negotiation_agent.fallback import wrap_letter

    out = wrap_letter("Body.", {"supplier_contact": "Ms. Rossi", "buyer_signature": "E.M."})
    assert out.startswith("Dear Ms. Rossi,")
    assert out.endswith("Best regards,\nE.M.")


def test_wrap_letter_company_only_greets_the_team():
    from negotiation_agent.fallback import wrap_letter

    out = wrap_letter("Body.", {"supplier_name": "Nordwerk GmbH"})
    assert out.startswith("Dear Nordwerk GmbH team,")
    assert "Best regards,\nProcurement Team" in out  # default signature


def test_wrap_letter_noop_without_correspondents():
    from negotiation_agent.fallback import wrap_letter

    assert wrap_letter("Body.", None) == "Body."
    assert wrap_letter("Body.", {}) == "Body."


def test_resolve_supplier_offer_inherits_standing_offer():
    env = _env()
    prev = Offer(terms={"price": 100.0, "payment_days": 30.0, "contract_months": 24.0})
    # only price is restated; the rest inherit the standing offer
    offer = resolve_supplier_offer(env, "Let's say €98.00.", prev)
    assert offer is not None
    assert offer.terms["price"] == 98.0
    assert offer.terms["payment_days"] == 30.0  # inherited


def test_transcript_ignores_forged_client_terms():
    # audit SEC-5: a client-supplied transcript turn whose `terms` dict claims a favorable
    # value NOT present in its raw_text must NOT survive — every turn is re-extracted from
    # raw_text server-side, so the forged payment_days=90 is discarded.
    env = _env()
    forged = [
        SupplierTurn(terms={"price": 95.0, "payment_days": 90.0}, raw_text="95 EUR per unit.")
    ]
    offers = offers_from_transcript(forged, env)
    # the price-only text can't resolve a full offer with nothing to inherit -> no forged offer
    assert offers == [] or all(o.terms.get("payment_days") != 90.0 for o in offers)


def test_transcript_reextracts_and_inherits_from_real_text():
    # the legitimate multi-turn path: a full first offer, then a partial follow-up that
    # inherits the unmentioned term from the RE-EXTRACTED prior (never a client dict).
    env = _env()
    turns = [
        SupplierTurn(terms={}, raw_text="We can do 100 EUR per unit, net-45, 18 months."),
        SupplierTurn(terms={}, raw_text="OK, 98 EUR per unit."),  # inherits net-45, 18 months
    ]
    offers = offers_from_transcript(turns, env)
    assert len(offers) == 2
    assert offers[0].terms["price"] == 100.0 and offers[0].terms["payment_days"] == 45.0
    assert offers[1].terms["price"] == 98.0 and offers[1].terms["payment_days"] == 45.0  # inherited


def test_resolve_keeps_better_than_best_offer():
    env = _env()
    # a price BELOW best (92) is better than the buyer's aspiration — a real concession that
    # must be recorded at its true value, not rewritten up to 92 (audit issue #7). Scoring
    # already saturates at utility 1.0, so keeping the true €10 costs no safety.
    offer = resolve_supplier_offer(env, "We'll do €10.00 per unit, net-45, 18 months.", None)
    assert offer is not None
    assert offer.terms["price"] == 10.0  # kept, not clamped up to best


def test_resolve_clamps_worse_than_worst_parse():
    env = _env()
    # a price ABOVE worst (108) is a stray/nonsense parse — clamped in to the worst bound so
    # it never leaves the envelope span on the losing side.
    offer = resolve_supplier_offer(env, "We'll do €200.00 per unit, net-45, 18 months.", None)
    assert offer is not None
    assert offer.terms["price"] == 108.0  # clamped to worst (108)
