"""Wire schemas — round-trip and the small pieces of behavior they carry."""

from __future__ import annotations

import json

from negotiation_agent.wire import (
    GuardAttempt,
    GuardAudit,
    MandateConfig,
    MandateEnvelope,
    SignedMandate,
    StepResponse,
    TranscriptView,
    TurnResult,
)


def test_guard_audit_redrafted_reflects_attempt_count():
    passed = GuardAudit(released_by="model", attempts=[GuardAttempt(draft="ok", ok=True)])
    assert passed.redrafted is False

    redrafted = GuardAudit(
        released_by="model",
        attempts=[
            GuardAttempt(draft="€10.20 …", ok=False, violations=["10.20"]),
            GuardAttempt(draft="€9.00 …", ok=True),
        ],
    )
    assert redrafted.redrafted is True


def test_signed_mandate_round_trips_through_json():
    m = MandateEnvelope(
        envelope={"negotiation_id": "n1", "version": 1},
        supplier_appetite={"price": 0.15, "payment_days": 0.85},
        config=MandateConfig(max_rounds=6, beta=2.5),
    )
    signed = SignedMandate(mandate=m, session_id="s1", iat=1000, exp=4600, sig="deadbeef")
    dumped = signed.model_dump_json()
    back = SignedMandate.model_validate_json(dumped)
    assert back == signed
    assert back.mandate.config.beta == 2.5


def test_default_mandate_config_matches_design():
    c = MandateConfig()
    assert (c.max_rounds, c.beta, c.stall_rounds, c.on_unknown_terms) == (6, 2.5, 3, "escalate")


def test_step_response_serializes_without_internal_when_absent():
    turn = TurnResult(
        outcome="counter",
        round_index=1,
        reason_tag="counter",
        approved_numbers={"price": 96.0},
        buyer_message="…",
    )
    resp = StepResponse(
        buyer_view=TranscriptView(turns=[]),
        supplier_view=TranscriptView(turns=[]),
        turn=turn,
        terminal=False,
    )
    payload = json.loads(resp.model_dump_json())
    # internal is null by default -> no buyer-private numbers on the wire
    assert payload["turn"]["internal"] is None
