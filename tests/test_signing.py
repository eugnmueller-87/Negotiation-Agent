"""Mandate signing — tamper detection, session binding, expiry.

The server re-runs the engine from the client's mandate, so the signature is the
only thing stopping a client from dropping its own reservation floor to extract a
below-floor deal. These tests pin that it can't.
"""

from __future__ import annotations

import pytest

from negotiation_agent.signing import MandateError, sign_mandate, verify_mandate
from negotiation_agent.wire import MandateConfig, MandateEnvelope

SECRET = "test-secret-value"


def _mandate(reservation=0.60):
    return MandateEnvelope(
        envelope={
            "negotiation_id": "n1", "version": 1, "signed_by": "buyer",
            "target_utility": 0.90, "reservation_utility": reservation,
            "terms": [
                {"name": "price", "term_type": "price", "direction": "minimize",
                 "best": 92.0, "worst": 108.0, "weight": 1.0},
            ],
        },
        supplier_appetite={"price": 0.5},
        config=MandateConfig(max_rounds=6, beta=2.5),
    )


def test_signed_mandate_verifies():
    m = _mandate()
    signed = sign_mandate(m, "sess-1", iat=1000, exp=4600, secret=SECRET)
    got = verify_mandate(signed, SECRET, now=2000)
    assert got == m


def test_tampered_mandate_is_rejected():
    signed = sign_mandate(_mandate(reservation=0.60), "sess-1", 1000, 4600, SECRET)
    # attacker lowers the reservation floor to 0.0 to extract a below-floor deal
    tampered = signed.model_copy(
        update={"mandate": _mandate(reservation=0.0)}
    )
    with pytest.raises(MandateError, match="verify"):
        verify_mandate(tampered, SECRET, now=2000)


def test_expired_mandate_is_rejected():
    signed = sign_mandate(_mandate(), "sess-1", iat=1000, exp=4600, secret=SECRET)
    with pytest.raises(MandateError, match="expired"):
        verify_mandate(signed, SECRET, now=5000)  # now > exp


def test_wrong_secret_is_rejected():
    signed = sign_mandate(_mandate(), "sess-1", 1000, 4600, SECRET)
    with pytest.raises(MandateError):
        verify_mandate(signed, "different-secret", now=2000)


def test_session_binding_prevents_replay():
    # a signature for session A must not verify when re-labelled as session B
    signed = sign_mandate(_mandate(), "sess-A", 1000, 4600, SECRET)
    replayed = signed.model_copy(update={"session_id": "sess-B"})
    with pytest.raises(MandateError):
        verify_mandate(replayed, SECRET, now=2000)
