"""Mandate signing — the tamper-check for the stateless negotiation server.

The server re-runs ``DealEngine.decide`` from the client's mandate every step, so a
client that could edit the mandate (drop ``reservation_utility`` to 0) could extract
a below-floor "deal". The signature binds the mandate to a session and a time window
so it can't be tampered with or replayed across sessions to dodge the per-session
spend cap. See ``docs/peitho-v2-architecture.md`` §3.3.

The HMAC secret is read from ``PEITHO_MANDATE_SECRET`` in the environment — never in
code. Verification uses ``hmac.compare_digest`` (constant-time). Canonicalization is
pinned to sorted, separator-fixed JSON over the *parsed* model so a float never
round-trips to a different byte string and breaks the MAC.
"""

from __future__ import annotations

import hashlib
import hmac
import json

from negotiation_agent.wire import MandateEnvelope, SignedMandate


class MandateError(Exception):
    """The signed mandate is missing, malformed, tampered, or expired."""


def _canonical(mandate: MandateEnvelope, session_id: str, iat: int, exp: int) -> bytes:
    payload = {
        "mandate": mandate.model_dump(mode="json"),
        "session_id": session_id,
        "iat": iat,
        "exp": exp,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign_mandate(
    mandate: MandateEnvelope, session_id: str, iat: int, exp: int, secret: str
) -> SignedMandate:
    """Produce a :class:`SignedMandate` with a session-scoped, time-boxed HMAC tag."""
    mac = hmac.new(
        secret.encode("utf-8"), _canonical(mandate, session_id, iat, exp), hashlib.sha256
    ).hexdigest()
    return SignedMandate(mandate=mandate, session_id=session_id, iat=iat, exp=exp, sig=mac)


def verify_mandate(signed: SignedMandate, secret: str, now: int) -> MandateEnvelope:
    """Verify the tag and expiry; return the mandate or raise :class:`MandateError`.

    ``now`` is passed in (not read from a clock) so verification stays pure and
    testable. Constant-time comparison prevents a timing side-channel on the MAC.
    """
    expected = hmac.new(
        secret.encode("utf-8"),
        _canonical(signed.mandate, signed.session_id, signed.iat, signed.exp),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(expected, signed.sig):
        raise MandateError("mandate signature does not verify")
    if now > signed.exp:
        raise MandateError("mandate has expired")
    return signed.mandate


def mandate_hash(mandate: MandateEnvelope) -> str:
    """A short, non-secret content hash for display ('ref' on the mandate card).

    Decoration, not a security control — it uses no secret and is not collision-proof;
    it just lets the UI show a stable reference for the signed mandate.
    """
    canonical = json.dumps(
        mandate.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()[:12]
