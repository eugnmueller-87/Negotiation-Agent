"""Mandate factory — compile one portfolio contract row into an envelope pair.

Phase 3 batch simulation: a category manager's row (baseline price, ONE
instruction — cancel, or renew within ±X% — annual spend, extraction
confidence) becomes a signable buyer :class:`Envelope` plus the hidden supplier
truth the deterministic ``ParametricSupplier`` negotiates from. PURE module:
no web framework, no LLM, no I/O, no clock — unit-testable and reusable from
the API, the CLI, and scripts.

## The scale fix (the honesty guarantee)

The v0 reference envelopes hardcode a 9..12 price band
(``simulator/scenarios.py``, copied into ``scripts/gen_tailspend_demo.py``).
Scored against a real contract price — say EUR 272 — every value lies far
outside that band, so ``linear_value`` CLAMPS: the buyer scores the real price
0.0, the supplier bot scores it 1.0, and the engine "negotiates" EUR ~10
counters against a EUR 272 contract. Convert that settlement to money and the
batch fabricates a ~96% saving nobody authorized.

Here BOTH envelopes derive their price band from the row's own baseline::

    buyer:    best = baseline*(1 - pct)   worst = baseline*(1 + pct)   MINIMIZE
    supplier: best = baseline*(1 + pct)   worst = baseline*(1 - pct)   MAXIMIZE

``linear_value`` is affine-invariant, so this is exactly the v0 reference
geometry re-based onto the contract's own price axis: the ZOPA structure the
reference matrix proves is preserved; every price ``fill_package`` can emit
lies INSIDE the scored span (``linear_inverse`` clamps to it); no synthetic
offer can saturate at utility 1.0; and the settled price converts to EUR
against the baseline exactly — bounded by the ±pct the human signed.

A row with no baseline price keeps the abstract reference band and is flagged
``price_scaled=False``: its result is utility-only and MUST never be converted
to EUR (the API layer enforces this).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from negotiation_agent.envelope import Direction, Envelope, TermSpec, TermType

# Rows below this extraction confidence route to a human, never to the engine —
# a guessed baseline must not become a mandate. Mirrors the threshold in
# ``intake.ContractExtraction.low_confidence``.
LOW_CONFIDENCE_THRESHOLD = 0.6

# Abstract reference price band for rows with NO baseline (utility-only results).
_REFERENCE_PRICE_BAND = (9.0, 12.0)

# Who "signs" the synthetic counterparty's hidden envelope. Explicitly labeled
# synthetic so an audit replay can never mistake it for a real supplier mandate.
_SYNTHETIC_SIGNER = "parametric-supplier (synthetic counterparty)"


class ContractRow(BaseModel):
    """One portfolio line: a contract, its baseline economics, ONE instruction."""

    model_config = {"frozen": True}

    row_id: str = Field(min_length=1, max_length=64)
    supplier_name: str | None = Field(default=None, max_length=200)
    instruction: Literal["cancel", "renew"]
    # "renew, keep pricing ±X%": the mandate half-width in percent of baseline.
    renew_pct: float | None = Field(default=None, gt=0.0, le=50.0)
    baseline_price: float | None = Field(default=None, gt=0.0, le=1e9)
    annual_spend_eur: float | None = Field(default=None, gt=0.0, le=1e12)
    extraction_confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_instruction(self) -> ContractRow:
        # Strict pairing catches CSV column-mapping bugs at the boundary instead
        # of silently negotiating under a half-formed mandate.
        if self.instruction == "renew" and self.renew_pct is None:
            raise ValueError(f"row {self.row_id!r}: 'renew' requires renew_pct")
        if self.instruction == "cancel" and self.renew_pct is not None:
            raise ValueError(f"row {self.row_id!r}: 'cancel' must not carry renew_pct")
        return self


class CompiledMandate(BaseModel):
    """Routing decision + (for renewals) the compiled envelope pair."""

    model_config = {"frozen": True}

    route: Literal["negotiate", "terminate", "human_confirm"]
    buyer_envelope: Envelope | None = None
    supplier_envelope: Envelope | None = None
    # False => the price band is the abstract reference scale: utility-only,
    # no EUR may ever be derived from the settlement.
    price_scaled: bool = False
    reasons: list[str] = Field(default_factory=list)


def compile_row(row: ContractRow, *, signed_by: str) -> CompiledMandate:
    """Route one row; for renewals, compile its baseline-scaled envelope pair.

    ``signed_by`` is the category manager authorizing the batch — it lands on
    every buyer envelope, so the audit trail names the human behind each mandate.

    Routing order is deliberate: ``cancel`` is decided BEFORE the low-confidence
    gate because termination uses no price — a low-confidence baseline is
    irrelevant to a cancel, so it correctly routes straight to the clock. Only a
    ``renew`` (which negotiates on price) is held for human confirmation.
    """
    if row.instruction == "cancel":
        return CompiledMandate(
            route="terminate",
            reasons=[
                "cancel → contract-termination clock; any figure is COST AVOIDANCE, "
                "never a negotiated saving"
            ],
        )
    if row.extraction_confidence < LOW_CONFIDENCE_THRESHOLD:
        return CompiledMandate(
            route="human_confirm",
            reasons=[
                f"extraction confidence {row.extraction_confidence:.2f} is below "
                f"{LOW_CONFIDENCE_THRESHOLD} — confirm the baseline price before it "
                "becomes a mandate; a guessed price is never negotiated"
            ],
        )

    reasons: list[str] = []
    if row.baseline_price is not None:
        pct = (row.renew_pct or 0.0) / 100.0
        price_best = row.baseline_price * (1.0 - pct)
        price_worst = row.baseline_price * (1.0 + pct)
        price_scaled = True
    else:
        price_best, price_worst = _REFERENCE_PRICE_BAND
        price_scaled = False
        reasons.append(
            "no baseline price — abstract reference band; result is utility-only "
            "and must not be converted to EUR"
        )

    negotiation_id = f"portfolio-{row.row_id}"
    return CompiledMandate(
        route="negotiate",
        buyer_envelope=_buyer_envelope(negotiation_id, signed_by, price_best, price_worst),
        # supplier MAXIMISEs price: its best is the buyer's worst end and vice-versa. Pass by
        # keyword so the mirror is explicit and a future edit can't silently un-mirror it (FIX 6).
        supplier_envelope=_supplier_envelope(
            negotiation_id, price_hi=price_worst, price_lo=price_best
        ),
        price_scaled=price_scaled,
        reasons=reasons,
    )


def settled_savings(
    *,
    baseline_price: float,
    settled_price: float,
    annual_spend_eur: float | None,
) -> tuple[float, float | None]:
    """EXACT savings of a simulated close against the contract's own baseline.

    ``ratio = (baseline - settled) / baseline``; ``eur = annual_spend * ratio``
    (constant-volume assumption: the row's annual spend was incurred at the
    baseline price). A NEGATIVE ratio — the simulation conceded a price increase
    within the ±pct mandate — is reported, never hidden. Returns ``(ratio, eur)``;
    ``eur`` is None when no annual spend was supplied (ratio-only result).
    """
    if baseline_price <= 0:
        raise ValueError("baseline_price must be positive")
    ratio = (baseline_price - settled_price) / baseline_price
    eur = round(annual_spend_eur * ratio, 2) if annual_spend_eur is not None else None
    return round(ratio, 6), eur


# --- envelope builders (reference geometry, price band parameterized) ----------
# Non-price bands are identical to the v0 reference matrix (simulator/scenarios.py),
# which zopa_check and the eval suite validate. They are in universal units
# (days / months / units / %) so they need no per-contract scaling — and they
# never feed the EUR math: only the price term converts to money.


def _buyer_envelope(
    negotiation_id: str, signed_by: str, price_best: float, price_worst: float
) -> Envelope:
    return Envelope(
        negotiation_id=negotiation_id,
        version=1,
        signed_by=signed_by,
        target_utility=0.95,
        reservation_utility=0.55,
        terms=[
            TermSpec(
                name="price",
                term_type=TermType.PRICE,
                direction=Direction.MINIMIZE,
                best=price_best,
                worst=price_worst,
                weight=0.45,
            ),
            TermSpec(
                name="payment_days",
                term_type=TermType.PAYMENT_DAYS,
                direction=Direction.MAXIMIZE,
                best=90,
                worst=30,
                weight=0.15,
            ),
            TermSpec(
                name="contract_months",
                term_type=TermType.CONTRACT_MONTHS,
                direction=Direction.MINIMIZE,
                best=12,
                worst=36,
                weight=0.10,
            ),
            TermSpec(
                name="volume_units",
                term_type=TermType.VOLUME_UNITS,
                direction=Direction.MINIMIZE,
                best=10000,
                worst=50000,
                weight=0.10,
            ),
            TermSpec(
                name="rebate_pct",
                term_type=TermType.REBATE_PCT,
                direction=Direction.MAXIMIZE,
                best=8.0,
                worst=0.0,
                weight=0.20,
            ),
        ],
    )


def _supplier_envelope(
    negotiation_id: str, *, price_hi: float, price_lo: float
) -> Envelope:
    # The hidden truth mirrors the reference supplier: price matters less than the
    # buyer thinks; the real pain is cash flow (payment days) and volume certainty.
    # Its price band is the exact mirror of the buyer's, so both agents score every
    # reachable price inside their span — the anti-clamping half of the scale fix.
    # The supplier MAXIMISEs price: best = the HIGH end, worst = the LOW end.
    return Envelope(
        negotiation_id=negotiation_id,
        version=1,
        signed_by=_SYNTHETIC_SIGNER,
        target_utility=0.92,
        reservation_utility=0.50,
        terms=[
            TermSpec(
                name="price",
                term_type=TermType.PRICE,
                direction=Direction.MAXIMIZE,
                best=price_hi,
                worst=price_lo,
                weight=0.20,
            ),
            TermSpec(
                name="payment_days",
                term_type=TermType.PAYMENT_DAYS,
                direction=Direction.MINIMIZE,
                best=30,
                worst=90,
                weight=0.30,
            ),
            TermSpec(
                name="contract_months",
                term_type=TermType.CONTRACT_MONTHS,
                direction=Direction.MAXIMIZE,
                best=36,
                worst=12,
                weight=0.10,
            ),
            TermSpec(
                name="volume_units",
                term_type=TermType.VOLUME_UNITS,
                direction=Direction.MAXIMIZE,
                best=50000,
                worst=10000,
                weight=0.30,
            ),
            TermSpec(
                name="rebate_pct",
                term_type=TermType.REBATE_PCT,
                direction=Direction.MINIMIZE,
                best=0.0,
                worst=8.0,
                weight=0.10,
            ),
        ],
    )
