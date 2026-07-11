"""Contract intelligence — the deep-extraction schema and the finding→adjustment rules.

Two zones (``docs/contract-intelligence-architecture.md`` §2.1):
  - **Zone A** — engine-negotiable numbers (the 5 ``TermType``s). These flow into an
    ``Offer`` via the existing ``ContractExtraction``.
  - **Zone B** — intelligence fields (expiration, renewal, licenses, UoM, SKUs, NDA/DPA,
    governing law, liability cap). NOT ``TermType``s, never forced into an ``Offer``;
    they are inputs to the mandate transform and context for the human.

Grounding is split by *what the fact is about*: a fact read from the uploaded document
(``DocumentGrounded``) is sourced by the document; a fact about the world (sanctions,
geopolitics — ``SourcedFinding``) needs a source + retrieval date, and the type won't
construct without them. The transform reads only the discrete ``assurance``, never the
LLM's continuous ``confidence`` — that's what keeps mandate construction deterministic.

The rule engine (``propose_adjustments``) is a pure function: findings in, a list of
``ProposedAdjustment`` (from :mod:`shaper`) out. No LLM, no I/O.
"""

from __future__ import annotations

import datetime as _dt
import re
from typing import Literal

from pydantic import BaseModel, Field

from negotiation_agent.envelope import Direction, TermType
from negotiation_agent.intake import ContractExtraction, extract_contract
from negotiation_agent.research import SupplierBrief
from negotiation_agent.shaper import (
    W_GIVE,
    AddGate,
    AddTerm,
    ProposedAdjustment,
    ShiftTarget,
    _add_term_spec,
    days_until,
)

Assurance = Literal["confirmed", "probable", "unknown"]

# assurance thresholds — the pure-Python collapse of continuous confidence
_CONFIRMED_MIN = 0.85
_PROBABLE_MIN = 0.60


def derive_assurance(confidence: float, quote_verified: bool) -> Assurance:
    """Collapse continuous confidence + quote-verification into a discrete assurance.

    The transform is only ever allowed to read this — never ``confidence`` — so LLM
    sampling drift can't change the shaped envelope.
    """
    if confidence >= _CONFIRMED_MIN and quote_verified:
        return "confirmed"
    if confidence >= _PROBABLE_MIN:
        return "probable"
    return "unknown"


class DocumentGrounded(BaseModel):
    """A fact read from the uploaded contract. Source = the document the human holds."""

    model_config = {"frozen": True}

    value: str | None = None  # the parsed/typed value as text (e.g. "2026-12-31", "true")
    quote: str = ""  # verbatim span, grounding
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    source: Literal["regex", "llm"] = "llm"
    assurance: Assurance = "unknown"


class SourcedFinding(BaseModel):
    """A fact about the WORLD (supplier, sanctions). Cannot be built without a source +
    retrieval date — the compliance rule ("no source + date = not a finding") as a type.
    """

    model_config = {"frozen": True}

    claim: str
    source_ref: str  # list name / URL / assessment id, e.g. "OFAC SDN"
    retrieved_at: str  # ISO-8601; REQUIRED, no default
    as_of: str | None = None  # when the underlying fact was dated, if known
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    provider: Literal["hades", "sample", "derived", "manual"]


class ContractLifecycle(BaseModel):
    model_config = {"frozen": True}
    effective_date: DocumentGrounded | None = None
    expiration_date: DocumentGrounded | None = None
    initial_term_months: DocumentGrounded | None = None
    auto_renews: DocumentGrounded | None = None
    renewal_notice_days: DocumentGrounded | None = None
    termination_notice_days: DocumentGrounded | None = None


class LegalFlags(BaseModel):
    model_config = {"frozen": True}
    has_nda: DocumentGrounded | None = None  # tri-state: True / False / None(unknown)
    has_dpa: DocumentGrounded | None = None
    governing_law: DocumentGrounded | None = None
    liability_cap: DocumentGrounded | None = None
    data_processing_location: DocumentGrounded | None = None


class LineItem(BaseModel):
    model_config = {"frozen": True}
    sku: DocumentGrounded | None = None
    description: DocumentGrounded | None = None
    quantity: DocumentGrounded | None = None
    unit: DocumentGrounded | None = None  # "pcs" | "kg" | "1000 tokens" | "user/month"


class ContractIntelligence(BaseModel):
    """The full picture: the negotiable numbers (Zone A) + the intelligence (Zone B)."""

    model_config = {"frozen": True}

    extraction: ContractExtraction  # Zone A — unchanged shape, backward compatible
    lifecycle: ContractLifecycle | None = None
    licenses: list[LineItem] = Field(default_factory=list)
    line_items: list[LineItem] = Field(default_factory=list)
    legal: LegalFlags | None = None
    units_of_measure: list[str] = Field(default_factory=list)
    extractor_used: Literal["regex", "regex+llm"] = "regex"
    llm_model: str | None = None
    conflicts: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def supplier_name(self) -> str | None:
        return self.extraction.supplier_name


# Expiry windows that drive the lifecycle adjustment rules. Named so a single edit re-tunes
# both the "expiring soon" (lower the floor to close) and "far out" (hold firm) thresholds.
_EXPIRING_SOON_DAYS = 30
_EXPIRING_FAR_DAYS = 180


# ── the rule engine — pure: findings → proposed adjustments ───────────────────
def propose_adjustments(
    intel: ContractIntelligence,
    brief: SupplierBrief | None,
    *,
    today: _dt.date,
) -> tuple[list[ProposedAdjustment], bool, str | None]:
    """Map findings to bounded, human-reviewable adjustments. Returns
    ``(adjustments, blocked, block_reason)``. Pure — no LLM, no I/O.

    A ``blocked`` proposal (sanctions/registry) emits no signable mandate change; the
    human must clear it. Rules fire only on confirmed document facts or sourced world
    findings — a sample-sourced brief is display-only and shapes nothing.
    """
    adjustments: list[ProposedAdjustment] = []
    blocked = False
    block_reason: str | None = None

    # ── critical: sanctions / registry block (only on a REAL, non-sample brief) ──
    if brief is not None and brief.source == "hades":
        if brief.sanctioned is True or brief.is_blocking:
            return (
                [],
                True,
                (
                    f"Supplier flagged (recommendation: {brief.recommendation or 'block'}). "
                    "A human must clear this before any mandate is signed."
                ),
            )
        if (brief.registry_status or "").lower().startswith(("dissolved", "insolvent")):
            return (
                [],
                True,
                (
                    f"Supplier registry status is '{brief.registry_status}'. "
                    "Block pending verification."
                ),
            )

    # ── supplier-risk rules (real Hades brief only; sample is display-only) ──
    if brief is not None and brief.source == "hades":
        if brief.lksg_signal == "red_flag":
            adjustments.append(
                ProposedAdjustment(
                    rule_id="R-LKSG-REDFLAG",
                    severity="high",
                    role="hedge",
                    delta=ShiftTarget(target_delta=0.03),
                    rationale=(
                        "LkSG/CSDDD red flag on this supplier. Anchor harder — the "
                        "oversight cost is real. Also require a remediation clause."
                    ),
                )
            )
            adjustments.append(
                ProposedAdjustment(
                    rule_id="R-LKSG-REDFLAG-GATE",
                    severity="high",
                    role="hold",
                    delta=AddGate(gate_id="lksg_remediation", label="LkSG remediation clause"),
                    rationale="Require a documented remediation plan as a signing precondition.",
                )
            )
        elif brief.lksg_signal == "needs_monitoring":
            adjustments.append(
                ProposedAdjustment(
                    rule_id="R-LKSG-MONITOR",
                    severity="low",
                    role="hold",
                    delta=AddGate(
                        gate_id="lksg_declaration",
                        label="Current LkSG risk-management declaration",
                        severity="preferred",
                    ),
                    rationale="LkSG monitoring advised. Request a current declaration.",
                )
            )

    # ── contract lifecycle rules (document-grounded, confirmed only) ──
    life = intel.lifecycle
    if life and life.expiration_date and life.expiration_date.assurance == "confirmed":
        n = days_until(life.expiration_date.value, today=today)
        if n is not None and 0 <= n < _EXPIRING_SOON_DAYS:
            adjustments.append(
                ProposedAdjustment(
                    rule_id="R-EXPIRING-SOON",
                    severity="medium",
                    role="hedge",
                    delta=ShiftTarget(target_delta=-0.03, reservation_delta=-0.02),
                    rationale=(
                        f"Contract expires in {n} days. Our no-deal alternative is worse "
                        "under time pressure — lower the floor so we can close, don't hold "
                        "out for the last basis point."
                    ),
                )
            )
        elif n is not None and n > _EXPIRING_FAR_DAYS:
            adjustments.append(
                ProposedAdjustment(
                    rule_id="R-EXPIRING-FAR",
                    severity="low",
                    role="hold",
                    delta=ShiftTarget(target_delta=0.02),
                    rationale=f"Plenty of runway ({n} days). Anchor harder — no urgency.",
                )
            )

    # ── legal-flag rules (explicit False + confirmed only; None never triggers) ──
    legal = intel.legal
    if (
        legal
        and legal.has_dpa
        and legal.has_dpa.value == "false"
        and legal.has_dpa.assurance == "confirmed"
    ):
        adjustments.append(
            ProposedAdjustment(
                rule_id="R-DPA-MISSING",
                severity="high",
                role="hold",
                delta=AddGate(gate_id="dpa_signed", label="Signed DPA (GDPR Art. 28)"),
                rationale=(
                    "No DPA found. GDPR Art. 28 requires one before processing personal "
                    "data — a non-negotiable signing precondition, not a price lever."
                ),
            )
        )
    if (
        legal
        and legal.has_nda
        and legal.has_nda.value == "false"
        and legal.has_nda.assurance == "confirmed"
    ):
        adjustments.append(
            ProposedAdjustment(
                rule_id="R-NDA-MISSING",
                severity="medium",
                role="hold",
                delta=AddGate(gate_id="nda_in_place", label="NDA in place"),
                rationale="No NDA found. Required before sharing volumes or roadmap.",
            )
        )

    # ── give-term rules (trade bait — pure upside the supplier can grant) ──
    extracted_names = {t.name for t in intel.extraction.terms}
    has_volume = "volume_units" in extracted_names
    if has_volume and "rebate_pct" not in extracted_names:
        adjustments.append(
            ProposedAdjustment(
                rule_id="R-NO-REBATE",
                severity="low",
                role="give",
                delta=AddTerm(
                    spec=_add_term_spec(
                        "rebate_pct", TermType.REBATE_PCT, Direction.MAXIMIZE, 8.0, 0.0, W_GIVE
                    ),
                    appetite=0.8,
                ),
                rationale=(
                    "No volume rebate despite committed volume. Adding a rebate ask is "
                    "pure upside — the supplier can grant it cheaply, we weight it lightly."
                ),
            )
        )

    return adjustments, blocked, block_reason


# ── regex Zone-B extractor — the deterministic path (the LLM extractor is v1-later) ──
# Real figures are short; bounded patterns keep this linear on hostile input.
_EXPIRY_RE = re.compile(
    r"(?:expir\w+|terminat\w+|end\w*|valid until|through)\D{0,20}"
    r"(\d{4}-\d{2}-\d{2}|\d{1,2}[./]\d{1,2}[./]\d{4})",
    re.I,
)
_DPA_POS_RE = re.compile(r"data[\s-]?processing agreement|\bDPA\b|Art(?:icle)?\.?\s*28", re.I)
_NDA_POS_RE = re.compile(r"non[\s-]?disclosure|confidentiality agreement|\bNDA\b", re.I)
_UOM_RE = re.compile(
    r"\bper\s+(unit|seat|user|licen[cs]e|month|year|1?,?000 tokens|kg|piece|hour)\b", re.I
)


def extract_intelligence(text: str) -> ContractIntelligence:
    """Deterministic (regex-only) contract intelligence — works offline, no LLM.

    Populates Zone A via the existing extractor and the Zone-B fields regex can reach
    (expiration date, NDA/DPA presence, units of measure). Fields it can't determine
    stay ``None`` — so the rule engine simply doesn't fire on them, per the design's
    graceful-degradation rule. The LLM extractor (v1-later) fills the rest behind the
    same ``ContractExtraction`` Protocol.
    """
    extraction = extract_contract(text)
    warnings = list(extraction.warnings)

    lifecycle: ContractLifecycle | None = None
    m = _EXPIRY_RE.search(text)
    if m:
        raw = m.group(1).replace("/", ".") if "/" in m.group(1) else m.group(1)
        # regex date match is a confirmed document fact (verbatim span present)
        lifecycle = ContractLifecycle(
            expiration_date=DocumentGrounded(
                value=raw, quote=m.group(0).strip(), source="regex", confidence=0.9,
                assurance="confirmed",
            )
        )

    # NDA/DPA: regex can confirm PRESENCE (positive match). It cannot confirm ABSENCE —
    # a missing keyword might mean the clause is worded differently. So a non-match is
    # left None (unknown), never a confirmed False. Only the LLM extractor asserts False.
    legal = LegalFlags(
        has_dpa=DocumentGrounded(value="true", quote="", source="regex", assurance="confirmed")
        if _DPA_POS_RE.search(text) else None,
        has_nda=DocumentGrounded(value="true", quote="", source="regex", assurance="confirmed")
        if _NDA_POS_RE.search(text) else None,
    )

    uoms = sorted({u.lower() for u in _UOM_RE.findall(text)})

    return ContractIntelligence(
        extraction=extraction, lifecycle=lifecycle,
        legal=legal if (legal.has_dpa or legal.has_nda) else None,
        units_of_measure=uoms, extractor_used="regex", warnings=warnings,
    )


class IntelligenceResult(BaseModel):
    """Everything the /intel endpoint returns: the picture, the brief, the proposed
    adjustments, and whether the supplier is blocked. Framework-free."""

    model_config = {"frozen": True}

    intelligence: ContractIntelligence
    brief: SupplierBrief | None = None
    adjustments: list[ProposedAdjustment] = Field(default_factory=list)
    blocked: bool = False
    block_reason: str | None = None
    research_note: str | None = None


def prepare_intelligence(
    contract_text: str,
    brief: SupplierBrief | None,
    *,
    today: _dt.date,
    research_note: str | None = None,
) -> IntelligenceResult:
    """Compose extraction + supplier brief into the full intelligence result. Pure.

    ``brief`` is passed in (fetched by the caller's ``HadesClient`` server-side) so this
    stays testable and framework-free. The rule engine turns findings into proposed
    adjustments the human reviews before signing.
    """
    intel = extract_intelligence(contract_text)
    adjustments, blocked, reason = propose_adjustments(intel, brief, today=today)
    return IntelligenceResult(
        intelligence=intel, brief=brief, adjustments=adjustments,
        blocked=blocked, block_reason=reason, research_note=research_note,
    )
