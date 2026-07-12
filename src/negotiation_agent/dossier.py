"""The canned due-diligence dossier — a worked example that runs through the REAL spines.

The cockpit's demo needs something to show without a paid LLM scan. This module supplies a
bundled sample MSA and a hand-authored set of findings — but it does NOT fake the machinery:

  - the sample contract is rendered to a real PDF and decomposed by :mod:`negotiation_agent.anchor`,
  - every finding's quote is verified against its cited block (so the "anchored ✓" badges and the
    deep-links are genuine — one finding quotes text that ISN'T in the doc and legitimately
    quarantines, to show the mechanism working),
  - severities are escalated and the legal gate is computed by :mod:`negotiation_agent.gate`.

So the numbers, badges, and gate verdict a viewer sees are real computation on canned inputs —
honest to label "example, not a live scan". The live version swaps the hand-authored findings for
LLM extraction passes; the anchor + gate code underneath is identical.

Pure and deterministic: no LLM, no network. The economic breakdown is likewise a fixed worked
example (price build-up + economic risks) — the shape the live pricing/TCO pass will produce.
"""

from __future__ import annotations

from pydantic import BaseModel

from . import anchor, gate

# ── the sample contract ──────────────────────────────────────────────────────────
# (section heading, clause text). Each entry becomes its own paragraph on the page, so the
# anchor layer gives each clause a distinct block id. Text is realistic MSA boilerplate chosen
# so the findings below have verbatim quotes to anchor to.
_CLAUSES: list[tuple[str, str]] = [
    (
        "1. Definitions and Scope",
        "This Master Services Agreement ('Agreement') governs the supply of cloud services by "
        "CloudVendor SE ('Supplier') to the Customer. It is effective as of the Effective Date and "
        "continues for an initial term of thirty-six (36) months unless terminated earlier.",
    ),
    (
        "4. Fees and Indexation",
        "The annual subscription fee is EUR 400,000 for the committed capacity set out in Schedule A. "
        "The Supplier may increase the annual fee each year by the change in the consumer price index "
        "(CPI) plus three percent (3%), with no ceiling on the annual increase.",
    ),
    (
        "5. Usage and Overage",
        "Usage above the committed capacity is billed at the overage rate of EUR 0.12 per unit. The "
        "Customer commits to a minimum annual volume of 3,000,000 units regardless of actual usage.",
    ),
    (
        "12. Limitation of Liability",
        "The Supplier's aggregate liability arising out of or in connection with this Agreement, "
        "including any breach of data protection obligations, shall not exceed the fees paid by the "
        "Customer in the three (3) months preceding the claim.",
    ),
    (
        "16. Data Protection",
        "The Supplier shall process personal data in accordance with applicable law. The parties "
        "acknowledge that sub-processors may be engaged by the Supplier as required to deliver the "
        "services, and the Supplier's list of sub-processors may be updated from time to time.",
    ),
    (
        "18. Indemnification",
        "The Customer shall indemnify the Supplier against any and all claims arising from the "
        "Customer's use of the services, without limit as to amount.",
    ),
    (
        "21. Security and Incident Response",
        "The Supplier maintains commercially reasonable security measures. The Supplier will use "
        "reasonable efforts to inform the Customer of material incidents affecting the services.",
    ),
    (
        "23. Term and Termination",
        "Either party may terminate this Agreement for material breach on sixty (60) days' written "
        "notice if the breach remains uncured. The Customer has no right to terminate for "
        "convenience during the initial term.",
    ),
    (
        "25. Governing Law and Venue",
        "This Agreement is governed by the laws of Ireland. The parties submit to the exclusive "
        "jurisdiction of the courts of Dublin for any dispute arising under it.",
    ),
    (
        "27. Code of Conduct",
        "The Supplier agrees to conduct business ethically. The Agreement does not require the "
        "Supplier to comply with the Customer's Supplier Code of Conduct or to cascade equivalent "
        "obligations to its own sub-suppliers.",
    ),
    (
        "29. Warranties",
        "The Supplier warrants that the services will conform in all material respects to the "
        "documentation. Except as expressly stated, the services are provided 'as is' and the "
        "Supplier disclaims all other warranties to the maximum extent permitted by law.",
    ),
]


class _CannedFinding(BaseModel):
    """A hand-authored finding before verification: the model's proposed severity + a quote that
    (usually) lives in the sample. ``anchor_hint`` is the section heading text used to locate the
    block — resolved to a real anchor_id at build time, NOT hardcoded, so a clause edit can't
    silently break the link."""

    model_config = {"frozen": True}

    category: gate.Category
    llm_severity: gate.Severity
    title: str
    quote: str
    anchor_hint: str  # a substring of the target block's text (usually the section heading)
    why_it_hurts: str
    suggested_position: str = ""
    fallback_position: str = ""


_FINDINGS: list[_CannedFinding] = [
    _CannedFinding(
        category="legal",
        llm_severity="medium",  # the model under-rates it; the rule will raise it to high
        title="Liability capped at 3 months' fees, incl. data-breach damages",
        quote="shall not exceed the fees paid by the Customer in the three (3) months preceding the claim",
        anchor_hint="Limitation of Liability",
        why_it_hurts="A 3-month cap on a EUR 400k/yr deal is well below the usual 12-month landing "
        "zone, and it swallows data-breach damages — a single GDPR incident could cost far more.",
        suggested_position="Cap at 12 months' fees; carve data-protection breaches OUT of the cap.",
        fallback_position="Cap at 12 months' fees with a super-cap (e.g. 2x) for data breaches.",
    ),
    _CannedFinding(
        category="gdpr",
        llm_severity="high",  # the rule will raise it to critical (no DPA)
        title="No Art. 28 DPA; sub-processors unrestricted",
        quote="sub-processors may be engaged by the Supplier as required to deliver the services",
        anchor_hint="Data Protection",
        why_it_hurts="No data-processing agreement and no sub-processor consent/notice mechanism — "
        "an Art. 28 GDPR gap. The Customer carries controller liability with no contractual cover.",
        suggested_position="Require a signed Art. 28 DPA; prior notice + objection right on new "
        "sub-processors; EU/adequacy transfer mechanism named.",
        fallback_position="DPA + 30-day sub-processor notice with a right to terminate on objection.",
    ),
    _CannedFinding(
        category="commercial",
        llm_severity="high",
        title="Uncapped annual indexation, CPI + 3%",
        quote="the change in the consumer price index (CPI) plus three percent (3%), with no ceiling",
        anchor_hint="Fees and Indexation",
        why_it_hurts="CPI + 3% with no ceiling compounds. Over the 36-month term this can lift the "
        "annual fee well above budget with no negotiation trigger.",
        suggested_position="Cap annual indexation at CPI, or a hard 3% ceiling whichever is lower.",
        fallback_position="CPI + 1.5% with a 5% annual hard cap and a benchmarking right.",
    ),
    _CannedFinding(
        category="legal",
        llm_severity="medium",
        title="One-sided, uncapped indemnity from Customer",
        quote="indemnify the Supplier against any and all claims arising from the Customer's use of "
        "the services, without limit as to amount",
        anchor_hint="Indemnification",
        why_it_hurts="The Customer indemnifies the Supplier without limit, but there is no reciprocal "
        "indemnity from the Supplier for IP or data-protection claims.",
        suggested_position="Make the indemnity mutual and subject to the liability cap.",
        fallback_position="Cap the Customer indemnity and add a Supplier IP-infringement indemnity.",
    ),
    _CannedFinding(
        category="infosec",
        llm_severity="medium",
        title="No firm breach-notification obligation",
        quote="use reasonable efforts to inform the Customer of material incidents",
        anchor_hint="Security and Incident Response",
        why_it_hurts="'Reasonable efforts' with no deadline undercuts the Customer's own GDPR 72-hour "
        "breach-notification duty and incident-response SLAs.",
        suggested_position="Notify without undue delay and within 24 hours of a security incident.",
        fallback_position="48-hour notification with a defined incident-severity matrix.",
    ),
    _CannedFinding(
        category="coc",
        llm_severity="medium",
        title="Supplier not bound by the Customer Code of Conduct",
        quote="does not require the Supplier to comply with the Customer's Supplier Code of Conduct",
        anchor_hint="Code of Conduct",
        why_it_hurts="No CoC flow-down means no contractual basis for LkSG/CSDDD due-diligence "
        "expectations to reach the Supplier or its sub-suppliers.",
        suggested_position="Bind the Supplier to the Customer CoC and require cascade to sub-suppliers.",
        fallback_position="CoC acknowledgement + annual self-assessment and audit right.",
    ),
    _CannedFinding(
        # This one's quote is NOT in the contract — it demonstrates the anchor gate quarantining a
        # finding the model asserted but couldn't ground. It must show as unverified, never anchored.
        category="infosec",
        llm_severity="high",
        title="Claimed: no encryption-at-rest commitment (UNVERIFIED)",
        quote="the Supplier makes no commitment to encrypt Customer data at rest in any environment",
        anchor_hint="Security and Incident Response",
        why_it_hurts="If true this is a material InfoSec gap — but the quote could not be located in "
        "the contract, so it is quarantined pending human review rather than shown as a fact.",
    ),
]


# ── the economic breakdown (a fixed worked example — the shape the live pricing pass produces) ──
class PriceComponent(BaseModel):
    model_config = {"frozen": True}
    label: str
    amount_eur: float
    note: str = ""


class EconomicRisk(BaseModel):
    model_config = {"frozen": True}
    label: str
    severity: gate.Severity
    detail: str
    lever: str = ""  # the negotiation lever that neutralises it


class EconomicBreakdown(BaseModel):
    model_config = {"frozen": True}
    currency: str = "EUR"
    term_months: int
    build_up: list[PriceComponent]
    year1_total_eur: float
    tco_over_term_eur: float
    risks: list[EconomicRisk]


def _economic_breakdown() -> EconomicBreakdown:
    """Year-1 price build-up + TCO over the term, and the economic risks that drive negotiation.

    The TCO applies the uncapped CPI+3% indexation across the 36-month term (assuming 2% CPI, so
    5% per year) — the worked math the cockpit shows so the buyer sees what the indexation clause
    actually costs, not just that it exists.
    """
    base = 400_000.0
    overage = 60_000.0  # illustrative annual overage at EUR 0.12/unit over the 3M commit
    year1 = base + overage
    # 5%/yr compounding (2% CPI + 3%) over 3 years on the base fee, plus flat overage each year
    y2 = base * 1.05 + overage
    y3 = base * (1.05**2) + overage
    tco = round(year1 + y2 + y3, 2)
    return EconomicBreakdown(
        term_months=36,
        build_up=[
            PriceComponent(label="Committed subscription (Schedule A)", amount_eur=base,
                           note="EUR 400,000/yr flat for committed capacity"),
            PriceComponent(label="Expected overage", amount_eur=overage,
                           note="~500k units/yr over the 3M commit at EUR 0.12/unit"),
        ],
        year1_total_eur=year1,
        tco_over_term_eur=tco,
        risks=[
            EconomicRisk(
                label="Uncapped CPI + 3% indexation",
                severity="high",
                detail="At 2% CPI the fee rises ~5%/yr compounding — ~EUR 41k of extra spend over "
                "the term versus a flat fee, with no ceiling if inflation runs higher.",
                lever="Cap indexation at CPI or a 3% hard ceiling.",
            ),
            EconomicRisk(
                label="Overage with no price protection",
                severity="medium",
                detail="Overage at EUR 0.12/unit has no volume tiers — a demand spike bills linearly "
                "with no discount.",
                lever="Tiered overage pricing + a true-up/roll-over of unused commit.",
            ),
            EconomicRisk(
                label="Minimum commit above realistic forecast",
                severity="medium",
                detail="The 3,000,000-unit annual minimum is billed regardless of usage — pay-for-"
                "nothing risk if adoption lags.",
                lever="Lower the commit or add carry-forward of unused units.",
            ),
        ],
    )


# ── the assembled dossier ──────────────────────────────────────────────────────────
class DossierFinding(BaseModel):
    """A finding after the real anchor + gate machinery has run over it."""

    model_config = {"frozen": True}

    category: gate.Category
    severity: gate.Severity  # post-escalation
    llm_severity: gate.Severity  # what the model proposed (shown as "model said X, raised to Y")
    title: str
    quote: str
    anchor_id: str | None
    page_display: int | None
    verified: bool
    raised_by: tuple[str, ...]
    why_it_hurts: str
    suggested_position: str
    fallback_position: str


class ViewerBlock(BaseModel):
    """One block for the in-tool document viewer — the deep-link target the risk boxes jump to."""

    model_config = {"frozen": True}

    anchor_id: str
    page_display: int
    text: str


class Dossier(BaseModel):
    """The full cockpit payload: findings (with real verdicts), the gate, the document for the
    viewer, and the economic breakdown. The canned example sets ``is_example=True``; the live
    scan (:mod:`negotiation_agent.scan`) builds the same shape with ``is_example=False``.

    ``is_complete`` is False when a live scan had a specialist window fail — the dossier is built
    from the windows that ran, never a silent short read. ``no_anchorable_findings`` is True when
    the gate counted zero findings, so a UI can distinguish "extraction grounded nothing" from
    "clean contract" (both show review_required=False otherwise)."""

    model_config = {"frozen": True}

    is_example: bool = True
    is_complete: bool = True
    no_anchorable_findings: bool = False
    doc_title: str
    page_count: int
    findings: list[DossierFinding]
    gate: gate.GateVerdict
    blocks: list[ViewerBlock]
    economics: EconomicBreakdown


def assemble_dossier(
    document: anchor.Document,
    findings_pre: list[tuple[gate.RiskFinding, gate.Severity]],
    economics: EconomicBreakdown,
    *,
    doc_title: str,
    is_example: bool,
    is_complete: bool = True,
    policy: gate.GatePolicy | None = None,
) -> Dossier:
    """The shared tail for BOTH the canned and live paths — the trust layer, byte-identical.

    ``findings_pre`` pairs each pre-escalation :class:`gate.RiskFinding` with the LLM's originally
    proposed severity (so the UI can show "model said X → raised to Y"). This runs
    ``escalate_all`` → ``legal_gate`` → zips the proposal against the escalated finding → maps each
    to its page → builds the viewer blocks. Keeping this one function is what guarantees the live
    scan gets exactly the same verification and gate the canned example was proven with."""
    escalated = gate.escalate_all([f for f, _ in findings_pre])
    gate_verdict = gate.legal_gate(escalated, policy or gate.GatePolicy(owner="E. Procurement"))
    page_by_anchor = {b.anchor_id: b.page_display for b in document.blocks}
    dossier_findings = [
        DossierFinding(
            category=e.category,
            severity=e.severity,
            llm_severity=llm_sev,
            title=e.title,
            quote=e.quote,
            anchor_id=e.anchor_id,
            page_display=page_by_anchor.get(e.anchor_id) if e.anchor_id else None,
            verified=e.verified,
            raised_by=e.raised_by,
            why_it_hurts=e.why_it_hurts,
            suggested_position=e.suggested_position,
            fallback_position=e.fallback_position,
        )
        for (_, llm_sev), e in zip(findings_pre, escalated, strict=True)
    ]
    blocks = [
        ViewerBlock(anchor_id=b.anchor_id, page_display=b.page_display, text=b.text)
        for b in document.blocks
    ]
    return Dossier(
        is_example=is_example,
        is_complete=is_complete,
        no_anchorable_findings=gate_verdict.counted == 0,
        doc_title=doc_title,
        page_count=document.page_count,
        findings=dossier_findings,
        gate=gate_verdict,
        blocks=blocks,
        economics=economics,
    )


def _render_sample_pdf() -> bytes:
    """Render the sample clauses to a real PDF, one clause per paragraph block, paginated."""
    import fitz

    doc = fitz.open()
    # A6-ish short page so the sample paginates across a few pages — the viewer then shows real
    # page navigation (Page 2 of N), the way a real 47-page MSA would.
    page_w, page_h = 460.0, 360.0
    page = doc.new_page(width=page_w, height=page_h)
    y = 48.0
    bottom = page_h - 48
    for heading, body in _CLAUSES:
        para = f"{heading}\n{body}"
        lines = max(1, len(body) // 78 + 2)
        height = 15 * lines + 10
        if y + height > bottom:  # start a new page when the current one is full
            page = doc.new_page(width=page_w, height=page_h)
            y = 48.0
        page.insert_textbox(fitz.Rect(40, y, page_w - 40, y + height), para, fontsize=10)
        y += height + 12
    data: bytes = doc.tobytes()
    doc.close()
    return data


def build_dossier(policy: gate.GatePolicy | None = None) -> Dossier:
    """Build the canned dossier by running the hand-authored findings through the REAL anchor +
    gate code. Deterministic, no LLM, no network. ``policy`` overrides the default legal-gate
    thresholds (the cockpit passes the named owner)."""
    document = anchor.blocks_from_pdf(_render_sample_pdf(), doc_id="MSA_CloudVendor")

    # Resolve each finding's anchor_hint to a real block, then verify its quote against that block.
    risk_findings: list[gate.RiskFinding] = []
    resolved: list[tuple[_CannedFinding, str | None]] = []
    for canned in _FINDINGS:
        block = next((b for b in document.blocks if canned.anchor_hint in b.text), None)
        anchor_id = block.anchor_id if block else None
        verdict = (
            anchor.verify_finding(document, anchor_id, canned.quote)
            if anchor_id
            else anchor.Verdict(status="quarantined", reason="section not found")
        )
        resolved.append((canned, anchor_id))
        risk_findings.append(
            gate.RiskFinding(
                category=canned.category,
                severity=canned.llm_severity,
                title=canned.title,
                quote=canned.quote,
                anchor_id=verdict.anchor_id,
                verified=verdict.status == "anchored",
                why_it_hurts=canned.why_it_hurts,
                suggested_position=canned.suggested_position,
                fallback_position=canned.fallback_position,
            )
        )

    findings_pre = [
        (rf, canned.llm_severity) for rf, (canned, _) in zip(risk_findings, resolved, strict=True)
    ]
    return assemble_dossier(
        document,
        findings_pre,
        _economic_breakdown(),
        doc_title="MSA_CloudVendor_2026.pdf",
        is_example=True,
        policy=policy,
    )
