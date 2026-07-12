"""Risk-finding severity + the legal-review gate — deterministic code deciding, LLM advising.

The cockpit's second load-bearing principle, after anchoring: **the LLM proposes a severity, a
deterministic rules layer sits on top that can only RAISE it, never lower it, and a named human
owns every threshold.** So a model that shrugs "low" at a missing DPA is overruled to Critical by
a rule; a model that panics can't be talked down below what a rule floors it at.

Two pure functions, no LLM, no I/O, no RNG:

  - :func:`escalate_severity` — apply the raise-only rules to one finding. Each rule that fires is
    recorded in ``raised_by`` so the escalation is auditable ("Critical because R-GDPR-NO-DPA").
  - :func:`legal_gate` — from the escalated findings and a *named, configurable* policy, compute
    whether legal review is required. The verdict carries the RULE that fired, not just a boolean —
    that is what makes it defensible when someone asks why legal got pulled in.

Severity is ordered ``low < medium < high < critical``. "Raise, never lower" is enforced by
:func:`_max_severity` — a rule can only move a finding up the ladder.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

Severity = Literal["low", "medium", "high", "critical"]
Category = Literal["legal", "gdpr", "infosec", "coc", "commercial"]

# The severity ladder. Index = rank; a rule may only move a finding to a HIGHER rank.
_SEVERITY_ORDER: tuple[Severity, ...] = ("low", "medium", "high", "critical")
_RANK: dict[Severity, int] = {s: i for i, s in enumerate(_SEVERITY_ORDER)}


def _max_severity(a: Severity, b: Severity) -> Severity:
    """Return the higher of two severities — the primitive that guarantees raise-never-lower."""
    return a if _RANK[a] >= _RANK[b] else b


class RiskFinding(BaseModel):
    """One risk the extraction surfaced, with the severity the LLM proposed and the anchor that
    proves where it lives. ``verified`` comes from :mod:`negotiation_agent.anchor` — an unverified
    finding is shown distinctly and never counts toward the legal gate (see :func:`legal_gate`).

    ``suggested_position``/``fallback_position`` turn a risk into a negotiation lever — the Peitho
    story: this is prep, not just a scanner. ``raised_by`` is empty until :func:`escalate_severity`
    records which rule(s) lifted the severity above what the model proposed.
    """

    model_config = {"frozen": True}

    category: Category
    severity: Severity  # as proposed by the LLM; escalate_severity may raise it
    title: str
    quote: str = ""  # the verbatim span the anchor layer verified
    anchor_id: str | None = None
    verified: bool = False
    why_it_hurts: str = ""
    suggested_position: str = ""
    fallback_position: str = ""
    raised_by: tuple[str, ...] = ()  # rule ids that raised this finding's severity


class EscalationRule(BaseModel):
    """A deterministic raise-only rule: if a finding in ``category`` matches ``pattern`` (against
    its title + quote), its severity is raised to at least ``floor``. Rules never lower severity.
    """

    model_config = {"frozen": True}

    rule_id: str
    category: Category
    pattern: str  # regex, matched case-insensitively against "title \n quote"
    floor: Severity
    rationale: str


# The built-in escalation rules. These encode procurement/legal judgement that must not depend on
# an LLM's mood: a missing DPA or uncapped buyer liability is Critical regardless of what the model
# said. Ordered by rule_id; all matching rules apply (severity ends at the highest floor fired).
DEFAULT_RULES: tuple[EscalationRule, ...] = (
    EscalationRule(
        rule_id="R-GDPR-NO-DPA",
        category="gdpr",
        pattern=r"\b(no|missing|without|absent|lacks?)\b.{0,40}"
        r"\b(dpa|data[ -]?processing agreement)\b"
        r"|\bdpa\b.{0,20}\b(missing|absent|not (in place|included|present))\b",
        floor="critical",
        rationale="A missing Art. 28 data-processing agreement is a GDPR compliance failure — "
        "Critical regardless of the model's proposed severity.",
    ),
    EscalationRule(
        rule_id="R-GDPR-SUBPROCESSOR",
        category="gdpr",
        pattern=r"\bsub[- ]?processor",
        floor="high",
        rationale="Unrestricted sub-processor terms need legal sign-off (Art. 28(2) consent).",
    ),
    EscalationRule(
        rule_id="R-LEGAL-UNCAPPED-LIABILITY",
        category="legal",
        pattern=r"\buncapped\b.{0,40}\bliabilit|\bunlimited\b.{0,40}\bliabilit"
        r"|\bliabilit.{0,40}\b(uncapped|unlimited|no (cap|limit))\b",
        floor="critical",
        rationale="Uncapped buyer liability is an unbounded risk — Critical, needs legal review.",
    ),
    EscalationRule(
        rule_id="R-LEGAL-LOW-LIABILITY-CAP",
        category="legal",
        # tolerate "three (3) months" — the number word AND a parenthesized digit before "months"
        pattern=r"\bliabilit.{0,80}\b(3|three)\b.{0,10}months?|\bcap(ped)?\b.{0,40}\b(3|three)\b.{0,10}months?",
        floor="high",
        rationale="A liability cap at a few months' fees is well below a normal landing zone — "
        "flag for legal review as a negotiation lever.",
    ),
    EscalationRule(
        rule_id="R-LEGAL-INDEMNITY-ONE-SIDED",
        category="legal",
        pattern=r"\bindemnif.{0,60}\b(all claims|any and all|without limit)",
        floor="high",
        rationale="A one-sided/open-ended indemnity shifts unbounded risk to the buyer.",
    ),
    EscalationRule(
        rule_id="R-INFOSEC-NO-BREACH-NOTICE",
        category="infosec",
        pattern=r"\b(no|without|missing)\b.{0,40}\bbreach\b.{0,20}\b(notif|notice)"
        r"|\bbreach notif.{0,20}\b(missing|absent|none)\b",
        floor="high",
        rationale="No breach-notification obligation undermines incident response and the "
        "GDPR 72-hour duty.",
    ),
)


def escalate_severity(
    finding: RiskFinding, rules: tuple[EscalationRule, ...] = DEFAULT_RULES
) -> RiskFinding:
    """Apply the raise-only rules to ``finding``. Returns a new finding whose severity is at least
    what the LLM proposed and at least the highest floor of every rule that matched, with the
    matching rule ids appended to ``raised_by``. Never lowers severity.
    """
    haystack = f"{finding.title}\n{finding.quote}"
    severity = finding.severity
    raised: list[str] = list(finding.raised_by)
    for rule in rules:
        if rule.category != finding.category:
            continue
        if not re.search(rule.pattern, haystack, re.IGNORECASE | re.DOTALL):
            continue
        new_severity = _max_severity(severity, rule.floor)
        if _RANK[new_severity] > _RANK[severity]:
            # only credit a rule that actually raised the severity — a rule whose floor was
            # already met didn't change the outcome, so it doesn't belong in the audit trail
            raised.append(rule.rule_id)
            severity = new_severity
    if severity == finding.severity and not (set(raised) - set(finding.raised_by)):
        return finding  # no change — return the same object
    return finding.model_copy(update={"severity": severity, "raised_by": tuple(raised)})


def escalate_all(
    findings: list[RiskFinding], rules: tuple[EscalationRule, ...] = DEFAULT_RULES
) -> list[RiskFinding]:
    """Apply :func:`escalate_severity` to every finding."""
    return [escalate_severity(f, rules) for f in findings]


class GatePolicy(BaseModel):
    """The *named, human-owned* thresholds that decide when legal must review. Every field is a
    dial a compliance owner sets — the gate's logic is fixed, the numbers are theirs. ``owner`` is
    surfaced next to the verdict so the mandate names who owns the rule that pulled legal in.
    """

    model_config = {"frozen": True}

    owner: str = "Procurement"  # the named human who owns these thresholds
    escalate_categories: tuple[Category, ...] = ("legal", "gdpr")  # any critical here → review
    critical_triggers_review: bool = True  # a critical in an escalate_category forces review
    high_count_threshold: int = Field(default=3, ge=1)  # >= this many highs → review
    # Only findings the anchor layer verified count toward the gate — an unverified finding
    # (quarantined quote) must not be able to trigger a legal review on its own.
    require_verified: bool = True


class GateVerdict(BaseModel):
    """The computed answer to "should legal see this", with the RULE that fired — not just a bool.
    ``rule`` is the human-readable reason ("2 Critical findings in Legal/GDPR"), so the decision is
    defensible when someone asks why legal got pulled in.
    """

    model_config = {"frozen": True}

    review_required: bool
    rule: str  # the specific threshold that fired (or why none did)
    owner: str
    critical_count: int
    high_count: int
    counted: int  # findings that counted toward the gate (verified, if require_verified)
    ignored_unverified: int  # findings excluded because they weren't anchored


def legal_gate(findings: list[RiskFinding], policy: GatePolicy | None = None) -> GateVerdict:
    """Compute whether legal review is required, from the (already escalated) findings and a named
    policy. Returns a :class:`GateVerdict` carrying the rule that fired. Pure and deterministic.

    Call :func:`escalate_all` first so the gate sees rule-raised severities, not raw LLM ones.
    """
    if policy is None:
        policy = GatePolicy()  # the default thresholds; GatePolicy is frozen so this is cheap
    if policy.require_verified:
        counted_findings = [f for f in findings if f.verified]
    else:
        counted_findings = list(findings)
    ignored = len(findings) - len(counted_findings)

    criticals = [f for f in counted_findings if f.severity == "critical"]
    highs = [f for f in counted_findings if f.severity == "high"]

    # Rule 1: any Critical in an escalate_category (Legal/GDPR by default) → review.
    if policy.critical_triggers_review:
        gating_criticals = [f for f in criticals if f.category in policy.escalate_categories]
        if gating_criticals:
            cats = ", ".join(sorted({f.category for f in gating_criticals}))
            n = len(gating_criticals)
            return GateVerdict(
                review_required=True,
                rule=f"{n} Critical finding{'s' if n != 1 else ''} in {cats} "
                f"(policy: any Critical in {'/'.join(policy.escalate_categories)})",
                owner=policy.owner,
                critical_count=len(criticals),
                high_count=len(highs),
                counted=len(counted_findings),
                ignored_unverified=ignored,
            )

    # Rule 2: >= high_count_threshold High findings (in any category) → review.
    if len(highs) >= policy.high_count_threshold:
        return GateVerdict(
            review_required=True,
            rule=f"{len(highs)} High findings (policy: >= {policy.high_count_threshold} High)",
            owner=policy.owner,
            critical_count=len(criticals),
            high_count=len(highs),
            counted=len(counted_findings),
            ignored_unverified=ignored,
        )

    return GateVerdict(
        review_required=False,
        rule="no threshold met — below the legal-review bar",
        owner=policy.owner,
        critical_count=len(criticals),
        high_count=len(highs),
        counted=len(counted_findings),
        ignored_unverified=ignored,
    )
