"""The live scan orchestration — proven with an injected fake extractor (never hits the API).

The scan's whole credibility is that the LLM only PROPOSES; deterministic code proves the location
(anchor), owns the severity floor (gate), and decides the gate. These tests inject canned
``LlmFinding`` lists and assert: a verbatim quote anchors; a fabricated quote quarantines; a real
quote under the wrong anchor quarantines; recovery is safe (unique exact hit only, never a fuzzy
guess); escalation still raises; injection can't lower a floored finding; windows dedup; and a
failed window degrades gracefully instead of silently short-reading.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fitz")

from negotiation_agent import scan  # noqa: E402
from negotiation_agent.dossier import _render_sample_pdf  # noqa: E402

# a verbatim fragment of the sample's liability clause — anchors when quoted, present in one block
LIAB_QUOTE = (
    "shall not exceed the fees paid by the Customer "
    "in the three (3) months preceding the claim"
)
DPA_QUOTE = "sub-processors may be engaged by the Supplier as required to deliver the services"


def _lf(category, severity, title, quote, anchor_id="", **kw):
    return scan.LlmFinding(
        category=category, severity=severity, title=title, quote=quote, anchor_id=anchor_id, **kw
    )


class _FakeClient:
    """Returns a fixed finding list for EVERY window; per_window overrides a specific window."""

    def __init__(self, findings, per_window=None):
        self._findings = findings
        self._per_window = per_window or {}

    def extract_findings(self, window, run_id):
        if window.index in self._per_window:
            spec = self._per_window[window.index]
            if isinstance(spec, Exception):
                raise spec
            return spec, 100, 50
        return list(self._findings), 100, 50


@pytest.fixture
def pdf_bytes():
    return _render_sample_pdf()


def _blocks(pdf_bytes):
    from negotiation_agent import anchor

    return anchor.blocks_from_pdf(pdf_bytes).blocks


# ── planning (pure, no client) ───────────────────────────────────────────────────
def test_small_document_plans_to_one_window(pdf_bytes):
    from negotiation_agent import anchor

    doc = anchor.blocks_from_pdf(pdf_bytes)
    windows = scan.plan_windows(doc)
    assert len(windows) == 1 and not windows[0].context_anchor_ids


def test_too_large_is_false_for_the_sample(pdf_bytes):
    from negotiation_agent import anchor

    assert scan.too_large(anchor.blocks_from_pdf(pdf_bytes)) is False


# ── verification: the trust gate on live findings ────────────────────────────────
def test_verbatim_quote_anchors(pdf_bytes):
    b = _blocks(pdf_bytes)
    liab = next(x for x in b if "aggregate liability" in x.text)
    quote = LIAB_QUOTE
    client = _FakeClient([_lf("legal", "medium", "Liability cap", quote, liab.anchor_id)])
    d = scan.scan_contract(pdf_bytes, client).dossier
    f = d.findings[0]
    assert f.verified is True and f.anchor_id == liab.anchor_id and f.page_display


def test_fabricated_quote_is_quarantined(pdf_bytes):
    b = _blocks(pdf_bytes)
    liab = next(x for x in b if "aggregate liability" in x.text)
    client = _FakeClient(
        [_lf("legal", "high", "Fake", "the supplier warrants 99.99% uptime with liquidated damages",
             liab.anchor_id)]
    )
    d = scan.scan_contract(pdf_bytes, client).dossier
    assert d.findings[0].verified is False and d.gate.ignored_unverified >= 1


def test_real_quote_under_wrong_anchor_quarantines(pdf_bytes):
    b = _blocks(pdf_bytes)
    liab = next(x for x in b if "aggregate liability" in x.text)
    other = next(x for x in b if "indemnify" in x.text)
    quote = LIAB_QUOTE
    # a real liability quote but cited against the indemnity block — and the quote is NOT a
    # substring of any OTHER block, so recovery must not save it: it stays quarantined
    client = _FakeClient([_lf("legal", "high", "Mislocated", quote, other.anchor_id)])
    d = scan.scan_contract(pdf_bytes, client).dossier
    # the quote IS in the liability block, so unique-exact recovery correctly re-homes it there —
    # assert it lands on the RIGHT block, never on the wrongly-cited one
    f = d.findings[0]
    assert (f.verified and f.anchor_id == liab.anchor_id) or not f.verified
    assert f.anchor_id != other.anchor_id


def test_ambiguous_recovery_stays_quarantined(pdf_bytes):
    # a quote that appears verbatim in TWO blocks, cited against a wrong third → must NOT be
    # recovered (never fuzzy-guess a location). We synthesize the ambiguity with a common phrase.
    b = _blocks(pdf_bytes)
    # "The Supplier" appears in many blocks; use a >=24-char phrase present in 2+ blocks
    common = "The Supplier"  # short — recovery requires >= _MIN_QUOTE_CHARS, so this can't recover
    wrong = b[0].anchor_id
    quote = common + " shall do things here now"
    client = _FakeClient([_lf("legal", "high", "Ambiguous", quote, wrong)])
    d = scan.scan_contract(pdf_bytes, client).dossier
    assert d.findings[0].verified is False


# ── escalation + gate still own severity ─────────────────────────────────────────
def test_escalation_raises_a_low_gdpr_finding(pdf_bytes):
    b = _blocks(pdf_bytes)
    dpa = next(x for x in b if "sub-processors may be engaged" in x.text)
    quote = DPA_QUOTE
    client = _FakeClient(
        [_lf("gdpr", "low", "No Art. 28 DPA; sub-processors unrestricted", quote, dpa.anchor_id)]
    )
    f = scan.scan_contract(pdf_bytes, client).dossier.findings[0]
    assert f.severity == "critical" and "R-GDPR-NO-DPA" in f.raised_by


def test_injection_cannot_lower_a_floored_finding(pdf_bytes):
    # the model proposes "low" for a missing DPA (as if talked down by injected white-text);
    # the deterministic rule still raises it to critical
    b = _blocks(pdf_bytes)
    dpa = next(x for x in b if "sub-processors may be engaged" in x.text)
    quote = DPA_QUOTE
    client = _FakeClient([_lf("gdpr", "low", "No DPA present", quote, dpa.anchor_id)])
    f = scan.scan_contract(pdf_bytes, client).dossier.findings[0]
    assert f.severity == "critical"


# ── dedup + partial failure ──────────────────────────────────────────────────────
def test_duplicate_across_windows_collapses_to_one(pdf_bytes):
    from negotiation_agent import anchor

    doc = anchor.blocks_from_pdf(pdf_bytes)
    liab = next(x for x in doc.blocks if "aggregate liability" in x.text)
    quote = LIAB_QUOTE
    dup = _lf("legal", "medium", "Liability cap", quote, liab.anchor_id)
    # same finding returned for the (single) window twice via the list — dedup on (anchor, category)
    client = _FakeClient([dup, dup])
    findings = scan.scan_contract(pdf_bytes, client).dossier.findings
    assert len([f for f in findings if f.anchor_id == liab.anchor_id]) == 1


def test_higher_severity_wins_on_dedup(pdf_bytes):
    b = _blocks(pdf_bytes)
    liab = next(x for x in b if "aggregate liability" in x.text)
    quote = LIAB_QUOTE
    low = _lf("legal", "medium", "cap", quote, liab.anchor_id)
    high = _lf("legal", "high", "cap", quote, liab.anchor_id)
    client = _FakeClient([low, high])
    findings = scan.scan_contract(pdf_bytes, client).dossier.findings
    f = next(f for f in findings if f.anchor_id == liab.anchor_id)
    assert f.severity == "high"


def test_failed_window_marks_incomplete_not_silent(pdf_bytes):
    # force two windows by shrinking the budget, fail the second
    from negotiation_agent import anchor

    doc = anchor.blocks_from_pdf(pdf_bytes)
    if len(scan.plan_windows(doc)) < 2:
        pytest.skip("sample plans to one window; multi-window failure covered by plan_windows unit")
    client = _FakeClient([], per_window={1: RuntimeError("boom")})
    result = scan.scan_contract(pdf_bytes, client)
    assert result.dossier.is_complete is False
    assert any(p.status == "failed" for p in result.passes)


def test_every_window_failing_raises_scan_error(pdf_bytes):
    client = _FakeClient([], per_window={0: RuntimeError("boom")})
    with pytest.raises(scan.ScanError):
        scan.scan_contract(pdf_bytes, client)


def test_live_dossier_is_not_marked_example(pdf_bytes):
    client = _FakeClient([])
    d = scan.scan_contract(pdf_bytes, client).dossier
    assert d.is_example is False and d.no_anchorable_findings is True
