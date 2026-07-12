"""The canned due-diligence dossier — proving it's REAL computation, not hardcoded output.

The dossier's whole credibility is that its anchored badges, escalated severities, and gate verdict
come from running the actual anchor + gate code over a bundled sample — so these tests assert the
machinery fired: findings anchor to real blocks, a model-underrated finding gets escalated, the
fabricated finding quarantines, and the gate fires on the escalated severities. The /dossier/ask
tests prove the paid path stays behind full mode.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fitz")

from negotiation_agent import dossier  # noqa: E402


@pytest.fixture(scope="module")
def built():
    return dossier.build_dossier()


# ── the dossier is real computation ──────────────────────────────────────────────
def test_findings_anchor_to_real_blocks(built):
    # every verified finding's anchor_id must exist among the document's blocks
    block_ids = {b.anchor_id for b in built.blocks}
    anchored = [f for f in built.findings if f.verified]
    assert anchored and all(f.anchor_id in block_ids for f in anchored)


def test_verified_findings_carry_a_page_number(built):
    anchored = [f for f in built.findings if f.verified]
    assert all(f.page_display and f.page_display >= 1 for f in anchored)


def test_the_fabricated_finding_is_quarantined(built):
    # one canned finding quotes text NOT in the contract — it must not anchor
    quarantined = [f for f in built.findings if not f.verified]
    assert len(quarantined) == 1 and quarantined[0].anchor_id is None


def test_missing_dpa_is_escalated_to_critical(built):
    dpa = next(f for f in built.findings if f.category == "gdpr")
    assert dpa.severity == "critical" and dpa.llm_severity != "critical"
    assert "R-GDPR-NO-DPA" in dpa.raised_by


def test_liability_cap_is_escalated_above_the_model_severity(built):
    cap = next(f for f in built.findings if "Liability cap" in f.title)
    assert cap.severity == "high" and cap.llm_severity == "medium"


def test_gate_requires_legal_review_and_names_its_rule(built):
    assert built.gate.review_required is True
    assert "Critical" in built.gate.rule


def test_unverified_finding_is_excluded_from_the_gate(built):
    # the fabricated finding must not count toward the review decision
    assert built.gate.ignored_unverified == 1


def test_document_paginates(built):
    # the sample spans multiple pages so the viewer shows real page navigation
    assert built.page_count >= 2


# ── economic breakdown ───────────────────────────────────────────────────────────
def test_economics_tco_exceeds_year_one(built):
    # TCO over the term (with compounding indexation) must be more than a single year
    assert built.economics.tco_over_term_eur > built.economics.year1_total_eur


def test_economics_lists_the_indexation_risk(built):
    labels = " ".join(r.label.lower() for r in built.economics.risks)
    assert "indexation" in labels


# ── the /dossier and /dossier/ask endpoints ──────────────────────────────────────
@pytest.fixture
def client(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from negotiation_agent import api

    monkeypatch.setenv("PEITHO_MANDATE_SECRET", "t")
    api._rate_hits.clear()
    yield TestClient(api.app)
    api._rate_hits.clear()


def test_dossier_endpoint_serves_the_example_anonymously(client):
    r = client.get("/dossier")
    assert r.status_code == 200 and r.json()["is_example"] is True


def test_ask_is_blocked_in_demo_mode(client):
    # no full token -> Ask-Opus (a paid path) must be refused, never call the LLM
    r = client.post("/dossier/ask", json={"question": "How bad is the liability cap?"})
    assert r.status_code == 403 and r.json()["error"]["code"] == "full_mode_only"


def test_full_token_reaches_the_ask_path(client, monkeypatch):
    # with the token the endpoint enters full mode and calls ask_opus (here, a stub)
    from negotiation_agent import ask

    monkeypatch.setenv("PEITHO_FULL_TOKEN", "tok")
    monkeypatch.setattr(
        ask, "ask_opus", lambda q, b, e=None: ask.AskAnswer(answer="Capped at 3mo [p2-b0].",
                                                            cited_anchors=["p2-b0"])
    )
    r = client.post("/dossier/ask", json={"question": "q"}, headers={"X-Peitho-Full": "tok"})
    assert r.status_code == 200 and "cited_anchors" in r.json()
