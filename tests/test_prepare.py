"""Pre-flight orchestration — contract intake composed with supplier research.

No network: the researcher is a fake with an ``investigate`` method, which is the
seam :func:`prepare_negotiation` calls. Research failure must never be fatal.
"""

from __future__ import annotations

from negotiation_agent.prepare import prepare_negotiation
from negotiation_agent.research import ResearchUnavailable, sample_brief

CONTRACT = """
SUPPLY AGREEMENT
Supplier: Nordwerk Verpackung GmbH
1. Price. The unit price shall be €11.50 per unit.
2. Payment. Invoices are payable net-30 days from receipt.
3. Term. This agreement runs for 24 months.
"""


class _FakeResearcher:
    def __init__(self, brief=None, raises=None):
        self._brief = brief
        self._raises = raises
        self.called_with = None

    def investigate(self, company, category="", country="DE"):
        self.called_with = company
        if self._raises is not None:
            raise self._raises
        return self._brief


def test_extracts_terms_and_attaches_brief():
    r = _FakeResearcher(brief=sample_brief("Nordwerk Verpackung GmbH"))
    result = prepare_negotiation(CONTRACT, researcher=r)
    terms = {t.name: t.value for t in result.extraction.terms}
    assert terms["price"] == 11.5
    assert terms["payment_days"] == 30
    assert result.supplier_name == "Nordwerk Verpackung GmbH"
    assert result.brief is not None and result.brief.risk_level == "Medium"
    assert r.called_with == "Nordwerk Verpackung GmbH"  # researched the extracted name


def test_research_failure_is_not_fatal():
    r = _FakeResearcher(raises=ResearchUnavailable("Supplier research service is unreachable."))
    result = prepare_negotiation(CONTRACT, researcher=r)
    # terms still come back; the brief is absent with a buyer-safe note
    assert result.extraction.terms  # extraction succeeded
    assert result.brief is None
    assert "unreachable" in result.research_note.lower()


def test_no_supplier_name_skips_research():
    r = _FakeResearcher(brief=sample_brief())
    # a contract with a price but no recognisable legal name
    result = prepare_negotiation("Unit price: €9.00 per unit.", researcher=r)
    assert result.brief is None
    assert r.called_with is None  # never called
    assert "no supplier name" in result.research_note.lower()


def test_research_flag_disables_lookup():
    r = _FakeResearcher(brief=sample_brief("Nordwerk Verpackung GmbH"))
    result = prepare_negotiation(CONTRACT, researcher=r, research=False)
    assert result.brief is None
    assert r.called_with is None
    assert "not requested" in result.research_note.lower()


def test_blocking_brief_is_flagged():
    blocked = sample_brief("Risky Corp").model_copy(update={"recommendation": "Block"})
    r = _FakeResearcher(brief=blocked)
    result = prepare_negotiation(CONTRACT, researcher=r)
    assert result.is_blocking is True


def test_to_offer_seeds_from_extraction():
    r = _FakeResearcher(brief=None)
    result = prepare_negotiation(CONTRACT, researcher=r)
    offer = result.extraction.to_offer(["price", "payment_days", "contract_months"])
    assert offer.terms == {"price": 11.5, "payment_days": 30.0, "contract_months": 24.0}


def test_no_researcher_returns_terms_only():
    result = prepare_negotiation(CONTRACT)  # no researcher at all
    assert result.extraction.terms
    assert result.brief is None
    assert result.research_note is not None
