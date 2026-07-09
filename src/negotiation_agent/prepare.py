"""Prepare a negotiation from a contract + supplier research — the v1 pre-flight.

Before the agent sends its first message, a buyer's team does two things: read the
current contract to know the supplier's standing position, and run due diligence on
who they're dealing with. This module composes those two existing pieces —
:mod:`negotiation_agent.intake` (contract → terms) and
:mod:`negotiation_agent.research` (supplier → :class:`SupplierBrief`) — into one
pre-flight result the caller uses to seed the negotiation and brief the human.

Design boundary, held here too: the research brief is **advisory context for the
human**. It is returned alongside the extracted terms but never merged into them —
the engine still decides only on the signed mandate. Research informs; it does not
negotiate.

This is a pure, framework-agnostic function. The optional FastAPI adapter in
:mod:`negotiation_agent.api` wraps it in an HTTP endpoint; nothing here imports a
web framework, so the core package keeps its single runtime dependency (pydantic).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from negotiation_agent.intake import ContractExtraction, ContractExtractor, extract_contract
from negotiation_agent.research import ResearchUnavailable, SupplierBrief


class PreparedNegotiation(BaseModel):
    """The pre-flight result: what the contract said + who the supplier is.

    ``extraction`` is the opening state the negotiation seeds from. ``brief`` is
    advisory — present when research succeeded, ``None`` when it was unavailable
    (the negotiation proceeds regardless). ``research_note`` explains a missing
    brief in words safe to show a buyer.
    """

    model_config = {"frozen": True}

    extraction: ContractExtraction
    brief: SupplierBrief | None = None
    research_note: str | None = None
    warnings: list[str] = Field(default_factory=list)

    @property
    def supplier_name(self) -> str | None:
        return self.extraction.supplier_name

    @property
    def is_blocking(self) -> bool:
        """True if research returned a do-not-proceed recommendation.

        A blocking brief does not stop the negotiation mechanically — that is a
        human's call — but the caller should surface it before starting.
        """
        return self.brief is not None and self.brief.is_blocking


def prepare_negotiation(
    contract_text: str,
    *,
    researcher: object | None = None,
    extractor: ContractExtractor | None = None,
    research: bool = True,
) -> PreparedNegotiation:
    """Read the contract and, if a researcher is supplied, brief the supplier.

    ``researcher`` is anything with an ``investigate(company, ...) -> SupplierBrief``
    method — typically a :class:`~negotiation_agent.research.HadesClient`. It is
    passed in (not constructed here) so this stays testable and so the API layer
    owns the credential. Research failure is never fatal: the brief comes back
    ``None`` with a note, and the extracted terms are still returned.

    Research runs only when ``research`` is true, a researcher was given, and the
    contract yielded a supplier name to look up.
    """
    extraction = extract_contract(contract_text, extractor)
    warnings = list(extraction.warnings)

    brief: SupplierBrief | None = None
    note: str | None = None

    if not research or researcher is None:
        note = "Supplier research was not requested."
    elif extraction.supplier_name is None:
        note = "No supplier name was extracted from the contract, so research was skipped."
    else:
        try:
            brief = researcher.investigate(extraction.supplier_name)  # type: ignore[attr-defined]
        except ResearchUnavailable as e:
            note = str(e)

    return PreparedNegotiation(
        extraction=extraction, brief=brief, research_note=note, warnings=warnings
    )
