"""A contract store — the corpus the agent 'pulls a contract from, reads, and strategises'.

For now this is a SIMULATED store: a handful of representative sample contracts, one per
procurement category, so the whole flow (pull → detect category → scope strategy) works
end-to-end without a real repository. It is deliberately behind a small ``ContractStore``
Protocol so a real store (a database, a document management system, the AI-Brain RAG) drops
in later by implementing the same two methods — no caller changes.

The sample texts are synthetic — no real supplier, price, or personal data — so this store
ships in the repo safely (unlike the vault-derived knowledge index).
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel


class StoredContract(BaseModel):
    """One contract in the store — enough text to detect a category and prime a negotiation."""

    model_config = {"frozen": True}

    contract_id: str
    title: str
    category: str  # the intended procurement category (also independently detectable)
    text: str


class ContractStore(Protocol):
    """The seam a real contract repository implements. Two reads, no writes from the agent."""

    def get(self, contract_id: str) -> StoredContract | None: ...

    def list_contracts(self) -> list[StoredContract]: ...


# ── the simulated store — synthetic sample contracts, one per category ──────────────
_SAMPLES: tuple[StoredContract, ...] = (
    StoredContract(
        contract_id="cloud-001",
        title="Cloud compute & hosting agreement",
        category="cloud_infrastructure",
        text=(
            "CLOUD SERVICES AGREEMENT. The provider supplies reserved vCPU compute, object "
            "storage, and multi-region hosting on its IaaS platform. Charges are per vCPU-hour "
            "with committed-use discounts; egress bandwidth billed per GB. Kubernetes managed "
            "control plane included. Term 24 months, auto-renews, 90 days' notice."
        ),
    ),
    StoredContract(
        contract_id="saas-001",
        title="Software licence & subscription agreement",
        category="software_licenses",
        text=(
            "SOFTWARE SUBSCRIPTION AGREEMENT. Grant of 500 named-user licences to the SaaS "
            "platform, billed annually per seat. Subscription auto-renews; a true-up reconciles "
            "added seats at renewal. Maintenance and support included. Entitlement is per named "
            "user, non-transferable. 36-month initial term."
        ),
    ),
    StoredContract(
        contract_id="hragency-001",
        title="Temporary staffing agency agreement",
        category="hr_staffing_agency",
        text=(
            "STAFFING SERVICES AGREEMENT. The agency supplies temporary and contingent workers "
            "(Zeitarbeit) for defined assignments. Charges are a markup on the worker pay rate; "
            "a placement fee applies on permanent conversion. Agency-worker equal-treatment "
            "rules apply after 9 months. Framework term 12 months."
        ),
    ),
    StoredContract(
        contract_id="legal-001",
        title="Outside counsel engagement letter",
        category="legal_services",
        text=(
            "ENGAGEMENT LETTER. The law firm provides outside counsel for litigation and "
            "commercial matters. Fees are on a billable-hour basis at partner and associate "
            "rates, subject to a monthly retainer. Rate card reviewed annually. Attorney-client "
            "privilege applies."
        ),
    ),
    StoredContract(
        contract_id="facility-001",
        title="Facility management services agreement",
        category="facility_services",
        text=(
            "FACILITY MANAGEMENT AGREEMENT. The supplier provides janitorial cleaning, HVAC "
            "maintenance, waste handling, and on-site security guard services across the campus. "
            "Priced per square metre per month with a service-level agreement and penalty regime. "
            "36-month term."
        ),
    ),
    StoredContract(
        contract_id="marketing-001",
        title="Agency-of-record marketing services",
        category="marketing",
        text=(
            "MARKETING SERVICES AGREEMENT. The agency acts as agency of record for advertising "
            "campaigns, media buying, and creative production. Media is billed at net cost plus a "
            "commission; a monthly retainer covers creative. Performance measured on impressions "
            "and CPM. 12-month term."
        ),
    ),
)


class SampleContractStore:
    """The default in-repo simulated store. Swap for a real ``ContractStore`` later."""

    def __init__(self, contracts: tuple[StoredContract, ...] = _SAMPLES) -> None:
        self._by_id = {c.contract_id: c for c in contracts}

    def get(self, contract_id: str) -> StoredContract | None:
        return self._by_id.get(contract_id)

    def list_contracts(self) -> list[StoredContract]:
        return list(self._by_id.values())
