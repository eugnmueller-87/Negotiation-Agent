"""``_detect_context`` — the seam that turns an uploaded contract into a strategy.

This is what "upload a contract to negotiate with that provider" rests on: the
contract's own text picks the procurement category (and thus the playbook), the
supplier's words set the register, and a maverick purchase with no contract still
gets a category from what the supplier said. Pure classification — no network.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from negotiation_agent.api import _detect_context  # noqa: E402
from negotiation_agent.wire import NegotiationContext  # noqa: E402

CLOUD_CONTRACT = (
    "CLOUD SERVICES AGREEMENT. Reserved vCPU compute, object storage, and multi-region "
    "IaaS hosting. Charges per vCPU-hour with committed-use discounts; egress per GB."
)


def test_uploaded_contract_text_drives_the_category():
    # the uploaded contract body classifies the deal -> the agent pulls that playbook
    ctx = NegotiationContext(contract_text=CLOUD_CONTRACT)
    category, _register = _detect_context(ctx, [])
    assert category == "cloud_infrastructure"


def test_contract_text_beats_the_free_text_hint():
    # a mislabeled hint must not override the contract's own signal
    ctx = NegotiationContext(contract_text=CLOUD_CONTRACT, category_hint="legal services")
    category, _ = _detect_context(ctx, [])
    assert category == "cloud_infrastructure"


def test_maverick_no_contract_detects_from_supplier_words():
    # no contract on file (a maverick spend) -> the supplier's own message picks the category
    ctx = NegotiationContext()
    supplier_messages = [
        "We provide outside counsel for litigation at partner and associate billable-hour rates."
    ]
    category, _ = _detect_context(ctx, supplier_messages)
    assert category == "legal_services"


def test_register_is_read_from_supplier_messages():
    ctx = NegotiationContext(contract_text=CLOUD_CONTRACT)
    _cat, formal = _detect_context(ctx, ["Dear Sir or Madam, please find our proposal enclosed."])
    assert formal == "formal"
    _cat, informal = _detect_context(ctx, ["Hi! Thanks so much, chat soon, cheers!"])
    assert informal == "informal"


def test_no_signal_at_all_is_unknown_and_formal():
    # anchor turn: no contract, no hint, no supplier message yet -> unknown category, formal
    category, register = _detect_context(NegotiationContext(), [])
    assert category == "unknown"
    assert register == "formal"
