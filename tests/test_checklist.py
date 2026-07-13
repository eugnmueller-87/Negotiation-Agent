"""The senior-counsel checklist — coverage + the no-fabricated-benchmark honesty rail.

The checklist drives what the scan hunts for and how findings read. Two things must hold: it
covers every category (so no category is a blind spot), and its negotiation positions are
HEURISTICS, never unsourced market-standard claims (the no-fabrication rule).
"""

from __future__ import annotations

import pytest

from negotiation_agent import checklist


def test_every_category_has_checklist_items():
    for category in ("legal", "gdpr", "infosec", "coc", "commercial"):
        assert checklist.CHECKLIST.get(category), f"{category} has no checklist items"


def test_prompt_block_names_every_category():
    block = checklist.checklist_prompt_block()
    for label in ("LEGAL", "GDPR", "INFORMATION SECURITY", "CODE OF CONDUCT", "COMMERCIAL"):
        assert label in block


def test_prompt_block_instructs_flagging_absence():
    # the senior-counsel behaviour: file a finding when a protection is MISSING, not only present
    block = checklist.checklist_prompt_block().lower()
    assert "absent" in block and "missing" in block


# The honesty rail: positions are negotiation guidance, never "the market standard is N".
_BENCHMARK_CLAIMS = ("market standard", "industry standard", "the standard is", "standard is ")


@pytest.mark.parametrize(
    "item",
    [item for items in checklist.CHECKLIST.values() for item in items],
    ids=lambda i: i.key,
)
def test_positions_make_no_unsourced_market_standard_claim(item):
    text = item.position.lower()
    assert not any(claim in text for claim in _BENCHMARK_CLAIMS), (
        f"{item.key} position asserts a market standard — must be a heuristic, not a benchmark"
    )
