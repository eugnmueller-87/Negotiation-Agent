"""The Phase-4 LLM draft-budget cap — a hard ceiling on paid buyer-drafts.

Full mode + within budget → the (fake, injected) paid drafter runs. Once the rolling budget is
exhausted, _drafter_for FAILS CLOSED to the deterministic templated drafter: the negotiation still
completes at $0, spend just stops. Gated on FastAPI; no network (the paid drafter is a fake).
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from negotiation_agent import api  # noqa: E402
from negotiation_agent.fallback import DeterministicDrafter  # noqa: E402


class _FakePaid:
    """Stands in for the paid client — distinguishable from DeterministicDrafter by type."""

    def draft_buyer(self, *a, **k):
        return "paid draft"

    def draft_supplier(self, *a, **k):
        return "..."


class _Req:
    """Minimal request stub carrying only the full-mode header."""

    def __init__(self, full: str) -> None:
        self.headers = {"X-Peitho-Full": full}


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    api._draft_hits.clear()
    monkeypatch.setattr(api, "draft_client_factory", lambda: _FakePaid())
    monkeypatch.setenv("PEITHO_FULL_TOKEN", "tok")
    yield
    api._draft_hits.clear()


def test_full_mode_within_budget_uses_the_paid_drafter(monkeypatch):
    monkeypatch.setattr(api, "_DRAFT_BUDGET", 5)
    d = api._drafter_for(_Req("tok"))
    assert isinstance(d, _FakePaid)


def test_budget_exhausted_fails_closed_to_templated(monkeypatch):
    monkeypatch.setattr(api, "_DRAFT_BUDGET", 3)
    # first 3 paid drafts are allowed
    for _ in range(3):
        assert isinstance(api._drafter_for(_Req("tok")), _FakePaid)
    # the 4th degrades to the $0 templated drafter — never errors
    assert isinstance(api._drafter_for(_Req("tok")), DeterministicDrafter)


def test_demo_mode_never_touches_the_budget(monkeypatch):
    monkeypatch.setattr(api, "_DRAFT_BUDGET", 1)
    # a non-full request always gets the templated drafter and consumes no budget
    for _ in range(5):
        assert isinstance(api._drafter_for(_Req("wrong-token")), DeterministicDrafter)
    assert api._draft_hits == []  # demo mode recorded zero paid drafts


def test_zero_budget_disables_the_cap(monkeypatch):
    monkeypatch.setattr(api, "_DRAFT_BUDGET", 0)
    for _ in range(50):
        assert isinstance(api._drafter_for(_Req("tok")), _FakePaid)
