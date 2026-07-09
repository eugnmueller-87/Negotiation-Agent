"""Golden-fixture regression: the reference numbers must stay reproducible.

The README quotes closure rate, capture ratio, and joint utility from the
reference simulation. This test pins those numbers to a committed artifact
(``tests/fixtures/reference_metrics.json``, produced by ``neg-sim --json``) so a
change that silently shifts the eval fails CI instead of quietly making the
README wrong.

To regenerate after an intentional change:
``neg-sim --json > tests/fixtures/reference_metrics.json``
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from negotiation_agent.engine import EngineConfig
from negotiation_agent.simulator.metrics import run_batch
from negotiation_agent.simulator.scenarios import reference_matrix

FIXTURE = Path(__file__).parent / "fixtures" / "reference_metrics.json"
TOL = 1e-9


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_batch_totals_match_fixture(golden):
    _, batch = run_batch(reference_matrix(), EngineConfig())
    exp = golden["batch"]
    assert batch.n == exp["n"]
    assert batch.closure_rate == pytest.approx(exp["closure_rate"], abs=TOL)
    assert batch.escalation_rate == pytest.approx(exp["escalation_rate"], abs=TOL)
    assert batch.joint_utility_mean == pytest.approx(exp["joint_utility_mean"], abs=TOL)
    assert batch.capture_ratio_mean == pytest.approx(exp["capture_ratio_mean"], abs=TOL)


def test_per_negotiation_matches_fixture(golden):
    metrics, _ = run_batch(reference_matrix(), EngineConfig())
    exp = {m["scenario"]: m for m in golden["per_negotiation"]}
    assert {m.scenario for m in metrics} == set(exp)
    for m in metrics:
        e = exp[m.scenario]
        assert m.status == e["status"], m.scenario
        assert m.rounds_used == e["rounds_used"], m.scenario
        if m.joint_utility is not None:
            assert m.joint_utility == pytest.approx(e["joint_utility"], abs=TOL), m.scenario


def test_readme_headline_numbers_hold(golden):
    """Guard the specific figures the README front page cites."""
    batch = golden["batch"]
    by_belief = batch["by_belief"]
    # Closure and the oracle > uniform joint-utility ordering are load-bearing.
    assert batch["closure_rate"] == pytest.approx(8 / 9, abs=1e-6)
    assert by_belief["oracle"]["joint_utility_mean"] > by_belief["uniform"]["joint_utility_mean"]
