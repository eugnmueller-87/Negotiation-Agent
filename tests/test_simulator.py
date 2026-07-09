"""Simulator + metrics — agent-vs-agent, ZOPA, and the eval batch."""

from __future__ import annotations

from negotiation_agent.engine import DealEngine, EngineConfig
from negotiation_agent.simulator.loop import run_negotiation
from negotiation_agent.simulator.metrics import aggregate, compute, run_batch
from negotiation_agent.simulator.scenarios import (
    reference_matrix,
    zopa_check,
)
from negotiation_agent.simulator.supplier import ParametricSupplier


def test_reference_matrix_has_zopa():
    scenarios = reference_matrix()
    z = zopa_check(scenarios[0].buyer_envelope, scenarios[0].supplier_envelope)
    # A real zone of agreement must exist or closure rate measures the fixture.
    assert z >= scenarios[0].supplier_envelope.reservation_utility


def test_negotiation_terminates_and_is_bounded():
    sc = reference_matrix()[0]
    cfg = EngineConfig()
    engine = DealEngine(sc.buyer_envelope, sc.belief, cfg)
    supplier = ParametricSupplier(sc.supplier_envelope, sc.persona)
    result = run_negotiation(
        sc.buyer_envelope, engine, supplier,
        supplier_envelope=sc.supplier_envelope,
        persona_name=sc.persona.name, belief_source=sc.belief_source, config=cfg,
    )
    assert result.status in ("closed_engine", "closed_supplier", "escalated", "walked")
    assert len(result.transcript.turns) <= 2 * cfg.max_rounds + 3


def test_cooperative_persona_closes():
    # Cooperative supplier + oracle belief should reliably close.
    sc = next(s for s in reference_matrix() if s.persona.name == "cooperative"
              and s.belief_source == "oracle")
    cfg = EngineConfig()
    engine = DealEngine(sc.buyer_envelope, sc.belief, cfg)
    supplier = ParametricSupplier(sc.supplier_envelope, sc.persona)
    result = run_negotiation(
        sc.buyer_envelope, engine, supplier,
        supplier_envelope=sc.supplier_envelope,
        persona_name=sc.persona.name, belief_source=sc.belief_source, config=cfg,
    )
    assert result.status in ("closed_engine", "closed_supplier")
    m = compute(result, sc)
    # A closed deal must clear the buyer's reservation.
    assert m.buyer_utility_final >= sc.buyer_envelope.reservation_utility - 1e-9


def test_oracle_belief_beats_inverted_on_joint_utility():
    """The logrolling proof: correct belief captures more joint utility than
    worst-case misclassification, aggregated across closed cooperative deals."""
    scenarios = reference_matrix()
    cfg = EngineConfig()

    def joint_for(condition):
        subset = [s for s in scenarios if s.belief_source == condition]
        metrics, _ = run_batch(subset, cfg)
        joints = [m.joint_utility for m in metrics if m.closed and m.joint_utility is not None]
        return sum(joints) / len(joints) if joints else 0.0

    oracle = joint_for("oracle")
    inverted = joint_for("inverted")
    # Oracle should not be worse than inverted on joint welfare of closed deals.
    assert oracle >= inverted - 1e-9


def test_batch_metrics_shapes():
    scenarios = reference_matrix()
    metrics, batch = run_batch(scenarios, EngineConfig())
    assert batch.n == len(scenarios)
    assert 0.0 <= batch.closure_rate <= 1.0
    assert 0.0 <= batch.escalation_rate <= 1.0
    # Rates partition the outcomes.
    assert abs(batch.closure_rate + batch.escalation_rate + batch.walk_rate - 1.0) < 1e-9
    assert set(batch.by_persona) == {"aggressive", "cooperative", "evasive"}
    assert set(batch.by_belief) == {"oracle", "uniform", "inverted"}


def test_run_batch_is_deterministic():
    scenarios = reference_matrix()
    m1, b1 = run_batch(scenarios, EngineConfig())
    m2, b2 = run_batch(scenarios, EngineConfig())
    assert [m.model_dump() for m in m1] == [m.model_dump() for m in m2]
    assert b1.model_dump() == b2.model_dump()


def test_aggregate_empty():
    b = aggregate([])
    assert b.n == 0 and b.closure_rate == 0.0
