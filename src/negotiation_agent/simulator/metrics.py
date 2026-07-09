"""Evaluation metrics — the eval suite, the dashboard content, the demo numbers.

Per negotiation we record whether it closed, how much buyer utility it captured
relative to the mandate span, how many rounds it took, and — for evaluation only,
from the hidden supplier envelope — the supplier's utility and the joint utility.
Joint utility is the win-win proxy: at equal buyer capture, logrolled deals
should show materially higher supplier utility than price-splitting would.
"""

from __future__ import annotations

from collections.abc import Callable
from statistics import mean, median

from pydantic import BaseModel, Field

from negotiation_agent.engine import DealEngine, EngineConfig
from negotiation_agent.envelope import Envelope, Offer
from negotiation_agent.simulator.loop import NegotiationResult, run_negotiation
from negotiation_agent.simulator.scenarios import Scenario
from negotiation_agent.simulator.supplier import ParametricSupplier


class NegotiationMetrics(BaseModel):
    model_config = {"frozen": True}

    negotiation_id: str
    scenario: str
    persona: str
    belief_source: str
    status: str
    closed: bool
    escalated: bool
    walked: bool
    rounds_used: int
    buyer_utility_final: float | None = None
    # (U - reservation) / (target - reservation); may exceed 1 on early supplier
    # acceptance of the anchor. Not clamped — the overshoot is real signal.
    capture_ratio: float | None = None
    supplier_utility_final: float | None = None  # eval-only, hidden envelope
    joint_utility: float | None = None


class BatchMetrics(BaseModel):
    n: int
    closure_rate: float
    escalation_rate: float
    walk_rate: float
    rounds_to_close_mean: float | None = None
    rounds_to_close_median: float | None = None
    capture_ratio_mean: float | None = None
    capture_ratio_min: float | None = None
    joint_utility_mean: float | None = None
    by_persona: dict[str, BatchMetrics] = Field(default_factory=dict)
    by_belief: dict[str, BatchMetrics] = Field(default_factory=dict)
    # v1 placeholder — populated once an LLM sits on the wire and the injection
    # red-team suite runs. None here means "not measured", not "zero".
    injection_pass_rate: float | None = None


def compute(result: NegotiationResult, scenario: Scenario) -> NegotiationMetrics:
    env = scenario.buyer_envelope
    closed = result.status in ("closed_engine", "closed_supplier")
    buyer_u = env.utility(result.final_deal) if result.final_deal is not None else None

    capture = None
    if buyer_u is not None:
        span = env.target_utility - env.reservation_utility
        capture = (buyer_u - env.reservation_utility) / span if span > 0 else None

    supplier_u = None
    joint = None
    if result.final_deal is not None:
        sup_env = scenario.supplier_envelope
        projected = _project(result.final_deal.terms, sup_env)
        supplier_u = sup_env.utility(projected)
        if buyer_u is not None:
            joint = buyer_u + supplier_u

    return NegotiationMetrics(
        negotiation_id=result.transcript.negotiation_id,
        scenario=scenario.name,
        persona=scenario.persona.name,
        belief_source=scenario.belief_source,
        status=result.status,
        closed=closed,
        escalated=result.status == "escalated",
        walked=result.status == "walked",
        rounds_used=result.rounds_used,
        buyer_utility_final=buyer_u,
        capture_ratio=capture,
        supplier_utility_final=supplier_u,
        joint_utility=joint,
    )


def aggregate(items: list[NegotiationMetrics], *, _top: bool = True) -> BatchMetrics:
    n = len(items)
    if n == 0:
        return BatchMetrics(n=0, closure_rate=0.0, escalation_rate=0.0, walk_rate=0.0)

    closed = [m for m in items if m.closed]
    close_rounds = [m.rounds_used for m in closed]
    captures = [m.capture_ratio for m in closed if m.capture_ratio is not None]
    joints = [m.joint_utility for m in closed if m.joint_utility is not None]

    batch = BatchMetrics(
        n=n,
        closure_rate=len(closed) / n,
        escalation_rate=sum(m.escalated for m in items) / n,
        walk_rate=sum(m.walked for m in items) / n,
        rounds_to_close_mean=mean(close_rounds) if close_rounds else None,
        rounds_to_close_median=median(close_rounds) if close_rounds else None,
        capture_ratio_mean=mean(captures) if captures else None,
        capture_ratio_min=min(captures) if captures else None,
        joint_utility_mean=mean(joints) if joints else None,
    )
    if _top:
        by_persona = _group(items, key=lambda m: m.persona)
        by_belief = _group(items, key=lambda m: m.belief_source)
        batch = batch.model_copy(update={"by_persona": by_persona, "by_belief": by_belief})
    return batch


def run_batch(
    scenarios: list[Scenario], config: EngineConfig | None = None
) -> tuple[list[NegotiationMetrics], BatchMetrics]:
    """Run every scenario and aggregate. Deterministic end to end."""
    cfg = config or EngineConfig()
    metrics: list[NegotiationMetrics] = []
    for sc in scenarios:
        engine = DealEngine(sc.buyer_envelope, sc.belief, cfg)
        supplier = ParametricSupplier(sc.supplier_envelope, sc.persona)
        result = run_negotiation(
            sc.buyer_envelope,
            engine,
            supplier,
            supplier_envelope=sc.supplier_envelope,
            persona_name=sc.persona.name,
            belief_source=sc.belief_source,
            config=cfg,
        )
        metrics.append(compute(result, sc))
    return metrics, aggregate(metrics)


def _group(
    items: list[NegotiationMetrics], key: Callable[[NegotiationMetrics], str]
) -> dict[str, BatchMetrics]:
    buckets: dict[str, list[NegotiationMetrics]] = {}
    for m in items:
        buckets.setdefault(key(m), []).append(m)
    return {k: aggregate(v, _top=False) for k, v in sorted(buckets.items())}


def _project(terms: dict[str, float], envelope: Envelope) -> Offer:
    """Restrict a deal to an envelope's terms, filling gaps from its floor."""
    return Offer(terms={n: terms.get(n, envelope.term_map[n].worst) for n in envelope.term_map})
