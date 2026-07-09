"""Headless agent-vs-agent simulator.

A parametric supplier bot with a hidden preference profile negotiates against the
deal engine. The supplier speaks through :class:`SupplierAgent`, a two-method
protocol, so a Claude-backed supplier drops in behind the same seam in v1
without any change upstream.
"""

from negotiation_agent.simulator.loop import (
    NegotiationResult,
    Transcript,
    Turn,
    run_negotiation,
)
from negotiation_agent.simulator.metrics import (
    BatchMetrics,
    NegotiationMetrics,
    aggregate,
    compute,
    run_batch,
)
from negotiation_agent.simulator.personas import (
    AGGRESSIVE,
    COOPERATIVE,
    EVASIVE,
    PersonaConfig,
)
from negotiation_agent.simulator.scenarios import Scenario, zopa_check
from negotiation_agent.simulator.supplier import (
    ParametricSupplier,
    SupplierAgent,
    SupplierMove,
)

__all__ = [
    "AGGRESSIVE",
    "COOPERATIVE",
    "EVASIVE",
    "PersonaConfig",
    "SupplierAgent",
    "SupplierMove",
    "ParametricSupplier",
    "Scenario",
    "zopa_check",
    "Turn",
    "Transcript",
    "NegotiationResult",
    "run_negotiation",
    "NegotiationMetrics",
    "BatchMetrics",
    "compute",
    "aggregate",
    "run_batch",
]
