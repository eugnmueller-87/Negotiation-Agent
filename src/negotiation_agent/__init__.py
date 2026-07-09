"""Negotiation Agent — deterministic procurement negotiation engine.

v0 exposes the pure-Python core: the deal envelope schema, the utility/value
functions, the deal engine (scoring + Boulware concession + logrolling
counteroffers), and a headless simulator for agent-vs-agent evaluation.
"""

from negotiation_agent.engine import DealEngine, EngineDecision, Outcome
from negotiation_agent.envelope import (
    Direction,
    Envelope,
    Offer,
    TermSpec,
    TermType,
)

__all__ = [
    "Direction",
    "Envelope",
    "Offer",
    "TermSpec",
    "TermType",
    "DealEngine",
    "EngineDecision",
    "Outcome",
]

__version__ = "0.1.0"
