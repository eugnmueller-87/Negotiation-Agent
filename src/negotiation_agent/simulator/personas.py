"""Supplier personas — parametric behavior profiles for the v0 bot.

Each persona is a fully deterministic parameter set. No RNG: the "evasive" stall
is driven by round parity, not randomness, so every negotiation replays exactly.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PersonaConfig(BaseModel):
    model_config = {"frozen": True}

    name: str
    beta_s: float = Field(description="supplier Boulware exponent; <1 = conceder, >1 = holdout")
    open_utility: float = Field(ge=0.0, le=1.0, description="own-utility of the first counter")
    accept_margin: float = Field(ge=0.0, description="slack below own threshold it will accept")
    stall_period: int = Field(
        ge=1, description="1 = never stalls; k = stalls unless round % k == 0"
    )
    walkaway_at_deadline: bool


# Concedes almost nothing until the very end, accepts only at/above its schedule,
# walks if the deadline passes without a deal.
AGGRESSIVE = PersonaConfig(
    name="aggressive",
    beta_s=10.0,
    open_utility=0.98,
    accept_margin=0.0,
    stall_period=1,
    walkaway_at_deadline=True,
)

# Concedes early (beta<1), accepts a little below its own schedule, never walks.
COOPERATIVE = PersonaConfig(
    name="cooperative",
    beta_s=0.6,
    open_utility=0.90,
    accept_margin=0.05,
    stall_period=1,
    walkaway_at_deadline=False,
)

# Holds firm and repeats its offer on non-multiple rounds (the stall), letting
# the engine's stall guard eventually fire.
EVASIVE = PersonaConfig(
    name="evasive",
    beta_s=6.0,
    open_utility=0.95,
    accept_margin=0.0,
    stall_period=3,
    walkaway_at_deadline=False,
)

ALL_PERSONAS: tuple[PersonaConfig, ...] = (AGGRESSIVE, COOPERATIVE, EVASIVE)
