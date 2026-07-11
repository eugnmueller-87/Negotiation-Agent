"""Register detection — read the counterpart's messages, decide formal vs informal.

A buyer email should mirror the counterpart's register: if the supplier writes "Hi Eugen,
thanks!" the agent greets informally; if "Dear Sir/Madam" it stays formal. This reads the
supplier's prior messages and returns a register, deterministically — no model, no guess.

Default is ``formal``: absent any signal (e.g. the opening turn), business correspondence
starts formal, and only relaxes once the counterpart signals informality. That's the safe
direction — over-formal reads as polite; over-familiar reads as presumptuous.
"""

from __future__ import annotations

import re
from typing import Literal

Register = Literal["formal", "informal"]

# Informal openers and markers — a message starting this way, or using these, signals the
# counterpart is comfortable being casual.
_INFORMAL_OPENERS = re.compile(
    r"^\s*(hi|hey|hello|hallo|servus|moin|hiya|yo)\b", re.IGNORECASE | re.MULTILINE
)
_INFORMAL_MARKERS = re.compile(
    r"\b(thanks!|cheers|no worries|sounds good|let'?s|gonna|wanna|btw|fyi|:\)|:-\)|great stuff)\b",
    re.IGNORECASE,
)
# Formal openers / markers — explicit formality that should keep the agent formal.
_FORMAL_OPENERS = re.compile(
    r"^\s*(dear\b|to whom it may concern|sehr geehrte)", re.IGNORECASE | re.MULTILINE
)
_FORMAL_MARKERS = re.compile(
    r"\b(kind regards|yours (sincerely|faithfully)|please find|we hereby|with reference to)\b",
    re.IGNORECASE,
)


def detect_register(messages: list[str]) -> Register:
    """Decide formal/informal from the counterpart's messages. Formal unless the
    counterpart clearly signals informality more than they signal formality."""
    if not messages:
        return "formal"
    joined = "\n".join(messages)
    informal = len(_INFORMAL_OPENERS.findall(joined)) * 2 + len(_INFORMAL_MARKERS.findall(joined))
    formal = len(_FORMAL_OPENERS.findall(joined)) * 2 + len(_FORMAL_MARKERS.findall(joined))
    # Require informal to clearly win — ties and no-signal stay formal (the safe default).
    return "informal" if informal > formal and informal > 0 else "formal"


def greeting_for(register: Register, *, contact: str, supplier: str) -> str:
    """The salutation line for a register + addressee. Mirrors the LLM's own rule so the
    fallback letter matches; a first name pairs with an informal 'Hello'."""
    contact = contact.strip()
    supplier = supplier.strip()
    if register == "informal":
        if contact:
            return f"Hello {contact},"
        if supplier:
            return f"Hi {supplier} team,"
        return "Hello,"
    if contact:
        return f"Dear {contact},"
    if supplier:
        return f"Dear {supplier} team,"
    return "Dear Sir or Madam,"
