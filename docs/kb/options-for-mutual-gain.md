---
title: Invent options for mutual gain
tag: negotiation-strategy
source: Fisher & Ury, principled negotiation (public framework), restated
---

# Invent options for mutual gain

Most negotiations are treated as fixed-pie: every euro you win, the other side loses. Most
deals are not actually fixed-pie, because the two parties **weight the terms differently.**
When you care a lot about term A and lightly about term B, and the supplier is the reverse,
there is a trade that makes *both* better off — you concede B (cheap to you, valuable to
them) to gain A (valuable to you, cheap to them). Finding those trades is the single
highest-leverage move in a negotiation, and it is invisible if you argue one term at a time.

The practical steps:

1. **Widen the terms on the table.** Price, payment days, contract length, volume, rebate,
   SLAs, references, ramp schedules, renewal rights. A one-term fight (price) has no room
   for a trade; a multi-term deal has many.
2. **Learn the other side's weights.** Which terms do they defend hardest, and which do
   they give up easily? (This is [[interests-not-positions]] applied to terms.)
3. **Trade across terms, not within one.** Offer movement on a term they value and you
   don't, explicitly *in exchange* for holding a term you value and they don't.
4. **Separate inventing from deciding.** Generate candidate packages first; judge them
   against your target and floor second — don't kill an option before it's fully formed.

## How this maps to the engine — logrolling

This is the core IP of the system. Given a buyer utility `threshold`, the package search
routes concessions to terms the buyer weights lightly and the supplier's belief values
highly — a fractional-knapsack solve that finds the cheapest-to-the-buyer package clearing
the threshold. That *is* "invent options for mutual gain," made deterministic: the engine
never just drops price; it looks for a cross-term trade that holds total buyer utility up.

The visible **give/get** on each counter ("gave: payment 30→45 · held: price €92") is this
principle surfaced — the collaborative trade made legible, so both sides can see the deal
is a package, not a single-number squeeze. Reaching a *collaborative* agreement — the buyer
walking away from a low-weight term to get a high-weight one — is mutual-gain bargaining,
not a concession. See [[batna-and-reservation-value]] for the floor these trades never cross
and [[objective-criteria]] for defending each term's value.
