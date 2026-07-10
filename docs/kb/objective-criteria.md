---
title: Insist on objective criteria
tag: negotiation-strategy
source: Fisher & Ury, principled negotiation (public framework), restated
---

# Insist on objective criteria

When two sides argue from will alone — "this is our best price" versus "that's too high" —
the deal goes to whoever is more stubborn, and both sides resent the outcome. **Objective
criteria** replace the contest of wills with a standard neither party controls: a market
benchmark, a published rate card, an independent index, a like-for-like competing quote,
a cost-plus model, precedent from a comparable deal.

The discipline is to agree on the *standard* before arguing the *number*. "What benchmark
should a fair price track?" is a question both sides can reason about; "give me 10% off"
is a demand only one side can win. Once a standard is agreed, the number often follows —
and the party who has to move can do so without losing face, because they are yielding to
a criterion, not to pressure.

## Sources of objective criteria in procurement

- **Market price benchmarks** — third-party rate data, analyst pricing, public tiers.
- **Comparable quotes** — a like-for-like offer from a credible alternative (this also
  *is* your BATNA; see [[batna-and-reservation-value]]).
- **The supplier's own precedent** — prior-year rate, published list less standard
  discount, the rate given to a comparable account.
- **Cost structure** — a should-cost or cost-plus model for a made-to-spec item.
- **Indexation** — tie escalation to a named index (CPI, a commodity index) rather than
  to the supplier's discretion.

## How this maps to the engine

Objective criteria are what make a `TermSpec`'s `best`/`worst` bounds *defensible* rather
than arbitrary: the target is where a fair benchmark says a good deal sits, the floor is
where the BATNA sits. A counteroffer argued from a benchmark ("comparable renewals cleared
at X") is one the guard can release, because the figure is engine-approved and the
*rationale* is a public standard — not a disclosed internal threshold. Pair with
[[interests-not-positions]] to know which criterion the other side will actually accept.
