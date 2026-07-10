---
title: BATNA and the reservation value
tag: negotiation-strategy
source: Fisher & Ury, principled negotiation (public framework), restated
---

# BATNA and the reservation value

Your **BATNA** — Best Alternative To a Negotiated Agreement — is what you will do if
this deal falls through. It is not a number you wish for; it is the concrete course of
action you would actually take: renew with the incumbent, switch to the runner-up quote,
run a new RFQ, insource, or do without. The BATNA is the only honest measure of your
power in a negotiation. You are never forced to accept terms worse than your BATNA,
because walking to the alternative is, by definition, better.

## Reservation value = the price of your BATNA

The **reservation value** (or walk-away point) is your BATNA expressed as a deal term:
the worst set of terms at which agreement is still better than walking. Below it, no
deal beats the alternative. Above it, a deal is worth doing. The reservation value is
therefore *derived* — improve your BATNA (line up a credible second supplier, start
earlier so you are not cornered by a deadline) and your reservation value moves in your
favour; let your BATNA decay (single-source, run out of time) and it moves against you.

Two disciplines follow:

1. **Know your BATNA before you open.** A number you set by feel drifts under pressure.
   A reservation value anchored to a real alternative holds.
2. **Never reveal it.** The reservation point is the one figure that, disclosed, hands
   the counterparty your floor. Argue from interests and criteria, not from "this is the
   lowest I can go."

## How this maps to the engine

In this system the reservation value is `reservation_utility` — a **hard floor**. The
Boulware concession curve decays from the target toward it but the engine **never
concedes past it**; at the deadline it accepts only an offer that clears the floor, else
it escalates to a human. That is BATNA discipline as code: the agent walks (escalates)
rather than take a deal worse than the alternative.

Note the direction of urgency, which trips people up: **urgency lowers the floor.** If a
deadline makes no-deal worse (a service lapses, a project stalls), your BATNA is weaker,
so you rationally accept a worse deal — the reservation value *drops*. A rule that raised
the floor under urgency would make the agent walk away *more* when it can least afford to.
See [[interests-not-positions]] and [[options-for-mutual-gain]].
