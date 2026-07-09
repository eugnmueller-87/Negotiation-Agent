# Negotiation Agent — v0 core

A deterministic procurement negotiation engine with an LLM advising at the edges
and **zero authority** at the core. v0 is pure Python — the envelope schema, the
deal engine, a headless agent-vs-agent simulator, and a pytest eval suite. No UI,
no LLM calls, no network. Provable in tests.

> **The deal engine is the IP, and it's deterministic code.** The LLM (added in
> v1) only extracts offers, classifies intent, and composes prose from
> engine-approved numbers. It can never invent a concession.

## Install & run

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"     # Windows
# source .venv/bin/activate && pip install -e ".[dev]"   # POSIX

pytest -q                     # 37 tests, all green
neg-sim                       # run the reference simulation matrix
neg-sim --json                # machine-readable metrics
neg-sim --beta 6 --max-rounds 12
```

## How it works

Every negotiable term (price, payment days, contract length, volume, rebate) has
a normalized value function `vᵢ(xᵢ) ∈ [0,1]` and a weight `wᵢ`. Any offer scores
as buyer utility

```
U = Σ wᵢ · vᵢ(xᵢ)        with Σ wᵢ = 1, so U ∈ [0,1]
```

**Acceptance** rides a Boulware concession curve — the per-round threshold decays
from `target_utility` toward `reservation_utility`, conceding late and steeply:

```
threshold(t) = reservation + (target − reservation) · (1 − (t/T)^β),   β > 1
```

The engine accepts iff an incoming offer's `U ≥ threshold(t)`. At the deadline
`t = T` the threshold equals reservation and the engine escalates rather than
counter.

**Counteroffers use logrolling, not price-splitting.** At the round's threshold
the engine searches for the package the supplier is likeliest to accept: it
concedes on terms the buyer weights lightly but the supplier values highly, and
holds terms the buyer weights heavily. Formally it solves a fractional-knapsack
LP whose exact optimum is a greedy fill ordered by the cost ratio `pᵢ/wᵢ`
(supplier appetite ÷ buyer weight). Integer terms (payment days, volume) snap
toward the buyer's ideal so rounding can never push a package below its
threshold; the surplus is handed back on continuous terms so the package lands on
the threshold exactly.

Supplier preferences enter the engine as **pure data** (`SupplierModel`) — never
free text. In v0 the simulator supplies the belief; in v1 the LLM classifier maps
supplier intent labels (cash flow, volume certainty, term length, margin) to
terms via a static table and produces the same numbers.

## Architecture

```
src/negotiation_agent/
  value.py            normalized value functions vᵢ and their inverse
  envelope.py         versioned, signed mandate: TermSpec, Offer, Envelope
  supplier_model.py   SupplierModel — belief about supplier appetite (pure data)
  packages.py         fill_package() — the logrolling LP + 3-phase integer snap
  engine.py           DealEngine — Boulware curve, accept rule, decide() FSM
  simulator/
    personas.py       aggressive / cooperative / evasive parameter sets
    supplier.py       SupplierAgent protocol + deterministic ParametricSupplier
    loop.py           run_negotiation() + Turn/Transcript/NegotiationResult
    scenarios.py      Scenario, zopa_check(), reference 3×3 matrix
    metrics.py        capture ratio, joint utility, batch aggregation
    cli.py            `neg-sim` report
```

Everything is a **pure function of its arguments** — no RNG, no wall clock, no
globals. State is threaded explicitly, so a negotiation replays bit-identically
from its transcript (the audit story).

## The eval suite

`neg-sim` runs a 3 personas × 3 belief-conditions matrix. Belief conditions:
**oracle** (told the truth), **uniform** (no info), **inverted** (worst-case
misclassification) — the sweep that proves the logrolling pitch. Reference run:

| metric | overall | oracle | uniform | inverted |
|---|---|---|---|---|
| closure rate | 88.9% | 100% | 100% | 66.7% |
| capture ratio μ | 0.49 | 0.46 | 0.46 | 0.57 |
| **joint utility μ** | 1.31 | **1.33** | 1.28 | 1.29 |
| escalation rate | 11.1% | 0% | 0% | 33.3% |

The signal: **oracle belief captures the most joint utility** (better win-win
trades), and the single escalation is an evasive supplier under a wrong belief —
exactly where an agent *should* hand off to a human buyer.

`capture_ratio = (U − reservation) / (target − reservation)` — where in the
mandate span the deal landed. `joint_utility` (buyer + supplier, from the hidden
supplier envelope) is the Pareto proxy that separates logrolling from haggling.

## Build order

- **v0 (this repo)** — envelope schema, engine, simulator, pytest evals. Pure
  Python, provable in days. ✅
- **v1** — FastAPI + Postgres backend, React magic-link chat portal, buyer
  dashboard. LLM extractor (Pydantic + per-field confidence), intent classifier,
  reply composer behind the numeric guard, injection red-team suite.
- **v2** — DocuSign close, n8n email channel, German/English switching, Art. 50
  disclosure banner.

## v1 seams already in place

- `SupplierAgent` protocol — a Claude-backed supplier drops in behind the same
  `respond()`; the deterministic bot is just the v0 implementation.
- `SupplierModel.from_intents()` — the LLM classifier's plug point.
- `approved_numbers` on every `EngineDecision` — the numeric guard's allowlist;
  no engine internals (thresholds, utilities) ever reach supplier-facing prose.
- `injection_pass_rate` placeholder in `BatchMetrics` — populated once an LLM
  sits on the wire.
