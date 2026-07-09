# Negotiation Agent — Project Instructions

> Deterministic procurement negotiation engine with an LLM-advised layer.
> The Operating Rules block below is my non-negotiable core — keep it verbatim.

## Operating Rules (non-negotiable)

**1. NEVER commit secrets.**
- API keys, tokens, passwords, connection strings live in environment variables or
  a git-ignored `.env` — NEVER in tracked files, NEVER hardcoded, NEVER in a commit,
  log, or code comment. Reference them as `os.environ["X"]` / `os.getenv("X")`,
  never as literals.
- New config var → add it to `.env.example` with a placeholder, document it, read it
  from the environment. The `.gitignore` and the scan-secrets hook back this up, but
  the rule is mine to hold first.
- If you ever see a real secret in a file I ask you to edit, STOP and tell me.

**2. No bullshit — verify before you claim.**
- Don't say something works until you've run it. Don't say a file/function/API exists
  until you've checked. Ran the test → report the real result; didn't → say so. No
  "this should work," no invented function signatures, no guessed library behavior.
- No filler, no flattery, no hedging ("try", "hope", "maybe", "probably"). Say what's
  true and what to do. Lead with the answer, then the reasoning.
- Cite where facts came from (file:line, command output, doc URL). If you're guessing,
  the word "guess" must appear.

**3. Report failures honestly.**
- When something breaks or you got it wrong: say so plainly and immediately. State
  what failed, the actual error, and the smallest next step.
- Never mask a failure as success. Never `except: pass`, `|| true`, or a silent
  fallback that hides breakage. A loud failure beats a quiet corruption.
- "I don't know yet" is a valid, respected answer — park it as a TODO, don't serve a
  guess dressed as fact.

**4. Work ADHD-aware.**
- Lead with the single thing that matters, then detail. Bullets over walls of text.
- When I'm stuck starting, hand me the smallest next step (one 5-minute action), not a
  10-item plan.
- Be my external working memory: restate open loops, resurface what I dropped, and
  nudge me to FINISH (I start fast, finish slow). Celebrate closing a loop.
- One thing at a time. If I'm scattering, name it and ask which one matters now. No
  shame, ever — dropped threads are normal, just facts to act on.

## Build status (verified 2026-07-09)

- **v0 is fully built and green.** `import negotiation_agent` succeeds; `pytest` →
  **37 passed**; `ruff check` clean; `neg-sim` runs the 9-scenario reference matrix
  end-to-end (text + `--json`), exit 0.
- **`ruff format --check` reports 10 files as unformatted** (most of `engine.py`,
  `packages.py`, the simulator, and several tests). Style only — not a bug. Run
  `ruff format .` when you want to normalize; it will produce a large whitespace diff.
- The LLM extraction/classification layer (v1) is **not** built yet — the engine is
  pure-Python and takes a `SupplierModel` belief as data. No live LLM calls anywhere,
  so v0 needs no secrets.

## Commands

```bash
# Env (Windows + Git Bash)
python -m venv .venv && source .venv/Scripts/activate
pip install -e ".[dev]"                 # editable install with pytest + coverage

# Test / Lint
pytest                                  # full suite (37 tests)
pytest tests/test_engine.py::test_name  # single test
ruff check .                            # lint (clean)
ruff format .                           # apply formatting (10 files currently drift)

# Run the reference simulation
neg-sim                                 # text metrics report over the scenario matrix
neg-sim --json                          # machine-readable metrics
neg-sim --max-rounds 8 --beta 4.0        # tune the Boulware schedule
```

## Architecture

- **Separation of powers is the core design.** The `Envelope` (the mandate: which terms
  are negotiable, their utility mapping, weights, and the reservation floor) is
  **deterministic and human-owned** — the LLM never edits it. Planned v1 LLM jobs are
  extraction (supplier free-text → `Offer`) and classification (supplier free-text →
  intent labels). All scoring, concession, and counteroffer math is pure Python.
- **`value.py`** — per-term value functions mapping a term value to utility in `[0,1]`
  via linear interpolation between `best` (1.0) and `worst` (0.0), clamped outside the
  span. `linear_inverse` answers "what term value yields this utility?".
- **`envelope.py`** — `TermSpec`, `Offer`, `Envelope` (the versioned mandate). Total
  buyer utility `U = Σ wᵢ·vᵢ(xᵢ)`, weights summing to 1, so `U ∈ [0,1]`.
- **`packages.py`** — the core IP: logrolling package search. Given a buyer utility
  `threshold`, route concessions to terms the buyer weights lightly and the supplier
  values highly. It's a fractional-knapsack LP (`min Σ pᵢvᵢ` s.t. `Σ wᵢvᵢ = θ`) solved
  by a greedy fill ordered by cost ratio `rᵢ = pᵢ/wᵢ`. Pure function, no RNG.
- **`supplier_model.py`** — `SupplierModel`: a *belief* about supplier concession
  appetite per term (numbers only; the engine never sees true supplier prefs). Holds
  the `INTENT_TO_TERM_TYPES` table the v1 LLM classifier will map labels through.
- **`engine.py`** — `DealEngine` / `EngineConfig` / `NegotiationState` / `EngineDecision`
  / `Outcome (ACCEPT/COUNTER/ESCALATE)`. Boulware threshold decay, accept/counter/escalate,
  stall + deadline + unknown-term guards. Emits `approved_numbers` (the ONLY numbers a
  downstream reply may contain). Pure: `decide(state, incoming) -> (decision, next_state)`.
- **`simulator/`** — headless agent-vs-agent harness: `loop` (alternating turns + audit
  `Turn` records), `supplier` bot, `personas`, `scenarios` (reference matrix + ZOPA
  check), `metrics` (batch scorecard), `cli` (`neg-sim`).

## Key Decisions

- **Envelopes are versioned + `signed_by`, never mutated.** Bumping any term, weight, or
  threshold is a new version. This is the audit trail: it proves which human authorized
  the mandate an agent negotiated under. Models are `frozen=True` to enforce this.
- **Utility is always computed against a specific envelope, never stored on the `Offer`.**
  The same offer can be re-scored under different mandates during audit replay.
- **Linear value functions are the honest v0 default**, not a limitation to hide —
  non-linear shapes (diminishing returns on payment terms) can be layered in later by
  swapping the interpolation. `value.py` says so explicitly.
- **Whole-number terms round to integers** in generated counteroffers (never propose
  "37.4 payment days"); price and rebate stay continuous. See `_INTEGER_TERMS`.
- **`reservation_utility` is a hard floor.** The Boulware threshold decays to it but never
  below; at the deadline (`round == max_rounds`) the engine accepts only if the offer
  clears the floor, otherwise it returns `Outcome.ESCALATE` (hand to a human buyer). The
  engine never concedes past reservation.
- **`approved_numbers` is the confidentiality line.** Every `EngineDecision` carries the
  only numeric values a downstream (v1) LLM-composed reply may contain. Thresholds, beta,
  utilities, and the reservation point are internal and must never reach supplier-facing
  prose.
- **Concessions never retract.** `NegotiationState.concession_caps` ratchets each term's
  offered utility down monotonically, so a mid-negotiation belief update can't walk back a
  concession already made.

## Domain Knowledge

- **Envelope** = the category manager's negotiation mandate expressed as data.
- **Target vs reservation utility** = aspiration (where the concession curve starts) vs.
  walk-away floor. Reservation must be strictly below target (validated).
- **Boulware concession** = a concession curve that holds firm early and gives ground
  late, decaying from target toward reservation over the negotiation.
- **Logrolling** = trading across terms — concede on a low-weight term the supplier values
  to gain on a high-weight term you value, keeping total utility up.
- **Direction** = which end of a term's range the buyer prefers: `MINIMIZE` (price) vs.
  `MAXIMIZE` (payment days — pay later is better).
- Procurement compliance context if it ever enters scope: LkSG/CSDDD (German/EU
  supply-chain due-diligence law), sanctions/OFAC screening — a compliance finding
  without a source + retrieval date is not a finding (see `data-privacy-procurement.md`).

## Don'ts

- **Never let the LLM edit the envelope, invent a utility number, or concede past
  `reservation_utility`.** The determinism of the mandate is the product's credibility.
- Don't add dependencies without asking — the runtime dep is `pydantic` only; dev adds
  `pytest`/`pytest-cov`. Check `pyproject.toml` before reaching for anything else.
- Don't make `envelope.py`/`value.py` non-deterministic or non-monotone — the engine's
  whole guarantee rests on those being pure and reproducible.
