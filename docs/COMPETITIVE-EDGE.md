# Peitho Competitive Edge — Strategy Doc

*Synthesized 2026-07 (Claude Fable 5, grounded in July-2026 market research + a read of
`src/negotiation_agent/`). Competitor facts are cited; codebase claims were verified against the
actual source. Read this before the next build phase.*

> **Provenance note:** market facts below come from public 2026 sources (Pactum's own docs,
> procurement-analyst reviews) linked at the end. Simulated outcome numbers are labelled as such —
> they are engine-quality evidence, NOT customer savings, and must never be marketed as customer
> results until earned from real usage.

---

## 1. The Thesis

Peitho's edge is **not the architecture** — Pactum's own docs describe the identical split
(rule-based AI decides, LLM only talks, every concession recorded), so "deterministic and
explainable" is table stakes independently confirmed by the market leader. The edge is **positional
and structural**: Peitho is the only negotiation-execution product a company under ~$500M revenue
can even *evaluate* — Pactum requires a formal sales/scoping engagement, no trial, enterprise
pricing; Keelvar never touches execution at all. That empty segment, combined with two things a
hosted enterprise SaaS structurally cannot offer — **buyer-verifiable (not vendor-asserted)
auditability** via a pure, replayable decision function, and **full data sovereignty** (the entire
decision path runs on the buyer's infrastructure) — is the strategy. The wedge into it is a workflow
no incumbent addresses: the auto-renewing indirect-spend contract, entered through a free
termination-clock alert Peitho already ships. The "model advises, engine decides" principle is not
traded away by any of this; it is the load-bearing wall of all three positions.

---

## 2. The Moat — Defensible vs. Table Stakes

**Table stakes (do NOT build the pitch here):**

- *Rules-decide / LLM-talks architecture.* Pactum markets exactly this. Independent convergence
  validates the design; it differentiates nothing.
- *"Learning engine improves over time."* Pactum claims it; Peitho's cross-negotiation priors match
  the mechanism (`priors.py` — per-category warm-start, confidence-gated at 5 samples). But Pactum's
  cross-customer volume dwarfs any single-tenant corpus. **Peitho cannot win a data-volume race.**
- *Multi-variable optimization.* Pactum optimizes price/payment/delivery/volume. Peitho's greedy
  fill (`packages.py`) is the exact LP optimum under linear value functions — parity, not advantage.

**Actually defensible** (each anchored to something an incumbent structurally won't do):

| Edge | Why Pactum can't follow |
|---|---|
| **A. Sub-$500M access** — self-serve, $0 real-engine demo, evaluate in an afternoon | Their unit economics require enterprise pricing + implementation projects; serving a €50M Mittelstand buyer destroys their sales motion |
| **B. Verifiable transparency** — buyer/auditor can *independently replay* every negotiation (pure `decide()`, frozen versioned `signed_by` envelopes, utility recomputable under any envelope version) and mathematically confirm no turn crossed `reservation_utility` | Their logic is their trade secret; opening it commoditizes their product. Pactum says "trust our records"; Peitho says "run the replay yourself" |
| **C. EU data sovereignty** — decision path is pure Python, LLM only at the language edge with cheap models; whole thing runs on-prem/EU-hosted; learning is per-tenant, privacy-minimized by design | Pactum is hosted US SaaS with (presumably) cross-customer learning — for a GDPR/LkSG/works-council buyer, "your negotiation history stays in your building" is a decision criterion |

**Honest status:** today Peitho has an engine, not a moat. Edges A–C are *positions to occupy*,
converted into a moat only via switching costs (a customer's accumulated envelope library, category
priors, audit history) before a funded copycat arrives. Transparency stops Pactum specifically, not
a copycat — and only wins where buyers pay for it (regulated/compliance-heavy EU procurement).

---

## 3. The Wedge — "Renewal Guard"

**ICP (specific, not "SMB"):** EU/German Mittelstand, 50–500 employees, €10M–€250M revenue, 0–3
procurement FTEs (or a CFO wearing the hat). Their reality: dozens-to-hundreds of auto-renewing
indirect contracts (logistics, MRO, facilities, IT/telecom/SaaS) that renew un-negotiated because
nobody has capacity. Pactum's Walmart case (3% avg gain + 35 days payment terms) quantifies what
that leaves on the table — and this ICP structurally cannot buy Pactum. Peitho already ships EU
number formats, LkSG context, and the German-law termination-notice clock. **Not** the ICP:
enterprises with P2P suites, and strategic direct-materials spend (needs per-deal customization no
one — including Peitho — can turnkey).

**First killer workflow (~70% already built):**

1. **Free:** upload the contract PDF/DOCX (shipped: intake + extractor + cockpit) → Peitho computes
   the notice deadline, alerts at T-60/T-30/T-14.
2. At the alert: Peitho presents a **draft mandate** the human signs (versioned, `signed_by`, never
   LLM-edited — architecture preserved).
3. Engine runs the counteroffer sequence (Boulware + logrolling + hard floor + `approved_numbers`) →
   **email drafts the human sends.** Draft-mode is the deliberate answer to the named category
   friction of supplier-side AI resistance: the supplier sees a normal email from a person.
   Autonomous send is a trust-earned upgrade, not v1.

**The motion incumbents can't match:**

- The $0-marginal-cost free tier is *proven*, not planned — demo mode runs the real engine,
  templated prose, `PEITHO_FULL_TOKEN` gates all spend fail-closed.
- The contract calendar is the retention loop; the T-30 alert **is** the upsell moment ("deadline in
  30 days — unlock Peitho to negotiate this one"). Urgency-monetization, not seats.
- Pricing sketch: free (intake + clock + demo) → ~€79/mo self-serve (N active negotiations, LLM
  prose + risk scan) → later, per-negotiation success option.
- Land: one renewal saved. Expand: all renewals → tail-spend 3-bids-and-a-buy → per-tenant priors
  make the engine measurably warmer over time.

**Named risk (a bet, not a fact):** mid-market willingness-to-pay for procurement tooling is
historically weak; self-serve PLG in procurement is unproven category-wide. Deadline-urgency
monetization is the strongest available counter.

---

## 4. Build Moves — Ranked by Impact/Effort

**#1 — Audit Replay + Proof-of-Floor certificate — LOCAL ($0)** → *Edge B*
Per-negotiation export (envelope version + turn log + `EngineDecision` rationale) that re-runs
deterministically offline; recompute every decision, assert utility ≥ reservation, emit pass/fail.
Add a hash-chained transcript (`hashlib`, zero deps) and a mandate timeline (`signed_by` + version
bumps). Pure packaging of existing primitives — days of work — and it makes "prove the machine never
exceeded the human mandate" a demoable artifact, exactly what EU AI Act-era procurement review will
demand. **The single highest-leverage move: it converts the architecture from claim to evidence.**

**#2 — Free termination-clock landing page — LOCAL ($0)** → *the wedge itself*
"Never miss a notice deadline — upload one contract." Every piece exists (intake, extractor, clock,
drafted notice). Cheapest possible test of whether the ICP self-serves at all. Distribution has to
be product-led — one developer cannot out-sell anyone — and this is the product-led front door.

**#3 — MESO menus (2–3 Pareto-equivalent packages per counter) — LOCAL ($0)** → *Edge A + supplier-friction*
Emit multiple packages all clearing the same θ, distributing concessions differently; which one the
supplier engages with is a choice signal far stronger than the movement deltas
`opponent_model.infer_appetite` reads today. Floor-safe by construction (same `fill_package`, same
θ), deterministic (permute near-tied cost-ratio order, no RNG). Best-documented outcome improver in
the multi-issue literature (Medvec/Galinsky); directly addresses supplier resistance. Low–medium
effort — generator loop + choice-signal update + `neg-sim` eval.

**#4 — Calibrated opening anchor from category priors — LOCAL ($0)** → *makes "learning" concrete*
First-offer anchoring explains the largest share of settlement variance (Galinsky & Mussweiler
2001). Peitho opens flat at target today. Use `typical_settled_utility` + `escalation_rate` (already
computed in `priors.py`) to lift/temper the opening θ. Anchoring only raises the opening; floor
untouched. Confidence-gated behind `_MIN_SAMPLES_FOR_CONFIDENCE` so thin history can't mislead.
Lowest effort of the four; caveat: zero edge on cold start — seed synthetic history for the demo.

**#5 — Renewal-negotiation email prose in full mode — needs-LLM (cheap model only)** → *completes the wedge*
The drafts the human sends at step 3. Language is the only place an LLM is genuinely needed;
`approved_numbers` remains the confidentiality line, engine decides everything numeric.

**Explicitly rejected:** large-scale contract-space search per turn. Under linear value functions the
greedy fill is already the exact optimum; enumerating thousands of candidates finds nothing better
and burns the determinism story. Only revisit if non-linear value shapes land. Also rejected *as a
claim* (not a build): supplier-trust improvement from visible give/get — plausible, zero evidence,
don't market it until a real negotiation shows it.

**Flagged strategic choice (not a default):** open-sourcing the decision core (engine/envelope/value;
SaaS + LLM layer closed). "The only negotiation agent whose decision function you can read" is a
position no funded competitor will match, and the one-dev IP moat is weak anyway. Decide
deliberately.

---

## 5. Where Peitho Cannot Compete — No Hedging

- **The Global 2000.** Walmart-class reference cases, ERP/P2P integrations, multi-supplier campaign
  orchestration, cross-customer learning volume, an implementation and sales org — Peitho has none of
  these and won't at portfolio stage. Risk-averse enterprise procurement picks the vendor with the
  Walmart case. **Exclude this lane from positioning entirely; concede it to Pactum and the suites.**
- **Direct materials / services / project spend.** The named category-wide gap exists *because* it
  needs per-deal customization. One developer can't turnkey it either. Park it.
- **Proof parity.** Pactum has 3% avg gain + 35 payment days at Walmart. Peitho has a 9-scenario
  simulator. Publish `neg-sim` won-value deltas per feature as a reproducible scorecard — that's
  honest evidence of *engine quality* — but simulated savings are not customer savings and must be
  labelled as such. Outcome numbers must be earned from real usage.
- **Suite-embedded AI** (Coupa/GEP/Ivalua/JAGGAER) will copy any single feature, including the
  contract-intake → clock → mandate pipeline. That pipeline is real differentiation but a feature,
  not a moat.

---

## 6. The 90-Day Focus — "Become Sellable"

**Weeks 1–2: Ship the proof artifact.** Build #1 (Audit Replay + Proof-of-Floor + hash-chained
transcript). Without it the trust pitch is rhetoric identical to Pactum's; with it, a compliance team
can verify the engine in 10 minutes, free, before any commercial conversation — the thing Pactum
structurally cannot offer.

**Weeks 3–4: Ship the front door.** Build #2 (standalone termination-clock landing page, free tier).
This is the ICP self-serve test. Instrument it: uploads, alert opt-ins, T-30 click-throughs. If
nobody uploads a contract, the wedge hypothesis is falsified cheaply — that's a win too.

**Weeks 5–8: Make the engine demonstrably better, measurably.** Build #3 (MESOs) and #4 (calibrated
anchor), each gated by a before/after `neg-sim` scorecard published as the reproducible evidence
base. Then #5 (cheap-LLM draft emails) to complete the renewal workflow end-to-end.

**Weeks 9–12: Get 3–5 real free-tier users** (German Mittelstand network, procurement communities,
LkSG-adjacent channels) and drive one of them to a real renewal negotiation in draft-mode. One real
saved renewal — even one triggered by the free clock alert alone — is worth more than every simulated
metric. Decide the open-core question before any funded conversation.

**The positioning sentence that survives honesty review:**
> "Pactum's architecture, but yours: an inspectable, replayable, on-your-infrastructure negotiation
> mandate for the mid-market buyer Pactum can't afford to sell to — verified by replay, not by vendor
> claim — with the audit trail EU compliance will soon require."

**Assumptions this doc rests on (monitor them):**
1. Mid-market procurement will self-serve at all — untested; weeks 3–4 test it.
2. EU regulated buyers will pay for verifiability — directionally supported by the AI Act / LkSG
   trajectory, not yet by a purchase order.
3. No funded copycat occupies the sub-$500M segment within the window — switching costs (envelope
   library, priors, audit history) are the only defence, so ship them early.

---

## Sources (competitor facts)

- [Pactum AI Review 2026 — procurementaiagents.com](https://procurementaiagents.com/agents/pactum-ai)
  (enterprise-only, "inaccessible below ~$500M", no self-service/freemium)
- [The Required Components of an Autonomous Negotiations Platform — Pactum](https://pactum.com/blog/the-required-components-of-autonomous-negotiations-platform-pactum-platform)
  (their own architecture: rule-based AI decides, LLM advises)
- [Best AI Procurement Tools 2026 — Suplari](https://suplari.com/blog/top-10-ai-procurement-tools)
  (vendor landscape, phase split; Keelvar = sourcing only)
- [Pactum: Why Procurement is the Proving Ground for Agentic AI — Procurement Magazine](https://procurementmag.com/news/procurement-proving-ground-agentic-ai)
- Walmart case (3% avg commercial gain + 35 days payment terms) — cited across the above.
