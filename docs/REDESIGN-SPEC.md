# Peitho Demo — Redesign Spec

*Written 2026-07 as the plan to execute after the pause. The current `demo/peitho-v2.html` grew
organically across many features and now has redundant panels, an unclear flow, and a look that
reads "dev demo" not "product". This spec fixes all four problems the redesign must solve: clutter,
doesn't showcase the new capabilities, not professional enough, confusing flow. ONE file serves both
the free demo and the token-unlocked full version, so this redesign applies to both automatically.*

---

## 1. The core problem (why a rework, not a patch)

Every feature this session bolted a panel onto the intake screen. Result: the same commercial facts
render up to 4× (key-points bullets, "commercial picture" grid, "extracted numbers pre-filled" box,
mandate breakdown), and the flow has no spine. The `c2e492b` cleanup removed the worst duplication,
but the *structure* is still additive-by-accident. The redesign gives the app a **deliberate
information architecture** built around the one thing that's actually true now:

> **Peitho is a 4-step pipeline: UNDERSTAND → MANDATE → NEGOTIATE → PROVE.**
> The current UI hides that pipeline. The redesign makes it the backbone.

---

## 2. The pipeline (the new spine)

A persistent **stepper** at the top, always showing where you are:

`①  Understand  →  ②  Set the mandate  →  ③  Negotiate  →  ④  Proof & audit`

Each step is one screen. No screen shows another step's data. This single decision kills the clutter
(each fact has exactly one home) and fixes the flow (the path is literally drawn at the top).

### Step ① — Understand the contract
- **Input:** the upload dropzone + paste (unchanged mechanics). Sample + Due-diligence-example
  buttons live here.
- **Output (after Read):** ONE panel — the **key-points bullets** (already built: counterparty,
  value/price, payment, term, legal flags with ⚠ on gaps, extraction notes). Full text behind the
  collapsible toggle (already built). Supplier-risk row + mandate-adjustment proposals stay here too
  (they're understanding, not mandate-setting).
- **Cut:** the "commercial picture" tile grid (done), the "no adjustments / pre-filled" box (done).
- **New capability to showcase:** a **"Scan for risks"** primary action that opens the due-diligence
  cockpit inline for THIS contract (demo → canned example; full → live scan). Right now the cockpit
  is a separate "Due-diligence example" button disconnected from the uploaded contract — wire it
  into step ① as the natural "now check the risks" beat.

### Step ② — Set the mandate
- The mandate form (supplier, category, target, floor, payment lever, term, emphasis, terms-on-table)
  + the **Commercial breakdown** panel (already built: Current → Target → Floor with the gaps).
- **The breakdown is the hero of this step** — it's what answers "why these numbers". Make it
  prominent, not a footnote below the form.
- **New capability to showcase:** when history exists, show the **learned prior** ("in this category,
  suppliers typically settle around X; they concede payment, defend price") from `priors.py` — the
  "gets smarter over time" story, visible. Confidence-gated (hide below 5 samples).
- Total-value-only contracts (Phenom): the honest "no unit basis, set it yourself" state (done) —
  keep, it's correct.

### Step ③ — Negotiate
- The existing run view (buyer/supplier/both tabs, transport controls, convergence chart).
- **New capability to showcase:** the **adaptive negotiator** — when `EngineConfig.adaptive` is on,
  surface the per-turn **tactic + rationale** ("held firm — supplier isn't moving", "traded — they're
  defending price") from the `EngineDecision.tactic`/`tactic_rationale` fields already built. This is
  the "senior negotiator as code" made visible — a huge differentiator currently invisible in the UI.
- Show the give/get on each counter (already partly built per the termination memo).

### Step ④ — Proof & audit  *(NEW SCREEN — the competitive edge made real)*
- Per the competitive-edge doc, this is build move #1 and the moat: after a negotiation closes,
  a **Proof-of-Floor certificate** — replay the transcript, show every decision + rationale, and the
  green "✓ the engine never crossed your reservation floor (verified by replay)" stamp.
- This is the screen no competitor offers. It should feel like a compliance artifact: clean, exportable.
- Backend for this is mostly primitives that exist (`decide()` is pure, envelopes are versioned/signed);
  it needs the replay+assert packaging (see COMPETITIVE-EDGE.md #1).

---

## 3. Visual system (make it look like a product)

Keep the existing warm palette + serif/mono pairing (it's already tasteful — see the `:root` tokens).
The problem isn't the palette, it's **hierarchy and spacing**. Fixes:

- **One card style, consistently applied.** Right now panels vary (`.essence`, `.keypts`, `.panel`,
  `.intel-out`, `.mx`). Unify into ONE card component (border, radius, header, body) and use it
  everywhere. This alone makes it look designed rather than assembled.
- **A real header/hero per step**, not just a section label. Step ① : "What's in this contract?".
  Step ② : "What's your walk-away?". Give each screen a one-line purpose.
- **Vertical rhythm:** consistent gap between cards (the current stack has uneven margins). Use a
  single `--gap` and fl/grid `gap`, not per-element margins.
- **Progressive disclosure:** advanced controls (emphasis sliders, extra terms, supplier persona)
  behind "Advanced" — the default screen shows the 4 numbers that matter, not 12 controls.
- **The mode banner** (demo vs full) stays — it's honest and good.
- **Empty/loading states:** every step needs a clean empty state (before upload) and a loading state
  (during read/scan/negotiate) — currently some are bare.

---

## 4. What each fact's ONE home is (kill duplication by rule)

| Fact | Lives ONLY in |
|---|---|
| Supplier, value/price, payment, term, legal flags | Step ① key-points bullets |
| Supplier risk (Hades / sample) | Step ① risk row |
| Proposed mandate adjustments | Step ① (they inform the mandate you set next) |
| Target / floor / current + the gaps | Step ② commercial breakdown |
| Learned category prior | Step ② (when confident) |
| Per-turn tactic + rationale | Step ③ negotiation view |
| Floor-never-crossed proof | Step ④ certificate |

**Rule for the rebuild:** if a fact would appear on two screens, it's on the wrong one. Pick the
step where the user ACTS on it.

---

## 5. Build order (when you resume)

1. **The stepper + screen split** — refactor `render()` to route the four steps; move existing panels
   to their home step. Mostly moving code, not writing it. This alone fixes clutter + flow.
2. **Unify the card component** — one CSS card, applied everywhere. Fixes "looks like a dev demo".
3. **Wire the cockpit into step ①** — "Scan for risks" on the uploaded contract, not a stray button.
4. **Surface the adaptive tactic + rationale in step ③** — the fields exist; render them.
5. **Surface the learned prior in step ②** — `priors.py` output, confidence-gated.
6. **Build step ④ (Proof-of-Floor)** — the replay+certificate (also COMPETITIVE-EDGE.md #1, so this
   does double duty: the redesign's capstone AND the competitive moat).

Steps 1–2 are a day and fix the "looks bad" complaint. Steps 3–5 surface capabilities already built.
Step 6 is the differentiator and overlaps the go-to-market plan.

---

## 6. What NOT to do

- Don't change the palette/fonts — they're fine; the problem was structure.
- Don't add new *capabilities* in the redesign — surface the ones already built. Scope creep here
  turns a 2-day rework into a 2-week one.
- Don't rebuild the negotiation engine view from scratch — it works; it just needs the tactic
  rationale surfaced and to live inside the stepper.
- Don't make the free demo and full version look different — one file, one design; mode only changes
  whether prose is templated vs LLM and whether the scan is canned vs live.

---

*Companion docs: `COMPETITIVE-EDGE.md` (why step ④ is the moat), and the memory notes on the
negotiator + learning layer (what steps ②/③ surface).*
