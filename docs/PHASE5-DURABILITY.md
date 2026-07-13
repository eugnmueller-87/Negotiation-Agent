# Phase 5 — Durability (verified status + what server-side would require)

> **Verified 2026-07 (this build):** the Railway deploy has **NO persistent volume** —
> `railway.toml` declares no `[[deploy.volumes]]` mount, so any file written on the box is
> **ephemeral** (wiped on every redeploy/restart). Server-side persistence therefore **cannot be
> built today** without first provisioning a volume. We did not fake it: writing "durable" data to
> ephemeral disk would be a silent corruption.

## What we shipped instead (durability without a server)

The whole portfolio + savings ledger already lives in **buyer-owned `localStorage`** — never on the
server, never in a prompt. For a privacy-first product this is arguably the *correct* posture, not a
gap (it's the GDPR-minimisation default: EUR + supplier data stay on the buyer's machine).

The real durability gap of localStorage-only is **backup / portability**, and that's fixed **without
a server**:

- **Export** (`exportPortfolio`) — download the entire store (`PROJECTS` + the savings ledger) as a
  `peitho.portfolio.v1` JSON file the buyer keeps.
- **Import** (`importPortfolio`) — restore that backup on any machine (guarded by a confirm; revives
  the `touched` Set and clears any in-flight busy flags).

## What server-side Phase 5 WOULD require (when durability is demanded)

Only build this once a persistent volume exists — otherwise it stores to ephemeral disk.

1. **Provision a Railway volume** and mount it (e.g. at `/data`); expose the path via an env var
   (`PEITHO_DATA_DIR`), fail-closed to no-persistence when unset.
2. **`ProjectStore`** — a `sqlite3` (stdlib) or append-only JSONL store at `$PEITHO_DATA_DIR`, keyed
   by `session_id`. It is a **single transcript writer layered ON TOP of** the existing SEC-5
   server-side re-extraction — never replacing it (the client transcript stays untrusted; the store
   is a convenience cache + audit record, not the source of truth for the fold).
3. **Buyer-owned `SavingsStore`** — the EUR ledger, server-persisted per buyer, **still never sent to
   a model** (GDPR). Keep it separate from the PII-free `OutcomeStore` (which stays dimensionless).
4. **Server-side `human_led` audit** — the takeover message + actor, persisted (Phase 1 keeps the
   takeover audit client-side today; the store phase moves it server-side).

Until step 1 is done, the honest answer is: **the buyer owns their data locally, and Export/Import is
the backup path.**
