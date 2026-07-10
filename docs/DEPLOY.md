# Deploying the Peitho v2 negotiation backend

The backend is a FastAPI app (`negotiation_agent.api:app`) that runs the real
deterministic engine, drafts buyer/supplier prose with Claude server-side, and
enforces the numeric guard before any message is returned. It is **built and
tested** — what remains is yours: supply the secrets, deploy, point the domain.

## What's already done (in the repo)

- `railway.toml` + `Procfile` — start command, health check, build with the `[web]` extra.
- `pyproject.toml` `[web]` extra — `fastapi`, `uvicorn`, `anthropic`.
- `GET /health` — liveness + which models are configured (no secrets leaked).
- Abuse gates: per-session+IP rate limit, mandate TTL, HMAC mandate signing.

## Your manual steps

### 1. Set the secrets in Railway (never in the repo)

In the Railway service → **Variables**, add:

| Variable | Value | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | your Anthropic key | drafts buyer (Opus) + supplier (Haiku) |
| `PEITHO_MANDATE_SECRET` | a random 32-byte hex | **required** — signs the mandate. Generate: `python -c "import secrets; print(secrets.token_hex(32))"` |
| `HADES_API_KEY` | your Hades key | optional — enables live `/prepare` supplier research |
| `HADES_URL` | `https://hades-production-b86a.up.railway.app` | optional |

Optional tuning (sensible defaults, override only if needed):
`PEITHO_MANDATE_TTL` (3600), `PEITHO_RATE_PER_MIN` (20). Set `DEMO_GODVIEW=1`
**only** on a demo instance to allow the god-view internal block.

> Add these same keys to `.env.example` in the repo (keys only, empty values) —
> it's write-protected by the secret-scan hook, so edit it by hand. The vars are
> `ANTHROPIC_API_KEY`, `PEITHO_MANDATE_SECRET`, `PEITHO_MANDATE_TTL`,
> `PEITHO_RATE_PER_MIN`, `DEMO_GODVIEW`, `HADES_API_KEY`, `HADES_URL`.

### 2. Deploy on Railway (Hobby plan)

Connect this GitHub repo to a Railway service. Railway reads `railway.toml`,
installs with `pip install -e '.[web]'`, and starts uvicorn. The health check
hits `/health`. The Hobby plan's included $5 credit covers a small always-on
service — the only genuinely new cost is Anthropic API usage (a few cents per
negotiation, bounded by the rate limit).

### 3. Point your Hostinger domain at it

Railway gives the service a `*.up.railway.app` URL. To use your own domain:
- In Railway → **Settings → Domains**, add a custom domain (e.g. `peitho.yourdomain.com`).
- In Hostinger DNS, add the **CNAME** record Railway shows, pointing the subdomain
  at the Railway target. (Hostinger shared hosting can't *run* the backend — it's
  a vanity front; Railway is the compute.)

### 4. Verify

```bash
curl https://<your-domain>/health
# → {"status":"ok","buyer_model":"claude-opus-4-8","supplier_model":"claude-haiku-4-5","mandate_configured":true}
```

If `mandate_configured` is `false`, `PEITHO_MANDATE_SECRET` isn't set — `/negotiate`
will 500 until it is.

## Cost guardrails (already coded)

- **Rate limit** per session+IP (`PEITHO_RATE_PER_MIN`, default 20/min) → `429`.
- **Mandate TTL** — signed mandates expire (`PEITHO_MANDATE_TTL`, default 1h).
- A closed game is rejected (`409`) **before** any LLM call — a finished negotiation costs zero tokens.

> **Not yet coded (documented in `peitho-v2-architecture.md` §9):** a hard monthly
> USD spend cap, and Redis-backed counters for a multi-replica deploy. For a
> single-instance Hobby demo, the in-memory rate limit is the guardrail; add a
> Railway usage alert as a backstop.

## Local run

```bash
pip install -e '.[web]'
PEITHO_MANDATE_SECRET=$(python -c "import secrets; print(secrets.token_hex(32))") \
  uvicorn negotiation_agent.api:app --reload
```
