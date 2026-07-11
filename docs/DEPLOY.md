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
| `ANTHROPIC_API_KEY` | your Anthropic key | drafts buyer (Opus) + supplier (Haiku) — **only used in full mode** (see `PEITHO_FULL_TOKEN`) |
| `PEITHO_MANDATE_SECRET` | a random 32-byte hex | **required** — signs the mandate. Generate: `python -c "import secrets; print(secrets.token_hex(32))"` |
| `PEITHO_FULL_TOKEN` | a random secret, or **leave unset** | **The demo/full switch.** UNSET ⇒ the public URL runs the cost-free DEMO (real engine, templated messages, no LLM/Hades — $0). Set it to a secret to enable the FULL version (Opus/Haiku/Hades) for yourself: open the demo with `#full=<token>` in the URL. **Fail-closed:** with no token set, nothing can trigger paid calls even though the keys are present. Rotate if the full link leaks. |
| `PEITHO_GODVIEW_TOKEN` | a random secret, or **leave unset** | **Leave UNSET on a public instance** — god-view exposes the buyer's reservation floor. Unset ⇒ god-view is OFF (fail-closed). Set it only to reveal internals to yourself: then open the demo with `#gv=<token>` in the URL. Do NOT set the old `DEMO_GODVIEW`; a client header no longer unlocks anything. |
| `PEITHO_ALLOWED_ORIGINS` | your demo origin(s), comma-separated | **set on a public deploy** — locks CORS to your front-end. Unset ⇒ `*` (with a startup warning). |
| `HADES_API_KEY` | your Hades key | optional — enables live `/prepare` supplier research |
| `HADES_URL` | `https://hades-production-b86a.up.railway.app` | optional — must be `https://` |

> **The demo UI** is served at the service root `/` (same app, same URL). Open
> `https://<your-service>.up.railway.app/` in a browser and it runs the live
> negotiation against Opus/Haiku. `/health` and the `/negotiate/*` API are on the
> same host.

Optional tuning (sensible defaults, override only if needed):
`PEITHO_MANDATE_TTL` (3600), `PEITHO_RATE_PER_MIN` (20),
`PEITHO_RESEARCH_RATE_PER_MIN` (4, the stricter cap on the paid Hades routes),
`PEITHO_TRUSTED_PROXIES` (1 — the number of reverse proxies in front of the app;
Railway = 1. This is how the rate limiter resolves the real client IP from
`X-Forwarded-For`; a wrong value here either lets the limiter be bypassed or keys
everyone to the proxy IP).

> **Security note (Fable-5 audit, 2026-07-11):** god-view is now gated on the
> secret `PEITHO_GODVIEW_TOKEN` (constant-time compare), never a client header —
> leave it unset on any internet-reachable instance so the reservation floor can't
> leak. The rate limiter keys on the edge-resolved IP (`PEITHO_TRUSTED_PROXIES`),
> not a spoofable `X-Forwarded-For` prefix.

> Add these same keys to `.env.example` in the repo (keys only, empty values) —
> it's write-protected by the secret-scan hook, so edit it by hand. The vars are
> `ANTHROPIC_API_KEY`, `PEITHO_MANDATE_SECRET`, `PEITHO_FULL_TOKEN`,
> `PEITHO_GODVIEW_TOKEN`, `PEITHO_ALLOWED_ORIGINS`, `PEITHO_MANDATE_TTL`,
> `PEITHO_RATE_PER_MIN`, `PEITHO_RESEARCH_RATE_PER_MIN`, `PEITHO_TRUSTED_PROXIES`,
> `HADES_API_KEY`, `HADES_URL`.

### Portfolio (demo) vs. full version

One deploy serves both. The **public portfolio URL** (`https://<service>/`) runs the real
deterministic engine with **templated messages and no research — $0 to run, nothing to abuse**.
The **full version** (Opus writes the buyer prose, Haiku plays the supplier, live Hades
due-diligence) is the same URL with the secret token in the fragment:

```
https://<service>/                    → demo   (free, for the portfolio)
https://<service>/#full=<PEITHO_FULL_TOKEN>   → full   (paid, keep this link private)
```

Put the plain URL in your portfolio; keep the `#full=…` link for yourself. If the full link
leaks, rotate `PEITHO_FULL_TOKEN` (the per-IP rate limits cap the damage until you do).

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
