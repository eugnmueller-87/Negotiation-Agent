# Agents

Agents are specialized Claude instances that run in **isolated context**. They don't see your conversation history or loaded rules. They only have their own system prompt and tools.

Claude delegates to agents automatically based on the task description, or you can invoke them with `@agent-name`.

Every reviewer is **read-only** (`Read, Grep, Glob, Bash`) — isolation is the point, so a reviewer can never edit the code it's judging.

Blueprint agents that don't fit this pure-Python engine/CLI project (`frontend-designer`, `data-pipeline-reviewer`, `performance-reviewer`) were pruned on install. Re-add any of them from the blueprint if the project grows a UI, a pandas/SQL pipeline, or a hot path.

## Available agents

Listed core-first: the first three run on virtually every code review; the rest activate when their subject appears in the diff.

### code-reviewer
General bug catcher for Python (primary), JS/TS, PowerShell, and Bash. Catches off-by-ones, None/null derefs, inverted conditions, race conditions, swallowed errors, and excessive complexity — with Python-specific traps (mutable default args, late-binding loop closures, `is` vs `==`) and shell traps (`$?`/`$LASTEXITCODE` after a redirected native exe, unquoted Windows paths, missing `set -euo pipefail`). Skips style nitpicks. Trigger: after any code change, before committing.

### silent-failure-hunter
Hunts the one bug class worse than a crash: code that fails without telling anyone. Empty `except`, errors masked as `[]`/`None`/empty DataFrame, floating `asyncio` tasks, `errors='coerce'`/`fillna(0)` that hides bad input, ignored exit codes in nightly `.ps1`/n8n steps. For each error path it asks: if this fails in production, who finds out? Trigger: any change to error handling, fallbacks, retries, async, or pipeline steps.

### pr-test-analyzer
Judges whether a diff's **pytest** tests actually verify the change — test critique, not test generation. Catches assertion-free tests, mock theater (mocking your own unit vs. mocking the LLM/DB boundary), tests that can't fail, `@pytest.mark.skip` left in, and weakened tolerances. Includes the AI angle: a changed prompt/chain with no test asserting the parsed output. Core question: if the implementation were wrong, would any test go red? Trigger: a diff adds/changes tests, or changes behavior without touching tests.

### security-reviewer
OWASP-style static analysis tuned to Python and the **NEVER-commit-secrets** rule. Secrets are category one: hardcoded `ANTHROPIC_API_KEY`/`OPENAI_API_KEY`/`SLACK_BOT_TOKEN`/`TELEGRAM_TOKEN`, `.env`/`*.pem` staged in the diff. Plus SQL f-string injection, `subprocess(shell=True)`, `pickle`/`yaml.load` on untrusted data, weak crypto, and missing rate limits on the Telegram bot. Severity-ranked with attack vector and fix. Trigger: changes to auth, input handling, queries, subprocess, deserialization, or crypto. Belt-and-suspenders with the `scan-secrets` hook.

### ai-agent-reviewer
Reviews LLM / agent / RAG **code** (Claude & OpenAI APIs, LangChain/LangGraph, n8n AI nodes, MCP servers). Checks prompt-injection exposure (untrusted content in the system prompt), tool-call safety (model args reaching subprocess/SQL, ungated destructive tools), token/cost control (missing `max_tokens`, unbounded agent loops), API reliability (timeouts, retry/backoff, truncated-response handling), structured-output/hallucination guards, and eval coverage. Does not assert current model ids — it instructs the reviewer to verify them. Trigger: changes to prompts, LLM calls, agent/tool definitions, or RAG pipelines.

### procurement-domain-reviewer
Sanity-checks procurement business logic. For this project that means the negotiation math itself: utility/value scoring, weight normalization (weights sum to 1), reservation-vs-target ordering, concession-curve decay, and logrolling counteroffers — plus the classic savings/baseline/currency/unit traps if savings reporting is added. Catches flipped signs, wrong percentage bases, broken baselines (silent 0 → fake savings), and fail-open compliance checks (LkSG/CSDDD/sanctions that log but don't block). States both readings rather than guessing a domain rule. Trigger: changes to scoring, concession, counteroffer, or compliance logic.

### doc-reviewer
Reviews documentation for accuracy (do docs match the Python source?), completeness (are required params and **env vars** documented?), staleness, and clarity. Cross-references docs against source with grep and file reads; catches env-var drift between README/CLAUDE.md and `os.environ` lookups. Trigger: after `.md`/docstring/README changes, or when code changes may have invalidated docs.

## Adding your own

Create a directory per agent — `agents/<name>/<name>.md` (Claude Code scans agents directories recursively; one dir per agent is what lets the plugin marketplace symlink each agent individually):

```yaml
---
name: your-agent-name
description: When Claude should delegate to this agent
tools:
  - Read
  - Grep
  - Glob
  - Bash
---

Your agent's system prompt here.
```

See [Claude Code docs](https://code.claude.com/docs/en/sub-agents) for all frontmatter options.
