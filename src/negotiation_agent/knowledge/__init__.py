"""The procurement knowledge base — a curated, fail-closed retrieval layer.

The buyer agent negotiates better with domain knowledge (real vendor cases, pricing
methodology, category strategy, and the Fisher-Ury negotiation framework). That knowledge
is curated from a private Obsidian vault, but the vault is laced with confidential and
personal data a supplier-facing agent must never surface. So the whole layer is built
**fail-closed**: an explicit allowlist (``data/kb_manifest.json``) names the only paths
that may be ingested; everything else is skipped, and a defense-in-depth exclude-glob
check rejects a listed path that still looks sensitive.

Retrieval is BM25 (lexical, dependency-free, deterministic) over the ingested chunks. The
knowledge **advises** the buyer agent's drafting — it never edits the mandate, never moves
a number. Same wall as the LLM: knowledge in, the deterministic engine still decides.

Pipeline: ``manifest`` (the gate) → ``ingest`` (read allowed files, strip frontmatter,
chunk) → ``bm25`` (build the index / query it). The index is built offline and shipped in
the repo (``data/kb_index.json``); the server loads it read-only.
"""

from __future__ import annotations

from negotiation_agent.knowledge.bm25 import Bm25Index, Hit
from negotiation_agent.knowledge.category import CATEGORY_LABELS, Category, detect_category
from negotiation_agent.knowledge.ingest import Chunk, ingest
from negotiation_agent.knowledge.manifest import Manifest, ManifestEntry, load_manifest
from negotiation_agent.knowledge.retrieve import (
    category_coverage,
    has_category_playbook,
    retrieve,
)
from negotiation_agent.knowledge.tone import Register, detect_register, greeting_for

__all__ = [
    "CATEGORY_LABELS",
    "Bm25Index",
    "Category",
    "Chunk",
    "Hit",
    "Manifest",
    "ManifestEntry",
    "Register",
    "category_coverage",
    "detect_category",
    "detect_register",
    "greeting_for",
    "has_category_playbook",
    "ingest",
    "load_manifest",
    "retrieve",
]
