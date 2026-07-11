"""The retriever — load the shipped index once, answer advisory queries at draft time.

The knowledge **advises** the buyer agent's drafting: retrieved passages give it real
negotiation strategy, lever ideas, and framing. It never edits the mandate, never supplies
a number the engine didn't approve — the guard still enforces ``approved_numbers`` on the
drafted text, so a retrieved figure can't reach the wire. Knowledge in, engine still decides.

The index (``data/kb_index.json``) is built offline and shipped in the repo; this loads it
lazily and read-only. If the file is absent (a build without it), retrieval returns nothing
and the agent drafts exactly as it did before — the layer degrades to a no-op, never errors.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

from negotiation_agent.knowledge.bm25 import Bm25Index, Hit

logger = logging.getLogger(__name__)

_INDEX_PATH = Path(__file__).resolve().parents[3] / "data" / "kb_index.json"


@lru_cache(maxsize=1)
def _load_index() -> Bm25Index | None:
    """Load the shipped BM25 index once, or None if it isn't present/parseable.

    Cached for the process lifetime. A missing or broken index is logged and treated as
    "no knowledge available" — the caller degrades to unadvised drafting, never crashes.
    """
    if not _INDEX_PATH.is_file():
        logger.info("KB index not found at %s — retrieval disabled", _INDEX_PATH)
        return None
    try:
        data = json.loads(_INDEX_PATH.read_text(encoding="utf-8"))
        return Bm25Index.from_dict(data)
    except (ValueError, KeyError) as e:  # malformed index — fail soft, don't take down drafting
        logger.warning("KB index failed to load (%s) — retrieval disabled", e)
        return None


def retrieve(
    query: str, *, tag: str | None = None, category: str | None = None, top_k: int = 3
) -> list[Hit]:
    """Top-``top_k`` knowledge chunks for ``query``, optionally scoped to a content ``tag``
    and/or a procurement ``category``.

    Returns [] when the index is absent — the layer is always optional.
    """
    index = _load_index()
    if index is None:
        return []
    return index.query(query, top_k=top_k, tag=tag, category=category)


# A category needs at least this many chunks to be worth scoping to; below it, retrieval
# falls back to general (unscoped) strategy and the caller flags the coverage gap.
_MIN_CATEGORY_CHUNKS = 4


def category_coverage(category: str) -> int:
    """How many chunks the index holds for a procurement category (0 if index absent)."""
    index = _load_index()
    if index is None:
        return 0
    return sum(1 for c in index.chunks if c.category == category)


def has_category_playbook(category: str) -> bool:
    """True if the KB has enough material to negotiate this category specifically."""
    if category == "unknown":
        return False
    return category_coverage(category) >= _MIN_CATEGORY_CHUNKS
