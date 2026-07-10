"""Build the shipped KB index from the vault + repo notes. Run locally, commit the output.

    PEITHO_VAULT_ROOT="/path/to/My AI Brain" python -m negotiation_agent.knowledge.build

Reads ``data/kb_manifest.json`` (the allowlist), ingests only permitted paths, builds the
BM25 index, and writes ``data/kb_index.json`` — which ships in the repo so the Railway
server has the knowledge without ever touching the local vault. If ``PEITHO_VAULT_ROOT``
is unset, only the repo-authored Fisher-Ury notes are indexed (a valid, smaller build) and
that is reported loudly, not hidden.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from negotiation_agent.knowledge.bm25 import Bm25Index
from negotiation_agent.knowledge.ingest import ingest
from negotiation_agent.knowledge.manifest import load_manifest

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MANIFEST = _REPO_ROOT / "data" / "kb_manifest.json"
_INDEX_OUT = _REPO_ROOT / "data" / "kb_index.json"


def build_index(*, vault_root: Path | None, repo_root: Path = _REPO_ROOT) -> Bm25Index:
    manifest = load_manifest(_MANIFEST)
    chunks = ingest(manifest, vault_root=vault_root, repo_root=repo_root)
    return Bm25Index.build(chunks)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    vault_env = os.getenv("PEITHO_VAULT_ROOT")
    vault_root = Path(vault_env) if vault_env else None
    if vault_root is None:
        logger.warning(
            "PEITHO_VAULT_ROOT is unset — building from the repo Fisher-Ury notes ONLY "
            "(no vault content). Set it to include the curated vault knowledge."
        )
    elif not vault_root.is_dir():
        logger.error("PEITHO_VAULT_ROOT=%s is not a directory — aborting.", vault_root)
        return 1

    index = build_index(vault_root=vault_root)
    _INDEX_OUT.write_text(json.dumps(index.to_dict(), ensure_ascii=False), encoding="utf-8")
    tags: dict[str, int] = {}
    for c in index.chunks:
        tags[c.tag] = tags.get(c.tag, 0) + 1
    logger.info(
        "Built %d chunks from %d sources -> %s\n  by tag: %s",
        index.n_docs,
        len({c.source for c in index.chunks}),
        _INDEX_OUT.relative_to(_REPO_ROOT),
        dict(sorted(tags.items(), key=lambda x: -x[1])),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
