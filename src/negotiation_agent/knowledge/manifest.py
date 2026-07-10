"""The allowlist manifest — the fail-closed gate for what may be ingested.

The single most important invariant in the knowledge layer: **a path is ingestible only
if it is on the allowlist AND does not match an exclude glob.** Both conditions, always.
An empty or missing allowlist ingests nothing (fail-closed), never everything.
"""

from __future__ import annotations

import fnmatch
import json
from pathlib import Path

from pydantic import BaseModel, Field

# A tag scopes retrieval (e.g. pricing-only for a quote teardown). Kept as a plain str
# set, not an enum, so a new tag in the manifest doesn't require a code change.
KnownTag = str


class ManifestEntry(BaseModel):
    model_config = {"frozen": True}

    path: str
    tag: KnownTag
    # Frontmatter is stripped from every file regardless; this flag adds the heavier
    # job-hunt/career SECTION scrub for files (Companies/*) that interleave personal data.
    scrub_personal: bool = False


class Manifest(BaseModel):
    """The curated allowlist. ``vault_include`` is read relative to the vault root
    (``vault_root_env``); ``repo_include`` is read relative to the repo root."""

    model_config = {"frozen": True}

    version: int
    description: str = ""
    vault_root_env: str = "PEITHO_VAULT_ROOT"
    vault_include: list[ManifestEntry] = Field(default_factory=list)
    repo_include: list[ManifestEntry] = Field(default_factory=list)
    exclude_globs: list[str] = Field(default_factory=list)

    def is_excluded(self, path: str) -> bool:
        """True if ``path`` matches any exclude glob — the defense-in-depth second gate.

        Matches against the whole path and every suffix, so a glob like ``People/**`` or
        ``_private/**`` rejects a path regardless of the root it is joined to.
        """
        norm = path.replace("\\", "/")
        parts = norm.split("/")
        candidates = [norm] + ["/".join(parts[i:]) for i in range(len(parts))]
        for glob in self.exclude_globs:
            g = glob.replace("\\", "/")
            for cand in candidates:
                if fnmatch.fnmatch(cand, g):
                    return True
                # fnmatch treats ** like * (no path semantics); also try the /** form
                if g.endswith("/**") and (cand == g[:-3] or cand.startswith(g[:-2])):
                    return True
        return False

    def allowed_vault_entries(self) -> list[ManifestEntry]:
        """Vault entries that pass BOTH gates — on the allowlist and not excluded."""
        return [e for e in self.vault_include if not self.is_excluded(e.path)]

    def allowed_repo_entries(self) -> list[ManifestEntry]:
        return [e for e in self.repo_include if not self.is_excluded(e.path)]


def load_manifest(manifest_path: Path) -> Manifest:
    """Load and validate the manifest. Raises on malformed JSON or a schema violation —
    a broken manifest must fail loudly, never degrade to ingesting everything."""
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return Manifest.model_validate(data)
