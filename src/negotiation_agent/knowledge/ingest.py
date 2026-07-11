"""Ingest — turn allowlisted files into chunks, stripping anything sensitive on the way.

Three safety layers, defense-in-depth, all non-optional:
  1. Only paths the manifest allows are ever opened (the gate lives in ``manifest``; this
     module trusts the resolved entry list and never globs the vault itself).
  2. Frontmatter is stripped from every file (metadata/ids/tags, never content); entries
     flagged ``scrub_personal`` additionally have job-hunt/career SECTIONS removed — the
     Companies/* vendor profiles interleave personal metadata with vendor intel.
  3. A final content gate drops any chunk still carrying a personal marker, regardless of
     source — so a marker that slips past 1 and 2 still cannot reach the index.

Chunking is paragraph-pack: accumulate paragraphs up to a character budget so a chunk is a
coherent passage, not a sentence fragment. Deterministic — no RNG, no model.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel

from negotiation_agent.knowledge.category import detect_category
from negotiation_agent.knowledge.manifest import Manifest, ManifestEntry

# Chunk sizing: big enough to hold a full lever/story, small enough to rank precisely.
_CHUNK_CHARS = 1100
_CHUNK_OVERLAP_PARAS = 1  # carry one paragraph of context into the next chunk

_FRONTMATTER_RE = re.compile(r"\A﻿?---\r?\n.*?\r?\n---\r?\n", re.DOTALL)

# Header text (case-insensitive) that marks a section as personal job-hunt / career data
# rather than vendor intel. A matching header drops its section — the header line and every
# line until the next header of the SAME OR HIGHER level (or end of file).
_JOBHUNT_HEADER_TERMS = (
    "meddic",
    "interviewer",
    "stakeholder",
    "pipeline",
    "if i join",
    "later rounds",
    "job-hunt",
    "jobhunt",
    "recruiter",
    "application",
    "hiring",
    "my angle",
    "talking points",
)
_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$")

# Defense-in-depth CONTENT scrub: a chunk containing any of these personal markers is
# dropped from the index entirely, no matter which file it came from or whether section
# stripping missed it. Fail-closed — a false drop costs one chunk; a false keep leaks PII.
_PERSONAL_MARKERS = (
    "wehner",  # Eugen's employment lawyer — a private legal matter
    "aufhebungsvertrag",  # the exit/severance agreement — personal employment
    "meddic",  # sales-qualification framework used only in job-hunt notes here
    "stakeholder_sentiment",
    "if i join",
    "gold for later rounds",
)


class Chunk(BaseModel):
    """One retrievable passage. ``source`` + ``tag`` + ``category`` are carried so retrieval
    can cite and scope; ``text`` is what BM25 indexes. ``tag`` is the CONTENT type (playbook,
    pricing…); ``category`` is the PROCUREMENT category (cloud, hr…) detected from the text."""

    model_config = {"frozen": True}

    chunk_id: str  # "<source>#<n>"
    source: str  # the manifest-relative path it came from
    tag: str
    text: str
    category: str = "unknown"


def strip_frontmatter(text: str) -> str:
    """Remove a leading YAML frontmatter block (``---`` … ``---``). No-op if absent."""
    return _FRONTMATTER_RE.sub("", text, count=1)


def strip_jobhunt_sections(text: str) -> str:
    """Remove markdown sections whose header names personal job-hunt/career content.

    A matched header drops its whole section — the header and every following line until a
    header of the same or higher level (or EOF). Line-based so nested headers and multiple
    sections are handled correctly. Applied to flagged Companies/* profiles where vendor
    intel and career metadata are interleaved.
    """
    out: list[str] = []
    skip_above_level: int | None = None  # while set, drop lines until a header <= this level
    for line in text.splitlines():
        header = _HEADER_RE.match(line)
        if header:
            level = len(header.group(1))
            title = header.group(2).lower()
            if skip_above_level is not None and level <= skip_above_level:
                skip_above_level = None  # this header closes the dropped section
            if skip_above_level is None and any(t in title for t in _JOBHUNT_HEADER_TERMS):
                skip_above_level = level  # start dropping this section
                continue
        if skip_above_level is None:
            out.append(line)
    return "\n".join(out)


def is_personal(text: str) -> bool:
    """True if a chunk contains a personal/PII marker — the final content gate.

    Runs on EVERY chunk regardless of source, so a marker that slips past frontmatter and
    section stripping still cannot reach the index. Fail-closed by design.
    """
    low = text.lower()
    return any(marker in low for marker in _PERSONAL_MARKERS)


# Contact-detail patterns redacted from EVERY chunk. A supplier-facing agent has no reason
# to surface an email or phone number; the surrounding vendor intel is kept, the contact is
# replaced with a placeholder. Redact (not drop) so the useful passage survives.
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"\+\d{1,3}[\s./-]?(?:\(?\d{1,4}\)?[\s./-]?){2,5}\d")


def redact_contacts(text: str) -> str:
    """Replace emails and phone numbers with placeholders — no contact detail in the index."""
    text = _EMAIL_RE.sub("[email redacted]", text)
    return _PHONE_RE.sub("[phone redacted]", text)


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\r?\n\s*\r?\n", text) if p.strip()]


def chunk_text(text: str, *, source: str, tag: str, category: str = "unknown") -> list[Chunk]:
    """Pack paragraphs into ~``_CHUNK_CHARS`` passages with a one-paragraph overlap."""
    paras = _paragraphs(text)
    chunks: list[Chunk] = []
    buf: list[str] = []
    size = 0
    n = 0

    def flush() -> None:
        nonlocal buf, size, n
        if not buf:
            return
        chunks.append(
            Chunk(
                chunk_id=f"{source}#{n}",
                source=source,
                tag=tag,
                text="\n\n".join(buf),
                category=category,
            )
        )
        n += 1

    for para in paras:
        if buf and size + len(para) > _CHUNK_CHARS:
            flush()
            buf = buf[-_CHUNK_OVERLAP_PARAS:] if _CHUNK_OVERLAP_PARAS else []
            size = sum(len(p) for p in buf)
        buf.append(para)
        size += len(para)
    flush()
    return chunks


def _ingest_entry(entry: ManifestEntry, root: Path) -> list[Chunk]:
    file_path = root / entry.path
    if not file_path.is_file():
        return []  # listed-but-absent is skipped, not an error (vault differs per machine)
    raw = file_path.read_text(encoding="utf-8", errors="replace")
    # Frontmatter is metadata (ids, tags, provenance) never negotiation content — strip it
    # from EVERY file. The per-entry flag adds the heavier job-hunt SECTION scrub on top.
    body = strip_frontmatter(raw)
    if entry.scrub_personal:
        body = strip_jobhunt_sections(body)
    body = redact_contacts(body)  # strip emails/phones from every file
    # An authored playbook pins its category in the manifest; otherwise detect it once per
    # file (whole-doc signal beats a chunk's). Stamp it on every chunk so retrieval can scope.
    if entry.category:
        category = entry.category
    else:
        category, _ = detect_category(body)
    # Final content gate: drop any chunk that still carries a personal marker. Applied to
    # every file, so the scrub can't be bypassed by a mis-flagged entry.
    return [
        c
        for c in chunk_text(body, source=entry.path, tag=entry.tag, category=category)
        if not is_personal(c.text)
    ]


def ingest(manifest: Manifest, *, vault_root: Path | None, repo_root: Path) -> list[Chunk]:
    """Read every allowed entry into chunks. Vault entries need ``vault_root``; if it's
    None (the vault isn't present on this machine) they're skipped and only the repo-
    authored notes are ingested — the build still produces a valid, smaller index.
    """
    chunks: list[Chunk] = []
    if vault_root is not None:
        for entry in manifest.allowed_vault_entries():
            chunks.extend(_ingest_entry(entry, vault_root))
    for entry in manifest.allowed_repo_entries():
        chunks.extend(_ingest_entry(entry, repo_root))
    return chunks
