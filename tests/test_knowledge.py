"""The procurement knowledge base — the fail-closed gate, the scrub, and BM25 retrieval.

The security-critical properties come first: a path is ingestible ONLY if allowlisted AND
not excluded; frontmatter and personal sections are stripped; a chunk with a personal
marker never survives; nothing from a hard-exclude zone can be indexed. Then the retrieval
behaviour (ranking, tag scoping, graceful empty).
"""

from __future__ import annotations

from pathlib import Path

from negotiation_agent.knowledge.bm25 import Bm25Index, tokenize
from negotiation_agent.knowledge.ingest import (
    Chunk,
    chunk_text,
    is_personal,
    redact_contacts,
    strip_frontmatter,
    strip_jobhunt_sections,
)
from negotiation_agent.knowledge.manifest import Manifest, ManifestEntry


# ── the fail-closed gate ────────────────────────────────────────────────────────
def _manifest(**kw) -> Manifest:
    base = {
        "version": 1,
        "vault_include": [{"path": "Notes/pricing.md", "tag": "pricing"}],
        "exclude_globs": ["_private/**", "People/**", "**/*.confidential.md"],
    }
    base.update(kw)
    return Manifest.model_validate(base)


def test_excluded_glob_blocks_a_path():
    m = _manifest()
    assert m.is_excluded("_private/msa-pricing.md") is True
    assert m.is_excluded("People/Herr Wehner.md") is True
    assert m.is_excluded("Companies/x.confidential.md") is True
    assert m.is_excluded("Notes/pricing.md") is False


def test_excluded_matches_regardless_of_join_root():
    # a path joined to any root prefix must still be caught by a suffix glob
    m = _manifest()
    assert m.is_excluded("C:/vault/_private/x.md") is True
    assert m.is_excluded("some/deep/People/y.md") is True


def test_allowlisted_but_excluded_path_is_dropped():
    # defense-in-depth: a path on the allowlist that ALSO matches an exclude glob is removed
    m = _manifest(vault_include=[{"path": "_private/leak.md", "tag": "pricing"}])
    assert m.allowed_vault_entries() == []


def test_empty_allowlist_ingests_nothing():
    # fail-closed: no allowlist -> nothing allowed, never "everything"
    m = _manifest(vault_include=[])
    assert m.allowed_vault_entries() == []


# ── the scrub ────────────────────────────────────────────────────────────────────
def test_strip_frontmatter_removes_yaml_block():
    text = "---\ntitle: x\ntags: [job-hunt]\n---\n\n# Real content\nbody"
    out = strip_frontmatter(text)
    assert "job-hunt" not in out
    assert "# Real content" in out


def test_strip_frontmatter_noop_without_block():
    text = "# No frontmatter\njust body"
    assert strip_frontmatter(text) == text


def test_strip_jobhunt_sections_removes_meddic_section():
    text = (
        "# Vendor\n\nUseful intel.\n\n## MEDDIC gaps\n"
        "interviewer notes here\n\n## Pricing\nlist price info"
    )
    out = strip_jobhunt_sections(text)
    assert "interviewer notes" not in out
    assert "Useful intel." in out
    assert "list price info" in out


def test_redact_contacts_strips_email_and_phone():
    text = "Reach the rep at dach_sales@jaggaer.com or call +49 30 1234 5678 for a quote."
    out = redact_contacts(text)
    assert "@jaggaer.com" not in out
    assert "1234 5678" not in out
    assert "[email redacted]" in out
    assert "for a quote" in out  # surrounding vendor content survives


def test_is_personal_flags_lawyer_and_exit_markers():
    assert is_personal("handled via Eugen's lawyer Wehner") is True
    assert is_personal("pursuing an Aufhebungsvertrag") is True
    assert is_personal("standard payment terms and rebate") is False


def test_ingest_entry_drops_personal_chunks(tmp_path: Path):
    # a file whose body still carries a marker after stripping yields no chunk for it
    from negotiation_agent.knowledge.ingest import _ingest_entry

    f = tmp_path / "profile.md"
    f.write_text(
        "# Vendor\n\nGood pricing intel about discounts.\n\nExit handled by lawyer Wehner.\n",
        encoding="utf-8",
    )
    entry = ManifestEntry(path="profile.md", tag="supplier-profile", scrub_personal=True)
    chunks = _ingest_entry(entry, tmp_path)
    joined = " ".join(c.text.lower() for c in chunks)
    assert "wehner" not in joined


# ── BM25 retrieval ──────────────────────────────────────────────────────────────
def _index() -> Bm25Index:
    chunks = [
        Chunk(
            chunk_id="a#0",
            source="a.md",
            tag="negotiation-strategy",
            text="Your BATNA is your best alternative and sets the reservation walk-away point.",
        ),
        Chunk(
            chunk_id="b#0",
            source="b.md",
            tag="pricing",
            text="Multi-year commitments unlock tiered discounts on a SaaS renewal.",
        ),
        Chunk(
            chunk_id="c#0",
            source="c.md",
            tag="pricing",
            text="Payment terms and rebates are levers to trade against headline price.",
        ),
    ]
    return Bm25Index.build(chunks)


def test_tokenize_is_lowercase_alphanumeric():
    assert tokenize("Multi-Year, 20% Discount!") == ["multi", "year", "20", "discount"]


def test_query_ranks_relevant_chunk_first():
    idx = _index()
    hits = idx.query("what is my BATNA walk-away point", top_k=1)
    assert hits[0].source == "a.md"


def test_query_tag_scopes_retrieval():
    idx = _index()
    hits = idx.query("discount price lever", tag="pricing", top_k=5)
    assert hits  # found something
    assert all(h.tag == "pricing" for h in hits)  # never a negotiation-strategy chunk


def test_query_empty_on_no_match():
    idx = _index()
    assert idx.query("zzzxqq wobblefram nonexistentword", top_k=3) == []


def test_query_deterministic_ranking():
    idx = _index()
    first = idx.query("discount price", top_k=3)
    second = idx.query("discount price", top_k=3)
    assert [h.chunk_id for h in first] == [h.chunk_id for h in second]


def test_index_roundtrips_through_dict():
    idx = _index()
    restored = Bm25Index.from_dict(idx.to_dict())
    assert restored.query("BATNA", top_k=1)[0].source == "a.md"


def test_chunk_text_packs_and_ids_sequentially():
    # each para ~120 chars; 20 of them exceed the ~1100-char chunk budget -> multiple chunks
    para = "This paragraph carries enough words and characters to push the packer past budget."
    text = "\n\n".join(f"{para} Item {i}." for i in range(20))
    chunks = chunk_text(text, source="x.md", tag="playbook")
    assert len(chunks) >= 2
    assert chunks[0].chunk_id == "x.md#0"
    assert chunks[1].chunk_id == "x.md#1"
