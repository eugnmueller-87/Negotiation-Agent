"""A dependency-free BM25 retriever over the ingested chunks.

BM25 is the standard lexical ranking function: it scores a chunk for a query by summing,
over the query terms, an IDF weight times a saturated term-frequency factor (so a term
appearing 10x isn't 10x as important) normalized by chunk length (so a long chunk isn't
favoured just for being long). Pure Python, ~no deps, fully deterministic — the same query
always returns the same ranking, which fits the engine's reproducibility guarantee.

The index serializes to JSON (``build`` → dict → ``data/kb_index.json``); the server loads
it read-only via ``from_dict`` and calls ``query``.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from pydantic import BaseModel

from negotiation_agent.knowledge.ingest import Chunk

# BM25 free parameters — the literature-standard defaults.
_K1 = 1.5  # term-frequency saturation
_B = 0.75  # length-normalization strength

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.ASCII)


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens. Deliberately simple and language-agnostic so the
    German + English content in the vault tokenizes the same way."""
    return _TOKEN_RE.findall(text.lower())


class Hit(BaseModel):
    model_config = {"frozen": True}

    chunk_id: str
    source: str
    tag: str
    text: str
    score: float


class Bm25Index(BaseModel):
    """A serializable BM25 index. Built offline from chunks; queried in-process."""

    model_config = {"frozen": True}

    chunks: list[Chunk]
    doc_tokens: list[list[str]]  # tokenized text, parallel to chunks
    doc_freq: dict[str, int]  # term -> number of chunks containing it
    avg_len: float
    n_docs: int

    @classmethod
    def build(cls, chunks: list[Chunk]) -> Bm25Index:
        doc_tokens = [tokenize(c.text) for c in chunks]
        df: Counter[str] = Counter()
        for toks in doc_tokens:
            for term in set(toks):
                df[term] += 1
        n = len(chunks)
        avg = (sum(len(t) for t in doc_tokens) / n) if n else 0.0
        return cls(
            chunks=chunks,
            doc_tokens=doc_tokens,
            doc_freq=dict(df),
            avg_len=avg,
            n_docs=n,
        )

    def _idf(self, term: str) -> float:
        # BM25 IDF with the standard +0.5 smoothing; floored at 0 so a term present in
        # nearly every chunk can't push a score negative.
        df = self.doc_freq.get(term, 0)
        return max(0.0, math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0))

    def query(self, text: str, *, top_k: int = 5, tag: str | None = None) -> list[Hit]:
        """Rank chunks for ``text``. ``tag`` scopes retrieval to one topic (e.g. only
        pricing chunks). Returns the top ``top_k`` by score, ties broken by chunk_id for
        determinism."""
        q_terms = tokenize(text)
        if not q_terms or self.n_docs == 0:
            return []
        q_idf = {t: self._idf(t) for t in set(q_terms)}

        scored: list[tuple[float, int]] = []
        for i, (chunk, toks) in enumerate(zip(self.chunks, self.doc_tokens, strict=True)):
            if tag is not None and chunk.tag != tag:
                continue
            if not toks:
                continue
            tf = Counter(toks)
            dl = len(toks)
            score = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if f == 0:
                    continue
                num = f * (_K1 + 1)
                den = f + _K1 * (1 - _B + _B * dl / self.avg_len)
                score += q_idf[term] * num / den
            if score > 0.0:
                scored.append((score, i))

        # sort by score desc, then chunk_id asc for a stable, reproducible ranking
        scored.sort(key=lambda si: (-si[0], self.chunks[si[1]].chunk_id))
        return [
            Hit(
                chunk_id=self.chunks[i].chunk_id,
                source=self.chunks[i].source,
                tag=self.chunks[i].tag,
                text=self.chunks[i].text,
                score=round(s, 4),
            )
            for s, i in scored[:top_k]
        ]

    def to_dict(self) -> dict[str, object]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Bm25Index:
        return cls.model_validate(data)
