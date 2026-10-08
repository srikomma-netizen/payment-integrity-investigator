"""Hybrid policy retrieval: BM25 + vector candidates, RRF fusion, metadata filter,
rerank, source validation, then context expansion.

Each stage writes to RetrievalResult.trace so retrieval misses can be told apart from generation misses.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Protocol

from .ingest import Chunk, ingest

ACCESS_RANK = {"standard": 0, "restricted": 1}
ROLE_ACCESS = {"analyst": 0, "investigator": 1, "siu_lead": 1, "auditor": 1}   # unknown roles fall back to 0
_TOKEN = re.compile(r"[a-z0-9]+")   # keeps CPT codes like 99215 as single tokens for BM25
_STOP = {"the", "a", "an", "and", "or", "of", "to", "is", "in", "for", "on", "by", "with", "are", "be", "as", "at", "any"}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


# Sparse dict vectors so the hashed embedding stays cheap; a dense provider would need cosine() reworked.
class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> dict[int, float]: ...


class HashedNgramEmbedding:
    """Deterministic sparse embedding over character trigrams of tokens.
    Captures morphology ("unbundled" ~ "unbundling") without a model, and runs offline."""

    # TODO: replace hashed n-grams with a real embedding model behind EmbeddingProvider

    def __init__(self, dims: int = 4096):
        self.dims = dims

    def embed(self, text: str) -> dict[int, float]:
        vec: Counter = Counter()
        for tok in tokenize(text):
            padded = f"#{tok}#"   # boundary markers so prefixes/suffixes get their own trigrams
            for i in range(len(padded) - 2):
                # blake2b, not hash(): Python's str hash is salted per process and would change vectors run to run
                h = int(hashlib.blake2b(padded[i:i + 3].encode(), digest_size=4).hexdigest(), 16) % self.dims
                vec[h] += 1.0
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {k: v / norm for k, v in vec.items()}


def cosine(a: dict[int, float], b: dict[int, float]) -> float:
    # vectors are already L2-normalised, so the dot product is the cosine; iterate the smaller one
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())


class BM25:
    def __init__(self, docs: dict[str, str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = {i: Counter(tokenize(t)) for i, t in docs.items()}
        self.len = {i: sum(c.values()) for i, c in self.tf.items()}
        self.avg = (sum(self.len.values()) / len(self.len)) if self.len else 1.0
        df: Counter = Counter()
        for c in self.tf.values():
            df.update(c.keys())
        n = len(docs)
        # the +1 inside the log (Lucene-style) keeps idf positive for terms in over half the docs
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def score(self, query: str, doc_id: str) -> float:
        s, tf, dl = 0.0, self.tf[doc_id], self.len[doc_id]
        for q in tokenize(query):
            if q in tf:
                f = tf[q]
                s += self.idf[q] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avg))
        return s


@dataclass
class RetrievalFilter:
    role: str = "analyst"   # least-privileged default
    as_of: str = "2025-12-31"   # pinned rather than today() so results are reproducible
    claim_type: str | None = None
    require_approved: bool = True


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    bm25_rank: int | None = None
    vector_rank: int | None = None
    rerank_score: float = 0.0
    expanded_from: str | None = None

    def to_dict(self) -> dict:
        c = self.chunk
        return {"chunk_id": c.chunk_id, "citation": c.citation, "policy_id": c.policy_id, "version": c.version,
                "section_id": c.section_id, "section_title": c.section_title, "text": c.text,
                "score": round(self.score, 4), "expanded_from": self.expanded_from}


@dataclass
class RetrievalResult:
    query: str
    chunks: list[RetrievedChunk]
    rejected: list[tuple[str, str]] = field(default_factory=list)   # (chunk_id, reason)
    trace: list[str] = field(default_factory=list)

    def ids(self) -> list[str]:
        return [r.chunk.chunk_id for r in self.chunks]


class HybridIndex:
    def __init__(self, chunks: list[Chunk] | None = None, embedder: EmbeddingProvider | None = None):
        self.chunks: dict[str, Chunk] = {c.chunk_id: c for c in (chunks or ingest())}
        self.embedder = embedder or HashedNgramEmbedding()
        # index titles along with the body; a part-2 chunk often doesn't repeat the section's topic words
        docs = {cid: f"{c.policy_title} {c.section_title}\n{c.text}" for cid, c in self.chunks.items()}
        self.bm25 = BM25(docs)
        self.vectors = {cid: self.embedder.embed(t) for cid, t in docs.items()}
        # latest approved version per policy, for "current" validation
        self.current_version: dict[str, int] = {}
        for c in self.chunks.values():
            if c.status == "approved":
                self.current_version[c.policy_id] = max(self.current_version.get(c.policy_id, 0), c.version)

    # ---- stages -----------------------------------------------------------
    def _filter_reason(self, c: Chunk, flt: RetrievalFilter) -> str | None:
        """Why this chunk can't be used, or None if it can. Reasons are kept for the trace, not just dropped."""
        if flt.require_approved and c.status != "approved":
            return f"status={c.status}"   # drafts/retired versions must never be cited
        if c.effective_date > flt.as_of:   # ISO date strings, so string compare is safe
            return f"not effective until {c.effective_date}"
        # current_version only counts approved versions, so a newer draft never supersedes an approved one
        if c.status == "approved" and c.version < self.current_version.get(c.policy_id, c.version):
            return f"superseded by v{self.current_version[c.policy_id]}"
        if ACCESS_RANK.get(c.access_level, 0) > ROLE_ACCESS.get(flt.role, 0):
            return f"access_level={c.access_level} exceeds role {flt.role}"
        # chunks with no claim_types apply to everything
        if flt.claim_type and c.claim_types and flt.claim_type not in c.claim_types:
            return f"claim_type {flt.claim_type} not in {c.claim_types}"
        return None

    def _rerank(self, query: str, c: Chunk) -> float:
        # Cheap stand-in for a cross-encoder: query-term coverage plus a bonus for exact code matches.
        q = set(tokenize(query))
        if not q:
            return 0.0
        text_tokens = set(tokenize(f"{c.section_title} {c.text}"))
        overlap = len(q & text_tokens) / len(q)
        codes = {t for t in q if t.isdigit() and len(t) >= 4}   # CPT-like codes; shorter numbers are usually section refs
        code_hit = 0.25 if codes and codes & text_tokens else 0.0
        return min(1.0, overlap + code_hit)

    def retrieve(self, query: str, flt: RetrievalFilter | None = None, *, k: int = 4,
                 candidates: int = 12, expand: bool = True) -> RetrievalResult:
        flt = flt or RetrievalFilter()
        trace: list[str] = []
        # Both retrievers run over the whole corpus and filtering happens after fusion, so the trace can
        # show which good matches were rejected and why. Fine at this size; pre-filter if the corpus grows.
        # BM25 catches exact codes/ids, the vector side catches paraphrase.
        bm25_ranked = sorted(self.chunks, key=lambda cid: -self.bm25.score(query, cid))[:candidates]
        qv = self.embedder.embed(query)
        vec_ranked = sorted(self.chunks, key=lambda cid: -cosine(qv, self.vectors[cid]))[:candidates]
        trace.append(f"bm25 top: {bm25_ranked[:3]}")
        trace.append(f"vector top: {vec_ranked[:3]}")

        # Reciprocal rank fusion: uses ranks only, so BM25 and cosine scores never need calibrating
        # against each other. k=60 is the usual constant; it flattens the gap between rank 1 and rank 5.
        rrf: dict[str, float] = defaultdict(float)
        ranks: dict[str, dict] = defaultdict(dict)
        for i, cid in enumerate(bm25_ranked):
            rrf[cid] += 1 / (60 + i); ranks[cid]["bm25"] = i + 1
        for i, cid in enumerate(vec_ranked):
            rrf[cid] += 1 / (60 + i); ranks[cid]["vec"] = i + 1

        rejected: list[tuple[str, str]] = []
        kept: list[RetrievedChunk] = []
        for cid, fused in rrf.items():
            c = self.chunks[cid]
            reason = self._filter_reason(c, flt)
            if reason:
                rejected.append((cid, reason))
                continue
            rr = self._rerank(query, c)
            # fused * 60 rescales RRF to roughly 0..2 (1.0 per retriever at rank 1) so it's comparable to rr in 0..1
            kept.append(RetrievedChunk(c, score=0.5 * fused * 60 + 0.5 * rr, bm25_rank=ranks[cid].get("bm25"),
                                       vector_rank=ranks[cid].get("vec"), rerank_score=rr))
        kept.sort(key=lambda r: -r.score)
        top = kept[:k]
        trace.append(f"filtered out {len(rejected)}; kept {len(kept)}; top {[r.chunk.chunk_id for r in top]}")

        # defense in depth: validate again right before anything is handed to the model
        # (redundant today, but it guards against a future stage adding chunks after the filter)
        validated = [r for r in top if self._filter_reason(r.chunk, flt) is None]

        if expand:
            # Pull in sibling parts of the same section and the parent section's opening chunk,
            # so the model reads a coherent section instead of a mid-paragraph fragment.
            seen = {r.chunk.chunk_id for r in validated}
            extra: list[RetrievedChunk] = []
            for r in validated:
                c = r.chunk
                for other in self.chunks.values():
                    # match on version too, or expansion could splice v1 text into a v2 answer
                    same_section_part = other.policy_id == c.policy_id and other.version == c.version and other.section_id == c.section_id
                    is_parent = c.parent_section and other.policy_id == c.policy_id and other.version == c.version \
                        and other.section_id == c.parent_section and other.part == 0
                    # expanded chunks go through the same filter: expansion must not be a side door around it
                    if (same_section_part or is_parent) and other.chunk_id not in seen and self._filter_reason(other, flt) is None:
                        seen.add(other.chunk_id)
                        # half the source score so expanded context never outranks what actually matched
                        extra.append(RetrievedChunk(other, score=r.score * 0.5, expanded_from=c.chunk_id))
            validated.extend(extra[: k])   # cap expansion so a long section can't flood the context
            trace.append(f"expanded +{min(len(extra), k)}")
        return RetrievalResult(query, validated, rejected, trace)
