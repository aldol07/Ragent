"""Hybrid retrieval: dense (FAISS, cosine) + sparse (BM25), fused with RRF,
optionally reranked by a cross-encoder."""
import math
import re
from collections import Counter

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from config import CANDIDATE_POOL, RRF_K, TOP_K_RESULTS

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "is", "are", "was", "were",
    "be", "by", "with", "as", "at", "that", "this", "it", "from", "what", "which", "how", "does",
}


def tokenize(text):
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


class Models:
    """Heavy models, loaded once per process and shared by every session's index."""

    def __init__(self, embedding_model, reranker_model=None):
        from sentence_transformers import CrossEncoder, SentenceTransformer

        self.embedder = SentenceTransformer(embedding_model)
        self.reranker = CrossEncoder(reranker_model) if reranker_model else None

    def embed(self, texts):
        vecs = self.embedder.encode(
            texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vecs, dtype="float32")


class HybridRetriever:
    """Per-session index over a list of Chunk objects."""

    def __init__(self, models, chunks):
        if not chunks:
            raise ValueError("Cannot build an index with no chunks.")
        self.models = models
        self.chunks = chunks
        vecs = models.embed([c.text for c in chunks])
        self.index = faiss.IndexFlatIP(vecs.shape[1])  # inner product on unit vectors = cosine
        self.index.add(vecs)
        self.bm25 = BM25Okapi([tokenize(c.text) or ["_"] for c in chunks])

    # -- info -------------------------------------------------------------
    @property
    def sources(self):
        counts = Counter(c.source for c in self.chunks)
        pages = {}
        for c in self.chunks:
            pages[c.source] = max(pages.get(c.source, 0), c.page)
        return [{"source": s, "chunks": n, "pages": pages[s]} for s, n in counts.items()]

    # -- individual retrievers --------------------------------------------
    def _allowed(self, source):
        if not source:
            return None
        allowed = {i for i, c in enumerate(self.chunks) if c.source.lower() == source.lower()}
        return allowed or None

    def dense(self, query, n, allowed=None):
        qv = self.models.embed([query])
        search_n = len(self.chunks) if allowed else min(n, len(self.chunks))
        scores, ids = self.index.search(qv, search_n)
        out = [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]
        if allowed:
            out = [(i, s) for i, s in out if i in allowed]
        return out[:n]

    def sparse(self, query, n, allowed=None):
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(-scores)
        out = [(int(i), float(scores[i])) for i in order if scores[i] > 0]
        if allowed:
            out = [(i, s) for i, s in out if i in allowed]
        return out[:n]

    # -- main entry point -------------------------------------------------
    def search(self, query, k=TOP_K_RESULTS, mode="hybrid", rerank=True, source=None):
        """Return up to k results: [{"chunk", "score", "retriever"}], best first.

        mode: "dense" | "bm25" | "hybrid". rerank applies the cross-encoder
        (if loaded) to the fused candidate pool.
        """
        allowed = self._allowed(source)
        pool = max(CANDIDATE_POOL, k)

        if mode == "dense":
            ranked = self.dense(query, pool, allowed)
        elif mode == "bm25":
            ranked = self.sparse(query, pool, allowed)
        else:
            fused = {}
            for results in (self.dense(query, pool, allowed), self.sparse(query, pool, allowed)):
                for rank, (i, _) in enumerate(results):
                    fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
            ranked = sorted(fused.items(), key=lambda x: -x[1])[:pool]

        label = mode
        if rerank and self.models.reranker is not None and ranked:
            ids = [i for i, _ in ranked]
            logits = self.models.reranker.predict([(query, self.chunks[i].text) for i in ids])
            ranked = sorted(
                ((i, 1 / (1 + math.exp(-float(s)))) for i, s in zip(ids, logits)),
                key=lambda x: -x[1],
            )
            label += "+rerank"

        return [
            {"chunk": self.chunks[i], "score": round(s, 4), "retriever": label}
            for i, s in ranked[:k]
        ]
