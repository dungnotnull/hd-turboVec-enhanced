"""
Hybrid Search Module — Dense vector search + BM25 sparse retrieval with RRF fusion.

Retrieval strategy:
  Dense:  HNSW (hnswlib) with BGE-large-en-v1.5 embeddings
  Sparse: BM25Okapi (rank_bm25 library) with NLTK tokenization
  Fusion: Reciprocal Rank Fusion (RRF), k=60 — no hyperparameter tuning required

Quality gate: RRF improves recall@10 vs dense-only on ≥ 3 BEIR datasets.
"""
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

RRF_K = 60  # Standard RRF constant (Cormack et al., 2009)


@dataclass
class HybridResult:
    id: str
    dense_score: float
    sparse_score: float
    rrf_score: float
    text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "score": self.rrf_score,
            "dense_score": self.dense_score,
            "sparse_score": self.sparse_score,
            "text": self.text,
            "metadata": self.metadata,
        }


class HybridSearch:
    """
    Hybrid dense+sparse search with Reciprocal Rank Fusion.

    Args:
        hnsw: HNSWSearch instance for dense retrieval
        hf: HFModelManager for embedding queries
        bm25_k1: BM25 term frequency saturation parameter (default 1.5)
        bm25_b: BM25 length normalization parameter (default 0.75)
        rrf_k: RRF constant k (default 60)
    """

    def __init__(
        self,
        hnsw: Any,
        hf: Any,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        rrf_k: int = RRF_K,
    ):
        self.hnsw = hnsw
        self.hf = hf
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.rrf_k = rrf_k

        self._bm25_index: Optional[Any] = None
        self._corpus_ids: List[str] = []
        self._corpus_texts: List[str] = []
        self._tokenizer = self._init_tokenizer()

    def _init_tokenizer(self):
        """Initialize NLTK tokenizer with stopword removal."""
        try:
            import nltk
            try:
                stopwords = set(nltk.corpus.stopwords.words("english"))
            except LookupError:
                nltk.download("stopwords", quiet=True)
                stopwords = set(nltk.corpus.stopwords.words("english"))
            return lambda text: [
                w.lower() for w in re.findall(r"\b\w+\b", text)
                if w.lower() not in stopwords and len(w) > 1
            ]
        except ImportError:
            logger.warning("nltk not installed; using simple tokenizer")
            return lambda text: re.findall(r"\b\w+\b", text.lower())

    # ──────────────────────── Indexing ────────────────────────

    def index(self, documents: List[str], embeddings: np.ndarray, ids: List[str]):
        """
        Build BM25 sparse index and insert into HNSW dense index.

        Args:
            documents: Raw text of each document chunk
            embeddings: Shape (N, dim) — pre-computed embeddings
            ids: List of N string IDs (must match documents and embeddings)
        """
        if len(documents) != len(ids) or len(documents) != len(embeddings):
            raise ValueError("documents, embeddings, and ids must have same length")

        self._corpus_ids = list(ids)
        self._corpus_texts = list(documents)

        tokenized = [self._tokenizer(doc) for doc in documents]
        try:
            from rank_bm25 import BM25Okapi
            self._bm25_index = BM25Okapi(
                tokenized,
                k1=self.bm25_k1,
                b=self.bm25_b,
            )
        except ImportError:
            logger.warning("rank_bm25 not installed; BM25 disabled (dense-only mode)")
            self._bm25_index = None

        self.hnsw.add(embeddings=embeddings, ids=ids)
        logger.info("Hybrid index built: %d documents", len(documents))

    def add(self, documents: List[str], embeddings: np.ndarray, ids: List[str]):
        """Incrementally add documents to the hybrid index."""
        if self._bm25_index is None:
            self.index(documents=documents, embeddings=embeddings, ids=ids)
            return

        self._corpus_ids.extend(ids)
        self._corpus_texts.extend(documents)

        tokenized_existing = [self._tokenizer(d) for d in self._corpus_texts[:-len(documents)]]
        tokenized_new = [self._tokenizer(d) for d in documents]
        all_tokenized = tokenized_existing + tokenized_new

        try:
            from rank_bm25 import BM25Okapi
            self._bm25_index = BM25Okapi(all_tokenized, k1=self.bm25_k1, b=self.bm25_b)
        except ImportError:
            pass

        self.hnsw.add(embeddings=embeddings, ids=ids)

    # ──────────────────────── Search ────────────────────────

    def search(
        self,
        query: str,
        query_embedding: Optional[np.ndarray] = None,
        k: int = 10,
        dense_k: int = 50,
        sparse_k: int = 50,
        alpha: float = 0.5,
        return_scores: bool = True,
    ) -> List[HybridResult]:
        """
        Hybrid search: dense + sparse retrieval fused via RRF.

        Args:
            query: Query text string
            query_embedding: Pre-computed query embedding (computed if None)
            k: Final number of results to return after fusion
            dense_k: Number of dense candidates to retrieve before fusion
            sparse_k: Number of sparse candidates to retrieve before fusion
            alpha: Weight for alpha interpolation (alternative to RRF, not default)
            return_scores: Include individual dense/sparse scores in results

        Returns:
            List of HybridResult sorted by RRF score descending
        """
        if query_embedding is None:
            query_embedding = self.hf.encode(query)

        dense_results = self._dense_search(query_embedding, k=dense_k)
        sparse_results = self._sparse_search(query, k=sparse_k)

        fused = self._rrf_fusion(dense_results, sparse_results, k=k)
        return fused

    def _dense_search(self, query_embedding: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """Dense HNSW search. Returns list of (id, distance)."""
        try:
            result = self.hnsw.search(query_embedding=query_embedding, k=k)
            return list(zip(result.ids, result.distances))
        except Exception as exc:
            logger.error("Dense search failed: %s", exc)
            return []

    def _sparse_search(self, query: str, k: int) -> List[Tuple[str, float]]:
        """BM25 sparse search. Returns list of (id, score)."""
        if self._bm25_index is None or not self._corpus_ids:
            return []
        try:
            tokens = self._tokenizer(query)
            scores = self._bm25_index.get_scores(tokens)
            top_k = min(k, len(scores))
            top_idx = np.argsort(-scores)[:top_k]
            return [
                (self._corpus_ids[i], float(scores[i]))
                for i in top_idx
                if scores[i] > 0
            ]
        except Exception as exc:
            logger.error("BM25 search failed: %s", exc)
            return []

    def _rrf_fusion(
        self,
        dense: List[Tuple[str, float]],
        sparse: List[Tuple[str, float]],
        k: int,
    ) -> List[HybridResult]:
        """
        Reciprocal Rank Fusion: score(d) = Σ 1/(rrf_k + rank_i(d))

        Each candidate that appears in either list gets an RRF score.
        Candidates only in one list still receive their single-list contribution.
        """
        rrf_scores: Dict[str, float] = {}
        dense_scores: Dict[str, float] = {}
        sparse_scores: Dict[str, float] = {}

        for rank, (doc_id, dist) in enumerate(dense):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            dense_scores[doc_id] = float(dist)

        for rank, (doc_id, score) in enumerate(sparse):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            sparse_scores[doc_id] = float(score)

        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:k]

        results = []
        for doc_id in sorted_ids:
            idx = self._corpus_ids.index(doc_id) if doc_id in self._corpus_ids else -1
            text = self._corpus_texts[idx] if idx >= 0 else None
            results.append(HybridResult(
                id=doc_id,
                dense_score=dense_scores.get(doc_id, 0.0),
                sparse_score=sparse_scores.get(doc_id, 0.0),
                rrf_score=rrf_scores[doc_id],
                text=text,
            ))
        return results

    # ──────────────────────── Stats ────────────────────────

    def get_stats(self) -> Dict:
        return {
            "corpus_size": len(self._corpus_ids),
            "bm25_available": self._bm25_index is not None,
            "hnsw_stats": self.hnsw.get_stats(),
            "rrf_k": self.rrf_k,
            "bm25_k1": self.bm25_k1,
            "bm25_b": self.bm25_b,
        }

    def clear(self):
        """Reset the hybrid index."""
        self._bm25_index = None
        self._corpus_ids = []
        self._corpus_texts = []
        from agent.modules.hnsw_search import HNSWSearch
        stats = self.hnsw.get_stats()
        self.hnsw._init_index()
        logger.info("Hybrid index cleared")
