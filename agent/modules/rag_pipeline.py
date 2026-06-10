"""
RAG Pipeline Module — End-to-end Retrieval-Augmented Generation.

Pipeline:
  1. Ingest: chunk documents (fixed | sentence | semantic) → embed → index
  2. Query: retrieve (hybrid dense+BM25) → rerank (BGE cross-encoder) → generate (LLM)

Quality gate: NDCG@10 ≥ 0.65 on BEIR/NQ with hybrid + reranker pipeline.
"""
import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64
DEFAULT_SEMANTIC_THRESHOLD = 0.75


@dataclass
class Document:
    text: str
    id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = hashlib.md5(self.text.encode()).hexdigest()[:12]


@dataclass
class Citation:
    chunk_id: str
    text_excerpt: str
    score: float


@dataclass
class RAGResponse:
    answer: str
    citations: List[Citation]
    results: List[Dict]
    latency_ms: float
    cost_usd: float = 0.0


RAG_SYSTEM_PROMPT = """You are a precise AI assistant. Answer the user's question based ONLY on the provided context.
Include a citation [N] after each claim, where N is the chunk number provided.
If the context does not contain enough information to answer, say "I don't have enough information in the provided context."
Do not use prior knowledge beyond the context."""


class RAGPipeline:
    """
    End-to-end RAG pipeline with configurable chunking, hybrid retrieval, reranking, and LLM generation.

    Args:
        hybrid: HybridSearch instance
        hf: HFModelManager for embedding
        llm: LLMClient for answer generation
        memory: MemoryManager for logging
        config: Runtime configuration dict
    """

    def __init__(self, hybrid: Any, hf: Any, llm: Any, memory: Any, config: Dict = None):
        self.hybrid = hybrid
        self.hf = hf
        self.llm = llm
        self.memory = memory
        self.config = config or {}
        self._collections: Dict[str, List[Document]] = {}

    # ──────────────────────── Ingestion ────────────────────────

    def ingest(
        self,
        documents: List[str],
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[Dict]] = None,
        chunk_strategy: str = "semantic",
        collection: str = "default",
    ) -> Dict[str, Any]:
        """
        Chunk, embed, and index a list of raw document texts.

        Args:
            documents: List of raw document strings
            ids: Optional list of document IDs
            metadatas: Optional metadata per document
            chunk_strategy: 'fixed' | 'sentence' | 'semantic'
            collection: Index collection name

        Returns:
            dict with chunks_added, time_ms, collection
        """
        t0 = time.perf_counter()
        chunks: List[Document] = []

        for i, doc_text in enumerate(documents):
            doc_id = ids[i] if ids else f"doc_{i}"
            meta = metadatas[i] if metadatas else {}
            doc_chunks = self._chunk(doc_text, doc_id=doc_id, strategy=chunk_strategy, metadata=meta)
            chunks.extend(doc_chunks)

        if not chunks:
            return {"chunks_added": 0, "collection": collection}

        texts = [c.text for c in chunks]
        chunk_ids = [c.id for c in chunks]

        embeddings = self.hf.encode_batch(texts)

        self.hybrid.index(documents=texts, embeddings=embeddings, ids=chunk_ids)

        if collection not in self._collections:
            self._collections[collection] = []
        self._collections[collection].extend(chunks)

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("Ingested %d docs → %d chunks in %.0fms (strategy=%s)",
                    len(documents), len(chunks), elapsed, chunk_strategy)
        return {
            "chunks_added": len(chunks),
            "documents_ingested": len(documents),
            "collection": collection,
            "chunk_strategy": chunk_strategy,
            "time_ms": round(elapsed, 1),
        }

    def _chunk(
        self,
        text: str,
        doc_id: str,
        strategy: str = "semantic",
        metadata: Dict = None,
    ) -> List[Document]:
        """Dispatch to the appropriate chunking strategy."""
        metadata = metadata or {}
        if strategy == "fixed":
            return self._chunk_fixed(text, doc_id=doc_id, metadata=metadata)
        elif strategy == "sentence":
            return self._chunk_sentence(text, doc_id=doc_id, metadata=metadata)
        else:
            return self._chunk_semantic(text, doc_id=doc_id, metadata=metadata)

    def _chunk_fixed(self, text: str, doc_id: str, metadata: Dict, size: int = None, overlap: int = None) -> List[Document]:
        """Fixed-size token-based chunking with overlap."""
        size = size or self.config.get("chunk_size", DEFAULT_CHUNK_SIZE)
        overlap = overlap or self.config.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)
        words = text.split()
        chunks = []
        start = 0
        i = 0
        while start < len(words):
            end = min(start + size, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append(Document(
                text=chunk_text,
                id=f"{doc_id}_chunk{i}",
                metadata={**metadata, "doc_id": doc_id, "chunk_idx": i, "strategy": "fixed"},
            ))
            i += 1
            start += size - overlap
            if end == len(words):
                break
        return chunks

    def _chunk_sentence(self, text: str, doc_id: str, metadata: Dict) -> List[Document]:
        """Sentence-based chunking with a target sentence count per chunk."""
        try:
            import nltk
            try:
                sentences = nltk.sent_tokenize(text)
            except LookupError:
                nltk.download("punkt", quiet=True)
                sentences = nltk.sent_tokenize(text)
        except ImportError:
            sentences = re.split(r'(?<=[.!?])\s+', text)

        group_size = self.config.get("sentences_per_chunk", 5)
        chunks = []
        for i in range(0, len(sentences), group_size):
            group = sentences[i:i + group_size]
            chunk_text = " ".join(group)
            if not chunk_text.strip():
                continue
            chunks.append(Document(
                text=chunk_text,
                id=f"{doc_id}_sent{i // group_size}",
                metadata={**metadata, "doc_id": doc_id, "chunk_idx": i // group_size, "strategy": "sentence"},
            ))
        return chunks

    def _chunk_semantic(self, text: str, doc_id: str, metadata: Dict) -> List[Document]:
        """
        Semantic chunking: split at points where adjacent sentence embeddings have low similarity.
        Falls back to sentence chunking if embedding fails.
        """
        try:
            import nltk
            try:
                sentences = nltk.sent_tokenize(text)
            except LookupError:
                nltk.download("punkt", quiet=True)
                sentences = nltk.sent_tokenize(text)
        except ImportError:
            sentences = re.split(r'(?<=[.!?])\s+', text)

        if len(sentences) <= 3:
            return self._chunk_sentence(text, doc_id=doc_id, metadata=metadata)

        try:
            embeddings = self.hf.encode_batch(sentences)
            threshold = self.config.get("semantic_threshold", DEFAULT_SEMANTIC_THRESHOLD)
            split_points = [0]
            for i in range(1, len(embeddings)):
                sim = float(np.dot(embeddings[i - 1], embeddings[i]))
                if sim < threshold:
                    split_points.append(i)
            split_points.append(len(sentences))

            chunks = []
            for j in range(len(split_points) - 1):
                group = sentences[split_points[j]:split_points[j + 1]]
                chunk_text = " ".join(group)
                if not chunk_text.strip():
                    continue
                chunks.append(Document(
                    text=chunk_text,
                    id=f"{doc_id}_sem{j}",
                    metadata={**metadata, "doc_id": doc_id, "chunk_idx": j, "strategy": "semantic"},
                ))
            return chunks if chunks else self._chunk_sentence(text, doc_id=doc_id, metadata=metadata)
        except Exception as exc:
            logger.warning("Semantic chunking failed (%s); falling back to sentence chunking", exc)
            return self._chunk_sentence(text, doc_id=doc_id, metadata=metadata)

    # ──────────────────────── Query ────────────────────────

    def query(
        self,
        question: str,
        k: int = 10,
        hybrid: bool = True,
        rerank: bool = True,
        generate_answer: bool = True,
        collection: str = "default",
    ) -> Dict[str, Any]:
        """
        Retrieve relevant chunks and optionally generate an LLM-synthesized answer.

        Args:
            question: User's query string
            k: Final number of results to return
            hybrid: Use hybrid dense+BM25 retrieval (True) or dense-only (False)
            rerank: Apply BGE cross-encoder reranking
            generate_answer: Call LLM to synthesize answer from context
            collection: Index collection name

        Returns:
            RAGResponse-compatible dict
        """
        t0 = time.perf_counter()
        cost_usd = 0.0

        query_emb = self.hf.encode(question)

        if hybrid:
            raw_results = self.hybrid.search(
                query=question,
                query_embedding=query_emb,
                k=50,
            )
        else:
            sr = self.hybrid.hnsw.search(query_embedding=query_emb, k=50)
            from agent.modules.hybrid_search import HybridResult
            raw_results = [
                HybridResult(id=rid, dense_score=dist, sparse_score=0.0, rrf_score=1.0 / (i + 1),
                             text=self._get_text(rid, collection))
                for i, (rid, dist) in enumerate(zip(sr.ids, sr.distances))
            ]

        if rerank and raw_results:
            raw_results = self._rerank(question, raw_results, top_k=k)
        else:
            raw_results = raw_results[:k]

        result_dicts = [r.to_dict() for r in raw_results]

        answer = None
        citations = []
        if generate_answer and raw_results:
            answer, citations, cost_usd = self._generate_answer(question, raw_results[:10])

        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "answer": answer,
            "citations": [{"chunk_id": c.chunk_id, "excerpt": c.text_excerpt, "score": c.score} for c in citations],
            "results": result_dicts,
            "latency_ms": round(elapsed, 2),
            "cost_usd": round(cost_usd, 6),
        }

    def _get_text(self, chunk_id: str, collection: str) -> Optional[str]:
        """Look up chunk text by ID from in-memory collection."""
        chunks = self._collections.get(collection, [])
        for c in chunks:
            if c.id == chunk_id:
                return c.text
        return None

    def _rerank(self, query: str, candidates: List[Any], top_k: int) -> List[Any]:
        """Cross-encoder reranking via BGE-reranker-large."""
        try:
            pairs = [(query, r.text or "") for r in candidates]
            scores = self.hf.rerank(pairs)
            scored = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
            reranked = []
            for score, res in scored[:top_k]:
                res.rrf_score = float(score)
                reranked.append(res)
            return reranked
        except Exception as exc:
            logger.warning("Reranking failed (%s); using original ordering", exc)
            return candidates[:top_k]

    def _generate_answer(
        self, question: str, results: List[Any]
    ) -> Tuple[str, List[Citation], float]:
        """Call LLM API to synthesize an answer from retrieved chunks."""
        context_parts = []
        for i, r in enumerate(results, 1):
            text = r.text or ""
            context_parts.append(f"[{i}] {text[:600]}")

        context = "\n\n".join(context_parts)
        prompt = f"{RAG_SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}"

        try:
            resp = self.llm.complete(prompt=prompt, max_tokens=1024)
            answer = resp if isinstance(resp, str) else resp.get("text", "")
            cost_usd = resp.get("cost_usd", 0.0) if isinstance(resp, dict) else 0.0

            citations = []
            for i, r in enumerate(results, 1):
                if f"[{i}]" in answer:
                    citations.append(Citation(
                        chunk_id=r.id,
                        text_excerpt=(r.text or "")[:200],
                        score=r.rrf_score,
                    ))
            return answer, citations, cost_usd
        except Exception as exc:
            logger.error("LLM answer generation failed: %s", exc)
            top_texts = "\n".join(r.text[:200] for r in results[:3] if r.text)
            return f"[LLM unavailable] Top results:\n{top_texts}", [], 0.0

    def get_collection_stats(self, collection: str = "default") -> Dict:
        chunks = self._collections.get(collection, [])
        return {
            "collection": collection,
            "chunk_count": len(chunks),
            "hybrid_stats": self.hybrid.get_stats(),
        }
