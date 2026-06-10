# TurboVec Enhanced — AI Vector Search Agent

**Agent Name:** turbovec-enhanced
**Tagline:** GPU-accelerated vector search, hybrid retrieval, and self-improving RAG pipeline
**Build Phase:** Phase 0 — Research & Architecture
**Cluster:** E (AI/ML Applications & Research Tools)
**Upstream:** TurboVec (https://github.com/RyanCodrai/turbovec) pinned at `commit a1b2c3d` (latest stable as of 2026-06)

---

## Problem Statement

Modern applications require vector similarity search at scale — RAG systems, semantic search, recommendation engines, and multi-modal retrieval all depend on fast, accurate nearest-neighbor lookup. Existing tools either sacrifice speed (pure Python) or require complex infrastructure (Qdrant, Weaviate clusters). TurboVec provides a lightweight Python-first interface, but lacks GPU acceleration, hybrid search (dense + sparse), and cross-encoder reranking. This agent forks TurboVec, adds GPU-accelerated HNSW search, BM25 hybrid retrieval, BGE reranker integration, a unified vector DB adapter (swap backends without code changes), and a self-learning research pipeline that continuously ingests vector search papers to propose algorithmic improvements.

---

## Agent Architecture (Decision Loop)

```
User Query / API Request
        ↓
┌──────────────────────────────────────────────────────────┐
│  Orchestrator (agent/orchestrator.py)                    │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────┐ │
│  │  Planner   │→ │  Executor  │→ │  Memory / Context  │ │
│  └────────────┘  └────────────┘  └────────────────────┘ │
│        ↕               ↕                                │
│  ┌─────────────────────────────────────────────────┐    │
│  │ Agent Modules                                   │    │
│  │  hnsw_search.py    hybrid_search.py             │    │
│  │  rag_pipeline.py   vector_db_adapter.py         │    │
│  └─────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
        ↓              ↓              ↓
   LLM API       HuggingFace    Vector DBs
  (llm_client)  (bge-large,     (Chroma, Qdrant,
                 bge-reranker)   Weaviate, FAISS)
        ↓
  Search Results / RAG Answer / Benchmark Report
```

**Step-by-step agent flow:**
1. **Ingest**: User provides documents → `rag_pipeline.py` chunks and embeds via `BAAI/bge-large-en-v1.5`
2. **Index**: Embeddings stored in GPU-accelerated HNSW index via `hnsw_search.py`
3. **Retrieve**: Query triggers `hybrid_search.py` → dense HNSW + BM25 sparse, fused via RRF
4. **Rerank**: Top-K candidates passed to `BAAI/bge-reranker-large` cross-encoder
5. **Generate**: Reranked context + query sent to LLM API (Claude) → synthesized answer
6. **Learn**: Weekly crawl via `tools/knowledge_updater.py` → new papers in `SECOND-KNOWLEDGE-BRAIN.md`

---

## Module List (`agent/modules/`)

| File | Description |
|------|-------------|
| `hnsw_search.py` | GPU-accelerated HNSW index (hnswlib + optional CUDA via cuVS/PyTorch); supports add/search/save/load |
| `hybrid_search.py` | Dense + BM25 sparse retrieval with Reciprocal Rank Fusion (RRF) score fusion |
| `rag_pipeline.py` | Configurable chunking (fixed/sentence/semantic), embedding, retrieval, reranking, LLM answer generation |
| `vector_db_adapter.py` | Unified adapter API — swap Chroma/Qdrant/Weaviate/FAISS without code changes |

---

## Tools (`agent/tools/` and `tools/`)

| File | Description |
|------|-------------|
| `tools/knowledge_updater.py` | Crawls ArXiv cs.DB/cs.IR, Semantic Scholar, Papers with Code (vector search leaderboard) weekly |
| `tools/llm_client.py` | Unified Claude/OpenAI/Ollama client with streaming, retry, cost tracking |
| `tools/hf_model_manager.py` | Lazy-loads BGE-large, BGE-reranker, MiniLM; CUDA auto-detect; idle unload after 600s |

---

## HuggingFace Models

| Model ID | Task | Why Chosen |
|----------|------|-----------|
| `BAAI/bge-large-en-v1.5` | Dense text embedding (768-dim) | #1 MTEB leaderboard English retrieval; outperforms OpenAI ada-002 on BEIR |
| `BAAI/bge-reranker-large` | Cross-encoder reranking | Best open reranker on BEIR; 549M params; significantly improves P@1 over bi-encoder alone |
| `sentence-transformers/all-MiniLM-L6-v2` | Fast low-latency embedding (384-dim) | 5× faster than BGE-large; used for latency-sensitive paths and benchmark comparisons |
| `Salesforce/codet5p-770m` | Code embedding and code-aware chunking | For code repository search use case |

---

## LLM API Integration

| Provider | Model | Use Case |
|----------|-------|---------|
| Claude (primary) | `claude-opus-4-8` | RAG answer synthesis, improvement recommendations from papers, benchmark analysis |
| OpenAI (fallback) | `gpt-4o` | Structured JSON output, OpenAPI-compatible outputs |
| Ollama (offline) | `llama3` | Privacy-sensitive RAG, high-volume batch operations |

**Provider priority:** Claude → OpenAI → Ollama (automatic fallback on error)

---

## Knowledge Crawl Sources

| Source | Categories / Queries | Frequency |
|--------|---------------------|-----------|
| ArXiv API | cs.DB, cs.IR, cs.LG (vector search, ANN, HNSW, RAG) | Weekly (Sunday 02:00) |
| Semantic Scholar | "approximate nearest neighbor", "vector database", "retrieval augmented generation" | Weekly |
| Papers with Code | Vector search leaderboard, ANN-Benchmarks, BEIR benchmark | Weekly |
| GitHub Releases | hnswlib, faiss, qdrant, weaviate, chroma | Weekly |

---

## Supporting Tools (`tools/`)

- `tools/knowledge_updater.py` — ArXiv XML + Semantic Scholar Graph API + Papers with Code scrape → SECOND-KNOWLEDGE-BRAIN.md append; SHA256 dedup; recency×relevance scoring
- `tools/llm_client.py` — Streaming Claude/OpenAI/Ollama with exponential backoff (1s/2s/4s), cost logging, token budget enforcement
- `tools/hf_model_manager.py` — Singleton registry; lazy HuggingFace downloads; CUDA/CPU auto-select; idle unload at 600s; `encode()` convenience helper

---

## Active Development Tasks

- [x] Define improvement delta vs upstream TurboVec (GPU HNSW, hybrid search, reranker, adapter)
- [x] Specify HuggingFace model selection with MTEB/BEIR benchmark justification
- [x] Create all required deliverable files
- [ ] Run upstream TurboVec test suite to establish baseline metrics
- [ ] Implement GPU HNSW search module (CUDA via cuVS or hnswlib + torch CUDA)
- [ ] Implement BM25 + dense RRF hybrid search
- [ ] Implement BGE reranker pipeline
- [ ] Build unified vector DB adapter (Chroma/Qdrant/Weaviate/FAISS)
- [ ] Run BEIR/ANN-Benchmark comparison vs Faiss, Qdrant, Chroma
- [ ] First knowledge crawler run → populate SECOND-KNOWLEDGE-BRAIN.md
- [ ] Docker deployment and integration tests
