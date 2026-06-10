# TurboVec Enhanced — Full Technical Specification

## Executive Summary

TurboVec Enhanced is a production-grade vector search agent that forks TurboVec and adds four major capabilities: (1) GPU-accelerated HNSW indexing via hnswlib + optional CUDA kernels, (2) hybrid dense+sparse retrieval with BM25 and Reciprocal Rank Fusion, (3) BGE cross-encoder reranking for precision improvement, and (4) a unified vector DB adapter that lets callers swap Chroma/Qdrant/Weaviate/FAISS backends without changing application code. A self-learning research agent crawls ArXiv, Semantic Scholar, and Papers with Code weekly to keep the AI layer improving automatically.

**Upstream:** TurboVec (https://github.com/RyanCodrai/turbovec)  
**Pinned version:** Latest stable commit (documented in `upstream/README.md`)  
**Improvement targets (quantified):**
1. Search latency: ≤ 5ms p99 for 1M 768-dim vectors on GPU (vs ~40ms CPU hnswlib baseline)
2. Retrieval quality: NDCG@10 ≥ 0.65 on BEIR/NQ with hybrid+reranker (vs ~0.48 bi-encoder only)
3. Reranker precision: P@1 improvement ≥ 15% over bi-encoder alone on BEIR/TREC-DL

---

## Problem Statement

Vector similarity search is the backbone of RAG systems, semantic search engines, recommendation systems, and multi-modal AI applications. The proliferation of vector DBs (Pinecone, Weaviate, Qdrant, Chroma) creates vendor lock-in — switching backends requires rewriting retrieval code. Additionally:

- Pure CPU HNSW is too slow for production workloads at scale (>1M vectors)
- Single-modality dense retrieval misses keyword-critical queries (BM25 still wins on ~30% of BEIR tasks)
- Bi-encoder recall@100 often 10-15% lower than cross-encoder precision@10 would suggest

TurboVec Enhanced solves all three with a unified, extensible Python agent.

---

## Target Users & Use Cases

| User | Trigger | Agent Action |
|------|---------|-------------|
| RAG application developer | Uploads 10k documents for semantic search | Agent chunks, embeds, indexes; answers queries with citations |
| ML engineer | Wants to compare vector DB backends | Agent runs BEIR benchmark across all adapters, returns comparison report |
| Research team | Needs daily paper recommendations | Agent crawls ArXiv daily, reranks by relevance, generates summaries |
| DevOps team | Migrating from Chroma to Qdrant | Swap adapter backend in config — zero code changes |

---

## Agent Architecture (ASCII Diagram)

```
┌──────────────────────────────────────────────────────────────────┐
│  INPUT: Document corpus / User query / Benchmark request         │
└──────────────────────────┬───────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│  agent/orchestrator.py  (TurboVecOrchestrator)                   │
│                                                                  │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────────────┐  │
│  │   Planner   │→  │   Executor   │→  │  Memory / Context    │  │
│  │  (decide    │   │  (dispatch   │   │  (SQLite + FAISS     │  │
│  │   pipeline) │   │   modules)   │   │   benchmark cache)   │  │
│  └─────────────┘   └──────────────┘   └──────────────────────┘  │
│         ↓                 ↓                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Agent Modules                                           │   │
│  │  ┌─────────────────┐  ┌──────────────────────────────┐  │   │
│  │  │  hnsw_search.py │  │  hybrid_search.py            │  │   │
│  │  │  GPU HNSW index │  │  BM25 + dense + RRF fusion  │  │   │
│  │  └─────────────────┘  └──────────────────────────────┘  │   │
│  │  ┌─────────────────┐  ┌──────────────────────────────┐  │   │
│  │  │ rag_pipeline.py │  │  vector_db_adapter.py        │  │   │
│  │  │ chunk→embed→    │  │  Chroma/Qdrant/Weaviate/FAISS│  │   │
│  │  │ retrieve→rerank │  │  unified API                 │  │   │
│  │  └─────────────────┘  └──────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────┬───────────────────────────────────────┘
                           ↓
         ┌─────────────────┼──────────────────┐
         ↓                 ↓                  ↓
    LLM API           HuggingFace         Vector Backends
  (Claude/GPT/       bge-large-en        Chroma/Qdrant/
   Ollama)           bge-reranker        Weaviate/FAISS/
                     MiniLM-L6-v2        hnswlib (GPU)
         ↓
  ┌──────────────────────────────────┐
  │  OUTPUT                          │
  │  - RAG answer with citations     │
  │  - Benchmark comparison report   │
  │  - Optimization recommendations  │
  │  - Index stats / search results  │
  └──────────────────────────────────┘
```

---

## Full Module Catalog

### `agent/modules/hnsw_search.py`

**Responsibility:** GPU-accelerated HNSW approximate nearest neighbor search

**Inputs:** 
- `add(embeddings: np.ndarray, ids: List[str])` — bulk insert vectors
- `search(query_embedding: np.ndarray, k: int, ef: int)` — ANN query

**Outputs:** 
- `SearchResult(ids, distances, latency_ms)`

**Implementation:**
- Primary: hnswlib (C++ backend) with GPU-side embedding via PyTorch CUDA
- Optional: cuVS/RAFT HNSW for full GPU indexing when CUDA available
- Fallback: CPU hnswlib with numpy
- Parameters: `M=32, ef_construction=400, ef_search=200`
- Supports save/load index to disk (.bin format)

**Quality gate:** p99 search latency ≤ 5ms for 1M 768-dim vectors on GPU; ≤ 50ms on CPU

---

### `agent/modules/hybrid_search.py`

**Responsibility:** Combine dense vector search with BM25 sparse retrieval

**Inputs:**
- `index(documents: List[str], embeddings: np.ndarray)` — build BM25 + HNSW index
- `search(query: str, query_embedding: np.ndarray, k: int, alpha: float)` — hybrid search

**Outputs:**
- `HybridResult(id, dense_score, sparse_score, rrf_score, text)`

**Implementation:**
- Dense: HNSW via `hnsw_search.py`
- Sparse: rank_bm25 (BM25Okapi with tokenized corpus)
- Fusion: Reciprocal Rank Fusion (RRF) — `score = Σ 1/(k + rank_i)` where k=60
- Alpha parameter: 0.0 = pure BM25, 1.0 = pure dense, 0.5 = balanced

**Quality gate:** RRF score correlation with human relevance judgments ≥ 0.75 on BEIR/NQ

---

### `agent/modules/rag_pipeline.py`

**Responsibility:** End-to-end RAG pipeline from raw documents to LLM-synthesized answers

**Inputs:**
- `ingest(documents: List[Document], chunk_strategy: str)` — chunk and embed
- `query(question: str, k: int, rerank: bool)` — retrieve, rerank, generate

**Outputs:**
- `RAGResponse(answer: str, citations: List[Citation], latency_ms: float, token_cost: float)`

**Implementation:**
- Chunking strategies: `fixed` (512 tokens), `sentence` (nltk), `semantic` (embedding similarity split)
- Embedding: `BAAI/bge-large-en-v1.5` via `hf_model_manager.py`
- Retrieval: hybrid search (dense + BM25 via `hybrid_search.py`)
- Reranking: `BAAI/bge-reranker-large` cross-encoder on top-50 candidates → keep top-10
- Generation: LLM API (Claude) with retrieved context as system prompt
- Citation extraction: map answer spans back to source document chunks

**Quality gate:** NDCG@10 ≥ 0.65 on BEIR/NQ benchmark

---

### `agent/modules/vector_db_adapter.py`

**Responsibility:** Unified API to swap vector DB backends without application code changes

**Inputs:**
- `connect(backend: str, config: dict)` — initialize backend
- `upsert(id, embedding, metadata)` — insert/update
- `search(query_embedding, k, filter)` — similarity search
- `delete(id)` → remove vector
- `get_stats()` → count, dimensions, backend info

**Outputs:**
- `AdapterResult(ids, distances, metadatas)`

**Supported backends:**
- `chroma` — Chroma DB (via chromadb SDK)
- `qdrant` — Qdrant (via qdrant-client)
- `weaviate` — Weaviate (via weaviate-client)
- `faiss` — Facebook AI Similarity Search (in-memory/mmap)
- `hnswlib` — Local hnswlib index (default, no external service needed)

**Quality gate:** Identical results (within floating-point tolerance) across all backends for same query

---

## HuggingFace Model Selection

| Model | Task | MTEB/BEIR Score | Why vs Alternatives |
|-------|------|----------------|---------------------|
| `BAAI/bge-large-en-v1.5` | Dense embedding (768-dim) | MTEB avg 64.2 (2024) | #1 on MTEB English; beats OpenAI ada-002 (61.0) and E5-large (62.1) |
| `BAAI/bge-reranker-large` | Cross-encoder reranking | BEIR NDCG@10 +8.5pp avg | Best open reranker; 549M params; beats ms-marco-MiniLM-L12 by 4pp |
| `sentence-transformers/all-MiniLM-L6-v2` | Fast embedding (384-dim) | MTEB avg 56.3 | 5× faster than BGE-large; used in benchmark comparison + latency-sensitive paths |
| `Salesforce/codet5p-770m` | Code embedding | CodeSearchNet 0.746 | Best open code embedding model for repository search use case |

---

## LLM API Integration Spec

**Provider chain:** Claude (`claude-opus-4-8`) → OpenAI (`gpt-4o`) → Ollama (`llama3`)

| Use Case | Prompt Template | Max Tokens | Provider |
|----------|----------------|-----------|----------|
| RAG answer synthesis | System: retrieved chunks. User: question | 2048 | Claude |
| Benchmark analysis | "Analyze these benchmark results and provide recommendations..." | 1024 | Claude |
| Improvement recommendations from papers | "Given these papers, suggest 3 improvements to HNSW search..." | 1500 | Claude |
| Structured JSON extraction | Schema-constrained output | 512 | OpenAI |
| Privacy-sensitive RAG | Local model, no external calls | 2048 | Ollama |

**Token budget per call:** RAG synthesis ≤ 8192 input + 2048 output = 10240 total

---

## E2E Execution Flow

### Flow A: Document Ingestion + RAG Query
1. User uploads documents via CLI (`turbovec-enhanced ingest --dir ./docs`)
2. `rag_pipeline.py::ingest()` reads files, detects format (PDF/TXT/MD/HTML)
3. Chunker splits into segments (default: semantic chunking)
4. `hf_model_manager.py` loads `BAAI/bge-large-en-v1.5` → batch embed chunks (GPU if available)
5. `hnsw_search.py::add()` inserts embeddings into HNSW index
6. BM25 index built over tokenized chunk texts
7. Index saved to disk (`.index/` directory)
8. User queries: `turbovec-enhanced query "What is the capital of France?"`
9. `hybrid_search.py::search()` — dense top-50 + BM25 top-50 → RRF fusion → top-50 unified
10. `bge-reranker-large` cross-encoder scores all 50 → top-10 selected
11. Top-10 chunks assembled into context
12. `llm_client.py` calls Claude API with context + question
13. Answer + citations returned; cost logged to SQLite

### Flow B: Vector DB Backend Benchmark
1. `turbovec-enhanced benchmark --backends chroma,qdrant,faiss --dataset beir/nq --k 10`
2. `vector_db_adapter.py` connects to each backend sequentially
3. Same 1000 queries run against each backend; latency + NDCG@10 recorded
4. Results stored in SQLite benchmark table
5. LLM API (Claude) generates comparison report with recommendations
6. Report saved as Markdown + JSON

### Flow C: Self-Learning Knowledge Update
1. APScheduler triggers `knowledge_updater.py` every Sunday 02:00
2. ArXiv XML API queried for cs.DB + cs.IR papers (last 7 days)
3. Semantic Scholar API queried for citation context
4. Papers with Code scrapes vector search leaderboard
5. Papers scored: recency (last 90d = 1.0) × relevance (keyword match count / max_count)
6. Top-20 new papers appended to `SECOND-KNOWLEDGE-BRAIN.md` with ISO date stamp
7. SHA256 dedup prevents re-adding known papers
8. LLM API synthesizes top-3 actionable improvements from new papers
9. Improvements logged to `improvement_recommendations.md`

---

## SECOND-KNOWLEDGE-BRAIN.md Integration

- **Sources:** ArXiv cs.DB/cs.IR, Semantic Scholar, Papers with Code leaderboard, GitHub release notes
- **Crawl schedule:** Weekly (Sunday 02:00 local time)
- **Dedup strategy:** SHA256 hash of `(title + DOI/URL)` stored in SQLite `knowledge_hashes` table
- **Scoring:** `score = recency_weight × relevance_score` where recency decays linearly over 90 days
- **Output format:** Structured Markdown tables with title, authors, year, venue, key finding, relevance

---

## Quality Gates

1. **Embedding latency:** BGE-large batch embedding ≤ 100ms/1000 tokens on GPU
2. **HNSW search latency:** p99 ≤ 5ms on GPU for 1M vectors; p99 ≤ 50ms on CPU
3. **Retrieval quality:** NDCG@10 ≥ 0.65 on BEIR/NQ (hybrid + reranker)
4. **Reranker improvement:** P@1 ≥ 15pp gain over bi-encoder alone on BEIR
5. **Backend consistency:** All 5 adapter backends return identical top-3 results for same query (within 1e-5 float tolerance)
6. **RAG faithfulness:** LLM answer contains ≥ 1 citation from retrieved context (verified by substring check)
7. **Knowledge update:** ≥ 5 new papers added per weekly crawl on average

---

## Test Scenarios

See `tests/test-scenarios.md` for 7 full end-to-end scenarios.

---

## Key Design Decisions

1. **hnswlib over Annoy/ScaNN**: HNSW provides best query-time/recall tradeoff; hnswlib is the de-facto C++ implementation; GPU acceleration via cuVS available as opt-in upgrade
2. **RRF over linear interpolation**: RRF does not require tuning alpha per dataset; robust across domain shifts
3. **BGE-large over OpenAI ada-002**: Better MTEB score, open-source, runs locally, no per-token cost
4. **BGE-reranker-large over MonoBERT**: Larger capacity; distilled from GPT-4 ranking signals; available on HuggingFace
5. **Adapter pattern for vector DBs**: Prevents vendor lock-in; enables backend migration without application changes
6. **Semantic chunking as default**: Embedding-similarity-based split produces more coherent chunks than fixed-size; better retrieval performance on open-domain QA
7. **SQLite for metadata and benchmarks**: Zero-dependency persistence; sufficient for single-node deployment
