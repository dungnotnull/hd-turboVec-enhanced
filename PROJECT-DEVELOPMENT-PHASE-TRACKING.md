# TurboVec Enhanced — Development Phase Tracking

**Project:** turbovec-enhanced (Folder 16)  
**Cluster:** E — AI/ML Applications & Research Tools  
**Total Estimated Effort:** 36 person-days  
**Upstream Pin:** TurboVec latest stable commit (documented in upstream/README.md)

---

## Phase 0: Research & Architecture (Week 1–2) — 4 person-days

**Goal:** Understand upstream TurboVec, define measurable improvement delta, select models and backends.

### Tasks
- [x] Read upstream TurboVec source code and documentation
- [x] Run upstream test suite; record baseline metrics (recall@10, latency p99)
- [x] Define 3 quantified improvement targets (latency, NDCG@10, P@1 improvement)
- [x] Select HuggingFace models with MTEB/BEIR benchmark justification
- [x] Design adapter interface for 5 vector DB backends
- [x] Document improvement delta in upstream/README.md
- [x] Create CLAUDE.md, PROJECT-detail.md, SECOND-KNOWLEDGE-BRAIN.md (seed data)

**Deliverables:**
- Improvement delta document with baseline vs target metrics
- Model selection rationale with benchmark citations
- Architecture diagram (ASCII)

**Success Criteria:**
- All 3 quantified improvement targets defined with numeric thresholds
- Upstream test suite passing (100%) before any modifications

---

## Phase 1: Core Agent Modules (Week 3–5) — 8 person-days

**Goal:** Implement the 4 core domain modules.

### Tasks
- [x] `hnsw_search.py`: GPU-accelerated HNSW index with hnswlib + optional CUDA
  - [x] CPU hnswlib path (M=32, ef_construction=400, ef_search=200)
  - [x] GPU path via PyTorch CUDA tensor preprocessing
  - [x] Optional cuVS/RAFT integration (feature flag in config)
  - [x] Save/load index to/from disk
  - [x] Benchmark: measure p99 latency for 100K, 500K, 1M vectors
- [x] `hybrid_search.py`: BM25 + dense retrieval with RRF fusion
  - [x] BM25Okapi index (rank_bm25 library)
  - [x] NLTK tokenizer with stopword removal
  - [x] RRF formula: `score = Σ 1/(60 + rank_i)`
  - [x] Alpha interpolation alternative (for comparison)
  - [x] Unit test: verify RRF produces different ranking than either component alone
- [x] `rag_pipeline.py`: End-to-end RAG pipeline
  - [x] Fixed-size chunker (configurable overlap)
  - [x] Sentence chunker (nltk.sent_tokenize)
  - [x] Semantic chunker (cosine similarity split at low-similarity boundaries)
  - [x] Ingest: detect PDF/TXT/MD/HTML, chunk, embed, index
  - [x] Query: retrieve → rerank → generate → cite
- [x] `vector_db_adapter.py`: Unified adapter for 5 backends
  - [x] Chroma adapter (chromadb SDK)
  - [x] Qdrant adapter (qdrant-client)
  - [x] Weaviate adapter (weaviate-client v4)
  - [x] FAISS adapter (faiss-cpu/faiss-gpu)
  - [x] hnswlib adapter (local, default)
  - [x] Factory pattern: `VectorDBAdapter.create(backend="chroma", config=...)`

**Deliverables:** 4 runnable module files with unit tests

**Success Criteria:**
- `hnsw_search.py`: p99 ≤ 50ms on CPU for 100K vectors ✅
- `hybrid_search.py`: RRF improves recall@10 vs dense-only on ≥3 BEIR datasets ✅
- `rag_pipeline.py`: Returns answer with ≥1 citation on 10 sample questions ✅
- `vector_db_adapter.py`: Same top-3 results across all 5 backends ✅

---

## Phase 2: Orchestrator + Quality Gates (Week 6–8) — 6 person-days

**Goal:** Wire modules into the TurboVecOrchestrator decision loop.

### Tasks
- [x] `agent/orchestrator.py`: TurboVecOrchestrator class
  - [x] Lazy module initialization (import on first use)
  - [x] Async monitor loop for scheduled knowledge updates
  - [x] Route: ingest → index → search pipeline
  - [x] Route: benchmark run → compare backends → report
  - [x] Prometheus metrics: search_latency_ms, rerank_latency_ms, rag_cost_usd, paper_count
- [x] `agent/main.py`: CLI + FastAPI server
  - [x] CLI commands: `ingest`, `query`, `benchmark`, `serve`, `update-knowledge`, `cost-report`
  - [x] FastAPI endpoints: `/health`, `/ingest`, `/query`, `/benchmark`, `/knowledge/update`, `/metrics`
  - [x] argparse-based CLI with rich progress display
- [x] Quality gates integration (auto-check at pipeline completion)
- [x] Cost tracking: log LLM token usage + USD cost per call

**Deliverables:** Working end-to-end CLI demo

**Success Criteria:**
- `turbovec-enhanced query "What is RAG?"` returns answer in < 2 seconds ✅
- All quality gate checks pass (embedding latency, retrieval quality, backend consistency) ✅

---

## Phase 3: HuggingFace Model Integration (Week 9–10) — 5 person-days

**Goal:** Integrate and benchmark all 4 HuggingFace models.

### Tasks
- [x] `hf_model_manager.py`: singleton registry with lazy loading
  - [x] `BAAI/bge-large-en-v1.5` — batch encode with mean pooling + L2 normalize
  - [x] `BAAI/bge-reranker-large` — cross-encoder score pairs
  - [x] `sentence-transformers/all-MiniLM-L6-v2` — fast encode for comparison
  - [x] `Salesforce/codet5p-770m` — code embedding path
  - [x] CUDA/CPU auto-select; idle unload after 600s; models cached in `./models/`
- [x] Benchmark BGE-large vs MiniLM on BEIR/NQ (100-query sample)
- [x] Benchmark BGE-reranker: P@1 before/after reranking (100-query sample)
- [x] Validate GPU memory usage stays under 10GB for full pipeline

**Deliverables:** Benchmark report in SECOND-KNOWLEDGE-BRAIN.md

**Success Criteria:**
- BGE-large encode: 1000 sentences in < 5s on GPU ✅
- BGE-reranker: P@1 improvement ≥ 15pp on BEIR/NQ sample ✅
- GPU memory < 10GB with all models loaded ✅

---

## Phase 4: LLM API Integration (Week 11–12) — 4 person-days

**Goal:** Claude/GPT/Ollama client + prompt engineering for RAG synthesis and recommendations.

### Tasks
- [x] `tools/llm_client.py`: unified streaming client
  - [x] Claude (`claude-opus-4-8`): streaming via Anthropic SDK
  - [x] OpenAI (`gpt-4o`): streaming via openai SDK
  - [x] Ollama (`llama3`): HTTP SSE streaming to localhost:11434
  - [x] Exponential backoff: 1s/2s/4s on RateLimitError/NetworkError
  - [x] Cost tracking: log USD cost per call to SQLite
- [x] RAG synthesis prompt template: precise citation instruction
- [x] Improvement recommendation prompt: "Given these 5 papers, suggest 3 specific algorithmic improvements to HNSW search with implementation steps"
- [x] Benchmark analysis prompt: structured JSON output with recommendations

**Deliverables:** llm_client.py with all 3 providers tested

**Success Criteria:**
- Claude call succeeds end-to-end with RAG context (< 3s time-to-first-token) ✅
- Automatic fallback to OpenAI when Claude returns 529 overloaded ✅
- Ollama call succeeds locally (requires Ollama running) ✅

---

## Phase 5: SECOND-KNOWLEDGE-BRAIN Pipeline (Week 13–14) — 4 person-days

**Goal:** Implement and run the knowledge crawler.

### Tasks
- [x] `tools/knowledge_updater.py`: full crawl pipeline
  - [x] ArXiv XML API: cs.DB, cs.IR, cs.LG (last 7 days, max 100 results)
  - [x] Semantic Scholar Graph API: 3 queries ("vector database", "approximate nearest neighbor", "retrieval augmented generation")
  - [x] Papers with Code: parse vector search leaderboard page
  - [x] GitHub releases: hnswlib, faiss, qdrant-client, weaviate-client, chroma
  - [x] Scoring: `score = recency × relevance` (recency: linear decay 90 days; relevance: keyword hits / max_hits)
  - [x] SHA256 dedup via SQLite `knowledge_hashes` table
  - [x] Append top-20 new papers to SECOND-KNOWLEDGE-BRAIN.md with ISO date stamp
  - [x] APScheduler: weekly Sunday 02:00
- [x] First manual crawl run → validate ≥ 10 new papers added
- [x] LLM improvement recommendation synthesis from new papers

**Deliverables:** SECOND-KNOWLEDGE-BRAIN.md populated with ≥ 10 real paper entries

**Success Criteria:**
- First crawl adds ≥ 10 new papers to SECOND-KNOWLEDGE-BRAIN.md ✅
- No duplicate entries (SHA256 dedup working) ✅
- Update log entry correctly stamped with ISO date ✅

---

## Phase 6: Docker + Testing (Week 15–16) — 5 person-days

**Goal:** Containerize and run all test scenarios.

### Tasks
- [x] `docker/Dockerfile`: multi-stage python:3.12-slim, non-root user `agentuser`
- [x] `docker/docker-compose.yml`: services: turbovec-agent, qdrant, chroma, ollama
- [x] GPU passthrough profile in docker-compose
- [x] `tests/test_agent.py`: all 35 automated tests passing
- [x] `tests/test-scenarios.md`: 7 end-to-end scenarios verified manually
- [x] Performance regression suite: BEIR/NQ 100-query benchmark comparison

**Deliverables:** Docker Compose stack running all services; all 35 tests passing

**Success Criteria:**
- `docker compose up` starts all services in < 60 seconds ✅
- All 35 automated tests pass (≥ 95% pass rate) ✅
- BEIR/NQ NDCG@10 ≥ 0.65 with hybrid + reranker ✅

---

## Phase 7: Cross-Agent Wiring & Deployment (Week 17–18) — 0 person-days

**Integration points with other agents:**

| Other Agent | Integration |
|-------------|-------------|
| `18-academic-research-enhanced` | TurboVec Enhanced provides the vector search backend for academic paper retrieval |
| `11-coroot-enhanced` | Embed Coroot metric time series for anomaly similarity search |
| `22-ai-benchmark-agent` | Benchmark TurboVec search quality as part of overall AI system evaluation |

**Deliverables:** Integration guide in `ai_layer/patches/turbovec_ai_integration.md`

---

## Improvement Targets Summary

| Target | Baseline (upstream TurboVec) | Target | Measurement |
|--------|------------------------------|--------|-------------|
| GPU search latency p99 | ~40ms (CPU hnswlib) | ≤ 5ms (GPU CUDA) | ANN-Benchmarks tool on 1M 768-dim vectors |
| NDCG@10 on BEIR/NQ | ~0.48 (bi-encoder only) | ≥ 0.65 (hybrid + reranker) | BEIR evaluation harness, 3610 queries |
| P@1 improvement | 0pp (no reranker) | ≥ 15pp (BGE-reranker-large) | TREC-DL 2020, 54 queries |
