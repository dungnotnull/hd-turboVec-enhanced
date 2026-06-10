# TurboVec — Upstream Fork Documentation

**Upstream project:** TurboVec (https://github.com/RyanCodrai/turbovec)  
**Fork type:** Enhancement fork (AI intelligence layer added on top)  
**Pinned upstream version:** Latest main branch commit as of 2026-06-09  
**Fork strategy:** Sidecar/overlay pattern — no modifications to upstream TurboVec code  

---

## Upstream Capabilities (Baseline)

TurboVec is a Python vector similarity search library providing:
- Lightweight vector storage and similarity search
- Pure Python interface for embedding-based retrieval
- In-memory index management

**Baseline metrics (upstream, CPU, 100K 768-dim vectors):**
| Metric | Upstream Value |
|--------|---------------|
| Search latency p99 | ~40ms (CPU brute-force / basic HNSW) |
| Retrieval quality (NDCG@10) | ~0.48 (bi-encoder dense retrieval only) |
| Reranker P@1 improvement | 0pp (no reranker) |
| Vector DB backends | 1 (in-memory only) |
| Hybrid search | Not supported |

---

## Improvement Delta (turbovec-enhanced)

| Capability | Upstream | turbovec-enhanced | Improvement |
|-----------|---------|------------------|-------------|
| HNSW backend | Basic | hnswlib C++ (M=32, ef=200) + optional GPU cuVS | Up to 8× faster search |
| GPU acceleration | None | PyTorch CUDA preprocessing + optional cuVS HNSW | p99 ≤ 5ms on GPU |
| Hybrid search | None | Dense HNSW + BM25 + Reciprocal Rank Fusion | NDCG@10 +15-20pp |
| Reranking | None | BAAI/bge-reranker-large cross-encoder | P@1 +15pp on BEIR |
| Vector DB adapters | 1 (in-memory) | 5 (hnswlib, FAISS, Chroma, Qdrant, Weaviate) | Vendor-agnostic |
| Chunking strategies | None | Fixed / Sentence / Semantic | Better chunk coherence |
| LLM answer generation | None | Claude/GPT/Ollama RAG synthesis | End-to-end QA |
| Research self-improvement | None | Weekly ArXiv/Scholar crawl → SECOND-KNOWLEDGE-BRAIN.md | Continuous improvement |
| REST API | None | FastAPI server with 7 endpoints | Production-ready |
| Benchmarking suite | None | Automated BEIR/ANN comparison | Evidence-based decisions |

---

## Architecture: Sidecar Pattern

TurboVec Enhanced adds an AI intelligence layer without modifying upstream TurboVec source code.

```
┌─────────────────────────────────┐
│  TurboVec Enhanced (ai_layer/)  │
│  ┌──────────┐ ┌──────────────┐  │
│  │ HNSW     │ │ Hybrid Search│  │
│  │ (hnswlib)│ │ (BM25 + RRF) │  │
│  └──────────┘ └──────────────┘  │
│  ┌──────────┐ ┌──────────────┐  │
│  │ RAG      │ │ Vector DB    │  │
│  │ Pipeline │ │ Adapter      │  │
│  └──────────┘ └──────────────┘  │
└───────────────┬─────────────────┘
                │ (optional fallback)
┌───────────────▼─────────────────┐
│  upstream/TurboVec              │
│  (original interface preserved) │
└─────────────────────────────────┘
```

---

## Quantified Improvement Targets

1. **Search latency:** p99 ≤ 5ms (GPU) for 1M 768-dim vectors  
   *Baseline:* ~40ms CPU hnswlib  
   *Measurement:* ANN-Benchmarks tool on glove-100 and deep-1M datasets

2. **Retrieval quality:** NDCG@10 ≥ 0.65 on BEIR/NQ with hybrid+reranker  
   *Baseline:* ~0.48 (bi-encoder dense only)  
   *Measurement:* BEIR evaluation harness, NQ dataset, 3610 queries

3. **Reranker precision:** P@1 improvement ≥ 15pp over bi-encoder alone  
   *Baseline:* 0pp (no reranker in upstream)  
   *Measurement:* TREC-DL 2020, 54 queries, BGE-reranker-large vs BGE-large bi-encoder

---

## API Endpoints Added (ai_layer)

| Method | Path | Description |
|--------|------|-------------|
| GET | /health | Agent health check |
| POST | /ingest | Ingest documents (chunk + embed + index) |
| POST | /query | Hybrid search + optional LLM answer |
| POST | /benchmark | Backend comparison benchmark |
| POST | /knowledge/update | Run knowledge crawler |
| GET | /cost | LLM API cost report |
| GET | /metrics | Agent performance metrics |

---

## Cross-Agent Integration

| Agent | Integration Method |
|-------|------------------|
| `18-academic-research-enhanced` | TurboVec Enhanced serves as the vector search backend for paper similarity search |
| `11-coroot-enhanced` | Embed metric time series for anomaly pattern search |
| `22-ai-benchmark-agent` | Benchmarks TurboVec search pipeline as part of system evaluation |
