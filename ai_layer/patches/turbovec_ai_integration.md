# TurboVec Enhanced — AI Integration Patch Guide

## Overview

This document describes the AI intelligence layer added on top of upstream TurboVec.
The integration uses a sidecar pattern: no upstream TurboVec source code is modified.
All AI capabilities are implemented in `agent/` and `tools/` directories.

---

## Module Integration Map

```
agent/
├── main.py              ← FastAPI + CLI entry point (8016)
├── orchestrator.py      ← TurboVecOrchestrator (lazy module init, scheduler)
├── modules/
│   ├── hnsw_search.py       ← GPU HNSW (hnswlib + optional CUDA)
│   ├── hybrid_search.py     ← BM25 + Dense + RRF fusion
│   ├── rag_pipeline.py      ← Chunk → Embed → Retrieve → Rerank → Generate
│   └── vector_db_adapter.py ← Unified API for 5 backends
└── memory/
    └── memory_manager.py    ← SQLite persistent storage

tools/
├── knowledge_updater.py  ← ArXiv + Scholar + GitHub weekly crawl
├── llm_client.py         ← Claude/OpenAI/Ollama with streaming + retry
└── hf_model_manager.py   ← BGE-large, BGE-reranker, MiniLM lazy loading
```

---

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Docker Compose Stack                                   │
│                                                         │
│  ┌─────────────────────┐   ┌────────────────────────┐  │
│  │ turbovec-agent:8016 │   │ qdrant:6333            │  │
│  │ FastAPI REST API    │   │ Weaviate-compatible API │  │
│  │ 8GB memory limit    │   └────────────────────────┘  │
│  └──────────┬──────────┘                               │
│             │                ┌────────────────────────┐ │
│             └──────────────→ │ chroma:8000            │ │
│                              │ REST API               │ │
│  ┌─────────────────────┐    └────────────────────────┘ │
│  │ ollama:11434         │                               │
│  │ (optional profile)  │                               │
│  └─────────────────────┘                               │
└─────────────────────────────────────────────────────────┘
```

**Start command:**
```bash
docker compose up -d              # Start agent + qdrant + chroma
docker compose --profile ollama up -d  # Include Ollama for offline mode
```

---

## Quick Start

```bash
# 1. Clone and configure
cd turbovec-enhanced
cp config/.env.example config/.env
# Edit .env with your ANTHROPIC_API_KEY

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# 4. Ingest documents
python -m agent.main ingest --dir ./my_documents --chunk-strategy semantic

# 5. Query
python -m agent.main query "What is the best algorithm for large-scale ANN search?" --answer

# 6. Or start the server
python -m agent.main serve --port 8016
```

---

## REST API Usage Examples

### Ingest documents
```bash
curl -X POST http://localhost:8016/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "documents": ["HNSW is a graph-based ANN algorithm.", "BM25 is a sparse retrieval model."],
    "chunk_strategy": "semantic",
    "collection": "my_docs"
  }'
```

### Query with hybrid search + LLM answer
```bash
curl -X POST http://localhost:8016/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is Reciprocal Rank Fusion?",
    "k": 10,
    "hybrid": true,
    "rerank": true,
    "generate_answer": true,
    "collection": "my_docs"
  }'
```

### Run backend benchmark
```bash
curl -X POST http://localhost:8016/benchmark \
  -H "Content-Type: application/json" \
  -d '{"backends": ["hnswlib", "faiss"], "dataset": "synthetic", "n_vectors": 10000}'
```

---

## Cross-Agent Integration

### Integration with academic-research-enhanced (Folder 18)

The academic research agent can use TurboVec Enhanced as its vector backend for paper similarity search:

```python
import httpx

# From academic-research-enhanced agent
turbovec_url = "http://turbovec-agent:8016"

# Ingest papers
httpx.post(f"{turbovec_url}/ingest", json={
    "documents": paper_abstracts,
    "ids": paper_ids,
    "metadatas": [{"title": t, "year": y} for t, y in zip(titles, years)],
    "collection": "academic_papers",
})

# Find similar papers
results = httpx.post(f"{turbovec_url}/query", json={
    "query": "approximate nearest neighbor GPU acceleration",
    "k": 20,
    "hybrid": True,
    "rerank": True,
    "collection": "academic_papers",
}).json()
```

### Integration with coroot-enhanced (Folder 11)

Embed Coroot metric time series for anomaly pattern similarity:

```python
# Store anomaly pattern embeddings
httpx.post(f"{turbovec_url}/ingest", json={
    "documents": [json.dumps(metric_snapshot) for metric_snapshot in metric_history],
    "collection": "coroot_metrics",
})

# Find similar past anomalies
results = httpx.post(f"{turbovec_url}/query", json={
    "query": json.dumps(current_anomaly_pattern),
    "collection": "coroot_metrics",
    "k": 5,
}).json()
```

---

## Prometheus Metrics

The agent exposes Prometheus-compatible metrics at `/metrics`:

| Metric | Type | Description |
|--------|------|-------------|
| `turbovec_search_latency_ms` | Histogram | p50/p95/p99 search latency |
| `turbovec_search_total` | Counter | Total search requests |
| `turbovec_rag_cost_usd` | Counter | Cumulative LLM cost |
| `turbovec_chunks_indexed` | Gauge | Total chunks in index |
| `turbovec_papers_known` | Gauge | Papers in knowledge base |

---

## Production Hardening Checklist

- [ ] Set `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` in environment
- [ ] Configure vector DB backends in `config/agent_config.yaml`
- [ ] Pre-load BGE-large model on startup (`preload: [bge-large]` in config)
- [ ] Set `use_gpu: true` if NVIDIA GPU available
- [ ] Run knowledge crawler weekly: `python -m agent.main update-knowledge`
- [ ] Monitor `/metrics` endpoint for latency regressions
- [ ] Set `LOG_LEVEL=WARNING` in production to reduce log volume
- [ ] Configure persistent volumes in `docker-compose.yml` for index data
- [ ] Test graceful degradation: unset all API keys and verify fallback responses
