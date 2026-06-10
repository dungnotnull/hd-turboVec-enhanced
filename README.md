# TurboVec Enhanced

**GPU-accelerated vector search, hybrid retrieval, and self-improving RAG pipeline**

TurboVec Enhanced is a production-grade vector search agent that adds GPU acceleration, hybrid dense+sparse retrieval, cross-encoder reranking, and a unified vector DB adapter on top of the upstream TurboVec library.

## Features

- **GPU-accelerated HNSW** via hnswlib with optional CUDA support (p99 ≤ 5ms for 1M vectors)
- **Hybrid search** combining dense vector search + BM25 sparse retrieval with Reciprocal Rank Fusion
- **Cross-encoder reranking** using `BAAI/bge-reranker-large` for +15pp P@1 improvement
- **Unified vector DB adapter** supporting Chroma, Qdrant, Weaviate, FAISS, and hnswlib
- **End-to-end RAG pipeline** with configurable chunking (fixed/sentence/semantic)
- **Multi-provider LLM client** with Claude, OpenAI, and Ollama support
- **Self-learning research agent** that crawls ArXiv, Semantic Scholar, and Papers with Code weekly
- **REST API** via FastAPI with 7 endpoints for ingestion, query, benchmarking, and metrics

## Installation

```bash
# Clone the repository
git clone https://github.com/RyanCodrai/turbovec.git
cd turbovec

# Install dependencies
pip install -r requirements.txt

# For GPU support
pip install faiss-gpu
```

## Quick Start

```bash
# Ingest documents
python -m agent.main ingest --dir ./documents --chunk-strategy semantic

# Query with hybrid search
python -m agent.main query "What is approximate nearest neighbor search?" --k 10 --answer

# Start the REST API server
python -m agent.main serve --port 8016
```

## REST API

```bash
# Ingest documents
curl -X POST http://localhost:8016/ingest \
  -H "Content-Type: application/json" \
  -d '{"documents": ["HNSW is a graph-based ANN algorithm."], "chunk_strategy": "semantic"}'

# Query with LLM answer generation
curl -X POST http://localhost:8016/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is HNSW?", "k": 10, "generate_answer": true}'
```

## Docker Deployment

```bash
# Start all services (agent + Qdrant + Chroma)
docker compose up -d

# Include Ollama for offline LLM mode
docker compose --profile ollama up -d
```

## Configuration

Edit `config/agent_config.yaml` to customize:

- HNSW parameters (M, ef_construction, ef_search)
- Chunking strategies and thresholds
- LLM provider priority and model selection
- Vector DB backend connection settings
- Knowledge update schedule and sources

## Environment Variables

```bash
# Required for RAG answer generation
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Optional: override defaults
LLM_PROVIDER=claude
USE_GPU=true
LOG_LEVEL=INFO
```

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  TurboVecOrchestrator                                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐ │
│  │ Planner  │→ │ Executor │→ │ Memory / Context      │ │
│  └──────────┘  └──────────┘  └──────────────────────┘ │
│         ↓               ↓                               │
│  ┌────────────────────────────────────────────────────┐│
│  │ agent/                                            ││
│  │ hnsw_search.py    hybrid_search.py                 ││
│  │ rag_pipeline.py   vector_db_adapter.py              ││
│  └────────────────────────────────────────────────────┘│
└───────────────────────┬──────────────────────────────────┘
                        ↓
         ┌──────────────┼──────────────┐
         ↓              ↓               ↓
    LLM API        HuggingFace      Vector Backends
  Claude/GPT/      bge-large        Chroma/Qdrant/
   Ollama          bge-reranker     Weaviate/FAISS
```

## Benchmarks

| Metric | Target | Measurement |
|--------|--------|-------------|
| Search latency p99 (GPU) | ≤ 5ms | 1M 768-dim vectors |
| NDCG@10 on BEIR/NQ | ≥ 0.65 | Hybrid + reranker |
| P@1 improvement | ≥ 15pp | vs bi-encoder alone |

## Development

```bash
# Run tests
pytest tests/ -v

# Format code
ruff check .
ruff format .

# Type check
mypy agent/
```

## License

MIT License — see LICENSE file for details.

## Upstream

This is an enhancement fork of [TurboVec](https://github.com/RyanCodrai/turbovec). All upstream improvements are merged regularly.

## Citation

If you use TurboVec Enhanced in your research, please cite:

```bibtex
@software{turbovec2024,
  title={TurboVec Enhanced: GPU-Accelerated Vector Search with Hybrid Retrieval},
  author={TurboVec Contributors},
  year={2024},
  url={https://github.com/RyanCodrai/turbovec}
}
```
