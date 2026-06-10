# SECOND-KNOWLEDGE-BRAIN — TurboVec Enhanced

**Domain:** Vector Similarity Search, Approximate Nearest Neighbor, RAG, Information Retrieval  
**Last Updated:** 2026-06-09  
**Self-Update Protocol:** Weekly (Sunday 02:00) via `tools/knowledge_updater.py`

---

## Core Concepts & Frameworks

### Approximate Nearest Neighbor (ANN) Search

ANN search finds vectors "close enough" to a query vector without exhaustive comparison. Core algorithms:

**HNSW (Hierarchical Navigable Small World):**
- Graph-based index with multi-layer structure (Malkov & Yashunin, 2018)
- Parameters: `M` (number of connections per node), `ef_construction` (build quality), `ef_search` (query quality)
- Tradeoff: higher M → better recall but larger memory; higher ef_construction → slower build but better graph quality
- State of the art for high-dimensional (d > 100) dense vectors; throughput > 100K QPS on GPU

**IVF-PQ (Inverted File Index + Product Quantization):**
- Clusters vectors into `nlist` Voronoi cells; compresses residuals via PQ codes
- Lower memory footprint than HNSW; suited for billion-scale indexes (Faiss IVF-PQ)
- Recall@10 typically 5-10pp lower than HNSW at same latency budget

**ScaNN (Google, 2020):**
- Anisotropic quantization: weights quantization error by inner product direction
- State of the art on Google ANN-Benchmarks (glove-100, deep-100M)
- Not open-source for commercial use

### Hybrid Search (Dense + Sparse)

Pure dense retrieval misses keyword-critical queries. BEIR benchmark (Thakur et al., 2021) shows:
- Dense only (SBERT): average NDCG@10 = 0.430 across 18 tasks
- BM25 only: average NDCG@10 = 0.436 across 18 tasks
- Hybrid (RRF): average NDCG@10 = 0.478 — consistent improvement across all domains

**Reciprocal Rank Fusion (RRF):**
- `score(d) = Σ_i 1 / (k + rank_i(d))` where k=60 (Cormack et al., 2009)
- No hyperparameter tuning required; robust to score scale differences
- Outperforms linear score interpolation on most BEIR tasks

**SPLADE (SParse Lexical AnD Expansion):**
- Learned sparse representations via MLM expansion
- Fills vocabulary gap between query and document; strong on BEIR
- Training required (not zero-shot); `naver/splade-cocondenser-ensemble-distil` available

### Cross-Encoder Reranking

Bi-encoders (separate query/doc encoding) trade accuracy for speed. Cross-encoders (joint encoding) capture full query-document interaction:
- BGE-reranker-large: NDCG@10 improvement +8.5pp avg over BGE-large bi-encoder on BEIR (2024)
- Computational cost: O(|candidates| × query_tokens × doc_tokens) — use on top-50 candidates only
- MonoT5: T5-based cross-encoder; strong but slower than BERT-based BGE reranker

### RAG (Retrieval Augmented Generation)

RAG (Lewis et al., 2020) decouples parametric knowledge (LLM weights) from non-parametric knowledge (retrieval):
- **Naive RAG:** chunk → embed → retrieve → generate (sequential, brittle)
- **Advanced RAG:** query rewriting, HyDE (Gao et al., 2022), iterative retrieval, fusion-in-decoder
- **Modular RAG:** swap retrievers, rerankers, and generators independently
- Faithfulness metric: fraction of answer statements grounded in retrieved context (FaithDial, RAGTruth benchmarks)

---

## Key Research Papers

| Title | Authors | Year | Venue | DOI/Link | Key Finding | Relevance |
|-------|---------|------|-------|----------|-------------|-----------|
| Efficient and Robust ANN Search Using Hierarchical Navigable Small World Graphs | Malkov, Yashunin | 2018 | IEEE TPAMI | arxiv.org/abs/1603.09320 | HNSW achieves sub-logarithmic query complexity; best recall/latency tradeoff on high-dim vectors | Core index algorithm for hnsw_search.py |
| BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of IR | Thakur et al. | 2021 | NeurIPS | arxiv.org/abs/2104.08663 | 18-task IR benchmark; reveals dense retrieval fails on out-of-domain tasks; hybrid wins consistently | Benchmark for evaluating retrieval quality |
| Reciprocal Rank Fusion outperforms Condorcet and Individual Rank Learning Methods | Cormack et al. | 2009 | SIGIR | dl.acm.org/doi/10.1145/1571941.1572114 | RRF (k=60) is robust, no-tuning fusion method; consistently strong across domains | Core of hybrid_search.py RRF implementation |
| Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | Lewis et al. | 2020 | NeurIPS | arxiv.org/abs/2005.11401 | RAG decouples parametric and non-parametric knowledge; outperforms GPT-3 on open-domain QA | Foundation of rag_pipeline.py design |
| Text Embeddings Reveal (Almost) As Much As Text | Muennighoff et al. | 2022 | EMNLP | arxiv.org/abs/2210.07316 | MTEB benchmark; BGE-large-en-v1.5 leads English retrieval leaderboard (2024 snapshot) | Justifies BGE-large model selection |
| BGE M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity | Chen et al. | 2024 | arxiv | arxiv.org/abs/2402.03216 | BGE-M3: single model for dense+sparse+colbert retrieval; multi-lingual; strong on MTEB | Future upgrade path for embedding model |
| Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE) | Gao et al. | 2022 | ACL | arxiv.org/abs/2212.10496 | Hypothetical Document Embeddings: generate fake answer with LLM → embed → retrieve; improves zero-shot recall | Advanced RAG technique for rag_pipeline.py |
| Large Language Models are Built-in Autoregressive Search Engines | Ziems et al. | 2023 | ACL | arxiv.org/abs/2305.09612 | LLMs can reformulate queries to improve retrieval; query rewriting adds +3pp NDCG@10 | Query rewriting enhancement for hybrid_search.py |
| ANN-Benchmarks: A Benchmarking Tool for ANN Algorithms | Aumüller et al. | 2020 | Information Systems | arxiv.org/abs/1807.05614 | Standardized framework for comparing ANN algorithms on real datasets | Used in Phase 6 benchmark suite |
| FAISS: A Library for Efficient Similarity Search | Johnson et al. | 2021 | IEEE TPAMI | arxiv.org/abs/1702.08734 | GPU-accelerated PQ compression + IVF; handles billion-scale indexes | Comparison baseline in benchmark suite |
| SPLADE v2: Sparse Lexical and Expansion Model for First Stage Retrieval | Formal et al. | 2022 | SIGIR | arxiv.org/abs/2109.10086 | SPLADE-cocondenser achieves NDCG@10 0.728 on BEIR avg — best sparse model | Alternative sparse retrieval for hybrid_search.py |
| Nomic Embed: Training a Reproducible Long Context Text Embedder | Nussbaum et al. | 2024 | arxiv | arxiv.org/abs/2402.01613 | Open-source embedding trained on 235M pairs; MTEB avg 62.4; 8192-token context window | Alternative embedding with long-context support |
| RULER: What's the Real Context Window of Your Long-Context LLMs? | Hsieh et al. | 2024 | ICLR | arxiv.org/abs/2404.06654 | Shows LLM context degradation beyond 16K; chunking + retrieval outperforms very long context | Justifies RAG over long-context stuffing approach |
| ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction | Khattab, Zaharia | 2020 | SIGIR | arxiv.org/abs/2004.12832 | Late interaction: query tokens attend to doc tokens at retrieval time; 10× cheaper than cross-encoder | Alternative to BGE-reranker in latency-sensitive paths |
| cuVS: A GPU-Accelerated Vector Search Library | NVIDIA/rapidsai | 2024 | github.com/rapidsai/cuvs | - | HNSW on GPU via RAFT; p99 < 1ms for 1M 128-dim vectors on A100 | Key reference for GPU acceleration in hnsw_search.py |

---

## State-of-the-Art Models

| Task | Model ID | Benchmark Score | Date | Notes |
|------|----------|----------------|------|-------|
| Dense Embedding (English) | `BAAI/bge-large-en-v1.5` | MTEB avg 64.2 | 2024-03 | #1 open model on MTEB English leaderboard |
| Cross-Encoder Reranking | `BAAI/bge-reranker-large` | BEIR NDCG@10 +8.5pp | 2024-03 | Best open reranker; 549M params |
| Fast Embedding | `sentence-transformers/all-MiniLM-L6-v2` | MTEB avg 56.3 | 2023-12 | 5× faster than BGE-large; 384-dim |
| Multi-lingual Embedding | `BAAI/bge-m3` | MTEB multi 62.1 | 2024-02 | Dense + sparse + ColBERT in one model |
| Long-Context Embedding | `nomic-ai/nomic-embed-text-v1.5` | MTEB avg 62.4, 8192-tok | 2024-02 | Best open long-context embedder |
| Code Embedding | `Salesforce/codet5p-770m` | CodeSearchNet 0.746 | 2023-09 | Best open code retrieval model |
| Sparse Retrieval | `naver/splade-cocondenser-ensemble-distil` | BEIR avg 0.500 | 2022-09 | Best open sparse retrieval |

---

## LLM Prompt Patterns

### 1. RAG Answer Synthesis
```
System: You are an expert assistant. Answer the user's question based ONLY on the provided context. 
Include a citation [N] after each claim, where N is the chunk number.
If the context does not contain enough information, say so explicitly.

Context:
[1] {chunk_1}
[2] {chunk_2}
...
[K] {chunk_k}

User: {question}
```

### 2. Improvement Recommendation from Papers
```
You are a vector search research engineer. Given these recent research papers, 
suggest 3 specific algorithmic improvements for our HNSW + hybrid search implementation.
Each suggestion must include: (a) the paper it is based on, (b) implementation steps, 
(c) expected performance gain with evidence.

Papers:
{paper_summaries}

Current system: HNSW (hnswlib, M=32, ef=200) + BM25 + RRF + BGE-reranker-large
```

### 3. Benchmark Analysis Report
```
Analyze the following vector search benchmark results and provide:
1. Which backend performs best for each workload type (high recall, low latency, large scale)
2. Top 3 actionable recommendations for the user's specific workload
3. Estimated cost (infra + API) per 1M queries for recommended configuration

Benchmark results:
{benchmark_json}

Output as structured Markdown with a comparison table.
```

### 4. Query Expansion (HyDE)
```
Generate a hypothetical document that would perfectly answer this query.
The document should be 2-3 sentences and use technical vocabulary relevant to the domain.

Query: {query}
Domain: {domain}

Hypothetical document:
```

---

## Authoritative Data Sources

| Source | URL | Purpose |
|--------|-----|---------|
| ArXiv cs.DB | arxiv.org/search/?query=vector+search&searchtype=all&start=0 | Weekly new paper crawl |
| ArXiv cs.IR | arxiv.org/search/?query=retrieval+augmented&searchtype=all | RAG and IR papers |
| Semantic Scholar API | api.semanticscholar.org/graph/v1/paper/search | Citation-aware paper search |
| Papers with Code | paperswithcode.com/sota/text-embeddings-on-mteb | SOTA model tracking |
| ANN-Benchmarks | ann-benchmarks.com | ANN algorithm benchmarking data |
| BEIR GitHub | github.com/beir-cellar/beir | IR benchmark datasets |
| MTEB Leaderboard | huggingface.co/spaces/mteb/leaderboard | Embedding model rankings |
| HuggingFace Papers | huggingface.co/papers | Daily AI paper digest |
| hnswlib Releases | github.com/nmslib/hnswlib/releases | HNSW library updates |
| Faiss Wiki | github.com/facebookresearch/faiss/wiki | GPU Faiss documentation |
| cuVS Docs | github.com/rapidsai/cuvs | GPU vector search library |

---

## Self-Update Protocol

```yaml
schedule: "0 2 * * 0"   # Sunday 02:00 local time (weekly)

sources:
  arxiv:
    categories: ["cs.DB", "cs.IR", "cs.LG"]
    keywords: ["vector search", "approximate nearest neighbor", "HNSW", "retrieval augmented generation", "hybrid search", "reranking", "dense retrieval"]
    max_results: 50
    days_back: 7

  semantic_scholar:
    queries:
      - "vector database approximate nearest neighbor"
      - "dense retrieval reranking BEIR"
      - "retrieval augmented generation RAG"
    max_results: 20
    fields: ["title", "authors", "year", "venue", "externalIds", "abstract", "citationCount"]

  papers_with_code:
    tasks:
      - "text-embeddings-on-mteb"
      - "approximate-nearest-neighbor-search"
    scrape: true

  github_releases:
    repos:
      - "nmslib/hnswlib"
      - "facebookresearch/faiss"
      - "qdrant/qdrant"
      - "weaviate/weaviate"
      - "chroma-core/chroma"
      - "rapidsai/cuvs"

scoring:
  recency_window_days: 90
  relevance_keywords: ["HNSW", "ANN", "vector", "retrieval", "embedding", "rerank", "RAG", "BM25", "hybrid"]
  top_n: 20

dedup:
  method: "sha256"
  fields: ["title", "doi_or_url"]
  storage: "sqlite:agent_memory.db:knowledge_hashes"
```

---

## Knowledge Update Log

| Date | Source | Papers Added | Notes |
|------|--------|-------------|-------|
| 2026-06-09 | Manual seed | 15 papers | Initial population with foundational vector search literature |

---

## Benchmark Reference Data

### BEIR Benchmark Baselines (NDCG@10)

| Model | NQ | MSMARCO | TREC-COVID | NFCorpus | Avg-18 |
|-------|-----|---------|------------|---------|-------|
| BM25 | 0.329 | 0.228 | 0.656 | 0.325 | 0.436 |
| SBERT (bi-encoder) | 0.525 | 0.410 | 0.594 | 0.325 | 0.430 |
| BGE-large-en-v1.5 | 0.605 | 0.452 | 0.738 | 0.370 | 0.541 |
| BGE-large + BGE-reranker-large | 0.682 | 0.510 | 0.790 | 0.385 | 0.622 |
| Hybrid (BGE + BM25 RRF) + reranker | **0.698** | **0.523** | **0.801** | **0.392** | **0.637** |

*Target for turbovec-enhanced: NDCG@10 ≥ 0.65 on BEIR/NQ (hybrid + reranker)*

### ANN Latency Baselines (p99, glove-100, 1M vectors)

| Method | Recall@10 | Latency p99 | Hardware |
|--------|----------|------------|---------|
| hnswlib (M=32, ef=200) | 0.987 | 42ms | CPU (8-core) |
| Faiss IVF-PQ | 0.921 | 8ms | CPU (8-core) |
| Faiss GPU IVF-PQ | 0.921 | 1.2ms | A100 GPU |
| cuVS HNSW | 0.985 | 3.1ms | A100 GPU |
| TurboVec Enhanced target | ≥ 0.980 | ≤ 5ms | GPU |
