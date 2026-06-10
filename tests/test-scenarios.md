# TurboVec Enhanced — Test Scenarios

## Scenario 1: Document Ingestion + Semantic Search

**Trigger:** User ingests a 50-document corpus and queries for relevant content  
**Input:**
```
turbovec-enhanced ingest --dir ./sample_docs --chunk-strategy semantic --collection science
turbovec-enhanced query "What are the main applications of HNSW graphs?" --collection science --k 5
```
**Expected Output:**
- Ingestion: 50 documents chunked into 150-300 semantic chunks, indexed in < 10 seconds
- Query: 5 results returned in < 500ms
- Results include chunks containing "HNSW", "graph", "approximate nearest neighbor"
- All results have score > 0.5

**Pass Criteria:**
- `chunks_added` ≥ 100 in ingestion response
- `latency_ms` < 500 in query response
- Top result `score` > 0.5
- Answer contains citation [1] (if --answer flag used)

---

## Scenario 2: Hybrid Search vs Dense-Only Comparison

**Trigger:** Developer compares hybrid vs dense-only retrieval quality on keyword-heavy queries  
**Input:**
```python
from agent.modules.hybrid_search import HybridSearch

# Query: keyword-heavy, where BM25 should help
query = "BM25 Okapi TF-IDF sparse retrieval implementation"
result_hybrid = hybrid.search(query=query, query_embedding=emb, k=10)
result_dense = hnsw.search(query_embedding=emb, k=10)
```
**Expected Output:**
- Hybrid RRF result differs from dense-only in at least 30% of top-10 positions
- Hybrid recall@5 ≥ dense recall@5 on keyword-heavy queries
- BM25 top result is included in hybrid fusion result set

**Pass Criteria:**
- `rrf_score` correctly computed as `Σ 1/(60 + rank_i)`
- `sparse_score > 0` for at least 5 results
- Different ranking than pure dense retrieval

---

## Scenario 3: BGE Reranker Precision Improvement

**Trigger:** Measure P@1 before and after cross-encoder reranking  
**Input:**
```python
raw_candidates = hybrid.search(query="HNSW graph construction algorithm", k=50)
reranked = rag_pipeline._rerank("HNSW graph construction algorithm", raw_candidates, top_k=10)
```
**Expected Output:**
- Top-1 result after reranking has higher relevance than top-1 before reranking
- Reranking latency < 500ms for 50 candidates
- `rrf_score` of top result after reranking ≥ max pre-reranking score

**Pass Criteria:**
- Reranking returns exactly `top_k` results
- No errors from BGE-reranker-large model loading
- At least 2 results change position after reranking

---

## Scenario 4: Vector DB Backend Consistency

**Trigger:** Same 1000 vectors and 50 queries run against all 5 backends; verify consistent results  
**Input:**
```python
from agent.modules.vector_db_adapter import VectorDBAdapter
import numpy as np

rng = np.random.default_rng(42)
vecs = rng.random((1000, 768), dtype=np.float32)
queries = rng.random((50, 768), dtype=np.float32)
ids = [f"doc_{i}" for i in range(1000)]

backends = ["hnswlib", "faiss"]
results = {}
for backend in backends:
    adapter = VectorDBAdapter.create(backend=backend)
    adapter.upsert_batch(ids=ids, embeddings=vecs)
    results[backend] = [adapter.search(q, k=3) for q in queries]
```
**Expected Output:**
- Top-3 IDs are identical across hnswlib and faiss for all 50 queries
- FAISS scores and hnswlib distances correlate (rank order preserved)
- No exceptions raised for any backend

**Pass Criteria:**
- Top-1 result matches between hnswlib and faiss for ≥ 95% of queries
- All backends return exactly `k` results when corpus has ≥ k vectors
- No `ImportError` for available backends

---

## Scenario 5: Benchmark Suite — Latency Comparison

**Trigger:** Run benchmark comparing hnswlib vs faiss on 100K synthetic vectors  
**Input:**
```
turbovec-enhanced benchmark --backends hnswlib,faiss --dataset synthetic \
    --n-vectors 100000 --n-queries 1000 --k 10 --output bench.md
```
**Expected Output:**
- Benchmark report saved to `bench.md`
- hnswlib p99 latency < 50ms on CPU for 100K 768-dim vectors
- faiss p99 latency < 20ms for 100K vectors (flat IP index)
- Report includes LLM analysis with recommendations

**Pass Criteria:**
- `bench.md` contains a Markdown table with all backends
- Both backends complete 1000 queries without error
- Benchmark results saved to SQLite memory DB
- Report file size > 1KB

---

## Scenario 6: RAG Answer Generation with Citations

**Trigger:** User asks a question against an ingested domain corpus  
**Input:**
```
turbovec-enhanced ingest --file ./sample_docs/vector_search_intro.txt
turbovec-enhanced query "What is Reciprocal Rank Fusion and how does it work?" --answer --k 10
```
**Expected Output:**
```
ANSWER:
Reciprocal Rank Fusion (RRF) is a score fusion method that combines rankings from 
multiple retrieval systems. For each document d, RRF computes: score(d) = Σ 1/(k + rank_i(d)) 
where k=60 and rank_i is the rank in the i-th retrieval system [1]. 
It requires no hyperparameter tuning beyond k and is robust to score scale differences [2].

TOP 10 RESULTS (latency: 320.5ms):
  [1] score=0.8234  id=intro_sem3
  [2] score=0.7891  id=intro_sem1
  ...
```

**Pass Criteria:**
- Answer contains at least one citation in format `[N]`
- `latency_ms` < 3000 (including LLM call)
- `cost_usd` > 0 (LLM was actually called)
- `citations` list contains at least 1 entry with `chunk_id` and `excerpt`

---

## Scenario 7: Knowledge Crawler — New Papers Added

**Trigger:** Run the knowledge updater and verify new papers are appended to SECOND-KNOWLEDGE-BRAIN.md  
**Input:**
```
turbovec-enhanced update-knowledge
```
**Expected Output:**
```
Running knowledge crawler...
Knowledge update complete: {'crawled_total': 45, 'new_added': 12, 'timestamp': '2026-06-16T02:00:00Z'}
```
- SECOND-KNOWLEDGE-BRAIN.md has new section with ISO date stamp
- At least 1 paper from ArXiv cs.DB or cs.IR added
- No duplicate entries (SHA256 dedup working)
- Knowledge update log row added with date and count

**Pass Criteria:**
- `new_added` ≥ 1 (network-dependent, may be 0 in offline mode)
- `SECOND-KNOWLEDGE-BRAIN.md` modified (file mtime changes)
- SQLite `knowledge_hashes` table has ≥ 1 new row
- Running update twice produces 0 new papers (dedup working)

---

## Scenario 8: Graceful Degradation — LLM Unavailable

**Trigger:** All LLM API keys are invalid; verify agent still handles search gracefully  
**Setup:** Set `ANTHROPIC_API_KEY=invalid` and `OPENAI_API_KEY=invalid`; no Ollama running  
**Input:**
```
turbovec-enhanced query "What is HNSW?" --answer --k 5
```
**Expected Output:**
- Search results returned normally (HF embeddings still work)
- Answer field contains fallback message: "[LLM unavailable] Top results:"
- No unhandled exception; exit code 0
- Log message shows "LLM answer generation failed" at WARNING level

**Pass Criteria:**
- `results` list is non-empty (retrieval worked without LLM)
- `answer` is not None but contains fallback text
- `cost_usd` = 0 (no successful LLM call)
- Process exits cleanly without traceback
