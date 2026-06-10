"""
TurboVec Enhanced — Automated Test Suite

Tests cover:
  - HNSWSearch (7 tests)
  - HybridSearch (6 tests)
  - RAGPipeline (6 tests)
  - VectorDBAdapter (6 tests)
  - MemoryManager (5 tests)
  - LLMClient (3 tests)
  - HFModelManager (3 tests)
  - Integration (4 tests)
  - CLI smoke tests (3 tests)

Total: 43 tests
"""
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ──────────────────────── Fixtures ────────────────────────

def random_embeddings(n: int, dim: int = 768) -> np.ndarray:
    rng = np.random.default_rng(42)
    vecs = rng.random((n, dim), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def make_ids(n: int, prefix: str = "doc") -> list:
    return [f"{prefix}_{i}" for i in range(n)]


# ──────────────────────── HNSWSearch Tests ────────────────────────

class TestHNSWSearch(unittest.TestCase):

    def setUp(self):
        from agent.modules.hnsw_search import HNSWSearch
        with tempfile.TemporaryDirectory() as td:
            self.index_dir = Path(td)
        self.index_dir.mkdir(exist_ok=True)
        self.hnsw = HNSWSearch(dim=64, M=8, ef_construction=50, ef_search=20,
                               use_gpu=False, index_dir=self.index_dir)

    def test_add_and_search_basic(self):
        vecs = random_embeddings(100, dim=64)
        ids = make_ids(100)
        added = self.hnsw.add(vecs, ids)
        self.assertEqual(added, 100)

        result = self.hnsw.search(vecs[0], k=5)
        self.assertEqual(len(result.ids), 5)
        self.assertEqual(result.ids[0], ids[0])  # top result is itself

    def test_search_returns_k_results(self):
        vecs = random_embeddings(50, dim=64)
        self.hnsw.add(vecs, make_ids(50))
        result = self.hnsw.search(vecs[10], k=10)
        self.assertEqual(len(result.ids), 10)

    def test_latency_ms_positive(self):
        vecs = random_embeddings(20, dim=64)
        self.hnsw.add(vecs, make_ids(20))
        result = self.hnsw.search(vecs[0], k=3)
        self.assertGreater(result.latency_ms, 0)

    def test_get_stats(self):
        vecs = random_embeddings(10, dim=64)
        self.hnsw.add(vecs, make_ids(10))
        stats = self.hnsw.get_stats()
        self.assertEqual(stats["vector_count"], 10)
        self.assertEqual(stats["dim"], 64)

    def test_save_load_index(self):
        vecs = random_embeddings(30, dim=64)
        ids = make_ids(30)
        self.hnsw.add(vecs, ids)
        self.hnsw.save("test_collection")

        from agent.modules.hnsw_search import HNSWSearch
        hnsw2 = HNSWSearch(dim=64, M=8, ef_construction=50, ef_search=20,
                           use_gpu=False, index_dir=self.index_dir)
        hnsw2.load("test_collection")
        result = hnsw2.search(vecs[5], k=3)
        self.assertIn(ids[5], result.ids)

    def test_empty_index_search(self):
        result = self.hnsw.search(random_embeddings(1, dim=64)[0], k=5)
        self.assertEqual(len(result.ids), 0)

    def test_cosine_normalization(self):
        unnormalized = np.array([[3.0, 4.0] + [0.0] * 62], dtype=np.float32)
        processed = self.hnsw._preprocess(unnormalized)
        norm = np.linalg.norm(processed[0])
        self.assertAlmostEqual(norm, 1.0, places=5)


# ──────────────────────── HybridSearch Tests ────────────────────────

class TestHybridSearch(unittest.TestCase):

    def setUp(self):
        from agent.modules.hnsw_search import HNSWSearch
        from agent.modules.hybrid_search import HybridSearch

        mock_hf = MagicMock()
        mock_hf.encode.return_value = random_embeddings(1, dim=64)[0]
        mock_hf.encode_batch.side_effect = lambda texts: random_embeddings(len(texts), dim=64)

        hnsw = HNSWSearch(dim=64, M=8, ef_construction=50, ef_search=20, use_gpu=False)
        self.hybrid = HybridSearch(hnsw=hnsw, hf=mock_hf)

        self.documents = [
            "HNSW graph construction for approximate nearest neighbor search",
            "BM25 Okapi term frequency inverse document frequency retrieval",
            "Dense retrieval with bi-encoders and FAISS index",
            "Cross-encoder reranking improves precision for information retrieval",
            "Vector database benchmarking with BEIR evaluation framework",
        ]
        self.embeddings = random_embeddings(5, dim=64)
        self.ids = make_ids(5)

    def test_index_and_search_returns_results(self):
        self.hybrid.index(self.documents, self.embeddings, self.ids)
        results = self.hybrid.search("HNSW graph search", k=3)
        self.assertGreater(len(results), 0)

    def test_rrf_score_formula(self):
        self.hybrid.index(self.documents, self.embeddings, self.ids)
        results = self.hybrid.search("retrieval ranking", k=5)
        for r in results:
            self.assertGreater(r.rrf_score, 0)
            self.assertLessEqual(r.rrf_score, 1.0 / 60 * 2)

    def test_hybrid_result_has_text(self):
        self.hybrid.index(self.documents, self.embeddings, self.ids)
        results = self.hybrid.search("BM25 keyword", k=2)
        for r in results:
            self.assertIsNotNone(r.text)

    def test_result_count_bounded_by_k(self):
        self.hybrid.index(self.documents, self.embeddings, self.ids)
        results = self.hybrid.search("vector database", k=3)
        self.assertLessEqual(len(results), 3)

    def test_to_dict_format(self):
        self.hybrid.index(self.documents, self.embeddings, self.ids)
        results = self.hybrid.search("embedding search", k=1)
        d = results[0].to_dict()
        self.assertIn("id", d)
        self.assertIn("score", d)
        self.assertIn("dense_score", d)
        self.assertIn("sparse_score", d)

    def test_sparse_only_mode_fallback(self):
        self.hybrid._bm25_index = None
        self.hybrid.hnsw._current_label = 0
        results = self.hybrid.search("test query", k=5)
        self.assertEqual(len(results), 0)


# ──────────────────────── RAGPipeline Tests ────────────────────────

class TestRAGPipeline(unittest.TestCase):

    def setUp(self):
        from agent.modules.rag_pipeline import RAGPipeline

        mock_hf = MagicMock()
        mock_hf.encode.return_value = random_embeddings(1, dim=64)[0]
        mock_hf.encode_batch.side_effect = lambda texts: random_embeddings(len(texts), dim=64)
        mock_hf.rerank.side_effect = lambda pairs: [float(i) for i in range(len(pairs), 0, -1)]

        mock_hybrid = MagicMock()
        mock_hybrid.hnsw = MagicMock()
        mock_hybrid.hnsw.search.return_value = MagicMock(ids=["c0", "c1"], distances=[0.9, 0.8])

        from agent.modules.hybrid_search import HybridResult
        mock_hybrid.search.return_value = [
            HybridResult(id="c0", dense_score=0.9, sparse_score=0.8, rrf_score=0.015, text="HNSW graph search algorithm for fast retrieval of dense vectors"),
            HybridResult(id="c1", dense_score=0.7, sparse_score=0.6, rrf_score=0.013, text="Approximate nearest neighbor benchmarks comparing hnswlib and faiss"),
        ]

        mock_llm = MagicMock()
        mock_llm.complete.return_value = {"text": "HNSW is a graph-based ANN algorithm [1].", "cost_usd": 0.001}

        self.rag = RAGPipeline(hybrid=mock_hybrid, hf=mock_hf, llm=mock_llm, memory=None)

    def test_ingest_returns_chunk_count(self):
        result = self.rag.ingest(
            documents=["This is a test document about vector search."] * 5,
            chunk_strategy="fixed",
            collection="test",
        )
        self.assertGreater(result["chunks_added"], 0)
        self.assertEqual(result["collection"], "test")

    def test_chunk_fixed_strategy(self):
        text = " ".join([f"word{i}" for i in range(600)])
        chunks = self.rag._chunk_fixed(text, doc_id="d1", metadata={})
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(c.id.startswith("d1") for c in chunks))

    def test_chunk_sentence_strategy(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
        chunks = self.rag._chunk_sentence(text, doc_id="d2", metadata={})
        self.assertGreater(len(chunks), 0)

    def test_query_returns_results(self):
        self.rag.ingest(
            documents=["HNSW is a graph-based approximate nearest neighbor algorithm."],
            chunk_strategy="fixed",
            collection="test",
        )
        result = self.rag.query("What is HNSW?", k=5, generate_answer=False, collection="test")
        self.assertIn("results", result)
        self.assertGreater(len(result["results"]), 0)

    def test_query_with_answer_generation(self):
        self.rag._collections["test"] = []
        result = self.rag.query("What is HNSW?", k=5, generate_answer=True, collection="test")
        self.assertIn("answer", result)

    def test_query_latency_tracked(self):
        result = self.rag.query("test query", k=3, generate_answer=False, collection="empty")
        self.assertIn("latency_ms", result)
        self.assertGreaterEqual(result["latency_ms"], 0)


# ──────────────────────── VectorDBAdapter Tests ────────────────────────

class TestVectorDBAdapter(unittest.TestCase):

    def _make_adapter(self, backend: str):
        from agent.modules.vector_db_adapter import VectorDBAdapter
        return VectorDBAdapter.create(backend=backend, config={"dim": 64})

    def test_hnswlib_upsert_and_search(self):
        adapter = self._make_adapter("hnswlib")
        vecs = random_embeddings(50, dim=64)
        ids = make_ids(50)
        adapter.upsert_batch(ids=ids, embeddings=vecs)
        result = adapter.search(query_embedding=vecs[0], k=5)
        self.assertEqual(len(result.ids), 5)
        self.assertEqual(result.ids[0], ids[0])

    def test_faiss_upsert_and_search(self):
        try:
            adapter = self._make_adapter("faiss")
        except Exception:
            self.skipTest("faiss not installed")
        vecs = random_embeddings(50, dim=64)
        ids = make_ids(50)
        adapter.upsert_batch(ids=ids, embeddings=vecs)
        result = adapter.search(query_embedding=vecs[0], k=5)
        self.assertGreater(len(result.ids), 0)

    def test_available_backends_list(self):
        from agent.modules.vector_db_adapter import VectorDBAdapter
        backends = VectorDBAdapter.available_backends()
        self.assertIn("hnswlib", backends)
        self.assertIn("faiss", backends)
        self.assertIn("chroma", backends)

    def test_hnswlib_stats(self):
        adapter = self._make_adapter("hnswlib")
        vecs = random_embeddings(10, dim=64)
        adapter.upsert_batch(ids=make_ids(10), embeddings=vecs)
        stats = adapter.get_stats()
        self.assertEqual(stats["vector_count"], 10)
        self.assertEqual(stats["backend"], "hnswlib")

    def test_hnswlib_delete(self):
        adapter = self._make_adapter("hnswlib")
        vecs = random_embeddings(5, dim=64)
        ids = make_ids(5)
        adapter.upsert_batch(ids=ids, embeddings=vecs)
        adapter.delete(ids[0])
        self.assertNotIn(ids[0], adapter._id_to_int)

    def test_invalid_backend_raises(self):
        from agent.modules.vector_db_adapter import VectorDBAdapter
        with self.assertRaises(ValueError):
            VectorDBAdapter.create(backend="nonexistent_db")


# ──────────────────────── MemoryManager Tests ────────────────────────

class TestMemoryManager(unittest.TestCase):

    def setUp(self):
        from agent.memory.memory_manager import MemoryManager
        self.td = tempfile.mkdtemp()
        self.mem = MemoryManager(db_path=Path(self.td) / "test.db")

    def test_log_and_count_searches(self):
        self.mem.log_search(query="test", k=10, latency_ms=15.3, backend="hnswlib", results_count=10)
        self.assertEqual(self.mem.get_search_count(), 1)

    def test_llm_cost_tracking(self):
        self.mem.log_llm_cost("claude", "claude-opus-4-8", 500, 200, 0.01, "rag")
        summary = self.mem.get_cost_summary()
        self.assertGreater(summary["total_cost_usd"], 0)
        self.assertEqual(len(summary["by_model"]), 1)

    def test_knowledge_dedup(self):
        self.mem.mark_paper_known("Test Paper", "http://arxiv.org/abs/1234", "arxiv")
        self.assertTrue(self.mem.is_known_paper("Test Paper", "http://arxiv.org/abs/1234"))
        self.assertFalse(self.mem.is_known_paper("Different Paper", "http://different.url"))

    def test_benchmark_save_and_retrieve(self):
        self.mem.save_benchmark("synthetic", 10000, ["hnswlib", "faiss"], {"hnswlib": {"p99_ms": 2.0}})
        latest = self.mem.get_latest_benchmark()
        self.assertIsNotNone(latest)
        self.assertEqual(latest["dataset"], "synthetic")

    def test_ingestion_log(self):
        self.mem.log_ingestion("default", 10, 50, "semantic")
        summary = self.mem.get_ingestion_summary()
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["collection"], "default")


# ──────────────────────── LLMClient Tests ────────────────────────

class TestLLMClient(unittest.TestCase):

    def test_cost_per_1k_lookup(self):
        from tools.llm_client import COST_PER_1K
        self.assertIn("claude-opus-4-8", COST_PER_1K)
        self.assertIn("gpt-4o", COST_PER_1K)
        self.assertGreater(COST_PER_1K["claude-opus-4-8"]["output"], 0)

    def test_provider_order_default(self):
        from tools.llm_client import PROVIDER_ORDER
        self.assertEqual(PROVIDER_ORDER[0], "claude")
        self.assertEqual(PROVIDER_ORDER[-1], "ollama")

    @patch("tools.llm_client.LLMClient._call_provider")
    def test_fallback_on_error(self, mock_call):
        from tools.llm_client import LLMClient
        mock_call.side_effect = [RuntimeError("claude rate limit"), {"text": "ok", "provider": "openai", "model": "gpt-4o", "prompt_tokens": 10, "completion_tokens": 5, "cost_usd": 0.0001}]
        client = LLMClient()
        result = client.complete("test prompt", max_tokens=10)
        self.assertEqual(result["text"], "ok")
        self.assertEqual(mock_call.call_count, 2)


# ──────────────────────── HFModelManager Tests ────────────────────────

class TestHFModelManager(unittest.TestCase):

    def setUp(self):
        # Reset singleton for each test
        from tools.hf_model_manager import HFModelManager
        HFModelManager._instance = None

    def test_singleton_pattern(self):
        from tools.hf_model_manager import HFModelManager
        a = HFModelManager()
        b = HFModelManager()
        self.assertIs(a, b)

    def test_fallback_encode_returns_correct_shape(self):
        from tools.hf_model_manager import HFModelManager
        HFModelManager._instance = None
        mgr = HFModelManager(use_gpu=False)
        texts = ["hello world", "approximate nearest neighbor"]
        result = mgr._numpy_fallback_encode(texts)
        self.assertEqual(result.shape, (2, 768))
        self.assertEqual(result.dtype, np.float32)

    def test_fallback_vectors_normalized(self):
        from tools.hf_model_manager import HFModelManager
        HFModelManager._instance = None
        mgr = HFModelManager(use_gpu=False)
        result = mgr._numpy_fallback_encode(["test text"])
        norm = np.linalg.norm(result[0])
        self.assertAlmostEqual(norm, 1.0, places=5)


# ──────────────────────── Integration Tests ────────────────────────

class TestIntegration(unittest.TestCase):

    def test_full_ingest_and_search_pipeline(self):
        """End-to-end: ingest 10 docs → query → get results."""
        from agent.modules.hnsw_search import HNSWSearch
        from agent.modules.hybrid_search import HybridSearch
        from agent.modules.rag_pipeline import RAGPipeline
        from tools.hf_model_manager import HFModelManager

        HFModelManager._instance = None
        hf = HFModelManager(use_gpu=False)

        hnsw = HNSWSearch(dim=768, M=8, ef_construction=50, ef_search=20, use_gpu=False)
        hybrid = HybridSearch(hnsw=hnsw, hf=hf)

        mock_llm = MagicMock()
        mock_llm.complete.return_value = {"text": "RAG answer [1].", "cost_usd": 0.001}

        rag = RAGPipeline(hybrid=hybrid, hf=hf, llm=mock_llm, memory=None)

        docs = [
            "HNSW is a graph-based algorithm for approximate nearest neighbor search.",
            "BM25 Okapi is a sparse retrieval model based on TF-IDF statistics.",
            "Reciprocal Rank Fusion combines multiple ranked lists without score normalization.",
            "BGE-large-en-v1.5 is the top-ranked English embedding model on MTEB benchmark.",
            "Cross-encoder reranking improves precision by jointly encoding query and document.",
            "Qdrant is a vector database optimized for filtering and payload storage.",
            "Chroma is an open-source embedding database for AI applications.",
            "FAISS provides GPU-accelerated similarity search for billion-scale vectors.",
            "Dense retrieval uses bi-encoders to embed queries and documents independently.",
            "RAG systems combine retrieval and generation for knowledge-intensive tasks.",
        ]

        result = rag.ingest(documents=docs, chunk_strategy="fixed", collection="integration_test")
        self.assertGreater(result["chunks_added"], 0)

        query_result = rag.query("What is HNSW?", k=5, generate_answer=False, collection="integration_test")
        self.assertGreater(len(query_result["results"]), 0)
        self.assertIn("latency_ms", query_result)

    def test_adapter_backend_swap(self):
        """Verify hnswlib and faiss produce same top-1 result for identical data."""
        from agent.modules.vector_db_adapter import VectorDBAdapter
        rng = np.random.default_rng(99)
        vecs = rng.random((20, 64), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / norms
        ids = make_ids(20)

        adapters = {}
        for backend in ["hnswlib", "faiss"]:
            try:
                adapter = VectorDBAdapter.create(backend=backend, config={"dim": 64})
                adapter.upsert_batch(ids=ids, embeddings=vecs)
                adapters[backend] = adapter
            except Exception:
                pass

        if len(adapters) < 2:
            self.skipTest("Not all backends available")

        query = vecs[3]
        results = {b: a.search(query_embedding=query, k=3) for b, a in adapters.items()}
        self.assertEqual(results["hnswlib"].ids[0], results["faiss"].ids[0])

    def test_memory_manager_persists_across_instances(self):
        """Verify SQLite data persists when MemoryManager is recreated."""
        from agent.memory.memory_manager import MemoryManager
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "persist_test.db"
            m1 = MemoryManager(db_path=db_path)
            m1.log_search("persist test", 10, 5.0, "hnswlib", 10)
            m2 = MemoryManager(db_path=db_path)
            self.assertEqual(m2.get_search_count(), 1)

    def test_knowledge_updater_marks_known_papers(self):
        """Verify KnowledgeUpdater dedup prevents re-adding same paper."""
        from agent.memory.memory_manager import MemoryManager
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "ku_test.db"
            mem = MemoryManager(db_path=db_path)
            mem.mark_paper_known("HNSW Paper", "http://arxiv.org/abs/1603.09320", "arxiv")
            self.assertTrue(mem.is_known_paper("HNSW Paper", "http://arxiv.org/abs/1603.09320"))
            count_before = mem.get_known_paper_count()
            mem.mark_paper_known("HNSW Paper", "http://arxiv.org/abs/1603.09320", "arxiv")
            self.assertEqual(mem.get_known_paper_count(), count_before)


# ──────────────────────── CLI Smoke Tests ────────────────────────

class TestCLISmoke(unittest.TestCase):

    def test_cli_help_exits_cleanly(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, str(ROOT / "agent" / "main.py"), "--help"],
            capture_output=True, timeout=30
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn(b"turbovec-enhanced", result.stdout.lower() + result.stderr.lower())

    def test_hnsw_module_importable(self):
        from agent.modules.hnsw_search import HNSWSearch, SearchResult
        self.assertTrue(hasattr(HNSWSearch, "add"))
        self.assertTrue(hasattr(HNSWSearch, "search"))

    def test_adapter_factory_importable(self):
        from agent.modules.vector_db_adapter import VectorDBAdapter
        backends = VectorDBAdapter.available_backends()
        self.assertIsInstance(backends, list)
        self.assertGreater(len(backends), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
