"""
TurboVec Enhanced — Core Orchestrator
Manages the vector search pipeline, benchmarking, and knowledge update scheduling.
"""
import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "config" / "agent_config.yaml"
MEMORY_PATH = ROOT / "agent" / "memory"


class TurboVecOrchestrator:
    """Core decision loop for TurboVec Enhanced agent."""

    def __init__(self, config_path: Path = CONFIG_PATH):
        self.config = self._load_config(config_path)
        self.gpu_available = self._detect_gpu()
        self.collections: Dict[str, Any] = {}
        self._hnsw: Optional[Any] = None
        self._hybrid: Optional[Any] = None
        self._rag: Optional[Any] = None
        self._adapter_cache: Dict[str, Any] = {}
        self._memory: Optional[Any] = None
        self._llm: Optional[Any] = None
        self._hf: Optional[Any] = None
        self._scheduler_started = False
        logger.info("TurboVecOrchestrator initialized (GPU=%s)", self.gpu_available)

    # ──────────────────────── Private Helpers ────────────────────────

    def _load_config(self, path: Path) -> Dict[str, Any]:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def _detect_gpu(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _get_memory(self):
        if self._memory is None:
            from agent.memory.memory_manager import MemoryManager
            self._memory = MemoryManager(db_path=MEMORY_PATH / "agent_memory.db")
        return self._memory

    def _get_llm(self):
        if self._llm is None:
            from tools.llm_client import LLMClient
            self._llm = LLMClient()
        return self._llm

    def _get_hf(self):
        if self._hf is None:
            from tools.hf_model_manager import HFModelManager
            self._hf = HFModelManager(use_gpu=self.gpu_available)
        return self._hf

    def _get_hnsw(self):
        if self._hnsw is None:
            from agent.modules.hnsw_search import HNSWSearch
            cfg = self.config.get("hnsw", {})
            self._hnsw = HNSWSearch(
                dim=cfg.get("dim", 768),
                M=cfg.get("M", 32),
                ef_construction=cfg.get("ef_construction", 400),
                ef_search=cfg.get("ef_search", 200),
                use_gpu=self.gpu_available,
            )
        return self._hnsw

    def _get_hybrid(self):
        if self._hybrid is None:
            from agent.modules.hybrid_search import HybridSearch
            self._hybrid = HybridSearch(hnsw=self._get_hnsw(), hf=self._get_hf())
        return self._hybrid

    def _get_rag(self):
        if self._rag is None:
            from agent.modules.rag_pipeline import RAGPipeline
            self._rag = RAGPipeline(
                hybrid=self._get_hybrid(),
                hf=self._get_hf(),
                llm=self._get_llm(),
                memory=self._get_memory(),
                config=self.config.get("rag", {}),
            )
        return self._rag

    def _get_adapter(self, backend: str):
        if backend not in self._adapter_cache:
            from agent.modules.vector_db_adapter import VectorDBAdapter
            cfg = self.config.get("backends", {}).get(backend, {})
            self._adapter_cache[backend] = VectorDBAdapter.create(backend=backend, config=cfg)
        return self._adapter_cache[backend]

    # ──────────────────────── Public API ────────────────────────

    async def ingest(
        self,
        documents: List[str],
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        chunk_strategy: str = "semantic",
        collection: str = "default",
    ) -> Dict[str, Any]:
        """Chunk, embed, and index documents."""
        t0 = time.perf_counter()
        rag = self._get_rag()
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: rag.ingest(
                documents=documents,
                ids=ids,
                metadatas=metadatas,
                chunk_strategy=chunk_strategy,
                collection=collection,
            ),
        )
        self.collections[collection] = True
        elapsed = (time.perf_counter() - t0) * 1000
        result["latency_ms"] = round(elapsed, 2)
        logger.info("Ingested %d docs → %d chunks in %.0fms",
                    len(documents), result.get("chunks_added", 0), elapsed)
        return result

    async def search(
        self,
        query: str,
        k: int = 10,
        collection: str = "default",
        hybrid: bool = True,
        rerank: bool = True,
        generate_answer: bool = False,
        backend: str = "hnswlib",
    ) -> Dict[str, Any]:
        """Search the vector index and optionally generate an LLM answer."""
        t0 = time.perf_counter()
        rag = self._get_rag()
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: rag.query(
                question=query,
                k=k,
                hybrid=hybrid,
                rerank=rerank,
                generate_answer=generate_answer,
                collection=collection,
            ),
        )
        elapsed = (time.perf_counter() - t0) * 1000
        result["latency_ms"] = round(elapsed, 2)
        mem = self._get_memory()
        mem.log_search(query=query, k=k, latency_ms=elapsed,
                       backend=backend, results_count=len(result.get("results", [])))
        return result

    async def run_benchmark(
        self,
        backends: List[str],
        dataset: str = "synthetic",
        n_vectors: int = 10_000,
        n_queries: int = 100,
        k: int = 10,
    ) -> str:
        """Benchmark multiple vector DB backends and return a Markdown report."""
        from agent.modules.vector_db_adapter import VectorDBAdapter
        import numpy as np

        logger.info("Starting benchmark: backends=%s dataset=%s n=%d", backends, dataset, n_vectors)

        dim = self.config.get("hnsw", {}).get("dim", 768)
        rng = np.random.default_rng(42)

        if dataset == "synthetic":
            corpus_embeddings = rng.random((n_vectors, dim), dtype=np.float32)
            corpus_embeddings /= np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
            query_embeddings = rng.random((n_queries, dim), dtype=np.float32)
            query_embeddings /= np.linalg.norm(query_embeddings, axis=1, keepdims=True)
            ground_truth = None
        else:
            corpus_embeddings, query_embeddings, ground_truth = \
                await self._load_beir_dataset(dataset, n_vectors, n_queries, dim)

        results: Dict[str, Dict] = {}
        for backend in backends:
            try:
                adapter = VectorDBAdapter.create(
                    backend=backend,
                    config=self.config.get("backends", {}).get(backend, {}),
                )
                ids = [f"doc_{i}" for i in range(n_vectors)]
                t_idx = time.perf_counter()
                adapter.upsert_batch(ids=ids, embeddings=corpus_embeddings)
                index_ms = (time.perf_counter() - t_idx) * 1000

                latencies = []
                recalls = []
                for qi, qemb in enumerate(query_embeddings):
                    t_q = time.perf_counter()
                    res = adapter.search(query_embedding=qemb, k=k)
                    latencies.append((time.perf_counter() - t_q) * 1000)
                    if ground_truth is not None:
                        relevant = set(ground_truth[qi])
                        retrieved = set(r["id"] for r in res)
                        recalls.append(len(relevant & retrieved) / max(len(relevant), 1))

                results[backend] = {
                    "index_time_ms": round(index_ms, 1),
                    "latency_p50_ms": round(sorted(latencies)[len(latencies) // 2], 2),
                    "latency_p99_ms": round(sorted(latencies)[int(len(latencies) * 0.99)], 2),
                    "recall_at_k": round(sum(recalls) / len(recalls), 4) if recalls else None,
                }
                adapter.close()
            except Exception as exc:
                results[backend] = {"error": str(exc)}

        report = self._render_benchmark_report(results, dataset, n_vectors, n_queries, k)

        mem = self._get_memory()
        mem.save_benchmark(dataset=dataset, n_vectors=n_vectors, backends=backends, results=results)

        try:
            llm = self._get_llm()
            analysis = llm.complete(
                prompt=f"Analyze these benchmark results and provide 3 actionable recommendations:\n{results}",
                max_tokens=512,
            )
            report += f"\n\n## LLM Analysis\n\n{analysis}"
        except Exception:
            pass

        return report

    def _render_benchmark_report(
        self, results: Dict, dataset: str, n_vectors: int, n_queries: int, k: int
    ) -> str:
        lines = [
            f"# TurboVec Enhanced — Benchmark Report",
            f"",
            f"**Dataset:** {dataset}  **Vectors:** {n_vectors:,}  **Queries:** {n_queries}  **k:** {k}",
            f"",
            "| Backend | Index Time (ms) | Latency p50 (ms) | Latency p99 (ms) | Recall@k |",
            "|---------|----------------|-----------------|-----------------|---------|",
        ]
        for backend, r in results.items():
            if "error" in r:
                lines.append(f"| {backend} | ERROR: {r['error']} | — | — | — |")
            else:
                recall = f"{r['recall_at_k']:.4f}" if r.get("recall_at_k") else "N/A"
                lines.append(
                    f"| {backend} | {r['index_time_ms']} | {r['latency_p50_ms']} "
                    f"| {r['latency_p99_ms']} | {recall} |"
                )
        return "\n".join(lines)

    async def _load_beir_dataset(self, dataset: str, n_vectors: int, n_queries: int, dim: int):
        import numpy as np
        logger.warning("BEIR dataset loading not implemented; using synthetic data")
        rng = np.random.default_rng(0)
        corpus = rng.random((n_vectors, dim), dtype=np.float32)
        queries = rng.random((n_queries, dim), dtype=np.float32)
        return corpus, queries, None

    async def update_knowledge(self) -> Dict[str, Any]:
        """Run the knowledge crawler and update SECOND-KNOWLEDGE-BRAIN.md."""
        from tools.knowledge_updater import KnowledgeUpdater
        updater = KnowledgeUpdater(
            brain_path=ROOT / "SECOND-KNOWLEDGE-BRAIN.md",
            db_path=MEMORY_PATH / "agent_memory.db",
        )
        result = await asyncio.get_event_loop().run_in_executor(None, updater.run)
        logger.info("Knowledge update complete: %s", result)
        return result

    async def get_cost_report(self) -> Dict[str, Any]:
        mem = self._get_memory()
        return mem.get_cost_summary()

    async def get_metrics(self) -> Dict[str, Any]:
        mem = self._get_memory()
        return {
            "collections": list(self.collections.keys()),
            "gpu_available": self.gpu_available,
            "search_count": mem.get_search_count(),
            "cost_summary": mem.get_cost_summary(),
        }

    def start_scheduler(self):
        """Start the APScheduler for weekly knowledge updates."""
        if self._scheduler_started:
            return
        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler
            scheduler = AsyncIOScheduler()
            scheduler.add_job(
                self.update_knowledge,
                trigger="cron",
                day_of_week="sun",
                hour=2,
                minute=0,
                id="knowledge_update",
                replace_existing=True,
            )
            scheduler.start()
            self._scheduler_started = True
            logger.info("APScheduler started: weekly knowledge update at Sunday 02:00")
        except ImportError:
            logger.warning("apscheduler not installed; scheduled updates disabled")
