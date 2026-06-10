"""
TurboVec Enhanced — Entry Point
CLI + FastAPI server for GPU-accelerated vector search, hybrid retrieval, and RAG.
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from agent.orchestrator import TurboVecOrchestrator

app = FastAPI(
    title="TurboVec Enhanced",
    description="GPU-accelerated vector search, hybrid retrieval, and RAG pipeline",
    version="1.0.0",
)

_orchestrator: Optional[TurboVecOrchestrator] = None


def get_orchestrator() -> TurboVecOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = TurboVecOrchestrator()
    return _orchestrator


# ──────────────────────── Pydantic schemas ────────────────────────

class IngestRequest(BaseModel):
    documents: List[str] = Field(..., description="List of document texts to ingest")
    ids: Optional[List[str]] = Field(None, description="Optional document IDs")
    metadatas: Optional[List[Dict[str, Any]]] = Field(None, description="Optional metadata per document")
    chunk_strategy: str = Field("semantic", description="Chunking strategy: fixed | sentence | semantic")
    collection: str = Field("default", description="Collection/index name")

class QueryRequest(BaseModel):
    query: str = Field(..., description="Search query text")
    k: int = Field(10, ge=1, le=100, description="Number of results to return")
    collection: str = Field("default", description="Collection/index name")
    hybrid: bool = Field(True, description="Use hybrid dense+BM25 search")
    rerank: bool = Field(True, description="Apply BGE cross-encoder reranking")
    generate_answer: bool = Field(False, description="Generate LLM answer from retrieved context")
    backend: str = Field("hnswlib", description="Vector DB backend: hnswlib | chroma | qdrant | faiss")

class BenchmarkRequest(BaseModel):
    backends: List[str] = Field(["hnswlib", "faiss"], description="Backends to benchmark")
    dataset: str = Field("synthetic", description="Dataset: synthetic | beir/nq | custom")
    n_vectors: int = Field(10000, description="Number of vectors to benchmark on")
    n_queries: int = Field(100, description="Number of queries to run")
    k: int = Field(10, description="Top-k for retrieval")

class SearchResult(BaseModel):
    id: str
    score: float
    text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    results: List[SearchResult]
    answer: Optional[str] = None
    citations: Optional[List[str]] = None
    latency_ms: float
    cost_usd: float = 0.0


# ──────────────────────── REST Endpoints ────────────────────────

@app.get("/health")
async def health():
    orch = get_orchestrator()
    return {
        "status": "ok",
        "collections": list(orch.collections.keys()),
        "gpu_available": orch.gpu_available,
    }


@app.post("/ingest")
async def ingest(req: IngestRequest):
    orch = get_orchestrator()
    try:
        result = await orch.ingest(
            documents=req.documents,
            ids=req.ids,
            metadatas=req.metadatas,
            chunk_strategy=req.chunk_strategy,
            collection=req.collection,
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    orch = get_orchestrator()
    try:
        result = await orch.search(
            query=req.query,
            k=req.k,
            collection=req.collection,
            hybrid=req.hybrid,
            rerank=req.rerank,
            generate_answer=req.generate_answer,
            backend=req.backend,
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/benchmark")
async def benchmark(req: BenchmarkRequest, background_tasks: BackgroundTasks):
    orch = get_orchestrator()
    background_tasks.add_task(
        orch.run_benchmark,
        backends=req.backends,
        dataset=req.dataset,
        n_vectors=req.n_vectors,
        n_queries=req.n_queries,
        k=req.k,
    )
    return {"status": "benchmark started", "backends": req.backends, "dataset": req.dataset}


@app.post("/knowledge/update")
async def knowledge_update(background_tasks: BackgroundTasks):
    orch = get_orchestrator()
    background_tasks.add_task(orch.update_knowledge)
    return {"status": "knowledge update started"}


@app.get("/cost")
async def cost_report():
    orch = get_orchestrator()
    return await orch.get_cost_report()


@app.get("/metrics")
async def metrics():
    orch = get_orchestrator()
    return await orch.get_metrics()


# ──────────────────────── CLI ────────────────────────

def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="turbovec-enhanced",
        description="TurboVec Enhanced — GPU vector search, hybrid retrieval, RAG",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest documents into the vector index")
    p_ingest.add_argument("--dir", type=Path, help="Directory of documents to ingest")
    p_ingest.add_argument("--file", type=Path, help="Single file to ingest")
    p_ingest.add_argument("--collection", default="default", help="Collection name")
    p_ingest.add_argument("--chunk-strategy", default="semantic",
                          choices=["fixed", "sentence", "semantic"], help="Chunking strategy")
    p_ingest.add_argument("--backend", default="hnswlib",
                          choices=["hnswlib", "chroma", "qdrant", "faiss"], help="Vector DB backend")

    # query
    p_query = sub.add_parser("query", help="Search the vector index")
    p_query.add_argument("question", help="Search query")
    p_query.add_argument("--collection", default="default")
    p_query.add_argument("--k", type=int, default=10)
    p_query.add_argument("--no-hybrid", action="store_true", help="Disable BM25 hybrid search")
    p_query.add_argument("--no-rerank", action="store_true", help="Disable BGE reranking")
    p_query.add_argument("--answer", action="store_true", help="Generate LLM answer")
    p_query.add_argument("--backend", default="hnswlib")

    # benchmark
    p_bench = sub.add_parser("benchmark", help="Run vector DB benchmark comparison")
    p_bench.add_argument("--backends", default="hnswlib,faiss",
                         help="Comma-separated list of backends")
    p_bench.add_argument("--dataset", default="synthetic",
                         choices=["synthetic", "beir/nq", "glove-100"])
    p_bench.add_argument("--n-vectors", type=int, default=10000)
    p_bench.add_argument("--n-queries", type=int, default=100)
    p_bench.add_argument("--k", type=int, default=10)
    p_bench.add_argument("--output", type=Path, default=Path("benchmark_report.md"))

    # serve
    p_serve = sub.add_parser("serve", help="Start the FastAPI REST server")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8016)
    p_serve.add_argument("--reload", action="store_true")

    # update-knowledge
    sub.add_parser("update-knowledge", help="Run the knowledge crawler immediately")

    # cost-report
    sub.add_parser("cost-report", help="Show LLM API cost summary")

    return parser


async def run_cli(args: argparse.Namespace):
    orch = get_orchestrator()

    if args.command == "ingest":
        documents = []
        if args.dir and args.dir.is_dir():
            for f in args.dir.rglob("*"):
                if f.suffix in {".txt", ".md", ".pdf", ".html"}:
                    documents.append(f.read_text(encoding="utf-8", errors="replace"))
        elif args.file and args.file.is_file():
            documents.append(args.file.read_text(encoding="utf-8", errors="replace"))
        else:
            print("ERROR: Provide --dir or --file", file=sys.stderr)
            sys.exit(1)

        result = await orch.ingest(
            documents=documents,
            chunk_strategy=args.chunk_strategy,
            collection=args.collection,
        )
        print(json.dumps(result, indent=2))

    elif args.command == "query":
        result = await orch.search(
            query=args.question,
            k=args.k,
            collection=args.collection,
            hybrid=not args.no_hybrid,
            rerank=not args.no_rerank,
            generate_answer=args.answer,
            backend=args.backend,
        )
        print(f"\n{'='*60}")
        if result.get("answer"):
            print(f"ANSWER:\n{result['answer']}\n")
        print(f"TOP {args.k} RESULTS (latency: {result['latency_ms']:.1f}ms):")
        for i, r in enumerate(result["results"], 1):
            print(f"  [{i}] score={r['score']:.4f}  id={r['id']}")
            if r.get("text"):
                print(f"      {r['text'][:120]}...")

    elif args.command == "benchmark":
        backends = [b.strip() for b in args.backends.split(",")]
        report = await orch.run_benchmark(
            backends=backends,
            dataset=args.dataset,
            n_vectors=args.n_vectors,
            n_queries=args.n_queries,
            k=args.k,
        )
        args.output.write_text(report, encoding="utf-8")
        print(f"Benchmark report saved to {args.output}")
        print(report[:2000])

    elif args.command == "update-knowledge":
        print("Running knowledge crawler...")
        result = await orch.update_knowledge()
        print(f"Knowledge update complete: {result}")

    elif args.command == "cost-report":
        report = await orch.get_cost_report()
        print(json.dumps(report, indent=2))


def main():
    parser = build_cli()
    args = parser.parse_args()

    if args.command == "serve":
        uvicorn.run(
            "agent.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
    else:
        asyncio.run(run_cli(args))


if __name__ == "__main__":
    main()
