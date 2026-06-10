"""
Memory Manager — Persistent SQLite storage for TurboVec Enhanced agent.

Tables:
  search_logs       - Per-query search metadata (latency, backend, results count)
  benchmark_results - Backend benchmark comparisons
  llm_cost_log      - Per-call LLM token usage and USD cost
  knowledge_hashes  - SHA256 dedup for SECOND-KNOWLEDGE-BRAIN.md crawler
  ingestion_log     - Document ingestion history
"""
import hashlib
import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS search_logs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    query       TEXT,
    k           INTEGER,
    latency_ms  REAL,
    backend     TEXT,
    results_count INTEGER,
    hybrid      INTEGER DEFAULT 1,
    reranked    INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS benchmark_results (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    dataset     TEXT,
    n_vectors   INTEGER,
    backends    TEXT,
    results_json TEXT
);

CREATE TABLE IF NOT EXISTS llm_cost_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT NOT NULL,
    provider        TEXT,
    model           TEXT,
    prompt_tokens   INTEGER,
    completion_tokens INTEGER,
    cost_usd        REAL,
    use_case        TEXT
);

CREATE TABLE IF NOT EXISTS knowledge_hashes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    hash_sha256 TEXT UNIQUE NOT NULL,
    title       TEXT,
    source      TEXT
);

CREATE TABLE IF NOT EXISTS ingestion_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT NOT NULL,
    collection      TEXT,
    documents_count INTEGER,
    chunks_count    INTEGER,
    chunk_strategy  TEXT
);
"""


class MemoryManager:
    """SQLite-backed persistent memory for TurboVec Enhanced."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(SCHEMA)
        logger.debug("Memory DB initialized at %s", self.db_path)

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # ──────────────────────── Search Logging ────────────────────────

    def log_search(
        self,
        query: str,
        k: int,
        latency_ms: float,
        backend: str,
        results_count: int,
        hybrid: bool = True,
        reranked: bool = True,
    ):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO search_logs (ts, query, k, latency_ms, backend, results_count, hybrid, reranked) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (self._now(), query[:500], k, latency_ms, backend, results_count,
                 int(hybrid), int(reranked)),
            )

    def get_search_count(self) -> int:
        with self._conn() as conn:
            row = conn.execute("SELECT COUNT(*) FROM search_logs").fetchone()
            return row[0] if row else 0

    def get_search_stats(self) -> Dict:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as n, AVG(latency_ms) as avg_ms, "
                "MIN(latency_ms) as min_ms, MAX(latency_ms) as max_ms "
                "FROM search_logs"
            ).fetchone()
            return dict(row) if row else {}

    # ──────────────────────── Benchmark Results ────────────────────────

    def save_benchmark(
        self,
        dataset: str,
        n_vectors: int,
        backends: List[str],
        results: Dict[str, Any],
    ):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO benchmark_results (ts, dataset, n_vectors, backends, results_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (self._now(), dataset, n_vectors, json.dumps(backends), json.dumps(results)),
            )

    def get_latest_benchmark(self, dataset: Optional[str] = None) -> Optional[Dict]:
        with self._conn() as conn:
            if dataset:
                row = conn.execute(
                    "SELECT * FROM benchmark_results WHERE dataset=? ORDER BY ts DESC LIMIT 1",
                    (dataset,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM benchmark_results ORDER BY ts DESC LIMIT 1"
                ).fetchone()
            if row:
                d = dict(row)
                d["results_json"] = json.loads(d["results_json"])
                d["backends"] = json.loads(d["backends"])
                return d
            return None

    # ──────────────────────── LLM Cost Tracking ────────────────────────

    def log_llm_cost(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        use_case: str = "",
    ):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO llm_cost_log "
                "(ts, provider, model, prompt_tokens, completion_tokens, cost_usd, use_case) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (self._now(), provider, model, prompt_tokens, completion_tokens, cost_usd, use_case),
            )

    def get_cost_summary(self) -> Dict:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT provider, model, "
                "SUM(prompt_tokens) as total_prompt_tokens, "
                "SUM(completion_tokens) as total_completion_tokens, "
                "SUM(cost_usd) as total_cost_usd, "
                "COUNT(*) as call_count "
                "FROM llm_cost_log GROUP BY provider, model ORDER BY total_cost_usd DESC"
            ).fetchall()
            total = conn.execute("SELECT SUM(cost_usd) as total FROM llm_cost_log").fetchone()
            return {
                "by_model": [dict(r) for r in rows],
                "total_cost_usd": round(total["total"] or 0.0, 6),
            }

    # ──────────────────────── Knowledge Deduplication ────────────────────────

    def is_known_paper(self, title: str, url: str = "") -> bool:
        raw = f"{title.lower().strip()}{url.lower().strip()}"
        h = hashlib.sha256(raw.encode()).hexdigest()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id FROM knowledge_hashes WHERE hash_sha256=?", (h,)
            ).fetchone()
            return row is not None

    def mark_paper_known(self, title: str, url: str = "", source: str = ""):
        raw = f"{title.lower().strip()}{url.lower().strip()}"
        h = hashlib.sha256(raw.encode()).hexdigest()
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO knowledge_hashes (ts, hash_sha256, title, source) "
                "VALUES (?, ?, ?, ?)",
                (self._now(), h, title[:500], source),
            )

    def get_known_paper_count(self) -> int:
        with self._conn() as conn:
            row = conn.execute("SELECT COUNT(*) FROM knowledge_hashes").fetchone()
            return row[0] if row else 0

    # ──────────────────────── Ingestion Log ────────────────────────

    def log_ingestion(
        self,
        collection: str,
        documents_count: int,
        chunks_count: int,
        chunk_strategy: str,
    ):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO ingestion_log (ts, collection, documents_count, chunks_count, chunk_strategy) "
                "VALUES (?, ?, ?, ?, ?)",
                (self._now(), collection, documents_count, chunks_count, chunk_strategy),
            )

    def get_ingestion_summary(self) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT collection, SUM(documents_count) as docs, SUM(chunks_count) as chunks "
                "FROM ingestion_log GROUP BY collection"
            ).fetchall()
            return [dict(r) for r in rows]
