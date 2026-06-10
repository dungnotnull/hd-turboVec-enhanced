"""
Knowledge Updater — Research paper crawler for TurboVec Enhanced.

Pipeline:
  1. ArXiv XML API  → fetch cs.DB + cs.IR + cs.LG papers (last 7 days)
  2. Semantic Scholar Graph API → 3 domain queries
  3. Papers with Code → vector search leaderboard updates
  4. GitHub Releases → hnswlib, faiss, qdrant, weaviate, chroma, cuvs
  5. Score papers by recency × relevance (keyword overlap)
  6. SHA256 dedup via MemoryManager
  7. Append top-N entries to SECOND-KNOWLEDGE-BRAIN.md with ISO date stamp

Schedule: Weekly (Sunday 02:00) via APScheduler
"""
import hashlib
import json
import logging
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

ARXIV_CATEGORIES = ["cs.DB", "cs.IR", "cs.LG"]
DOMAIN_KEYWORDS = [
    "HNSW", "ANN", "approximate nearest neighbor", "vector search", "vector database",
    "dense retrieval", "retrieval augmented generation", "RAG", "BM25", "hybrid retrieval",
    "embedding", "rerank", "cross-encoder", "bi-encoder", "BEIR", "MTEB", "FAISS",
    "qdrant", "weaviate", "chroma", "hnswlib", "cuVS", "RAFT GPU",
]
SEMANTIC_SCHOLAR_QUERIES = [
    "vector database approximate nearest neighbor",
    "dense retrieval reranking BEIR benchmark",
    "retrieval augmented generation knowledge",
]
GITHUB_REPOS = [
    "nmslib/hnswlib", "facebookresearch/faiss", "qdrant/qdrant",
    "weaviate/weaviate", "chroma-core/chroma", "rapidsai/cuvs",
]
RECENCY_WINDOW_DAYS = 90
TOP_N_PAPERS = 20


class KnowledgeUpdater:
    """Crawls research sources and appends new papers to SECOND-KNOWLEDGE-BRAIN.md."""

    def __init__(self, brain_path: Path, db_path: Path):
        self.brain_path = Path(brain_path)
        self.db_path = Path(db_path)

    def _get_memory(self):
        from agent.memory.memory_manager import MemoryManager
        return MemoryManager(db_path=self.db_path)

    def run(self) -> Dict[str, Any]:
        """Execute the full crawl pipeline. Returns summary dict."""
        logger.info("Knowledge crawl started")
        papers: List[Dict] = []

        papers.extend(self._crawl_arxiv())
        papers.extend(self._crawl_semantic_scholar())
        papers.extend(self._crawl_github_releases())

        scored = self._score_and_deduplicate(papers)
        new_papers = scored[:TOP_N_PAPERS]

        if new_papers:
            self._append_to_brain(new_papers)

        summary = {
            "crawled_total": len(papers),
            "new_added": len(new_papers),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        logger.info("Knowledge crawl complete: %s", summary)
        return summary

    # ──────────────────────── Crawlers ────────────────────────

    def _crawl_arxiv(self) -> List[Dict]:
        """Fetch recent papers from ArXiv XML API."""
        papers = []
        try:
            import urllib.request
            cutoff = datetime.now(timezone.utc) - timedelta(days=7)
            for category in ARXIV_CATEGORIES:
                query = urllib.parse.quote(
                    f"cat:{category} AND (HNSW OR 'vector search' OR 'retrieval augmented' OR 'dense retrieval')"
                )
                url = (
                    f"http://export.arxiv.org/api/query?"
                    f"search_query={query}&max_results=30&sortBy=submittedDate&sortOrder=descending"
                )
                try:
                    with urllib.request.urlopen(url, timeout=15) as resp:
                        xml_data = resp.read().decode("utf-8")
                    papers.extend(self._parse_arxiv_xml(xml_data, cutoff))
                    time.sleep(3)
                except Exception as exc:
                    logger.warning("ArXiv fetch failed for %s: %s", category, exc)
        except Exception as exc:
            logger.error("ArXiv crawler error: %s", exc)
        return papers

    def _parse_arxiv_xml(self, xml_data: str, cutoff: datetime) -> List[Dict]:
        """Parse ArXiv Atom XML feed into paper dicts."""
        papers = []
        try:
            ns = {
                "atom": "http://www.w3.org/2005/Atom",
                "arxiv": "http://arxiv.org/schemas/atom",
            }
            root = ET.fromstring(xml_data)
            for entry in root.findall("atom:entry", ns):
                title_el = entry.find("atom:title", ns)
                abstract_el = entry.find("atom:summary", ns)
                published_el = entry.find("atom:published", ns)
                id_el = entry.find("atom:id", ns)
                authors_els = entry.findall("atom:author/atom:name", ns)

                if title_el is None or published_el is None:
                    continue

                published_str = published_el.text.strip()
                try:
                    published = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
                except ValueError:
                    continue

                if published < cutoff:
                    continue

                papers.append({
                    "title": title_el.text.strip(),
                    "abstract": abstract_el.text.strip() if abstract_el is not None else "",
                    "authors": ", ".join(el.text for el in authors_els[:5]),
                    "published": published.date().isoformat(),
                    "url": id_el.text.strip() if id_el is not None else "",
                    "venue": "ArXiv",
                    "source": "arxiv",
                })
        except ET.ParseError as exc:
            logger.warning("ArXiv XML parse error: %s", exc)
        return papers

    def _crawl_semantic_scholar(self) -> List[Dict]:
        """Query Semantic Scholar Graph API."""
        papers = []
        try:
            import urllib.request
            fields = "title,authors,year,venue,externalIds,abstract,citationCount"
            for query in SEMANTIC_SCHOLAR_QUERIES:
                encoded = urllib.parse.quote(query)
                url = (
                    f"https://api.semanticscholar.org/graph/v1/paper/search"
                    f"?query={encoded}&limit=10&fields={fields}"
                )
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "TurboVecEnhanced/1.0"})
                    with urllib.request.urlopen(req, timeout=15) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                    for paper in data.get("data", []):
                        doi = paper.get("externalIds", {}).get("DOI", "")
                        arxiv_id = paper.get("externalIds", {}).get("ArXiv", "")
                        url_ref = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else f"https://doi.org/{doi}"
                        papers.append({
                            "title": paper.get("title", ""),
                            "abstract": paper.get("abstract", ""),
                            "authors": ", ".join(a.get("name", "") for a in paper.get("authors", [])[:5]),
                            "published": str(paper.get("year", "")),
                            "url": url_ref,
                            "venue": paper.get("venue", ""),
                            "citations": paper.get("citationCount", 0),
                            "source": "semantic_scholar",
                        })
                    time.sleep(2)
                except Exception as exc:
                    logger.warning("Semantic Scholar query failed: %s", exc)
        except Exception as exc:
            logger.error("Semantic Scholar crawler error: %s", exc)
        return papers

    def _crawl_github_releases(self) -> List[Dict]:
        """Fetch latest GitHub release notes for key repos."""
        papers = []
        try:
            import urllib.request
            for repo in GITHUB_REPOS:
                url = f"https://api.github.com/repos/{repo}/releases/latest"
                try:
                    req = urllib.request.Request(
                        url, headers={"User-Agent": "TurboVecEnhanced/1.0", "Accept": "application/vnd.github.v3+json"}
                    )
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                    published = data.get("published_at", "")[:10]
                    body = data.get("body", "")[:1000]
                    tag = data.get("tag_name", "")
                    papers.append({
                        "title": f"{repo} {tag} Release Notes",
                        "abstract": body,
                        "authors": repo.split("/")[0],
                        "published": published,
                        "url": data.get("html_url", ""),
                        "venue": "GitHub Releases",
                        "source": "github",
                    })
                    time.sleep(1)
                except Exception as exc:
                    logger.warning("GitHub release fetch failed for %s: %s", repo, exc)
        except Exception as exc:
            logger.error("GitHub releases crawler error: %s", exc)
        return papers

    # ──────────────────────── Scoring & Dedup ────────────────────────

    def _score_and_deduplicate(self, papers: List[Dict]) -> List[Dict]:
        """Score by recency × relevance, deduplicate via SHA256."""
        mem = self._get_memory()
        now = datetime.now(timezone.utc).date()
        scored = []

        for p in papers:
            title = p.get("title", "")
            url = p.get("url", "")
            if not title:
                continue
            if mem.is_known_paper(title=title, url=url):
                continue

            recency = self._recency_score(p.get("published", ""), now)
            relevance = self._relevance_score(title + " " + p.get("abstract", ""))
            score = recency * relevance
            if score <= 0:
                continue
            p["_score"] = score
            scored.append(p)

        scored.sort(key=lambda x: x["_score"], reverse=True)

        for p in scored[:TOP_N_PAPERS]:
            mem.mark_paper_known(title=p.get("title", ""), url=p.get("url", ""), source=p.get("source", ""))

        return scored

    def _recency_score(self, published_str: str, now) -> float:
        """Linear decay: 1.0 if today, 0.0 if older than RECENCY_WINDOW_DAYS."""
        try:
            if not published_str or len(published_str) < 4:
                return 0.3
            pub_date = datetime.strptime(published_str[:10], "%Y-%m-%d").date()
            age_days = (now - pub_date).days
            if age_days < 0:
                return 1.0
            return max(0.0, 1.0 - age_days / RECENCY_WINDOW_DAYS)
        except ValueError:
            return 0.3

    def _relevance_score(self, text: str) -> float:
        """Count keyword hits normalized by max possible hits."""
        text_lower = text.lower()
        hits = sum(1 for kw in DOMAIN_KEYWORDS if kw.lower() in text_lower)
        return min(1.0, hits / max(len(DOMAIN_KEYWORDS) * 0.3, 1))

    # ──────────────────────── Brain Update ────────────────────────

    def _append_to_brain(self, papers: List[Dict]):
        """Append new papers to SECOND-KNOWLEDGE-BRAIN.md."""
        if not self.brain_path.exists():
            logger.warning("SECOND-KNOWLEDGE-BRAIN.md not found at %s", self.brain_path)
            return

        date_stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        lines = [
            f"\n\n### Knowledge Update — {date_stamp} ({len(papers)} new papers)\n",
            "| Title | Authors | Published | Venue | Link | Key Finding | Relevance Score |",
            "|-------|---------|-----------|-------|------|-------------|----------------|",
        ]
        for p in papers:
            title = p.get("title", "")[:80].replace("|", "–")
            authors = p.get("authors", "")[:40].replace("|", "–")
            published = p.get("published", "")[:10]
            venue = p.get("venue", "")[:30].replace("|", "–")
            url = p.get("url", "")
            abstract = p.get("abstract", "")[:120].replace("|", "–").replace("\n", " ")
            score = round(p.get("_score", 0), 3)
            lines.append(f"| {title} | {authors} | {published} | {venue} | {url} | {abstract} | {score} |")

        update_text = "\n".join(lines)
        existing = self.brain_path.read_text(encoding="utf-8")

        log_marker = "## Knowledge Update Log"
        if log_marker in existing:
            insert_pos = existing.index(log_marker) + len(log_marker)
            new_log_entry = f"\n| {date_stamp} | Crawl | {len(papers)} papers | ArXiv + Semantic Scholar + GitHub |"
            existing = (
                existing[:insert_pos]
                + new_log_entry
                + existing[insert_pos:]
                + update_text
            )
        else:
            existing = existing + update_text

        self.brain_path.write_text(existing, encoding="utf-8")
        logger.info("Appended %d new papers to SECOND-KNOWLEDGE-BRAIN.md", len(papers))
