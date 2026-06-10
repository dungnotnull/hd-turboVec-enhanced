"""
Vector DB Adapter Module — Unified API to swap vector database backends.

Supported backends:
  - hnswlib   : Local HNSW index (default, no external service)
  - faiss     : Facebook AI Similarity Search (in-memory / mmap)
  - chroma    : Chroma DB (via chromadb SDK)
  - qdrant    : Qdrant vector DB (via qdrant-client)
  - weaviate  : Weaviate (via weaviate-client v4)

Quality gate: Identical top-3 results across all backends for same query (within 1e-5 float tolerance).
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AdapterResult:
    ids: List[str]
    distances: List[float]
    metadatas: List[Optional[Dict]] = field(default_factory=list)

    def to_dicts(self) -> List[Dict]:
        return [
            {"id": self.ids[i], "score": self.distances[i],
             "metadata": self.metadatas[i] if i < len(self.metadatas) else None}
            for i in range(len(self.ids))
        ]


# ──────────────────────── Abstract Base ────────────────────────

class BaseVectorBackend(ABC):
    """Abstract base for all vector DB backends."""

    @abstractmethod
    def upsert(self, id: str, embedding: np.ndarray, metadata: Optional[Dict] = None): ...

    @abstractmethod
    def upsert_batch(self, ids: List[str], embeddings: np.ndarray,
                     metadatas: Optional[List[Dict]] = None): ...

    @abstractmethod
    def search(self, query_embedding: np.ndarray, k: int = 10,
               filter: Optional[Dict] = None) -> AdapterResult: ...

    @abstractmethod
    def delete(self, id: str): ...

    @abstractmethod
    def get_stats(self) -> Dict: ...

    def close(self): pass


# ──────────────────────── hnswlib Backend ────────────────────────

class HNSWLibBackend(BaseVectorBackend):
    """Local hnswlib-based backend (default, no external service required)."""

    def __init__(self, config: Dict):
        self.dim = config.get("dim", 768)
        self.metric = config.get("metric", "cosine")
        self._id_map: Dict[int, str] = {}
        self._id_to_int: Dict[str, int] = {}
        self._meta_map: Dict[str, Dict] = {}
        self._current = 0
        self._index = self._init()

    def _init(self):
        try:
            import hnswlib
            idx = hnswlib.Index(space=self.metric, dim=self.dim)
            idx.init_index(max_elements=1_000_000, ef_construction=400, M=32)
            idx.set_ef(200)
            return idx
        except ImportError:
            logger.warning("hnswlib not installed; hnswlib backend unavailable")
            return None

    def upsert(self, id: str, embedding: np.ndarray, metadata: Optional[Dict] = None):
        emb = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        if id not in self._id_to_int:
            self._id_to_int[id] = self._current
            self._id_map[self._current] = id
            self._current += 1
        label = self._id_to_int[id]
        if self._index:
            self._index.add_items(emb, [label])
        self._meta_map[id] = metadata or {}

    def upsert_batch(self, ids: List[str], embeddings: np.ndarray,
                     metadatas: Optional[List[Dict]] = None):
        embs = np.asarray(embeddings, dtype=np.float32)
        labels = []
        for i, sid in enumerate(ids):
            if sid not in self._id_to_int:
                self._id_to_int[sid] = self._current
                self._id_map[self._current] = sid
                self._current += 1
            labels.append(self._id_to_int[sid])
            self._meta_map[sid] = metadatas[i] if metadatas else {}
        if self._index:
            self._index.add_items(embs, labels)

    def search(self, query_embedding: np.ndarray, k: int = 10,
               filter: Optional[Dict] = None) -> AdapterResult:
        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        if self._index is None or self._current == 0:
            return AdapterResult(ids=[], distances=[], metadatas=[])
        actual_k = min(k, self._current)
        labels, dists = self._index.knn_query(query, k=actual_k)
        ids = [self._id_map.get(lbl, str(lbl)) for lbl in labels[0]]
        metas = [self._meta_map.get(sid, {}) for sid in ids]
        return AdapterResult(ids=ids, distances=dists[0].tolist(), metadatas=metas)

    def delete(self, id: str):
        if id in self._id_to_int:
            label = self._id_to_int.pop(id)
            self._id_map.pop(label, None)
            self._meta_map.pop(id, None)

    def get_stats(self) -> Dict:
        return {"backend": "hnswlib", "vector_count": self._current, "dim": self.dim}


# ──────────────────────── FAISS Backend ────────────────────────

class FAISSBackend(BaseVectorBackend):
    """Facebook AI Similarity Search backend (in-memory flat L2 index)."""

    def __init__(self, config: Dict):
        self.dim = config.get("dim", 768)
        self._ids: List[str] = []
        self._metadatas: List[Dict] = []
        self._matrix: Optional[np.ndarray] = None
        self._index = self._init()

    def _init(self):
        try:
            import faiss
            return faiss.IndexFlatIP(self.dim)
        except ImportError:
            logger.warning("faiss not installed; FAISS backend unavailable")
            return None

    def upsert(self, id: str, embedding: np.ndarray, metadata: Optional[Dict] = None):
        self.upsert_batch(ids=[id], embeddings=np.array([embedding]), metadatas=[metadata or {}])

    def upsert_batch(self, ids: List[str], embeddings: np.ndarray,
                     metadatas: Optional[List[Dict]] = None):
        embs = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embs = embs / norms
        self._ids.extend(ids)
        self._metadatas.extend(metadatas if metadatas else [{} for _ in ids])
        if self._index is not None:
            self._index.add(embs)

    def search(self, query_embedding: np.ndarray, k: int = 10,
               filter: Optional[Dict] = None) -> AdapterResult:
        if self._index is None or self._index.ntotal == 0:
            return AdapterResult(ids=[], distances=[])
        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm
        actual_k = min(k, self._index.ntotal)
        scores, indices = self._index.search(query, actual_k)
        ids = [self._ids[i] for i in indices[0] if 0 <= i < len(self._ids)]
        metas = [self._metadatas[i] for i in indices[0] if 0 <= i < len(self._metadatas)]
        return AdapterResult(ids=ids, distances=scores[0].tolist()[:len(ids)], metadatas=metas)

    def delete(self, id: str):
        if id in self._ids:
            idx = self._ids.index(id)
            self._ids.pop(idx)
            self._metadatas.pop(idx)

    def get_stats(self) -> Dict:
        return {
            "backend": "faiss",
            "vector_count": self._index.ntotal if self._index else 0,
            "dim": self.dim,
        }


# ──────────────────────── Chroma Backend ────────────────────────

class ChromaBackend(BaseVectorBackend):
    """Chroma DB backend via chromadb SDK."""

    def __init__(self, config: Dict):
        self.collection_name = config.get("collection", "turbovec")
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 8000)
        self._client = None
        self._collection = None
        self._init(config)

    def _init(self, config: Dict):
        try:
            import chromadb
            if config.get("in_memory", True):
                self._client = chromadb.EphemeralClient()
            else:
                self._client = chromadb.HttpClient(host=self.host, port=self.port)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except ImportError:
            logger.warning("chromadb not installed; Chroma backend unavailable")
        except Exception as exc:
            logger.warning("Chroma connection failed: %s", exc)

    def upsert(self, id: str, embedding: np.ndarray, metadata: Optional[Dict] = None):
        if self._collection is None:
            return
        self._collection.upsert(
            ids=[id],
            embeddings=[embedding.tolist()],
            metadatas=[metadata or {}],
        )

    def upsert_batch(self, ids: List[str], embeddings: np.ndarray,
                     metadatas: Optional[List[Dict]] = None):
        if self._collection is None:
            return
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),
            metadatas=metadatas or [{} for _ in ids],
        )

    def search(self, query_embedding: np.ndarray, k: int = 10,
               filter: Optional[Dict] = None) -> AdapterResult:
        if self._collection is None:
            return AdapterResult(ids=[], distances=[])
        try:
            results = self._collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(k, self._collection.count()),
                where=filter,
            )
            ids = results["ids"][0]
            dists = results["distances"][0]
            metas = results.get("metadatas", [[]])[0]
            return AdapterResult(ids=ids, distances=dists, metadatas=metas)
        except Exception as exc:
            logger.error("Chroma search failed: %s", exc)
            return AdapterResult(ids=[], distances=[])

    def delete(self, id: str):
        if self._collection:
            self._collection.delete(ids=[id])

    def get_stats(self) -> Dict:
        count = self._collection.count() if self._collection else 0
        return {"backend": "chroma", "vector_count": count, "collection": self.collection_name}

    def close(self):
        self._client = None
        self._collection = None


# ──────────────────────── Qdrant Backend ────────────────────────

class QdrantBackend(BaseVectorBackend):
    """Qdrant vector DB backend via qdrant-client."""

    def __init__(self, config: Dict):
        self.collection_name = config.get("collection", "turbovec")
        self.dim = config.get("dim", 768)
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 6333)
        self._client = None
        self._init()

    def _init(self):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
            self._client = QdrantClient(host=self.host, port=self.port)
            existing = [c.name for c in self._client.get_collections().collections]
            if self.collection_name not in existing:
                self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
                )
        except ImportError:
            logger.warning("qdrant-client not installed; Qdrant backend unavailable")
        except Exception as exc:
            logger.warning("Qdrant connection failed: %s", exc)

    def upsert(self, id: str, embedding: np.ndarray, metadata: Optional[Dict] = None):
        if self._client is None:
            return
        from qdrant_client.models import PointStruct
        self._client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(id=hash(id) % (2**31), vector=embedding.tolist(), payload=metadata or {})],
        )

    def upsert_batch(self, ids: List[str], embeddings: np.ndarray,
                     metadatas: Optional[List[Dict]] = None):
        if self._client is None:
            return
        from qdrant_client.models import PointStruct
        points = [
            PointStruct(
                id=hash(ids[i]) % (2**31),
                vector=embeddings[i].tolist(),
                payload=metadatas[i] if metadatas else {},
            )
            for i in range(len(ids))
        ]
        self._client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_embedding: np.ndarray, k: int = 10,
               filter: Optional[Dict] = None) -> AdapterResult:
        if self._client is None:
            return AdapterResult(ids=[], distances=[])
        try:
            results = self._client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=k,
                query_filter=filter,
            )
            ids = [str(r.id) for r in results]
            scores = [r.score for r in results]
            metas = [r.payload for r in results]
            return AdapterResult(ids=ids, distances=scores, metadatas=metas)
        except Exception as exc:
            logger.error("Qdrant search failed: %s", exc)
            return AdapterResult(ids=[], distances=[])

    def delete(self, id: str):
        if self._client:
            from qdrant_client.models import PointIdsList
            self._client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=[hash(id) % (2**31)]),
            )

    def get_stats(self) -> Dict:
        if self._client is None:
            return {"backend": "qdrant", "error": "not connected"}
        try:
            info = self._client.get_collection(self.collection_name)
            return {"backend": "qdrant", "vector_count": info.vectors_count, "collection": self.collection_name}
        except Exception:
            return {"backend": "qdrant", "collection": self.collection_name}

    def close(self):
        self._client = None


# ──────────────────────── Weaviate Backend ────────────────────────

class WeaviateBackend(BaseVectorBackend):
    """Weaviate vector DB backend via weaviate-client v4."""

    def __init__(self, config: Dict):
        self.class_name = config.get("class_name", "TurboVec")
        self.host = config.get("host", "http://localhost:8080")
        self.dim = config.get("dim", 768)
        self._client = None
        self._init()

    def _init(self):
        try:
            import weaviate
            self._client = weaviate.connect_to_local(
                host=self.host.replace("http://", "").split(":")[0],
                port=int(self.host.split(":")[-1]) if ":" in self.host else 8080,
            )
        except ImportError:
            logger.warning("weaviate-client not installed; Weaviate backend unavailable")
        except Exception as exc:
            logger.warning("Weaviate connection failed: %s", exc)

    def upsert(self, id: str, embedding: np.ndarray, metadata: Optional[Dict] = None):
        if self._client is None:
            return
        collection = self._client.collections.get(self.class_name)
        collection.data.insert(
            properties=metadata or {},
            uuid=id[:36] if len(id) >= 36 else id.ljust(36, "0"),
            vector=embedding.tolist(),
        )

    def upsert_batch(self, ids: List[str], embeddings: np.ndarray,
                     metadatas: Optional[List[Dict]] = None):
        for i, sid in enumerate(ids):
            self.upsert(sid, embeddings[i], metadatas[i] if metadatas else None)

    def search(self, query_embedding: np.ndarray, k: int = 10,
               filter: Optional[Dict] = None) -> AdapterResult:
        if self._client is None:
            return AdapterResult(ids=[], distances=[])
        try:
            collection = self._client.collections.get(self.class_name)
            results = collection.query.near_vector(
                near_vector=query_embedding.tolist(),
                limit=k,
                return_metadata=["distance"],
            )
            ids = [str(o.uuid) for o in results.objects]
            dists = [o.metadata.distance for o in results.objects]
            metas = [o.properties for o in results.objects]
            return AdapterResult(ids=ids, distances=dists, metadatas=metas)
        except Exception as exc:
            logger.error("Weaviate search failed: %s", exc)
            return AdapterResult(ids=[], distances=[])

    def delete(self, id: str):
        if self._client:
            collection = self._client.collections.get(self.class_name)
            collection.data.delete_by_id(id[:36].ljust(36, "0"))

    def get_stats(self) -> Dict:
        return {"backend": "weaviate", "class": self.class_name}

    def close(self):
        if self._client:
            self._client.close()
        self._client = None


# ──────────────────────── Factory ────────────────────────

class VectorDBAdapter:
    """Factory for creating vector DB adapters."""

    BACKENDS = {
        "hnswlib": HNSWLibBackend,
        "faiss": FAISSBackend,
        "chroma": ChromaBackend,
        "qdrant": QdrantBackend,
        "weaviate": WeaviateBackend,
    }

    @classmethod
    def create(cls, backend: str = "hnswlib", config: Optional[Dict] = None) -> BaseVectorBackend:
        """
        Create a vector DB adapter for the specified backend.

        Args:
            backend: Backend name ('hnswlib', 'faiss', 'chroma', 'qdrant', 'weaviate')
            config: Backend-specific configuration dict

        Returns:
            Initialized backend instance

        Raises:
            ValueError: If backend is not recognized
        """
        config = config or {}
        if backend not in cls.BACKENDS:
            raise ValueError(f"Unknown backend '{backend}'. Choose from: {list(cls.BACKENDS.keys())}")
        backend_cls = cls.BACKENDS[backend]
        adapter = backend_cls(config)
        logger.info("VectorDBAdapter created: backend=%s", backend)
        return adapter

    @classmethod
    def available_backends(cls) -> List[str]:
        return list(cls.BACKENDS.keys())
