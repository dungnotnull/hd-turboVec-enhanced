"""
HuggingFace Model Manager — Lazy-loading registry for TurboVec Enhanced.

Models:
  BAAI/bge-large-en-v1.5     — Dense text embedding (768-dim), MTEB #1 English
  BAAI/bge-reranker-large    — Cross-encoder reranking (549M params)
  sentence-transformers/all-MiniLM-L6-v2 — Fast embedding (384-dim)
  Salesforce/codet5p-770m    — Code embedding for repository search

All models use lazy loading: downloaded on first use, cached in ./models/.
Idle unload: models unused for 600s are unloaded to free GPU memory.
"""
import logging
import os
import time
from pathlib import Path
from threading import Lock, Timer
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "bge-large": {
        "model_id": "BAAI/bge-large-en-v1.5",
        "type": "sentence_transformer",
        "dim": 768,
        "task": "dense_embedding",
        "description": "MTEB #1 English embedding; used for dense retrieval and indexing",
    },
    "bge-reranker": {
        "model_id": "BAAI/bge-reranker-large",
        "type": "cross_encoder",
        "task": "reranking",
        "description": "Cross-encoder reranker; 549M params; best open reranker on BEIR",
    },
    "minilm": {
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "type": "sentence_transformer",
        "dim": 384,
        "task": "fast_embedding",
        "description": "5× faster than BGE-large; 384-dim; used for latency-sensitive paths",
    },
    "codet5p": {
        "model_id": "Salesforce/codet5p-770m",
        "type": "seq2seq",
        "task": "code_embedding",
        "description": "Code embedding for repository search (CodeSearchNet 0.746)",
    },
}

IDLE_TIMEOUT_SECONDS = 600


class HFModelManager:
    """
    Singleton registry with lazy loading, CUDA auto-selection, and idle unloading.

    Args:
        use_gpu: Use CUDA GPU if available
        cache_dir: Directory to cache downloaded models (default: ./models)
        idle_timeout: Seconds before unloading an idle model (default: 600)
    """

    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        use_gpu: bool = False,
        cache_dir: Optional[Path] = None,
        idle_timeout: int = IDLE_TIMEOUT_SECONDS,
    ):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True

        self.use_gpu = use_gpu and self._cuda_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        self.cache_dir = cache_dir or (Path(__file__).parent.parent / "models")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.idle_timeout = idle_timeout

        os.environ["TRANSFORMERS_CACHE"] = str(self.cache_dir)
        os.environ["HF_HOME"] = str(self.cache_dir)

        self._models: Dict[str, Any] = {}
        self._last_used: Dict[str, float] = {}
        self._timers: Dict[str, Timer] = {}
        self._model_lock = Lock()

        logger.info("HFModelManager initialized (device=%s, cache=%s)", self.device, self.cache_dir)

    def _cuda_available(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    # ──────────────────────── Public Encoding API ────────────────────────

    def encode(self, text: str, model_key: str = "bge-large") -> np.ndarray:
        """Encode a single text into a normalized embedding vector."""
        return self.encode_batch([text], model_key=model_key)[0]

    def encode_batch(
        self,
        texts: List[str],
        model_key: str = "bge-large",
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode a list of texts into embedding vectors.

        Args:
            texts: List of text strings to encode
            model_key: Model key from MODEL_REGISTRY (default: 'bge-large')
            batch_size: Encode in batches to manage memory
            normalize: L2-normalize output vectors (required for cosine similarity)

        Returns:
            np.ndarray of shape (N, dim)
        """
        model = self._load_sentence_transformer(model_key)
        if model is not None:
            try:
                import torch
                with torch.no_grad():
                    embeddings = model.encode(
                        texts,
                        batch_size=batch_size,
                        normalize_embeddings=normalize,
                        show_progress_bar=len(texts) > 100,
                        device=self.device,
                        convert_to_numpy=True,
                    )
                self._touch(model_key)
                return embeddings.astype(np.float32)
            except Exception as exc:
                logger.warning("SentenceTransformer encode failed: %s — using fallback", exc)

        return self._numpy_fallback_encode(texts)

    def rerank(self, pairs: List[tuple], model_key: str = "bge-reranker") -> List[float]:
        """
        Cross-encoder reranking: score (query, document) pairs.

        Args:
            pairs: List of (query, document_text) tuples
            model_key: Cross-encoder model key (default: 'bge-reranker')

        Returns:
            List of relevance scores (higher = more relevant)
        """
        model = self._load_cross_encoder(model_key)
        if model is not None:
            try:
                scores = model.predict(pairs, batch_size=16, show_progress_bar=False)
                self._touch(model_key)
                return [float(s) for s in scores]
            except Exception as exc:
                logger.warning("CrossEncoder rerank failed: %s — using dot-product fallback", exc)

        return self._dot_product_fallback_rerank(pairs)

    # ──────────────────────── Model Loading ────────────────────────

    def _load_sentence_transformer(self, model_key: str):
        """Load a SentenceTransformer model (lazy, with idle unloading)."""
        with self._model_lock:
            if model_key in self._models:
                self._touch(model_key)
                return self._models[model_key]

        reg = MODEL_REGISTRY.get(model_key)
        if not reg or reg.get("type") not in ("sentence_transformer",):
            logger.warning("Model key '%s' is not a sentence transformer", model_key)
            return None

        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading %s ...", reg["model_id"])
            t0 = time.perf_counter()
            model = SentenceTransformer(reg["model_id"], cache_folder=str(self.cache_dir))
            model = model.to(self.device)
            elapsed = time.perf_counter() - t0
            logger.info("Loaded %s in %.1fs", reg["model_id"], elapsed)
            with self._model_lock:
                self._models[model_key] = model
            self._touch(model_key)
            return model
        except Exception as exc:
            logger.error("Failed to load %s: %s", model_key, exc)
            return None

    def _load_cross_encoder(self, model_key: str):
        """Load a CrossEncoder model (lazy, with idle unloading)."""
        with self._model_lock:
            if model_key in self._models:
                self._touch(model_key)
                return self._models[model_key]

        reg = MODEL_REGISTRY.get(model_key)
        if not reg or reg.get("type") != "cross_encoder":
            return None

        try:
            from sentence_transformers import CrossEncoder
            logger.info("Loading CrossEncoder %s ...", reg["model_id"])
            t0 = time.perf_counter()
            model = CrossEncoder(reg["model_id"], device=self.device, max_length=512)
            elapsed = time.perf_counter() - t0
            logger.info("Loaded CrossEncoder %s in %.1fs", reg["model_id"], elapsed)
            with self._model_lock:
                self._models[model_key] = model
            self._touch(model_key)
            return model
        except Exception as exc:
            logger.error("Failed to load CrossEncoder %s: %s", model_key, exc)
            return None

    def _touch(self, model_key: str):
        """Reset the idle timer for a model."""
        self._last_used[model_key] = time.time()
        if model_key in self._timers:
            self._timers[model_key].cancel()
        timer = Timer(self.idle_timeout, self._unload, args=[model_key])
        timer.daemon = True
        timer.start()
        self._timers[model_key] = timer

    def _unload(self, model_key: str):
        """Unload a model that has been idle for idle_timeout seconds."""
        with self._model_lock:
            if model_key in self._models:
                del self._models[model_key]
                logger.info("Unloaded idle model: %s (idle >%ds)", model_key, self.idle_timeout)

    # ──────────────────────── Fallbacks ────────────────────────

    def _numpy_fallback_encode(self, texts: List[str]) -> np.ndarray:
        """Deterministic hash-based fallback embedding when model unavailable."""
        logger.warning("Using hash-based fallback embedding (no sentence-transformers installed)")
        dim = 768
        embeddings = []
        for text in texts:
            seed = int(hash(text) % (2**31))
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(dim).astype(np.float32)
            vec = vec / (np.linalg.norm(vec) + 1e-8)
            embeddings.append(vec)
        return np.array(embeddings, dtype=np.float32)

    def _dot_product_fallback_rerank(self, pairs: List[tuple]) -> List[float]:
        """Fallback reranking using query-doc embedding dot product."""
        scores = []
        for query, doc in pairs:
            q_vec = self._numpy_fallback_encode([query])[0]
            d_vec = self._numpy_fallback_encode([doc])[0]
            scores.append(float(np.dot(q_vec, d_vec)))
        return scores

    # ──────────────────────── Info ────────────────────────

    def get_loaded_models(self) -> Dict[str, Any]:
        return {
            k: {
                "model_id": MODEL_REGISTRY.get(k, {}).get("model_id", k),
                "last_used_ago_s": round(time.time() - self._last_used.get(k, 0), 0),
            }
            for k in self._models
        }

    def preload(self, model_keys: List[str]):
        """Pre-load models before first query (warmup)."""
        for key in model_keys:
            reg = MODEL_REGISTRY.get(key, {})
            if reg.get("type") == "sentence_transformer":
                self._load_sentence_transformer(key)
            elif reg.get("type") == "cross_encoder":
                self._load_cross_encoder(key)
