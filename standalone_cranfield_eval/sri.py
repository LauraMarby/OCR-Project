"""
sri.py — Retriever híbrido BM25 + E5 + RRF, autocontenido.

Este módulo NO importa nada del proyecto OCR. Replica la estructura
del SRI en producción para poder evaluarla independientemente sobre
colecciones IR estándar (Cranfield, TREC, etc.).

Arquitectura:
  - BM25Okapi (rank_bm25): re-ranking léxico.
      Hiperparámetros: k1=1.5, b=0.75 (defaults de Lucene/Elasticsearch).
  - multilingual-e5-small (sentence-transformers): embeddings densos.
      Prefijos requeridos por el modelo: "query: " y "passage: ".
      Embeddings normalizados L2 → cosine sim = dot product.
  - Reciprocal Rank Fusion (Cormack et al. 2009): fusión de rankings.
      score_RRF(d) = sum_r 1 / (k + rank_r(d)), con k=60 por convención.

Referencias:
  Robertson, S. & Walker, S. (1994). "Some Simple Effective Approximations
    to the 2-Poisson Model for Probabilistic Weighted Retrieval".
  Cormack, G. V., Clarke, C. L. A., & Büttcher, S. (2009).
    "Reciprocal rank fusion outperforms condorcet and individual rank
    learning methods". SIGIR 2009.
  Wang, L. et al. (2024). "Multilingual E5 Text Embeddings: A Technical
    Report". arXiv:2402.05672.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Hiperparámetros por defecto (configurables al instanciar) ────────────

DEFAULT_BM25_K1 = 1.5
DEFAULT_BM25_B  = 0.75
DEFAULT_RRF_K   = 60
DEFAULT_BATCH_SIZE = 32

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)

def tokenize_for_bm25(text: str) -> list[str]:
    """
    Tokenizador mínimo: minúsculas + split en no-alfanumérico, descarta
    tokens de longitud 1. NO aplica stemming ni quita stop-words —
    consistente con el SRI en producción.

    Si quisieras añadir stemming para una ablación, lo harías aquí.
    Recomendado para tesis: comparar con y sin stemming Porter.
    """
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1]

@dataclass
class RetrievalRun:
    """
    Resultado de ejecutar un retriever sobre un conjunto de consultas.

    runs: {qid -> [(doc_id, score), ...]}  ordenado por score desc.
    name: identificador legible ("bm25", "semantic", "hybrid_rrf").
    """
    name: str
    runs: dict[str, list[tuple[str, float]]] = field(default_factory=dict)

    def top_k(self, qid: str, k: int) -> list[tuple[str, float]]:
        return self.runs.get(qid, [])[:k]

    def doc_ids(self, qid: str, k: int | None = None) -> list[str]:
        items = self.runs.get(qid, [])
        return [d for d, _ in (items[:k] if k else items)]

class HybridRetriever:
    """
    Retriever híbrido BM25 + E5 + RRF.

    Uso típico:
        retriever = HybridRetriever(model_path="/ruta/a/e5")
        retriever.index(doc_ids=[...], texts=[...])
        run_bm25     = retriever.search_bm25_batch(queries, top_k=1000)
        run_semantic = retriever.search_semantic_batch(queries, top_k=1000)
        run_hybrid   = retriever.search_hybrid_batch(queries, top_k=1000)

    El método `index()` construye AMBOS índices (BM25 + densos). Si no
    se pasó `model_path`, sólo se construye BM25; `search_semantic_batch`
    y `search_hybrid_batch` devolverán resultados vacíos (o sólo BM25
    en el caso híbrido).
    """

    def __init__(self,
                 model_path: Optional[str | Path] = None,
                 *,
                 k1: float = DEFAULT_BM25_K1,
                 b:  float = DEFAULT_BM25_B,
                 rrf_k: int = DEFAULT_RRF_K,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 device: Optional[str] = None,
                 max_seq_length: int = 512):
        self.model_path  = str(model_path) if model_path else None
        self.k1          = k1
        self.b           = b
        self.rrf_k       = rrf_k
        self.batch_size  = batch_size
        self.device      = device  # None = autoselect (gpu si disponible)
        self.max_seq_length = max_seq_length

        # Estado tras indexar
        self.doc_ids: list[str] = []
        self.texts:   list[str] = []
        self._id_to_idx: dict[str, int] = {}
        self._bm25 = None
        self._emb_matrix: Optional[np.ndarray] = None  # (N, D) L2-norm
        self._st_model = None  # SentenceTransformer (lazy)

    def _load_st_model(self):
        if self._st_model is not None:
            return self._st_model
        if self.model_path is None:
            return None
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        except ImportError:
            raise ImportError(
                "sentence-transformers no está instalado. "
                "pip install sentence-transformers"
            )
        logger.info("Cargando modelo E5 desde %s ...", self.model_path)
        m = SentenceTransformer(self.model_path, device=self.device)
        # max_seq_length: E5 admite 512; truncar evita errores con docs largos.
        m.max_seq_length = self.max_seq_length
        self._st_model = m
        return m

    @property
    def has_semantic(self) -> bool:
        return self.model_path is not None

    def index(self, doc_ids: list[str], texts: list[str],
              show_progress: bool = True) -> None:
        """
        Construye índice BM25 + (si hay modelo) E5.

        Args:
            doc_ids: lista de identificadores de docs (string para
                máxima compatibilidad con formatos TREC).
            texts: lista de textos a indexar (uno por doc).
            show_progress: pasa progress_bar a sentence-transformers.
        """
        if len(doc_ids) != len(texts):
            raise ValueError(
                f"len(doc_ids)={len(doc_ids)} != len(texts)={len(texts)}"
            )
        self.doc_ids = list(doc_ids)
        self.texts = list(texts)
        self._id_to_idx = {d: i for i, d in enumerate(self.doc_ids)}

        # 1. BM25
        try:
            from rank_bm25 import BM25Okapi  # noqa: PLC0415
        except ImportError:
            raise ImportError("rank_bm25 no instalado. pip install rank_bm25")

        logger.info("Tokenizando %d documentos para BM25...", len(texts))
        tokenized = [tokenize_for_bm25(t) for t in texts]
        logger.info("Construyendo índice BM25 (k1=%.2f, b=%.2f)...",
                    self.k1, self.b)
        self._bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)

        # 2. E5 (opcional)
        if self.model_path is not None:
            model = self._load_st_model()
            # Prefijo "passage: " requerido por la familia E5
            prefixed = [f"passage: {t}" for t in texts]
            logger.info("Encoding %d passages con E5 (batch_size=%d)...",
                        len(prefixed), self.batch_size)
            embs = model.encode(
                prefixed,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=True,   # L2-norm → dot=cosine
                convert_to_numpy=True,
            ).astype(np.float32, copy=False)
            self._emb_matrix = embs
            logger.info("Index E5 listo: matriz %s", embs.shape)
        else:
            logger.info("Sin modelo E5 — sólo BM25 disponible.")

    def search_bm25(self, query: str, top_k: int = 1000) -> list[tuple[str, float]]:
        if self._bm25 is None:
            raise RuntimeError("Hay que llamar index() antes de search.")
        q_tokens = tokenize_for_bm25(query)
        if not q_tokens:
            return []
        scores = np.asarray(self._bm25.get_scores(q_tokens), dtype=np.float32)
        return self._top_k_from_scores(scores, top_k,
                                       drop_zero_or_negative=True)

    def search_bm25_batch(self, queries: dict[str, str],
                          top_k: int = 1000) -> RetrievalRun:
        out: dict[str, list[tuple[str, float]]] = {}
        for qid, q in queries.items():
            out[qid] = self.search_bm25(q, top_k)
        return RetrievalRun(name="bm25", runs=out)

    def search_semantic(self, query: str, top_k: int = 1000) -> list[tuple[str, float]]:
        if self._emb_matrix is None:
            return []
        model = self._load_st_model()
        if model is None:
            return []
        prefixed = f"query: {query}"
        q_emb = model.encode(
            [prefixed],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0].astype(np.float32, copy=False)
        # cos = dot por estar normalizados
        scores = self._emb_matrix @ q_emb
        # En semántico NO cortamos negativos: la similitud puede ser
        # negativa y aun así informativa. Pero sí limitamos top_k.
        return self._top_k_from_scores(scores, top_k,
                                       drop_zero_or_negative=False)

    def search_semantic_batch(self, queries: dict[str, str],
                              top_k: int = 1000,
                              show_progress: bool = True) -> RetrievalRun:
        """
        Versión batched que aprovecha el GPU/CPU encoding en bloque.
        Suele ser 5-20× más rápida que llamar search_semantic en loop.
        """
        if self._emb_matrix is None or not self.has_semantic:
            return RetrievalRun(name="semantic", runs={})
        model = self._load_st_model()
        qids = list(queries.keys())
        prefixed = [f"query: {queries[q]}" for q in qids]
        q_embs = model.encode(
            prefixed,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        ).astype(np.float32, copy=False)
        # scores: (Q, N) = (Q, D) @ (D, N)
        all_scores = q_embs @ self._emb_matrix.T
        out: dict[str, list[tuple[str, float]]] = {}
        for i, qid in enumerate(qids):
            out[qid] = self._top_k_from_scores(all_scores[i], top_k,
                                               drop_zero_or_negative=False)
        return RetrievalRun(name="semantic", runs=out)

    def rrf_fuse(self,
                 runs: list[list[tuple[str, float]]],
                 top_k: int = 1000,
                 k: Optional[int] = None) -> list[tuple[str, float]]:
        """
        Reciprocal Rank Fusion de múltiples rankings para UNA consulta.
        Cada `runs[i]` es la lista ranked de (doc_id, score) de un retriever.

        RRF score = sum_r 1 / (k + rank_r), con rank empezando en 1.
        Los docs ausentes de un ranker no aportan score de ese ranker.
        """
        if k is None:
            k = self.rrf_k
        agg: dict[str, float] = {}
        for ranking in runs:
            for rank, (doc_id, _) in enumerate(ranking, start=1):
                agg[doc_id] = agg.get(doc_id, 0.0) + 1.0 / (k + rank)
        fused = sorted(agg.items(), key=lambda x: -x[1])
        return fused[:top_k]

    def search_hybrid(self, query: str, top_k: int = 1000,
                      top_k_per_method: int = 1000) -> list[tuple[str, float]]:
        """
        BM25 + E5 + RRF. Si no hay modelo, devuelve sólo BM25.
        """
        bm = self.search_bm25(query, top_k_per_method)
        if not self.has_semantic:
            return bm[:top_k]
        sem = self.search_semantic(query, top_k_per_method)
        return self.rrf_fuse([bm, sem], top_k=top_k)

    def search_hybrid_batch(self,
                            queries: dict[str, str],
                            top_k: int = 1000,
                            top_k_per_method: int = 1000,
                            show_progress: bool = True) -> RetrievalRun:
        bm_run  = self.search_bm25_batch(queries, top_k=top_k_per_method)
        if not self.has_semantic:
            # Híbrido con un solo retriever = ese retriever truncado
            return RetrievalRun(
                name="hybrid_rrf",
                runs={q: bm_run.runs[q][:top_k] for q in queries},
            )
        sem_run = self.search_semantic_batch(queries, top_k=top_k_per_method,
                                             show_progress=show_progress)
        out: dict[str, list[tuple[str, float]]] = {}
        for qid in queries:
            out[qid] = self.rrf_fuse(
                [bm_run.runs.get(qid, []), sem_run.runs.get(qid, [])],
                top_k=top_k,
            )
        return RetrievalRun(name="hybrid_rrf", runs=out)

    def _top_k_from_scores(self, scores: np.ndarray, top_k: int,
                           drop_zero_or_negative: bool) -> list[tuple[str, float]]:
        if len(scores) == 0:
            return []
        n_take = min(top_k, len(scores))
        # argpartition + sort sobre el subset es O(N) en lugar de O(N log N)
        idx = np.argpartition(-scores, n_take - 1)[:n_take]
        idx = idx[np.argsort(-scores[idx])]
        out: list[tuple[str, float]] = []
        for i in idx:
            s = float(scores[i])
            if drop_zero_or_negative and s <= 0:
                break
            out.append((self.doc_ids[i], s))
        return out
