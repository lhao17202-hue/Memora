"""RAG indexing and hybrid retrieval orchestration."""

from __future__ import annotations

import math
from typing import Any

from .config import MemoryConfig
from .embeddings import (
    RESERVED_EMBEDDING_PROVIDERS,
    BgeM3EmbeddingProvider,
    EmbeddingProvider,
    HashEmbeddingProvider,
    memory_embedding_text,
    sha256_text,
)
from .errors import MemoryValidationError
from .reranker import RESERVED_RERANKERS, DeterministicReranker, NoOpReranker, Reranker
from .retriever import MemoryRetriever
from .schema import MemoryItem, MemoryQuery, MemorySearchResult
from .stores import MemoryCandidateStore, MemoryStore
from .vector_store import RESERVED_VECTOR_STORES, QdrantVectorStore, QdrantVectorStoreConfig, SQLiteVectorStore, VectorSearchHit, VectorStore


class ReservedEmbeddingProvider:
    def __init__(self, name: str):
        raise MemoryValidationError(f"embedding_provider '{name}' is reserved but not implemented in RAG v1")


class ReservedVectorStore:
    def __init__(self, name: str):
        raise MemoryValidationError(f"vector_store '{name}' is reserved but not implemented in RAG v1")


class ReservedReranker:
    def __init__(self, name: str):
        raise MemoryValidationError(f"reranker '{name}' is reserved but not implemented in RAG v1")


def _validate_retrieval_config(config: MemoryConfig, embedder: EmbeddingProvider | None = None) -> None:
    if config.retrieval_mode not in {"dense", "hybrid"}:
        raise MemoryValidationError(f"unsupported retrieval_mode: {config.retrieval_mode}")
    if config.keyword_recall not in {"auto", "fts", "scan", "off"}:
        raise MemoryValidationError(f"unsupported keyword_recall: {config.keyword_recall}")
    if config.retrieval_mode == "hybrid" and not config.embedding_sparse:
        raise MemoryValidationError("retrieval_mode 'hybrid' requires embedding_sparse=True")
    if config.retrieval_mode == "hybrid" and config.vector_store != "qdrant":
        raise MemoryValidationError("retrieval_mode 'hybrid' requires vector_store 'qdrant'")
    if config.retrieval_mode == "hybrid" and embedder is not None and not getattr(embedder, "supports_sparse", False):
        raise MemoryValidationError(f"embedding_provider '{embedder.name}' does not support sparse embeddings required for hybrid retrieval")


def _embedding_dimension(config: MemoryConfig) -> int:
    if config.embedding_provider == "bge" and config.embedding_dimension == 384:
        return 1024
    return config.embedding_dimension


def build_embedding_provider(config: MemoryConfig) -> EmbeddingProvider:
    _validate_retrieval_config(config)
    if config.embedding_provider == "hash":
        embedder = HashEmbeddingProvider(dimension=config.embedding_dimension, model=config.embedding_model)
    elif config.embedding_provider == "bge":
        model = config.embedding_model if config.embedding_model != "memora-hash-v1" else "bge-m3"
        embedder = BgeM3EmbeddingProvider(
            model_path=config.embedding_model_path,
            model=model,
            dimension=_embedding_dimension(config),
            batch_size=config.embedding_batch_size,
            fp16=config.embedding_fp16,
            return_sparse=config.embedding_sparse,
        )
    elif config.embedding_provider in RESERVED_EMBEDDING_PROVIDERS:
        ReservedEmbeddingProvider(config.embedding_provider)
    else:
        raise MemoryValidationError(f"unsupported embedding_provider for RAG v1: {config.embedding_provider}")
    _validate_retrieval_config(config, embedder)
    return embedder


def build_vector_store(config: MemoryConfig) -> VectorStore:
    _validate_retrieval_config(config)
    if config.vector_store == "sqlite":
        return SQLiteVectorStore(config)
    if config.vector_store == "qdrant":
        qdrant_config = QdrantVectorStoreConfig.from_options(
            config.vector_store_options,
            dimension=_embedding_dimension(config),
            retrieval_mode=config.retrieval_mode,
            hybrid_prefetch_limit=config.hybrid_prefetch_limit,
        )
        return QdrantVectorStore(qdrant_config)
    if config.vector_store in RESERVED_VECTOR_STORES:
        ReservedVectorStore(config.vector_store)
    raise MemoryValidationError(f"unsupported vector_store for RAG v1: {config.vector_store}")


def build_reranker(config: MemoryConfig) -> Reranker:
    if config.reranker == "none":
        return NoOpReranker()
    if config.reranker == "deterministic":
        return DeterministicReranker()
    if config.reranker in RESERVED_RERANKERS:
        ReservedReranker(config.reranker)
    raise MemoryValidationError(f"unsupported reranker for RAG v1: {config.reranker}")


def build_vector_metadata(item: MemoryItem, embedder: EmbeddingProvider, text: str, config: MemoryConfig | None = None) -> dict[str, Any]:
    metadata = {
        "memory_id": item.id,
        "user_id": item.user_id,
        "project_id": item.project_id,
        "workspace_id": item.workspace_id,
        "type": item.type,
        "status": item.status,
        "tags": item.tags,
        "provider": embedder.name,
        "model": embedder.model,
        "dimension": embedder.dimension,
        "text_hash": sha256_text(text),
        "updated_at": item.updated_at.isoformat() if item.updated_at else None,
    }
    if config is not None:
        metadata.update(
            {
                "vector_store": config.vector_store,
                "retrieval_mode": config.retrieval_mode,
                "sparse_enabled": config.embedding_sparse,
            }
        )
    return metadata


class RagIndex:
    def __init__(self, memory_store: MemoryStore, embedder: EmbeddingProvider, vector_store: VectorStore, config: MemoryConfig | None = None):
        self.memory_store = memory_store
        self.embedder = embedder
        self.vector_store = vector_store
        self.config = config or MemoryConfig()

    def init_storage(self) -> None:
        self.vector_store.init_storage()

    def sync_memory(self, item: MemoryItem) -> None:
        if item.status != "active":
            self.vector_store.delete(item.id)
            return
        text = memory_embedding_text(item)
        vector = self.embedder.embed([text])[0]
        self.vector_store.upsert(item.id, vector, build_vector_metadata(item, self.embedder, text, self.config))

    def delete_memory(self, memory_id: str) -> None:
        self.vector_store.delete(memory_id)

    def rebuild(self) -> None:
        self.vector_store.init_storage()
        active_ids = set()
        for item in self.memory_store.list_memories(include_archived=True):
            if item.status == "active":
                active_ids.add(item.id)
                self.sync_memory(item)
            else:
                self.vector_store.delete(item.id)
        report = self.vector_store.verify(active_ids)
        for orphan_id in report.get("vector_orphans", []):
            self.vector_store.delete(orphan_id)

    def verify(self) -> dict[str, Any]:
        active_items = [item for item in self.memory_store.list_memories(include_archived=True) if item.status == "active"]
        expected_ids = {item.id for item in active_items}
        report = self.vector_store.verify(expected_ids)
        mismatches = []
        for item in active_items:
            text = memory_embedding_text(item)
            metadata = build_vector_metadata(item, self.embedder, text, self.config)
            hit_metadata = self.vector_store.get_metadata(item.id)
            if hit_metadata is None:
                continue
            for key in ("provider", "model", "dimension", "text_hash"):
                if hit_metadata.get(key) != metadata.get(key):
                    mismatches.append({"memory_id": item.id, "field": key})
                    break
        report["embedding_mismatches"] = mismatches
        report["vector_ok"] = not report.get("vector_missing") and not report.get("vector_orphans") and not report.get("vector_errors") and not mismatches
        return report


class RagRetriever:
    def __init__(
        self,
        memory_store: MemoryStore,
        candidate_store: MemoryCandidateStore | None,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
        retriever: MemoryRetriever,
        reranker: Reranker,
        config: MemoryConfig,
    ):
        self.memory_store = memory_store
        self.candidate_store = candidate_store
        self.embedder = embedder
        self.vector_store = vector_store
        self.retriever = retriever
        self.reranker = reranker
        self.config = config

    def retrieve(self, query: MemoryQuery) -> list[MemorySearchResult]:
        allowed = self._allowed_memories(query)
        vector_hits = self._vector_recall(query, allowed)
        keyword_items = self._keyword_recall(query, allowed)
        merged = self._merge(allowed, vector_hits, keyword_items)
        scored = self._score(query, merged)
        try:
            ranked = self.reranker.rank(query, scored)
        except Exception:  # noqa: BLE001 - retrieval should degrade
            ranked = sorted(scored, key=lambda result: result.final_score, reverse=True)
        return ranked[: query.top_k]

    def _allowed_memories(self, query: MemoryQuery) -> dict[str, MemoryItem]:
        allowed = {}
        for item in self.memory_store.list_memories(include_archived=True):
            if item.user_id != query.user_id:
                continue
            if query.project_id is not None and item.project_id != query.project_id:
                continue
            if query.workspace_id is not None and item.workspace_id != query.workspace_id:
                continue
            if item.status == "deleted":
                continue
            if item.status != "active" and not query.include_archived:
                continue
            if query.memory_types and item.type not in query.memory_types:
                continue
            if query.tags and not set(query.tags).intersection(item.tags):
                continue
            if item.type == "knowledge" and not query.include_knowledge:
                continue
            allowed[item.id] = item
        return allowed

    def _vector_recall(self, query: MemoryQuery, allowed: dict[str, MemoryItem]) -> list[VectorSearchHit]:
        try:
            query_vector = self.embedder.embed([query.query])[0]
            mode = "hybrid" if self.config.retrieval_mode == "hybrid" and query_vector.sparse is not None else "dense"
            hits = self.vector_store.search(
                query_vector,
                top_k=self.config.vector_candidate_limit,
                filters={
                    "user_id": query.user_id,
                    "project_id": query.project_id,
                    "workspace_id": query.workspace_id,
                    "status": "active" if not query.include_archived else None,
                    "types": query.memory_types,
                    "tags": query.tags,
                },
                mode=mode,
            )
        except Exception:  # noqa: BLE001 - retrieval should degrade
            return []
        return [hit for hit in hits if hit.memory_id in allowed]

    def _keyword_recall(self, query: MemoryQuery, allowed: dict[str, MemoryItem]) -> list[MemoryItem]:
        mode = self.config.keyword_recall
        if mode == "off":
            return []
        if mode in {"auto", "fts"}:
            candidates = self._fts_keyword_recall(query, allowed)
            if candidates or mode == "fts":
                return candidates
        if mode in {"auto", "scan"}:
            return self._scan_keyword_recall(query, allowed)
        return []

    def _fts_keyword_recall(self, query: MemoryQuery, allowed: dict[str, MemoryItem]) -> list[MemoryItem]:
        if self.candidate_store is None:
            return []
        try:
            candidates = self.candidate_store.search_candidates(query)
        except Exception:  # noqa: BLE001 - retrieval should degrade
            return []
        candidates = [item for item in candidates if item.id in allowed]
        return candidates[: self.config.keyword_candidate_limit]

    def _scan_keyword_recall(self, query: MemoryQuery, allowed: dict[str, MemoryItem]) -> list[MemoryItem]:
        results = []
        for item in allowed.values():
            result = self.retriever.score(item, query)
            if result is not None:
                results.append(result)
        results.sort(key=lambda result: result.final_score, reverse=True)
        return [result.memory for result in results[: self.config.keyword_candidate_limit]]

    def _merge(
        self,
        allowed: dict[str, MemoryItem],
        vector_hits: list[VectorSearchHit],
        keyword_items: list[MemoryItem],
    ) -> dict[str, dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for hit in vector_hits:
            if hit.memory_id in allowed:
                merged.setdefault(hit.memory_id, {"memory": allowed[hit.memory_id], "semantic_score": 0.0, "keyword_candidate": False, "vector_source": "dense"})
                merged[hit.memory_id]["semantic_score"] = max(float(hit.score), merged[hit.memory_id]["semantic_score"])
                merged[hit.memory_id]["vector_source"] = hit.source
        for item in keyword_items:
            if item.id in allowed:
                merged.setdefault(item.id, {"memory": item, "semantic_score": 0.0, "keyword_candidate": False, "vector_source": "dense"})
                merged[item.id]["keyword_candidate"] = True
        return merged

    def _score(self, query: MemoryQuery, merged: dict[str, dict[str, Any]]) -> list[MemorySearchResult]:
        results = []
        for data in merged.values():
            memory = data["memory"]
            semantic_score = float(data.get("semantic_score") or 0.0)
            keyword_score = 0.0
            reason = "matched_hybrid_vector" if semantic_score > 0 and data.get("vector_source") == "hybrid" else "matched_vector" if semantic_score > 0 else ""
            keyword_result = self.retriever.score(memory, query)
            if keyword_result is not None:
                keyword_score = keyword_result.similarity_score
                reason = keyword_result.reason
                importance_score = keyword_result.importance_score
                recency_score = keyword_result.recency_score
                access_score = keyword_result.access_score
            else:
                importance_score = min(max(memory.weight, 1), 10) / 10
                recency_score = self.retriever._recency_score(memory)
                access_score = min(math.log1p(memory.access_count) / math.log1p(20), 1.0)
            if semantic_score <= 0 and keyword_score <= 0:
                continue
            if keyword_score <= 0 and semantic_score < self.config.min_semantic_score:
                continue
            similarity_score = max(semantic_score, keyword_score)
            final_score = (
                semantic_score * 0.25
                + keyword_score * 0.35
                + importance_score * 0.15
                + recency_score * 0.15
                + access_score * 0.10
            )
            results.append(
                MemorySearchResult(
                    memory=memory,
                    similarity_score=similarity_score,
                    importance_score=importance_score,
                    recency_score=recency_score,
                    access_score=access_score,
                    final_score=final_score,
                    reason=reason,
                    semantic_score=semantic_score,
                    keyword_score=keyword_score,
                )
            )
        return results
