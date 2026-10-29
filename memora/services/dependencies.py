"""Construction helpers for MemoryManager internals."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import MemoryConfig
from ..errors import MemoryValidationError
from ..rag import RagIndex, RagRetriever, build_embedding_provider, build_reranker, build_vector_store
from ..relations import SemanticMemoryRelationResolver
from ..retriever import MemoryRetriever
from ..sqlite_store import SQLiteMemoryStore
from ..stores import FileMemoryStore, FileSessionStore, MemoryCandidateStore, MemoryStore, SessionStore
from ..vector_store import VectorStore


@dataclass(frozen=True)
class ManagerDependencies:
    memory_store: MemoryStore
    session_store: SessionStore
    relation_resolver: SemanticMemoryRelationResolver | None
    rag_index: RagIndex | None
    rag_retriever: RagRetriever | None


def build_memory_store(config: MemoryConfig) -> MemoryStore:
    if config.memory_backend == "file":
        return FileMemoryStore(config)
    if config.memory_backend == "sqlite":
        return SQLiteMemoryStore(config)
    raise MemoryValidationError(f"unsupported memory backend: {config.memory_backend}")


def build_manager_dependencies(
    config: MemoryConfig,
    *,
    memory_store: MemoryStore | None = None,
    session_store: SessionStore | None = None,
    retriever: MemoryRetriever | None = None,
    vector_store: VectorStore | None = None,
) -> ManagerDependencies:
    resolved_memory_store = memory_store or build_memory_store(config)
    resolved_session_store = session_store or FileSessionStore(config)
    relation_resolver = None
    rag_index = None
    rag_retriever = None

    relation_resolution_enabled = config.semantic_write_relations_enabled or config.llm_relation_judge_enabled
    embedder = None
    if config.rag_enabled or relation_resolution_enabled:
        embedder = build_embedding_provider(config)

    if relation_resolution_enabled and embedder is not None:
        relation_resolver = SemanticMemoryRelationResolver(embedder, config)

    if config.rag_enabled and embedder is not None:
        resolved_vector_store = vector_store or build_vector_store(config)
        reranker = build_reranker(config)
        rag_index = RagIndex(resolved_memory_store, embedder, resolved_vector_store, config)
        candidate_store = resolved_memory_store if isinstance(resolved_memory_store, MemoryCandidateStore) else None
        rag_retriever = RagRetriever(
            resolved_memory_store,
            candidate_store,
            embedder,
            resolved_vector_store,
            retriever or MemoryRetriever(),
            reranker,
            config,
        )

    return ManagerDependencies(
        memory_store=resolved_memory_store,
        session_store=resolved_session_store,
        relation_resolver=relation_resolver,
        rag_index=rag_index,
        rag_retriever=rag_retriever,
    )

