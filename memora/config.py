"""Configuration defaults for Memora."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .taxonomy import MEMORY_TYPE_POLICIES


@dataclass
class MemoryConfig:
    root_dir: str | Path = ".memora"
    memory_backend: str = "file"
    sqlite_path: str | Path | None = None
    fts_enabled: bool = True
    fts_candidate_limit: int = 100
    rag_enabled: bool = False
    embedding_provider: str = "hash"
    embedding_model: str = "memora-hash-v1"
    embedding_dimension: int = 384
    embedding_model_path: str | Path | None = None
    embedding_batch_size: int = 8
    embedding_fp16: bool = False
    embedding_sparse: bool = False
    vector_store: str = "sqlite"
    vector_path: str | Path | None = None
    retrieval_mode: str = "dense"
    hybrid_prefetch_limit: int = 100
    qdrant_url: str | None = None
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: str | None = None
    qdrant_collection: str = "memora_memories"
    qdrant_timeout: float = 5.0
    qdrant_prefer_grpc: bool = False
    qdrant_recreate_collection: bool = False
    vector_candidate_limit: int = 50
    keyword_candidate_limit: int = 50
    min_semantic_score: float = 0.25
    semantic_write_relations_enabled: bool = False
    semantic_relation_threshold: float = 0.78
    semantic_merge_threshold: float = 0.82
    semantic_conflict_threshold: float = 0.90
    allow_high_confidence_conflict_replace: bool = True
    high_confidence_conflict_threshold: float = 0.90
    llm_relation_judge_enabled: bool = False
    llm_relation_confidence_threshold: float = 0.80
    llm_merge_confidence_threshold: float = 0.75
    llm_conflict_auto_replace_threshold: float = 0.90
    reranker: str = "deterministic"
    rerank_candidate_limit: int = 100
    max_retrieved_memories: int = 8
    max_memory_prompt_tokens: int = 2000
    max_memory_content_chars: int = 4000
    default_preference_weight: int = MEMORY_TYPE_POLICIES["preference"].default_weight
    default_project_weight: int = MEMORY_TYPE_POLICIES["project"].default_weight
    default_episodic_weight: int = MEMORY_TYPE_POLICIES["episodic"].default_weight
    default_reflective_weight: int = MEMORY_TYPE_POLICIES["reflective"].default_weight
    default_tool_weight: int = MEMORY_TYPE_POLICIES["tool"].default_weight
    default_knowledge_weight: int = MEMORY_TYPE_POLICIES["knowledge"].default_weight
    default_general_weight: int = MEMORY_TYPE_POLICIES["general"].default_weight
    project_fact_review_days: int = 180
    archive_cold_days: int = 180
    consolidate_memory_count: int = 50
    consolidate_summary_count: int = 20
    allow_auto_save_user_preferences: bool = True
    allow_auto_save_project_facts: bool = False
    require_confirmation_for_conflicts: bool = True
