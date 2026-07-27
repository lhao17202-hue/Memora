"""Configuration defaults for Memora."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
    vector_store: str = "sqlite"
    vector_path: str | Path | None = None
    vector_candidate_limit: int = 50
    keyword_candidate_limit: int = 50
    min_semantic_score: float = 0.25
    reranker: str = "deterministic"
    rerank_candidate_limit: int = 100
    max_retrieved_memories: int = 8
    max_memory_prompt_tokens: int = 2000
    max_memory_content_chars: int = 4000
    default_preference_weight: int = 9
    default_project_weight: int = 7
    default_episodic_weight: int = 5
    default_reflective_weight: int = 7
    default_tool_weight: int = 6
    default_knowledge_weight: int = 6
    default_general_weight: int = 4
    episodic_expire_days: int = 90
    tool_expire_days: int = 180
    project_fact_review_days: int = 180
    archive_cold_days: int = 180
    consolidate_memory_count: int = 50
    consolidate_summary_count: int = 20
    allow_auto_save_user_preferences: bool = True
    allow_auto_save_project_facts: bool = False
    require_confirmation_for_conflicts: bool = True
