"""Configuration defaults for Memora."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MemoryConfig:
    root_dir: str = ".memora"
    max_retrieved_memories: int = 8
    max_memory_prompt_tokens: int = 2000
    max_memory_content_chars: int = 4000
    default_user_weight: int = 9
    default_feedback_weight: int = 8
    default_project_weight: int = 7
    default_summary_weight: int = 4
    default_tool_experience_weight: int = 5
    session_summary_expire_days: int = 90
    tool_experience_expire_days: int = 180
    project_fact_review_days: int = 180
    archive_cold_days: int = 180
    consolidate_memory_count: int = 50
    consolidate_summary_count: int = 20
    allow_auto_save_user_preferences: bool = True
    allow_auto_save_project_facts: bool = False
    require_confirmation_for_conflicts: bool = True
