"""Core data structures for Memora."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

MemoryType = Literal[
    "user",
    "feedback",
    "project",
    "decision",
    "entity",
    "session_summary",
    "tool_experience",
    "reference",
    "knowledge",
]

MemoryStatus = Literal["active", "archived", "deleted"]

CandidateAction = Literal["create", "update", "archive", "delete", "reject", "ask_user"]


@dataclass
class MemoryItem:
    id: str
    name: str
    description: str
    type: MemoryType
    content: str
    user_id: str = "default"
    project_id: str | None = None
    workspace_id: str | None = None
    tags: list[str] = field(default_factory=list)
    source: str = "unknown"
    confidence: float = 1.0
    weight: int = 5
    status: MemoryStatus = "active"
    created_at: datetime | None = None
    updated_at: datetime | None = None
    last_accessed_at: datetime | None = None
    access_count: int = 0
    expires_at: datetime | None = None
    supersedes: list[str] = field(default_factory=list)
    related: list[str] = field(default_factory=list)


@dataclass
class MemoryCandidate:
    action: CandidateAction
    name: str
    description: str
    type: MemoryType
    content: str
    user_id: str = "default"
    project_id: str | None = None
    workspace_id: str | None = None
    tags: list[str] = field(default_factory=list)
    source: str = "conversation"
    confidence: float = 1.0
    weight: int = 5
    target_memory_id: str | None = None
    reason: str = ""


@dataclass
class MemoryQuery:
    query: str
    user_id: str = "default"
    project_id: str | None = None
    workspace_id: str | None = None
    memory_types: list[MemoryType] | None = None
    tags: list[str] | None = None
    top_k: int = 8
    max_tokens: int = 2000
    include_archived: bool = False
    include_knowledge: bool = True


@dataclass
class MemorySearchResult:
    memory: MemoryItem
    similarity_score: float
    importance_score: float
    recency_score: float
    access_score: float
    final_score: float
    reason: str = ""


@dataclass
class SessionMessage:
    role: str
    content: str
    name: str | None = None
    args: dict | None = None
    metadata: dict | None = None
    created_at: datetime | None = None


@dataclass
class WorkingMemoryState:
    task_summary: str = ""
    current_goal: str = ""
    open_questions: list[str] = field(default_factory=list)
    recent_files: list[str] = field(default_factory=list)
    file_summaries: dict[str, str] = field(default_factory=dict)
    process_notes: list[str] = field(default_factory=list)
    tool_failures: list[str] = field(default_factory=list)
    next_step: str = ""
