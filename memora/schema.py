"""Core data structures for Memora."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from .errors import MemoryValidationError
from .taxonomy import MemoryType, VALID_MEMORY_TYPES

MemoryStatus = Literal["active", "archived", "deleted"]

CandidateAction = Literal["create", "update", "supersede", "archive", "delete", "reject", "ask_user"]
SuggestedAction = Literal["create", "update", "supersede"]
RelationKind = Literal["none", "duplicate", "merge", "conflict", "supersede"]

VALID_MEMORY_STATUSES = ("active", "archived", "deleted")

VALID_CANDIDATE_ACTIONS = ("create", "update", "supersede", "archive", "delete", "reject", "ask_user")

VALID_SUGGESTED_ACTIONS = ("create", "update", "supersede")

VALID_RELATION_KINDS = ("none", "duplicate", "merge", "conflict", "supersede")

VALID_SESSION_ROLES = ("user", "assistant", "system", "tool")


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
    weight: int | None = None
    target_memory_id: str | None = None
    target_updated_at: datetime | None = None
    target_name: str | None = None
    suggested_action: SuggestedAction | None = None
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
class MemoryRelation:
    kind: RelationKind = "none"
    target_memory_id: str | None = None
    target_updated_at: datetime | None = None
    similarity_score: float = 0.0
    reason: str = ""


@dataclass
class MemoryRelationDecision:
    kind: RelationKind = "none"
    confidence: float = 0.0
    reason: str = ""
    merged_name: str | None = None
    merged_description: str | None = None
    merged_content: str | None = None
    merged_tags: list[str] | None = None


@dataclass
class MemorySearchResult:
    memory: MemoryItem
    similarity_score: float
    importance_score: float
    recency_score: float
    access_score: float
    final_score: float
    reason: str = ""
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    rerank_score: float | None = None


@dataclass
class MemoryWriteResult:
    action: str
    memory: MemoryItem | None = None
    candidate: MemoryCandidate | None = None
    reason: str = ""
    target_memory_id: str | None = None
    relation_kind: RelationKind | None = None
    relation_confidence: float | None = None
    relation_reason: str = ""
    relation_judge_status: str | None = None
    relation_judge_error: str | None = None
    rag_sync_errors: list[dict[str, str]] = field(default_factory=list)


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
    task: str = ""
    tool_notes: list[str] = field(default_factory=list)
    recent_files: list[str] = field(default_factory=list)
    file_summaries: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    trace: str = ""


def _require_non_empty_string(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise MemoryValidationError(f"{field_name} must be a non-empty string")


def validate_memory_type(value: str) -> None:
    if value not in VALID_MEMORY_TYPES:
        raise MemoryValidationError(f"invalid memory type: {value}")


def validate_memory_status(value: str) -> None:
    if value not in VALID_MEMORY_STATUSES:
        raise MemoryValidationError(f"invalid memory status: {value}")


def validate_candidate_action(value: str) -> None:
    if value not in VALID_CANDIDATE_ACTIONS:
        raise MemoryValidationError(f"invalid candidate action: {value}")


def validate_suggested_action(value: str) -> None:
    if value not in VALID_SUGGESTED_ACTIONS:
        raise MemoryValidationError(f"invalid suggested action: {value}")


def validate_relation_kind(value: str) -> None:
    if value not in VALID_RELATION_KINDS:
        raise MemoryValidationError(f"invalid relation kind: {value}")


def _validate_weight(weight: int) -> None:
    if isinstance(weight, bool) or not isinstance(weight, int) or weight < 1 or weight > 10:
        raise MemoryValidationError("weight must be an integer from 1 to 10")


def _validate_unit_score(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float) or value < 0.0 or value > 1.0:
        raise MemoryValidationError(f"{field_name} must be from 0.0 to 1.0")


def _validate_confidence(confidence: float) -> None:
    _validate_unit_score(confidence, "confidence")


def _validate_string_list(values: list[str], field_name: str) -> None:
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise MemoryValidationError(f"{field_name} must be a list of strings")


def _validate_access_count(access_count: int) -> None:
    if isinstance(access_count, bool) or not isinstance(access_count, int) or access_count < 0:
        raise MemoryValidationError("access_count must be an integer >= 0")


def _validate_positive_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MemoryValidationError(f"{field_name} must be an integer > 0")


def _validate_optional_string(value: str | None, field_name: str) -> None:
    if value is not None and not isinstance(value, str):
        raise MemoryValidationError(f"{field_name} must be a string or null")


def _validate_optional_non_empty_string(value: str | None, field_name: str) -> None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise MemoryValidationError(f"{field_name} must be a non-empty string or null")


def validate_memory_item(item: MemoryItem) -> None:
    _require_non_empty_string(item.id, "memory id")
    _require_non_empty_string(item.name, "memory name")
    _require_non_empty_string(item.description, "memory description")
    _require_non_empty_string(item.content, "memory content")
    validate_memory_type(item.type)
    validate_memory_status(item.status)
    _validate_string_list(item.tags, "tags")
    _validate_string_list(item.supersedes, "supersedes")
    _validate_string_list(item.related, "related")
    _validate_weight(item.weight)
    _validate_confidence(item.confidence)
    _validate_access_count(item.access_count)


def validate_memory_candidate(candidate: MemoryCandidate) -> None:
    validate_candidate_action(candidate.action)
    _require_non_empty_string(candidate.name, "candidate name")
    _require_non_empty_string(candidate.description, "candidate description")
    _require_non_empty_string(candidate.content, "candidate content")
    validate_memory_type(candidate.type)
    if candidate.suggested_action is not None:
        validate_suggested_action(candidate.suggested_action)
    if candidate.weight is not None:
        _validate_weight(candidate.weight)
    _validate_confidence(candidate.confidence)
    _validate_string_list(candidate.tags, "tags")


def validate_memory_query(query: MemoryQuery) -> None:
    if not isinstance(query.query, str):
        raise MemoryValidationError("query must be a string")
    _validate_positive_int(query.top_k, "top_k")
    _validate_positive_int(query.max_tokens, "max_tokens")
    if query.memory_types:
        for memory_type in query.memory_types:
            validate_memory_type(memory_type)
    if query.tags is not None:
        _validate_string_list(query.tags, "tags")


def validate_memory_relation(relation: MemoryRelation) -> None:
    validate_relation_kind(relation.kind)
    _validate_unit_score(relation.similarity_score, "similarity_score")
    _validate_optional_string(relation.reason, "relation reason")
    if relation.kind == "none":
        if relation.target_memory_id is not None and not isinstance(relation.target_memory_id, str):
            raise MemoryValidationError("target_memory_id must be a string or null")
        return
    _validate_optional_non_empty_string(relation.target_memory_id, "target_memory_id")
    if relation.target_memory_id is None:
        raise MemoryValidationError("target_memory_id is required for relation kind")


def validate_memory_relation_decision(decision: MemoryRelationDecision) -> None:
    validate_relation_kind(decision.kind)
    _validate_confidence(decision.confidence)
    _validate_optional_string(decision.reason, "relation decision reason")
    _validate_optional_non_empty_string(decision.merged_name, "merged_name")
    _validate_optional_non_empty_string(decision.merged_description, "merged_description")
    _validate_optional_non_empty_string(decision.merged_content, "merged_content")
    if decision.merged_tags is not None:
        _validate_string_list(decision.merged_tags, "merged_tags")
    if decision.kind == "merge" and (not decision.merged_description or not decision.merged_content):
        raise MemoryValidationError("merge relation decision requires merged description and content")


def validate_session_message(message: SessionMessage) -> None:
    if message.role not in VALID_SESSION_ROLES:
        raise MemoryValidationError(f"invalid session role: {message.role}")
    if not isinstance(message.content, str):
        raise MemoryValidationError("session message content must be a string")
