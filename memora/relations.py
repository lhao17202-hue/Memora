"""Embedding-backed write-time memory relation detection."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .config import MemoryConfig
from .embeddings import EmbeddingProvider, memory_embedding_text
from .errors import MemoryValidationError
from .schema import MemoryCandidate, MemoryItem, MemoryRelation
from .vector_store import cosine_similarity


def candidate_embedding_text(candidate: MemoryCandidate) -> str:
    return (
        f"name: {candidate.name}\n"
        f"type: {candidate.type}\n"
        f"description: {candidate.description}\n"
        f"tags: {', '.join(candidate.tags)}\n"
        f"content: {candidate.content}"
    )


@dataclass(frozen=True)
class RelationThresholds:
    relation: float
    merge: float
    conflict: float


class SemanticMemoryRelationResolver:
    def __init__(self, embedder: EmbeddingProvider, config: MemoryConfig):
        self.embedder = embedder
        self.thresholds = RelationThresholds(
            relation=config.semantic_relation_threshold,
            merge=config.semantic_merge_threshold,
            conflict=config.semantic_conflict_threshold,
        )
        _validate_thresholds(self.thresholds)

    def resolve(self, candidate: MemoryCandidate, existing: list[MemoryItem]) -> MemoryRelation:
        comparable = [
            item
            for item in existing
            if item.status == "active"
            and item.type == candidate.type
            and item.user_id == candidate.user_id
            and item.project_id == candidate.project_id
            and item.workspace_id == candidate.workspace_id
        ]
        if not comparable:
            return MemoryRelation()

        candidate_vector = self.embedder.embed([candidate_embedding_text(candidate)])[0]
        item_vectors = self.embedder.embed([memory_embedding_text(item) for item in comparable])
        scored = [
            (cosine_similarity(candidate_vector, item_vector), item)
            for item, item_vector in zip(comparable, item_vectors, strict=True)
        ]
        scored.sort(key=lambda match: match[0], reverse=True)
        similarity, target = scored[0]

        if similarity < self.thresholds.relation:
            return MemoryRelation(similarity_score=similarity, reason="below_semantic_relation_threshold")
        if _same_durable_text(candidate, target):
            return _target_relation("duplicate", target, similarity, "semantic_duplicate")
        if similarity >= self.thresholds.conflict and _has_conflict_evidence(candidate, target):
            return _target_relation("conflict", target, similarity, "semantic_conflict")
        if similarity >= self.thresholds.merge:
            return _target_relation("merge", target, similarity, "semantic_merge")
        return MemoryRelation(similarity_score=similarity, reason="below_semantic_merge_threshold")


def _target_relation(kind: str, target: MemoryItem, similarity: float, reason: str) -> MemoryRelation:
    return MemoryRelation(
        kind=kind,
        target_memory_id=target.id,
        target_updated_at=target.updated_at,
        similarity_score=similarity,
        reason=reason,
    )


def _validate_thresholds(thresholds: RelationThresholds) -> None:
    values = (thresholds.relation, thresholds.merge, thresholds.conflict)
    if not all(isinstance(value, int | float) and not isinstance(value, bool) and 0.0 <= value <= 1.0 for value in values):
        raise MemoryValidationError("semantic relation thresholds must be numbers from 0.0 to 1.0")
    if not thresholds.relation <= thresholds.merge <= thresholds.conflict:
        raise MemoryValidationError("semantic relation thresholds must satisfy relation <= merge <= conflict")


def _same_durable_text(candidate: MemoryCandidate, item: MemoryItem) -> bool:
    return _normalize(candidate.content) == _normalize(item.content) or (
        _normalize(candidate.description) == _normalize(item.description)
        and _normalize(candidate.content) in {_normalize(item.content), _normalize(item.description)}
    )


def _has_conflict_evidence(candidate: MemoryCandidate, item: MemoryItem) -> bool:
    candidate_text = _normalize(f"{candidate.name} {candidate.description} {candidate.content}")
    item_text = _normalize(f"{item.name} {item.description} {item.content}")
    return _opposes_language_preference(candidate_text, item_text) or _opposes_boolean_preference(candidate_text, item_text)


def _opposes_language_preference(left: str, right: str) -> bool:
    language_pairs = (
        ("chinese", "english"),
        ("zh", "en"),
        ("\u4e2d\u6587", "\u82f1\u6587"),
    )
    return any((first in left and second in right) or (second in left and first in right) for first, second in language_pairs)


def _opposes_boolean_preference(left: str, right: str) -> bool:
    positive_markers = ("prefer ", "prefers ", "always ", "use ", "enable ")
    negative_markers = ("do not ", "don't ", "avoid ", "never ", "disable ", "no longer ")
    shared_terms = set(re.findall(r"[a-z0-9_]{4,}", left)) & set(re.findall(r"[a-z0-9_]{4,}", right))
    if not shared_terms:
        return False
    left_positive = any(marker in left for marker in positive_markers)
    right_positive = any(marker in right for marker in positive_markers)
    left_negative = any(marker in left for marker in negative_markers)
    right_negative = any(marker in right for marker in negative_markers)
    return (left_positive and right_negative) or (left_negative and right_positive)


def _normalize(text: str) -> str:
    return " ".join((text or "").casefold().split())
