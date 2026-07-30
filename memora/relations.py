"""Embedding-backed write-time memory relation detection."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

from .config import MemoryConfig
from .embeddings import EmbeddingProvider, embedding_dense, memory_embedding_text
from .errors import MemoryValidationError
from .schema import MemoryCandidate, MemoryItem, MemoryRelation, MemoryRelationDecision, validate_memory_relation_decision
from .vector_store import cosine_similarity

RELATION_JUDGE_SYSTEM_PROMPT = """Judge whether one extracted candidate memory changes, duplicates, merges with, conflicts with, supersedes, or is unrelated to one existing memory.
Return JSON only. Do not include markdown.
Allowed kind values: none, duplicate, merge, conflict, supersede.
Use "none" when the candidate should be written as a separate memory.
Use "duplicate" when the candidate says the same durable fact.
Use "merge" when the candidate refines or extends the existing memory without contradiction.
Use "conflict" when the candidate invalidates or contradicts the existing memory but replacement intent is unclear.
Use "supersede" when the candidate explicitly replaces the existing memory and the old memory should be archived.
Judge only the candidate and existing memory provided by the runtime. Do not invent another target memory.
Never use "merge" for contradictory facts. Prefer "conflict" or "supersede" for changed preferences, obsolete project facts, or "from now on" instructions.
For merge, include a "merged" object with name, description, content, and tags.
The JSON shape is:
{"kind":"merge","confidence":0.86,"reason":"brief reason","merged":{"name":"...","description":"...","content":"...","tags":["..."]}}"""


class LLMRelationClient(Protocol):
    def complete(self, messages: Sequence[Mapping[str, str]]) -> str:
        """Return a JSON-only relation decision response."""


class MemoryRelationJudge(Protocol):
    def judge(
        self,
        candidate: MemoryCandidate,
        target: MemoryItem,
        relation: MemoryRelation,
    ) -> MemoryRelationDecision:
        ...


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
            (cosine_similarity(embedding_dense(candidate_vector), embedding_dense(item_vector)), item)
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


class DeterministicRelationJudge:
    def judge(
        self,
        candidate: MemoryCandidate,
        target: MemoryItem,
        relation: MemoryRelation,
    ) -> MemoryRelationDecision:
        if relation.kind == "merge":
            return MemoryRelationDecision(
                kind=relation.kind,
                confidence=relation.similarity_score,
                reason=relation.reason or "deterministic_relation",
                merged_name=candidate.name or target.name,
                merged_description=candidate.description,
                merged_content=candidate.content,
                merged_tags=list(candidate.tags),
            )
        return MemoryRelationDecision(
            kind=relation.kind,
            confidence=relation.similarity_score,
            reason=relation.reason or "deterministic_relation",
        )


class LLMMemoryRelationJudge:
    def __init__(self, client: LLMRelationClient):
        self.client = client

    def judge(
        self,
        candidate: MemoryCandidate,
        target: MemoryItem,
        relation: MemoryRelation,
    ) -> MemoryRelationDecision:
        raw_text = self.client.complete(relation_judge_prompt_messages(candidate, target, relation))
        return parse_relation_decision_json(raw_text)


def relation_judge_prompt_messages(
    candidate: MemoryCandidate,
    target: MemoryItem,
    relation: MemoryRelation,
) -> list[dict[str, str]]:
    payload = {
        "candidate": {
            "name": candidate.name,
            "type": candidate.type,
            "description": candidate.description,
            "content": candidate.content,
            "tags": candidate.tags,
            "confidence": candidate.confidence,
        },
        "existing_memory": {
            "id": target.id,
            "name": target.name,
            "type": target.type,
            "description": target.description,
            "content": target.content,
            "tags": target.tags,
            "updated_at": target.updated_at.isoformat() if target.updated_at else None,
        },
        "embedding_relation": {
            "kind": relation.kind,
            "similarity_score": relation.similarity_score,
            "reason": relation.reason,
        },
    }
    return [
        {"role": "system", "content": RELATION_JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def parse_relation_decision_json(raw_text: str) -> MemoryRelationDecision:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise MemoryValidationError(f"invalid_relation_decision_json:{exc.msg}") from exc
    if not isinstance(payload, dict):
        raise MemoryValidationError("relation_decision_payload_must_be_object")

    kind = payload.get("kind")
    if kind not in {"none", "duplicate", "merge", "conflict", "supersede"}:
        raise MemoryValidationError(f"invalid relation decision kind: {kind}")

    confidence = payload.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, int | float) or confidence < 0.0 or confidence > 1.0:
        raise MemoryValidationError("relation decision confidence must be from 0.0 to 1.0")

    reason = payload.get("reason", "")
    if reason is None:
        reason = ""
    if not isinstance(reason, str):
        raise MemoryValidationError("relation decision reason must be a string")

    merged_name = None
    merged_description = None
    merged_content = None
    merged_tags = None
    raw_merged = payload.get("merged")
    if raw_merged is not None:
        if not isinstance(raw_merged, dict):
            raise MemoryValidationError("relation decision merged must be an object")
        merged_name = _optional_non_empty_string(raw_merged, "name")
        merged_description = _optional_non_empty_string(raw_merged, "description")
        merged_content = _optional_non_empty_string(raw_merged, "content")
        if raw_merged.get("tags") is not None:
            raw_tags = raw_merged.get("tags")
            if not isinstance(raw_tags, list) or not all(isinstance(tag, str) for tag in raw_tags):
                raise MemoryValidationError("relation decision merged tags must be a list of strings")
            merged_tags = raw_tags
    if kind == "merge" and (not merged_description or not merged_content):
        raise MemoryValidationError("merge relation decision requires merged description and content")

    decision = MemoryRelationDecision(
        kind=kind,
        confidence=float(confidence),
        reason=reason,
        merged_name=merged_name,
        merged_description=merged_description,
        merged_content=merged_content,
        merged_tags=merged_tags,
    )
    validate_memory_relation_decision(decision)
    return decision


def _target_relation(kind: str, target: MemoryItem, similarity: float, reason: str) -> MemoryRelation:
    return MemoryRelation(
        kind=kind,
        target_memory_id=target.id,
        target_updated_at=target.updated_at,
        similarity_score=similarity,
        reason=reason,
    )


def _optional_non_empty_string(data: dict, field_name: str) -> str | None:
    value = data.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise MemoryValidationError(f"relation decision merged {field_name} must be a non-empty string")
    return value


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
