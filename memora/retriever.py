"""Deterministic keyword retrieval and ranking."""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone

from .schema import MemoryItem, MemoryQuery, MemorySearchResult

HALF_LIFE_DAYS = {
    "preference": 365,
    "project": 180,
    "episodic": 45,
    "reflective": 180,
    "tool": 120,
    "knowledge": 365,
    "general": 90,
}

FIELD_WEIGHTS = {
    "name": 1.00,
    "tags": 0.95,
    "description": 0.85,
    "content": 0.65,
}

_REASON_PRIORITY = {
    "exact_name": 7,
    "exact_description": 6,
    "phrase_content": 5,
    "tokens_tags": 4,
    "partial_name": 3,
    "partial_description": 2,
    "partial_content": 1,
}


def _tokens(text: str) -> set[str]:
    lowered = (text or "").lower()
    words = set(re.findall(r"[a-z0-9_]+", lowered))
    split_words = {part for word in words for part in word.split("_") if part}
    chinese_chunks = set(re.findall(r"[一-鿿]{2,}", lowered))
    chars = {char for char in lowered if "一" <= char <= "鿿"}
    return words | split_words | chinese_chunks | chars


def _normalize_text(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9一-鿿]+", " ", (text or "").casefold())
    return " ".join(normalized.split())


def _query_terms(text: str) -> list[str]:
    normalized = _normalize_text(text)
    return re.findall(r"[a-z0-9]+|[一-鿿]", normalized)


def _contains_adjacent_terms(query_terms: list[str], field_terms: list[str]) -> bool:
    if not query_terms or len(query_terms) > len(field_terms):
        return False
    width = len(query_terms)
    return any(field_terms[index : index + width] == query_terms for index in range(len(field_terms) - width + 1))


def _contains_ordered_terms(query_terms: list[str], field_terms: list[str]) -> bool:
    if not query_terms:
        return False
    position = 0
    for term in field_terms:
        if term == query_terms[position]:
            position += 1
            if position == len(query_terms):
                return True
    return False


def _field_reason(field_name: str, exact_or_phrase: bool, coverage: float) -> str:
    if field_name == "name" and exact_or_phrase:
        return "exact_name"
    if field_name == "description" and exact_or_phrase:
        return "exact_description"
    if field_name == "content" and exact_or_phrase:
        return "phrase_content"
    if field_name == "tags" and coverage > 0:
        return "tokens_tags"
    if field_name == "name" and coverage > 0:
        return "partial_name"
    if field_name == "description" and coverage > 0:
        return "partial_description"
    if field_name == "content" and coverage > 0:
        return "partial_content"
    return ""


def _query_has_ascii_word(query_terms: list[str]) -> bool:
    return any(re.fullmatch(r"[a-z0-9_]+", term) for term in query_terms)


def _query_has_chinese(query_terms: list[str]) -> bool:
    return any(any("一" <= char <= "鿿" for char in term) for term in query_terms)


def _field_score(query: str, field_text: str, field_name: str) -> tuple[float, str]:
    query_terms = _query_terms(query)
    if not query_terms:
        return 0.0, ""

    query_norm = _normalize_text(query)
    field_norm = _normalize_text(field_text)
    field_terms = _query_terms(field_text)
    field_tokens = _tokens(field_text)
    query_tokens = _tokens(query)
    coverage = len(query_tokens & field_tokens) / len(query_tokens) if query_tokens else 0.0

    exact = bool(query_norm and field_norm and query_norm in field_norm and not _query_has_ascii_word(query_terms))
    adjacent = _contains_adjacent_terms(query_terms, field_terms)
    ordered = _contains_ordered_terms(query_terms, field_terms)

    if exact or adjacent:
        if field_name == "name":
            base_score = 1.0
        elif field_name == "description":
            base_score = 0.95
        elif field_name == "tags":
            base_score = 0.90
        else:
            base_score = 0.82
        reason = _field_reason(field_name, exact_or_phrase=True, coverage=coverage)
    elif ordered:
        base_score = min(max(0.80, coverage), 0.90)
        reason = _field_reason(field_name, exact_or_phrase=False, coverage=coverage)
    elif coverage > 0:
        if field_name == "tags":
            base_score = max(0.80, coverage)
        elif field_name == "content":
            base_score = min(coverage, 0.70)
        else:
            base_score = coverage
        reason = _field_reason(field_name, exact_or_phrase=False, coverage=coverage)
    else:
        return 0.0, ""

    return min(base_score * FIELD_WEIGHTS[field_name], 1.0), reason


class MemoryRetriever:
    def retrieve(self, memories: list[MemoryItem], query: MemoryQuery) -> list[MemorySearchResult]:
        results_by_id: dict[str, MemorySearchResult] = {}
        for memory in memories:
            scored = self.score(memory, query)
            if scored is None:
                continue
            existing = results_by_id.get(memory.id)
            if existing is None or scored.final_score > existing.final_score:
                results_by_id[memory.id] = scored
        results = list(results_by_id.values())
        results.sort(key=lambda result: result.final_score, reverse=True)
        return results[: query.top_k]

    def score(self, memory: MemoryItem, query: MemoryQuery) -> MemorySearchResult | None:
        if memory.status == "deleted":
            return None
        if memory.status != "active" and not query.include_archived:
            return None
        if query.memory_types and memory.type not in query.memory_types:
            return None
        if query.tags and not set(query.tags).intersection(memory.tags):
            return None
        if memory.type == "knowledge" and not query.include_knowledge:
            return None

        field_texts = {
            "name": memory.name,
            "tags": " ".join(memory.tags),
            "description": memory.description,
            "content": memory.content,
        }
        field_scores = []
        for field_name, field_text in field_texts.items():
            score, reason = _field_score(query.query, field_text, field_name)
            if score > 0:
                field_scores.append((score, reason))
        if not field_scores:
            return None

        field_scores.sort(key=lambda item: (item[0], _REASON_PRIORITY.get(item[1], 0)), reverse=True)
        best_score, reason = field_scores[0]
        aggregate_bonus = min(0.12, sum(score for score, _ in field_scores[1:]) * 0.15)
        similarity_score = min(best_score + aggregate_bonus, 1.0)
        if similarity_score <= 0:
            return None

        importance_score = min(max(memory.weight, 1), 10) / 10
        recency_score = self._recency_score(memory)
        access_score = min(math.log1p(memory.access_count) / math.log1p(20), 1.0)
        final_score = (
            similarity_score * 0.45
            + importance_score * 0.25
            + recency_score * 0.20
            + access_score * 0.10
        )
        return MemorySearchResult(
            memory=memory,
            similarity_score=similarity_score,
            importance_score=importance_score,
            recency_score=recency_score,
            access_score=access_score,
            final_score=final_score,
            reason=reason,
        )

    def _recency_score(self, memory: MemoryItem) -> float:
        updated = memory.updated_at or datetime.now(timezone.utc)
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
        age_days = max((datetime.now(timezone.utc) - updated).days, 0)
        half_life = HALF_LIFE_DAYS.get(memory.type, 180)
        return math.exp(-age_days / half_life)
