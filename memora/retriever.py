"""Deterministic keyword retrieval and ranking."""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone

from .schema import MemoryItem, MemoryQuery, MemorySearchResult

HALF_LIFE_DAYS = {
    "user": 365,
    "feedback": 180,
    "project": 90,
    "decision": 180,
    "session_summary": 30,
    "tool_experience": 90,
    "reference": 180,
    "knowledge": 180,
    "entity": 180,
}

FIELD_WEIGHTS = {
    "name": 1.00,
    "tags": 0.95,
    "description": 0.85,
    "content": 0.65,
}


def _tokens(text: str) -> set[str]:
    lowered = (text or "").lower()
    words = set(re.findall(r"[a-z0-9_]+", lowered))
    chinese_chunks = set(re.findall(r"[一-鿿]{2,}", lowered))
    chars = {char for char in lowered if "一" <= char <= "鿿"}
    return words | chinese_chunks | chars


class MemoryRetriever:
    def retrieve(self, memories: list[MemoryItem], query: MemoryQuery) -> list[MemorySearchResult]:
        results = []
        for memory in memories:
            scored = self.score(memory, query)
            if scored is not None:
                results.append(scored)
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

        query_tokens = _tokens(query.query)
        field_texts = {
            "name": memory.name,
            "tags": " ".join(memory.tags),
            "description": memory.description,
            "content": memory.content,
        }
        similarity_score = 0.0
        reason = ""
        if query_tokens:
            for field_name, field_text in field_texts.items():
                field_tokens = _tokens(field_text)
                coverage = len(query_tokens & field_tokens) / len(query_tokens)
                weighted_score = coverage * FIELD_WEIGHTS[field_name]
                if weighted_score > similarity_score:
                    similarity_score = min(weighted_score, 1.0)
                    reason = f"matched_{field_name}"
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
