"""Deterministic RAG rerankers."""

from __future__ import annotations

from typing import Protocol

from .schema import MemoryQuery, MemorySearchResult

SUPPORTED_RERANKERS = ("none", "deterministic")
RESERVED_RERANKERS = ("cross-encoder", "cohere", "llm")
RERANKER_CHOICES = SUPPORTED_RERANKERS + RESERVED_RERANKERS


class Reranker(Protocol):
    name: str

    def rank(self, query: MemoryQuery, candidates: list[MemorySearchResult]) -> list[MemorySearchResult]:
        ...


class NoOpReranker:
    name = "none"

    def rank(self, query: MemoryQuery, candidates: list[MemorySearchResult]) -> list[MemorySearchResult]:
        return candidates


class DeterministicReranker:
    name = "deterministic"

    def rank(self, query: MemoryQuery, candidates: list[MemorySearchResult]) -> list[MemorySearchResult]:
        return sorted(candidates, key=lambda result: result.final_score, reverse=True)
