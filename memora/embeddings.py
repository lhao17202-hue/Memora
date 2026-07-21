"""Deterministic local embedding providers for Memora RAG."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

from .schema import MemoryItem

SUPPORTED_EMBEDDING_PROVIDERS = ("hash",)
RESERVED_EMBEDDING_PROVIDERS = ("openai", "cohere", "voyage", "sentence-transformers", "bge", "e5", "ollama")
EMBEDDING_PROVIDER_CHOICES = SUPPORTED_EMBEDDING_PROVIDERS + RESERVED_EMBEDDING_PROVIDERS


class EmbeddingProvider(Protocol):
    name: str
    model: str
    dimension: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def memory_embedding_text(item: MemoryItem) -> str:
    return (
        f"name: {item.name}\n"
        f"type: {item.type}\n"
        f"description: {item.description}\n"
        f"tags: {', '.join(item.tags)}\n"
        f"content: {item.content}"
    )


def _embedding_tokens(text: str) -> list[str]:
    lowered = (text or "").lower()
    words = re.findall(r"[a-z0-9_]+", lowered)
    chinese_chunks = re.findall(r"[一-鿿]{2,}", lowered)
    chars = [char for char in lowered if "一" <= char <= "鿿"]
    return words + chinese_chunks + chars


class HashEmbeddingProvider:
    name = "hash"

    def __init__(self, dimension: int = 384, model: str = "memora-hash-v1"):
        if dimension <= 0:
            raise ValueError("embedding dimension must be > 0")
        self.dimension = dimension
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in _embedding_tokens(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:8], "big") % self.dimension
            sign = 1.0 if digest[8] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]
