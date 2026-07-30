"""Deterministic local embedding providers for Memora RAG."""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any, Protocol

from .errors import MemoryValidationError
from .schema import MemoryItem

SUPPORTED_EMBEDDING_PROVIDERS = ("hash", "bge")
RESERVED_EMBEDDING_PROVIDERS = ("openai", "cohere", "voyage", "sentence-transformers", "e5", "ollama")
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


class BgeM3EmbeddingProvider:
    name = "bge"

    def __init__(
        self,
        model_path: str | Path | None,
        model: str = "bge-m3",
        dimension: int = 1024,
        batch_size: int = 8,
        fp16: bool = False,
    ):
        if not model_path:
            raise MemoryValidationError("embedding_model_path is required when embedding_provider is 'bge'")
        if dimension <= 0:
            raise MemoryValidationError("embedding dimension must be > 0")
        if batch_size <= 0:
            raise MemoryValidationError("embedding_batch_size must be > 0")
        try:
            from FlagEmbedding import BGEM3FlagModel
        except Exception as exc:  # noqa: BLE001 - optional dependency may fail during import
            raise MemoryValidationError("embedding_provider 'bge' requires optional dependency FlagEmbedding") from exc
        self.model = model
        self.dimension = dimension
        self.batch_size = batch_size
        self.model_path = str(model_path)
        self._model = BGEM3FlagModel(self.model_path, use_fp16=fp16)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        result = self._model.encode(
            sentences=texts,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
            batch_size=self.batch_size,
        )
        dense = result.get("dense_vecs") if isinstance(result, dict) else getattr(result, "dense_vecs", None)
        if dense is None:
            raise MemoryValidationError("bge embedding result missing dense_vecs")
        vectors = dense.tolist() if hasattr(dense, "tolist") else dense
        normalized = [[float(value) for value in vector] for vector in vectors]
        for vector in normalized:
            if len(vector) != self.dimension:
                raise MemoryValidationError(f"bge embedding dimension mismatch: expected {self.dimension}, got {len(vector)}")
        return normalized
