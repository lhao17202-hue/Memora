"""Deterministic local embedding providers for Memora RAG."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .errors import MemoryValidationError
from .schema import MemoryItem

SUPPORTED_EMBEDDING_PROVIDERS = ("hash", "bge")
RESERVED_EMBEDDING_PROVIDERS = ("openai", "cohere", "voyage", "sentence-transformers", "e5", "ollama")
EMBEDDING_PROVIDER_CHOICES = SUPPORTED_EMBEDDING_PROVIDERS + RESERVED_EMBEDDING_PROVIDERS


@dataclass(frozen=True)
class SparseVector:
    indices: list[int]
    values: list[float]


@dataclass(frozen=True)
class EmbeddingVector:
    dense: list[float]
    sparse: SparseVector | None = None


class EmbeddingProvider(Protocol):
    name: str
    model: str
    dimension: int
    supports_sparse: bool

    def embed(self, texts: list[str]) -> list[EmbeddingVector]:
        ...


def embedding_dense(vector: EmbeddingVector | list[float]) -> list[float]:
    if isinstance(vector, EmbeddingVector):
        return vector.dense
    return vector


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
    supports_sparse = False

    def __init__(self, dimension: int = 384, model: str = "memora-hash-v1"):
        if dimension <= 0:
            raise ValueError("embedding dimension must be > 0")
        self.dimension = dimension
        self.model = model

    def embed(self, texts: list[str]) -> list[EmbeddingVector]:
        return [EmbeddingVector(dense=self._embed_one(text)) for text in texts]

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
        return_sparse: bool = False,
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
        self.supports_sparse = True
        self.return_sparse = return_sparse
        self._model = BGEM3FlagModel(self.model_path, use_fp16=fp16)

    def embed(self, texts: list[str]) -> list[EmbeddingVector]:
        if not texts:
            return []
        result = self._model.encode(
            sentences=texts,
            return_dense=True,
            return_sparse=self.return_sparse,
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
        sparse_vectors = self._sparse_vectors(result, len(normalized)) if self.return_sparse else [None] * len(normalized)
        return [EmbeddingVector(dense=dense_vector, sparse=sparse_vector) for dense_vector, sparse_vector in zip(normalized, sparse_vectors, strict=True)]

    def _sparse_vectors(self, result: Any, expected_count: int) -> list[SparseVector]:
        raw_sparse = result.get("lexical_weights") if isinstance(result, dict) else getattr(result, "lexical_weights", None)
        if raw_sparse is None:
            raise MemoryValidationError("bge embedding result missing lexical_weights for sparse embeddings")
        if len(raw_sparse) != expected_count:
            raise MemoryValidationError("bge sparse embedding count does not match dense embedding count")
        return [_lexical_weights_to_sparse_vector(weights) for weights in raw_sparse]


def _lexical_weights_to_sparse_vector(weights: Any) -> SparseVector:
    if not isinstance(weights, dict):
        raise MemoryValidationError("bge lexical_weights entries must be mappings")
    pairs = []
    for raw_index, raw_value in weights.items():
        value = float(raw_value)
        if value == 0:
            continue
        pairs.append((int(raw_index), value))
    pairs.sort(key=lambda pair: pair[0])
    return SparseVector(indices=[index for index, _ in pairs], values=[value for _, value in pairs])
