"""SQLite vector store for Memora RAG."""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .config import MemoryConfig
from .utils import now_utc

SUPPORTED_VECTOR_STORES = ("sqlite",)
RESERVED_VECTOR_STORES = ("sqlite-vec", "qdrant", "chroma", "pgvector", "faiss", "milvus", "weaviate")
VECTOR_STORE_CHOICES = SUPPORTED_VECTOR_STORES + RESERVED_VECTOR_STORES


@dataclass
class VectorSearchHit:
    memory_id: str
    score: float
    metadata: dict[str, Any]


class VectorStore(Protocol):
    name: str

    def init_storage(self) -> None:
        ...

    def upsert(self, memory_id: str, vector: list[float], metadata: dict[str, Any]) -> None:
        ...

    def delete(self, memory_id: str) -> None:
        ...

    def search(self, vector: list[float], top_k: int, filters: dict[str, Any] | None = None) -> list[VectorSearchHit]:
        ...

    def get_metadata(self, memory_id: str) -> dict[str, Any] | None:
        ...

    def verify(self, expected_memory_ids: set[str]) -> dict[str, Any]:
        ...


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(left * right for left, right in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(value * value for value in a))
    norm_b = math.sqrt(sum(value * value for value in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SQLiteVectorStore:
    name = "sqlite"

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.root = Path(config.root_dir)
        if config.vector_path is not None:
            self.db_path = Path(config.vector_path)
        elif config.sqlite_path is not None:
            self.db_path = Path(config.sqlite_path)
        else:
            self.db_path = self.root / "memora.sqlite3"

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def init_storage(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_vectors (
                    memory_id TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    text_hash TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_vectors_provider_model
                ON memory_vectors(provider, model)
                """
            )

    def upsert(self, memory_id: str, vector: list[float], metadata: dict[str, Any]) -> None:
        self.init_storage()
        dimension = int(metadata.get("dimension") or len(vector))
        if len(vector) != dimension:
            raise ValueError("vector length must match metadata dimension")
        metadata = dict(metadata)
        metadata["memory_id"] = memory_id
        metadata["dimension"] = dimension
        with self._connection() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO memory_vectors(
                    memory_id, provider, model, dimension, text_hash, vector_json, metadata_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    str(metadata.get("provider") or ""),
                    str(metadata.get("model") or ""),
                    dimension,
                    str(metadata.get("text_hash") or ""),
                    json.dumps(vector),
                    json.dumps(metadata, ensure_ascii=False),
                    str(metadata.get("updated_at") or now_utc().isoformat()),
                ),
            )

    def delete(self, memory_id: str) -> None:
        self.init_storage()
        with self._connection() as connection:
            connection.execute("DELETE FROM memory_vectors WHERE memory_id = ?", (memory_id,))

    def search(self, vector: list[float], top_k: int, filters: dict[str, Any] | None = None) -> list[VectorSearchHit]:
        if top_k <= 0:
            return []
        self.init_storage()
        hits = []
        with self._connection() as connection:
            rows = connection.execute("SELECT memory_id, vector_json, metadata_json FROM memory_vectors").fetchall()
        for row in rows:
            try:
                metadata = dict(json.loads(row["metadata_json"] or "{}"))
                stored_vector = list(json.loads(row["vector_json"] or "[]"))
            except (TypeError, json.JSONDecodeError):
                continue
            if not self._matches_filters(metadata, filters):
                continue
            score = cosine_similarity(vector, [float(value) for value in stored_vector])
            hits.append(VectorSearchHit(memory_id=str(row["memory_id"]), score=score, metadata=metadata))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]

    def get_metadata(self, memory_id: str) -> dict[str, Any] | None:
        self.init_storage()
        with self._connection() as connection:
            row = connection.execute("SELECT metadata_json FROM memory_vectors WHERE memory_id = ?", (memory_id,)).fetchone()
        if row is None:
            return None
        try:
            metadata = json.loads(row["metadata_json"] or "{}")
        except (TypeError, json.JSONDecodeError):
            return None
        return dict(metadata) if isinstance(metadata, dict) else None

    def verify(self, expected_memory_ids: set[str]) -> dict[str, Any]:
        self.init_storage()
        report: dict[str, Any] = {
            "vector_missing": [],
            "vector_orphans": [],
            "vector_errors": [],
        }
        vector_ids = set()
        with self._connection() as connection:
            rows = connection.execute("SELECT * FROM memory_vectors ORDER BY memory_id").fetchall()
        for row in rows:
            memory_id = str(row["memory_id"])
            vector_ids.add(memory_id)
            try:
                vector = json.loads(row["vector_json"] or "[]")
                metadata = json.loads(row["metadata_json"] or "{}")
                if not isinstance(vector, list):
                    raise ValueError("vector_json must be a list")
                if not isinstance(metadata, dict):
                    raise ValueError("metadata_json must be an object")
                if int(row["dimension"]) != len(vector):
                    raise ValueError("dimension does not match vector length")
            except Exception as exc:  # noqa: BLE001 - verification reports row errors
                report["vector_errors"].append({"memory_id": memory_id, "error": str(exc)})
        report["vector_missing"] = sorted(expected_memory_ids - vector_ids)
        report["vector_orphans"] = sorted(vector_ids - expected_memory_ids)
        return report

    def _matches_filters(self, metadata: dict[str, Any], filters: dict[str, Any] | None) -> bool:
        if not filters:
            return True
        for key in ("user_id", "project_id", "workspace_id", "status"):
            expected = filters.get(key)
            if expected is not None and metadata.get(key) != expected:
                return False
        expected_types = filters.get("types")
        if expected_types and metadata.get("type") not in expected_types:
            return False
        expected_tags = filters.get("tags")
        if expected_tags:
            tags = metadata.get("tags") or []
            if not set(expected_tags).intersection(tags):
                return False
        return True
