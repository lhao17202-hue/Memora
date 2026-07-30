"""Vector stores for Memora RAG."""

from __future__ import annotations

import json
import math
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .config import MemoryConfig
from .embeddings import EmbeddingVector, SparseVector, embedding_dense
from .errors import MemoryValidationError
from .utils import now_utc

SUPPORTED_VECTOR_STORES = ("sqlite", "qdrant")
RESERVED_VECTOR_STORES = ("sqlite-vec", "chroma", "pgvector", "faiss", "milvus", "weaviate")
VECTOR_STORE_CHOICES = SUPPORTED_VECTOR_STORES + RESERVED_VECTOR_STORES

QDRANT_DENSE_VECTOR_NAME = "dense"
QDRANT_SPARSE_VECTOR_NAME = "sparse"


@dataclass
class VectorSearchHit:
    memory_id: str
    score: float
    metadata: dict[str, Any]
    dense_score: float | None = None
    sparse_score: float | None = None
    source: str = "dense"


class VectorStore(Protocol):
    name: str

    def init_storage(self) -> None:
        ...

    def upsert(self, memory_id: str, vector: EmbeddingVector | list[float], metadata: dict[str, Any]) -> None:
        ...

    def delete(self, memory_id: str) -> None:
        ...

    def search(self, vector: EmbeddingVector | list[float], top_k: int, filters: dict[str, Any] | None = None, mode: str = "dense") -> list[VectorSearchHit]:
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


def qdrant_point_id(memory_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"memora:{memory_id}"))


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

    def upsert(self, memory_id: str, vector: EmbeddingVector | list[float], metadata: dict[str, Any]) -> None:
        self.init_storage()
        dense = embedding_dense(vector)
        dimension = int(metadata.get("dimension") or len(dense))
        if len(dense) != dimension:
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
                    json.dumps(dense),
                    json.dumps(metadata, ensure_ascii=False),
                    str(metadata.get("updated_at") or now_utc().isoformat()),
                ),
            )

    def delete(self, memory_id: str) -> None:
        self.init_storage()
        with self._connection() as connection:
            connection.execute("DELETE FROM memory_vectors WHERE memory_id = ?", (memory_id,))

    def search(self, vector: EmbeddingVector | list[float], top_k: int, filters: dict[str, Any] | None = None, mode: str = "dense") -> list[VectorSearchHit]:
        if top_k <= 0:
            return []
        query_dense = embedding_dense(vector)
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
            score = cosine_similarity(query_dense, [float(value) for value in stored_vector])
            hits.append(VectorSearchHit(memory_id=str(row["memory_id"]), score=score, metadata=metadata, dense_score=score, source="dense"))
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


class QdrantVectorStore:
    name = "qdrant"

    def __init__(self, config: MemoryConfig):
        self.config = config
        try:
            from qdrant_client import QdrantClient, models
        except Exception as exc:  # noqa: BLE001 - optional dependency may fail during import
            raise MemoryValidationError("vector_store 'qdrant' requires optional dependency qdrant-client; install with: pip install -e \".[qdrant]\"") from exc
        self.models = models
        self.collection = config.qdrant_collection
        client_kwargs: dict[str, Any] = {
            "api_key": config.qdrant_api_key,
            "timeout": config.qdrant_timeout,
            "prefer_grpc": config.qdrant_prefer_grpc,
        }
        if config.qdrant_url:
            client_kwargs["url"] = config.qdrant_url
        else:
            client_kwargs["host"] = config.qdrant_host
            client_kwargs["port"] = config.qdrant_port
        self.client = QdrantClient(**{key: value for key, value in client_kwargs.items() if value is not None})

    def init_storage(self) -> None:
        exists = self._collection_exists()
        if exists and self.config.qdrant_recreate_collection:
            self.client.delete_collection(collection_name=self.collection)
            exists = False
        if exists:
            self._validate_collection()
            return
        vectors_config = {
            QDRANT_DENSE_VECTOR_NAME: self.models.VectorParams(size=self.config.embedding_dimension, distance=self.models.Distance.COSINE)
        }
        kwargs: dict[str, Any] = {"collection_name": self.collection, "vectors_config": vectors_config}
        if self.config.retrieval_mode == "hybrid":
            kwargs["sparse_vectors_config"] = {QDRANT_SPARSE_VECTOR_NAME: self.models.SparseVectorParams()}
        self.client.create_collection(**kwargs)

    def upsert(self, memory_id: str, vector: EmbeddingVector | list[float], metadata: dict[str, Any]) -> None:
        self.init_storage()
        embedding = _as_embedding_vector(vector)
        qdrant_vector: dict[str, Any] = {QDRANT_DENSE_VECTOR_NAME: embedding.dense}
        if self.config.retrieval_mode == "hybrid":
            if embedding.sparse is None:
                raise MemoryValidationError("hybrid retrieval requires sparse embedding vectors")
            qdrant_vector[QDRANT_SPARSE_VECTOR_NAME] = self._sparse_vector(embedding.sparse)
        payload = dict(metadata)
        payload["memory_id"] = memory_id
        point = self.models.PointStruct(id=qdrant_point_id(memory_id), vector=qdrant_vector, payload=payload)
        self.client.upsert(collection_name=self.collection, points=[point])

    def delete(self, memory_id: str) -> None:
        self.init_storage()
        self.client.delete(collection_name=self.collection, points_selector=[qdrant_point_id(memory_id)])

    def search(self, vector: EmbeddingVector | list[float], top_k: int, filters: dict[str, Any] | None = None, mode: str = "dense") -> list[VectorSearchHit]:
        if top_k <= 0:
            return []
        self.init_storage()
        embedding = _as_embedding_vector(vector)
        qdrant_filter = self._filter(filters)
        if mode == "hybrid":
            if embedding.sparse is None:
                raise MemoryValidationError("hybrid retrieval requires sparse query vector")
            response = self.client.query_points(
                collection_name=self.collection,
                prefetch=[
                    self.models.Prefetch(query=embedding.dense, using=QDRANT_DENSE_VECTOR_NAME, limit=self.config.hybrid_prefetch_limit),
                    self.models.Prefetch(query=self._sparse_vector(embedding.sparse), using=QDRANT_SPARSE_VECTOR_NAME, limit=self.config.hybrid_prefetch_limit),
                ],
                query=self.models.FusionQuery(fusion=self.models.Fusion.RRF),
                query_filter=qdrant_filter,
                limit=top_k,
            )
            return self._hits(response, source="hybrid")
        response = self.client.query_points(
            collection_name=self.collection,
            query=embedding.dense,
            using=QDRANT_DENSE_VECTOR_NAME,
            query_filter=qdrant_filter,
            limit=top_k,
        )
        return self._hits(response, source="dense")

    def get_metadata(self, memory_id: str) -> dict[str, Any] | None:
        self.init_storage()
        points = self.client.retrieve(collection_name=self.collection, ids=[qdrant_point_id(memory_id)], with_payload=True, with_vectors=False)
        if not points:
            return None
        payload = getattr(points[0], "payload", None) or {}
        return dict(payload) if isinstance(payload, dict) else None

    def verify(self, expected_memory_ids: set[str]) -> dict[str, Any]:
        self.init_storage()
        report: dict[str, Any] = {"vector_missing": [], "vector_orphans": [], "vector_errors": []}
        vector_ids = set()
        offset = None
        while True:
            points, offset = self.client.scroll(collection_name=self.collection, limit=256, offset=offset, with_payload=True, with_vectors=False)
            for point in points:
                payload = getattr(point, "payload", None) or {}
                memory_id = payload.get("memory_id")
                if isinstance(memory_id, str):
                    vector_ids.add(memory_id)
                else:
                    report["vector_errors"].append({"memory_id": str(getattr(point, "id", "unknown")), "error": "payload missing memory_id"})
            if offset is None:
                break
        report["vector_missing"] = sorted(expected_memory_ids - vector_ids)
        report["vector_orphans"] = sorted(vector_ids - expected_memory_ids)
        return report

    def _collection_exists(self) -> bool:
        if hasattr(self.client, "collection_exists"):
            return bool(self.client.collection_exists(collection_name=self.collection))
        collections = self.client.get_collections().collections
        return any(getattr(collection, "name", None) == self.collection for collection in collections)

    def _validate_collection(self) -> None:
        if not hasattr(self.client, "get_collection"):
            return
        try:
            info = self.client.get_collection(collection_name=self.collection)
        except Exception as exc:  # noqa: BLE001 - report incompatible/unreadable collection clearly
            raise MemoryValidationError(f"unable to validate qdrant collection '{self.collection}': {exc}") from exc
        config = getattr(getattr(info, "config", None), "params", None)
        vectors = getattr(config, "vectors", None)
        if isinstance(vectors, dict):
            dense = vectors.get(QDRANT_DENSE_VECTOR_NAME)
            size = getattr(dense, "size", None) if dense is not None else None
            if size is not None and int(size) != self.config.embedding_dimension:
                raise MemoryValidationError(f"qdrant collection '{self.collection}' dense vector dimension mismatch: expected {self.config.embedding_dimension}, got {size}")
        if self.config.retrieval_mode == "hybrid":
            sparse_vectors = getattr(config, "sparse_vectors", None)
            if isinstance(sparse_vectors, dict) and QDRANT_SPARSE_VECTOR_NAME not in sparse_vectors:
                raise MemoryValidationError(f"qdrant collection '{self.collection}' does not define sparse vector '{QDRANT_SPARSE_VECTOR_NAME}'")

    def _sparse_vector(self, sparse: SparseVector) -> Any:
        return self.models.SparseVector(indices=sparse.indices, values=sparse.values)

    def _filter(self, filters: dict[str, Any] | None) -> Any:
        if not filters:
            return None
        conditions = []
        for key in ("user_id", "project_id", "workspace_id", "status"):
            value = filters.get(key)
            if value is not None:
                conditions.append(self.models.FieldCondition(key=key, match=self.models.MatchValue(value=value)))
        types = filters.get("types")
        if types:
            conditions.append(self.models.FieldCondition(key="type", match=self.models.MatchAny(any=types)))
        tags = filters.get("tags")
        if tags:
            conditions.append(self.models.FieldCondition(key="tags", match=self.models.MatchAny(any=tags)))
        if not conditions:
            return None
        return self.models.Filter(must=conditions)

    def _hits(self, response: Any, source: str) -> list[VectorSearchHit]:
        points = getattr(response, "points", response)
        hits = []
        for point in points or []:
            payload = getattr(point, "payload", None) or {}
            memory_id = payload.get("memory_id")
            if not isinstance(memory_id, str):
                continue
            score = float(getattr(point, "score", 0.0) or 0.0)
            hits.append(
                VectorSearchHit(
                    memory_id=memory_id,
                    score=score,
                    metadata=dict(payload),
                    dense_score=score if source == "dense" else None,
                    source=source,
                )
            )
        return hits


def _as_embedding_vector(vector: EmbeddingVector | list[float]) -> EmbeddingVector:
    if isinstance(vector, EmbeddingVector):
        return vector
    return EmbeddingVector(dense=vector)
