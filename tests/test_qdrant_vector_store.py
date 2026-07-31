import sys
import types

import pytest

from memora.embeddings import EmbeddingVector, SparseVector
from memora.errors import MemoryValidationError
from memora.vector_store import QdrantVectorStore, QdrantVectorStoreConfig, qdrant_point_id


class FakeModels:
    class Distance:
        COSINE = "Cosine"

    class Fusion:
        RRF = "rrf"

    class VectorParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class SparseVectorParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class SparseVector:
        def __init__(self, indices, values):
            self.indices = indices
            self.values = values

    class PointStruct:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FieldCondition:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class MatchValue:
        def __init__(self, value):
            self.value = value

    class MatchAny:
        def __init__(self, any):
            self.any = any

    class Filter:
        def __init__(self, must):
            self.must = must

    class Prefetch:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FusionQuery:
        def __init__(self, fusion):
            self.fusion = fusion


class FakePoint:
    def __init__(self, id="point", score=0.0, payload=None):
        self.id = id
        self.score = score
        self.payload = payload or {}


class FakeQueryResponse:
    def __init__(self, points):
        self.points = points


class FakeQdrantClient:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.collections = set()
        self.created = []
        self.deleted_collections = []
        self.upserts = []
        self.deletes = []
        self.queries = []
        self.retrieved = []
        self.scroll_points = []
        self.collection_info = None
        self.__class__.instances.append(self)

    def collection_exists(self, collection_name):
        return collection_name in self.collections

    def create_collection(self, **kwargs):
        self.collections.add(kwargs["collection_name"])
        self.created.append(kwargs)

    def delete_collection(self, collection_name):
        self.collections.discard(collection_name)
        self.deleted_collections.append(collection_name)

    def get_collection(self, collection_name):
        if self.collection_info is None:
            return types.SimpleNamespace(config=types.SimpleNamespace(params=types.SimpleNamespace(vectors={}, sparse_vectors={})))
        return self.collection_info

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)

    def delete(self, **kwargs):
        self.deletes.append(kwargs)

    def query_points(self, **kwargs):
        self.queries.append(kwargs)
        return FakeQueryResponse([FakePoint(score=0.75, payload={"memory_id": "mem_1", "user_id": "alice"})])

    def retrieve(self, **kwargs):
        self.retrieved.append(kwargs)
        return [FakePoint(payload={"memory_id": "mem_1", "provider": "hash"})]

    def scroll(self, **kwargs):
        return self.scroll_points, None


@pytest.fixture
def fake_qdrant(monkeypatch):
    FakeQdrantClient.instances = []
    module = types.ModuleType("qdrant_client")
    module.QdrantClient = FakeQdrantClient
    module.models = FakeModels
    monkeypatch.setitem(sys.modules, "qdrant_client", module)
    return FakeQdrantClient


def test_qdrant_vector_store_reports_missing_optional_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "qdrant_client", None)

    with pytest.raises(MemoryValidationError, match="qdrant-client"):
        QdrantVectorStore(QdrantVectorStoreConfig())


def test_qdrant_init_creates_dense_collection(fake_qdrant):
    store = QdrantVectorStore(QdrantVectorStoreConfig(dimension=4, url="http://localhost:6333"))

    store.init_storage()

    client = fake_qdrant.instances[0]
    assert client.kwargs["url"] == "http://localhost:6333"
    assert client.created[0]["collection_name"] == "memora_memories"
    assert client.created[0]["vectors_config"]["dense"].size == 4
    assert "sparse_vectors_config" not in client.created[0]


def test_qdrant_init_creates_hybrid_collection(fake_qdrant):
    store = QdrantVectorStore(QdrantVectorStoreConfig(dimension=4, retrieval_mode="hybrid"))

    store.init_storage()

    created = fake_qdrant.instances[0].created[0]
    assert created["vectors_config"]["dense"].size == 4
    assert "sparse" in created["sparse_vectors_config"]


def test_qdrant_upsert_dense_and_sparse_vectors(fake_qdrant):
    store = QdrantVectorStore(QdrantVectorStoreConfig(retrieval_mode="hybrid", dimension=2))

    store.upsert(
        "mem_1",
        EmbeddingVector(dense=[1.0, 0.0], sparse=SparseVector(indices=[3], values=[0.5])),
        {"provider": "bge", "model": "bge-m3", "dimension": 2, "text_hash": "hash"},
    )

    point = fake_qdrant.instances[0].upserts[0]["points"][0]
    assert point.id == qdrant_point_id("mem_1")
    assert point.vector["dense"] == [1.0, 0.0]
    assert point.vector["sparse"].indices == [3]
    assert point.vector["sparse"].values == [0.5]
    assert point.payload["memory_id"] == "mem_1"


def test_qdrant_hybrid_upsert_requires_sparse_vector(fake_qdrant):
    store = QdrantVectorStore(QdrantVectorStoreConfig(retrieval_mode="hybrid", dimension=2))

    with pytest.raises(MemoryValidationError, match="sparse"):
        store.upsert("mem_1", EmbeddingVector(dense=[1.0, 0.0]), {"dimension": 2})


def test_qdrant_dense_search_uses_named_dense_vector_and_filters(fake_qdrant):
    store = QdrantVectorStore(QdrantVectorStoreConfig(dimension=2))

    hits = store.search(EmbeddingVector(dense=[1.0, 0.0]), top_k=5, filters={"user_id": "alice", "types": ["preference"], "tags": ["tag"]})

    query = fake_qdrant.instances[0].queries[0]
    assert query["using"] == "dense"
    assert query["query"] == [1.0, 0.0]
    assert query["limit"] == 5
    assert [condition.key for condition in query["query_filter"].must] == ["user_id", "type", "tags"]
    assert hits[0].memory_id == "mem_1"
    assert hits[0].source == "dense"


def test_qdrant_hybrid_search_uses_dense_sparse_prefetch_and_fusion(fake_qdrant):
    store = QdrantVectorStore(QdrantVectorStoreConfig(retrieval_mode="hybrid", dimension=2, hybrid_prefetch_limit=25))

    hits = store.search(EmbeddingVector(dense=[1.0, 0.0], sparse=SparseVector(indices=[3], values=[0.5])), top_k=3, mode="hybrid")

    query = fake_qdrant.instances[0].queries[0]
    assert [prefetch.using for prefetch in query["prefetch"]] == ["dense", "sparse"]
    assert query["prefetch"][0].limit == 25
    assert query["prefetch"][1].query.indices == [3]
    assert query["query"].fusion == "rrf"
    assert query["limit"] == 3
    assert hits[0].source == "hybrid"


def test_qdrant_delete_retrieve_and_verify_use_payload_memory_ids(fake_qdrant):
    store = QdrantVectorStore(QdrantVectorStoreConfig())
    client = fake_qdrant.instances[0]
    client.scroll_points = [FakePoint(payload={"memory_id": "mem_1"}), FakePoint(payload={"memory_id": "orphan"})]

    store.delete("mem_1")
    metadata = store.get_metadata("mem_1")
    report = store.verify({"mem_1", "mem_2"})

    assert client.deletes[0]["points_selector"] == [qdrant_point_id("mem_1")]
    assert metadata["memory_id"] == "mem_1"
    assert report["vector_missing"] == ["mem_2"]
    assert report["vector_orphans"] == ["orphan"]


def test_qdrant_config_builds_from_vector_store_options():
    config = QdrantVectorStoreConfig.from_options(
        {
            "url": "http://localhost:6333",
            "api_key": "secret",
            "collection": "custom_memories",
            "timeout": 7.5,
            "prefer_grpc": True,
            "recreate_collection": True,
        },
        dimension=1024,
        retrieval_mode="hybrid",
        hybrid_prefetch_limit=25,
    )

    assert config.url == "http://localhost:6333"
    assert config.api_key == "secret"
    assert config.collection == "custom_memories"
    assert config.timeout == 7.5
    assert config.prefer_grpc is True
    assert config.recreate_collection is True
    assert config.dimension == 1024
    assert config.retrieval_mode == "hybrid"
    assert config.hybrid_prefetch_limit == 25


def test_qdrant_config_rejects_unknown_options():
    with pytest.raises(MemoryValidationError, match="unknown vector_store_options"):
        QdrantVectorStoreConfig.from_options({"bad": "value"}, dimension=384, retrieval_mode="dense", hybrid_prefetch_limit=100)
