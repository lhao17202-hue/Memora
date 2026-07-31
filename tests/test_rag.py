import sys
import types
from datetime import datetime, timedelta, timezone

import pytest

from memora.config import MemoryConfig
from memora.errors import MemoryValidationError
from memora.manager import MemoryManager
from memora.rag import RagRetriever, build_embedding_provider, build_reranker, build_vector_metadata, build_vector_store
from memora.reranker import DeterministicReranker, NoOpReranker
from memora.retriever import MemoryRetriever
from memora.schema import MemoryItem, MemoryQuery, MemorySearchResult
from memora.vector_store import QdrantVectorStoreConfig, VectorSearchHit


class _DenseResult:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


class _FakeBgeModel:
    def __init__(self, model_path, use_fp16=False):
        self.model_path = model_path
        self.use_fp16 = use_fp16

    def encode(self, **kwargs):
        return {"dense_vecs": _DenseResult([[1.0] * 1024 for _ in kwargs["sentences"]]), "lexical_weights": []}


@pytest.fixture
def fake_flag_embedding(monkeypatch):
    module = types.ModuleType("FlagEmbedding")
    module.BGEM3FlagModel = _FakeBgeModel
    monkeypatch.setitem(sys.modules, "FlagEmbedding", module)


def test_rag_factories_support_only_v1_values(tmp_path, fake_flag_embedding):
    config = MemoryConfig(root_dir=tmp_path / ".memora", rag_enabled=True)

    assert build_embedding_provider(config).name == "hash"
    assert build_vector_store(config).name == "sqlite"
    assert build_reranker(config).name == "deterministic"

    bge = build_embedding_provider(
        MemoryConfig(
            rag_enabled=True,
            embedding_provider="bge",
            embedding_model_path="C:/Download/bge-m3",
        )
    )
    assert bge.name == "bge"
    assert bge.model == "bge-m3"
    assert bge.dimension == 1024

    with pytest.raises(MemoryValidationError, match="reserved but not implemented"):
        build_embedding_provider(MemoryConfig(rag_enabled=True, embedding_provider="openai"))
    with pytest.raises(MemoryValidationError, match="qdrant-client"):
        build_vector_store(MemoryConfig(rag_enabled=True, vector_store="qdrant"))
    with pytest.raises(MemoryValidationError, match="reserved but not implemented"):
        build_reranker(MemoryConfig(rag_enabled=True, reranker="llm"))
    with pytest.raises(MemoryValidationError, match="embedding_provider"):
        build_embedding_provider(MemoryConfig(rag_enabled=True, embedding_provider="unknown"))


def test_build_vector_store_passes_options_to_qdrant(fake_flag_embedding, monkeypatch):
    captured = {}

    class FakeQdrantStore:
        name = "qdrant"

        def __init__(self, config):
            captured["config"] = config

    monkeypatch.setattr("memora.rag.QdrantVectorStore", FakeQdrantStore)

    store = build_vector_store(
        MemoryConfig(
            rag_enabled=True,
            vector_store="qdrant",
            embedding_dimension=1024,
            retrieval_mode="hybrid",
            embedding_sparse=True,
            hybrid_prefetch_limit=25,
            vector_store_options={"url": "http://localhost:6333", "collection": "custom"},
        )
    )

    assert store.name == "qdrant"
    assert isinstance(captured["config"], QdrantVectorStoreConfig)
    assert captured["config"].url == "http://localhost:6333"
    assert captured["config"].collection == "custom"
    assert captured["config"].dimension == 1024
    assert captured["config"].retrieval_mode == "hybrid"
    assert captured["config"].hybrid_prefetch_limit == 25


def test_build_vector_store_uses_bge_default_dimension_for_qdrant(fake_flag_embedding, monkeypatch):
    captured = {}

    class FakeQdrantStore:
        name = "qdrant"

        def __init__(self, config):
            captured["config"] = config

    monkeypatch.setattr("memora.rag.QdrantVectorStore", FakeQdrantStore)
    config = MemoryConfig(
        rag_enabled=True,
        embedding_provider="bge",
        embedding_model_path="C:/Download/bge-m3",
        vector_store="qdrant",
    )

    embedder = build_embedding_provider(config)
    build_vector_store(config)

    assert embedder.dimension == 1024
    assert captured["config"].dimension == embedder.dimension


def test_rerankers_are_deterministic():
    item = MemoryItem(id="mem_1", name="one", description="desc", type="preference", content="content")
    low = MemorySearchResult(item, 0.1, 0.1, 0.1, 0.0, 0.1)
    high = MemorySearchResult(item, 0.9, 0.1, 0.1, 0.0, 0.9)
    query = MemoryQuery(query="content")

    assert NoOpReranker().rank(query, [low, high]) == [low, high]
    assert DeterministicReranker().rank(query, [low, high]) == [high, low]


def test_build_vector_metadata_tracks_hash_and_provider(tmp_path):
    manager = MemoryManager(MemoryConfig(root_dir=tmp_path / ".memora", rag_enabled=True))
    item = manager.save_memory("preference", "用户偏好中文回答。", "用户偏好中文。", name="language")
    text = "first text"
    changed = "changed text"

    metadata = build_vector_metadata(item, build_embedding_provider(manager.config), text)
    changed_metadata = build_vector_metadata(item, build_embedding_provider(manager.config), changed)

    assert metadata["memory_id"] == item.id
    assert metadata["provider"] == "hash"
    assert metadata["model"] == "memora-hash-v1"
    assert metadata["dimension"] == 384
    assert metadata["text_hash"] != changed_metadata["text_hash"]


def test_build_vector_metadata_tracks_bge_provider(fake_flag_embedding):
    item = MemoryItem(id="mem_1", name="invoice", description="desc", type="knowledge", content="发票OCR提取规范")
    embedder = build_embedding_provider(
        MemoryConfig(
            rag_enabled=True,
            embedding_provider="bge",
            embedding_model_path="C:/Download/bge-m3",
        )
    )

    metadata = build_vector_metadata(item, embedder, "text")

    assert metadata["provider"] == "bge"
    assert metadata["model"] == "bge-m3"
    assert metadata["dimension"] == 1024


def test_config_defaults_min_semantic_score_for_rag():
    assert MemoryConfig().min_semantic_score == 0.25


def test_rag_enabled_manager_retrieves_vector_synced_memory(tmp_path):
    manager = MemoryManager(MemoryConfig(root_dir=tmp_path / ".memora", rag_enabled=True))
    manager.init_storage()
    item = manager.save_memory("preference", "alpha retrieval marker", "stored vector memory", name="vector-memory")

    results = manager.retrieve_memory("alpha")
    report = manager.verify_memories()

    assert [result.memory.id for result in results] == [item.id]
    assert results[0].semantic_score > 0
    assert report["vector_ok"] is True
    assert report["vector_missing"] == []
    assert report["vector_orphans"] == []
    assert report["embedding_mismatches"] == []


def test_rag_revalidates_vector_hits_by_user_scope(tmp_path):
    manager = MemoryManager(MemoryConfig(root_dir=tmp_path / ".memora", rag_enabled=True))
    manager.init_storage()
    alice = manager.save_memory("preference", "shared retrieval marker", "Alice memory", name="alice-memory", user_id="alice")
    bob = manager.save_memory("preference", "shared retrieval marker", "Bob memory", name="bob-memory", user_id="bob")

    alice_results = manager.retrieve_memory("shared retrieval marker", user_id="alice")
    bob_results = manager.retrieve_memory("shared retrieval marker", user_id="bob")

    assert {result.memory.id for result in alice_results} == {alice.id}
    assert {result.memory.id for result in bob_results} == {bob.id}


def test_rag_archive_delete_restore_and_rebuild_vector_index(tmp_path):
    manager = MemoryManager(MemoryConfig(root_dir=tmp_path / ".memora", rag_enabled=True))
    manager.init_storage()
    item = manager.save_memory("preference", "restore retrieval marker", "restorable", name="restorable")

    manager.archive_memory("restorable")
    assert manager.verify_memories()["vector_missing"] == []
    assert manager.verify_memories()["vector_orphans"] == []
    assert manager.retrieve_memory("restore retrieval marker") == []

    manager.restore_memory("restorable")
    assert manager.retrieve_memory("restore retrieval marker")[0].memory.id == item.id

    manager.rag_index.delete_memory(item.id)
    assert manager.verify_memories()["vector_missing"] == [item.id]

    manager.rebuild_index()
    assert manager.verify_memories()["vector_ok"] is True

    manager.delete_memory("restorable")
    assert manager.verify_memories()["vector_missing"] == []
    assert manager.verify_memories()["vector_orphans"] == []
    assert manager.retrieve_memory("restore retrieval marker") == []


def test_rag_import_syncs_vector_index(tmp_path):
    source = MemoryManager(MemoryConfig(root_dir=tmp_path / "source", rag_enabled=True))
    target = MemoryManager(MemoryConfig(root_dir=tmp_path / "target", rag_enabled=True))
    export_path = tmp_path / "memories.json"
    source.init_storage()
    target.init_storage()
    source.save_memory("preference", "import retrieval marker", "imported memory", name="imported")
    source.export_memories(export_path)

    report = target.import_memories(export_path)
    verify = target.verify_memories()
    results = target.retrieve_memory("import retrieval marker")

    assert report["imported"] == 1
    assert verify["vector_ok"] is True
    assert verify["vector_missing"] == []
    assert [result.memory.name for result in results] == ["imported"]
    assert results[0].semantic_score > 0


def test_rag_import_rebuild_failure_preserves_import_report(tmp_path):
    source = MemoryManager(MemoryConfig(root_dir=tmp_path / "source", rag_enabled=True))
    target = MemoryManager(MemoryConfig(root_dir=tmp_path / "target", rag_enabled=True))
    export_path = tmp_path / "memories.json"
    source.init_storage()
    target.init_storage()
    source.save_memory("preference", "import failure marker", "imported memory", name="imported")
    source.export_memories(export_path)

    class BrokenRagIndex:
        def rebuild(self):
            raise RuntimeError("rebuild broken")

        def verify(self):
            return {
                "vector_ok": True,
                "vector_missing": [],
                "vector_orphans": [],
                "vector_errors": [],
                "embedding_mismatches": [],
            }

    target.rag_index = BrokenRagIndex()

    report = target.import_memories(export_path)
    verify = target.verify_memories()

    assert report["imported"] == 1
    assert target.get_memory("imported") is not None
    assert verify["vector_ok"] is False
    assert verify["rag_sync_errors"][0]["operation"] == "rebuild"


def test_rag_clean_expired_memory_removes_archived_vectors(tmp_path):
    manager = MemoryManager(MemoryConfig(root_dir=tmp_path / ".memora", rag_enabled=True))
    manager.init_storage()
    expired = manager.save_memory("project", "expired retrieval marker", "expired", name="expired")
    expired.expires_at = datetime.now(timezone.utc) - timedelta(days=1)
    manager.memory_store.update_memory(expired)

    report = manager.clean_expired_memory()
    verify = manager.verify_memories()

    assert report["archived"] == 1
    assert verify["vector_ok"] is True
    assert verify["vector_missing"] == []
    assert verify["vector_orphans"] == []
    assert manager.retrieve_memory("expired retrieval marker") == []


def test_rag_sync_failure_does_not_break_memory_write(tmp_path):
    manager = MemoryManager(MemoryConfig(root_dir=tmp_path / ".memora", rag_enabled=True))

    class BrokenRagIndex:
        def sync_memory(self, item):
            raise RuntimeError("broken")

        def verify(self):
            return {
                "vector_ok": True,
                "vector_missing": [],
                "vector_orphans": [],
                "vector_errors": [],
                "embedding_mismatches": [],
            }

    manager.rag_index = BrokenRagIndex()

    item = manager.save_memory("preference", "content", "description", name="safe-write")
    report = manager.verify_memories()

    assert item.name == "safe-write"
    assert manager.get_memory("safe-write") is not None
    assert report["vector_ok"] is False
    assert report["rag_sync_errors"][0]["operation"] == "sync"
    assert report["rag_sync_errors"][0]["memory_id"] == item.id

    manager.rag_index = build_vector_index = MemoryManager(MemoryConfig(root_dir=tmp_path / "fixed", rag_enabled=True)).rag_index
    build_vector_index.memory_store = manager.memory_store
    manager.rebuild_index()
    repaired = manager.verify_memories()

    assert repaired["vector_ok"] is True
    assert repaired["rag_sync_errors"] == []


def test_rag_filters_low_semantic_only_candidates(tmp_path):
    manager = MemoryManager(MemoryConfig(root_dir=tmp_path / ".memora", rag_enabled=True, min_semantic_score=0.25))
    manager.init_storage()
    weak = manager.save_memory("preference", "unrelated alpha", "unrelated", name="weak-vector")
    allowed = {weak.id: weak}
    merged = manager.rag_retriever._merge(
        allowed,
        [VectorSearchHit(memory_id=weak.id, score=0.10, metadata={})],
        [],
    )

    results = manager.rag_retriever._score(MemoryQuery(query="missing lexical evidence"), merged)

    assert results == []


def test_rag_keeps_low_semantic_hybrid_candidate_when_keyword_matches(tmp_path):
    manager = MemoryManager(MemoryConfig(root_dir=tmp_path / ".memora", rag_enabled=True, min_semantic_score=0.25))
    manager.init_storage()
    hybrid = manager.save_memory("preference", "pytest fixture details", "pytest fixture", name="hybrid")
    allowed = {hybrid.id: hybrid}
    merged = manager.rag_retriever._merge(
        allowed,
        [VectorSearchHit(memory_id=hybrid.id, score=0.10, metadata={})],
        [hybrid],
    )

    results = manager.rag_retriever._score(MemoryQuery(query="pytest fixture"), merged)

    assert [result.memory.id for result in results] == [hybrid.id]
    assert results[0].semantic_score == 0.10
    assert results[0].keyword_score > 0


def test_rag_keyword_match_ranks_above_hash_vector_only_candidate(tmp_path):
    manager = MemoryManager(MemoryConfig(root_dir=tmp_path / ".memora", rag_enabled=True, min_semantic_score=0.25))
    manager.init_storage()
    vector_only = manager.save_memory("preference", "unrelated content", "unrelated", name="vector-only", weight=1)
    keyword = manager.save_memory("preference", "pytest fixture details", "pytest fixture", name="keyword", weight=1)
    allowed = {vector_only.id: vector_only, keyword.id: keyword}
    merged = manager.rag_retriever._merge(
        allowed,
        [
            VectorSearchHit(memory_id=vector_only.id, score=0.80, metadata={}),
            VectorSearchHit(memory_id=keyword.id, score=0.10, metadata={}),
        ],
        [keyword],
    )

    results = manager.rag_retriever._score(MemoryQuery(query="pytest fixture"), merged)
    results.sort(key=lambda result: result.final_score, reverse=True)

    assert results[0].memory.id == keyword.id
    assert results[0].reason in {"exact_name", "exact_description", "phrase_content", "tokens_tags", "partial_content"}


def test_rag_merge_deduplicates_vector_and_keyword_candidates_by_memory_id(tmp_path):
    manager = MemoryManager(MemoryConfig(root_dir=tmp_path / ".memora", rag_enabled=True))
    manager.init_storage()
    shared = manager.save_memory("preference", "shared retrieval marker", "shared retrieval marker", name="shared")
    allowed = {shared.id: shared}
    merged = manager.rag_retriever._merge(
        allowed,
        [
            VectorSearchHit(memory_id=shared.id, score=0.20, metadata={}),
            VectorSearchHit(memory_id=shared.id, score=0.40, metadata={}),
        ],
        [shared, shared],
    )

    results = manager.rag_retriever._score(MemoryQuery(query="shared retrieval marker"), merged)
    ids = [result.memory.id for result in results]

    assert ids == [shared.id]
    assert results[0].semantic_score == 0.40
    assert results[0].keyword_score > 0


class _FakeCandidateStore:
    def __init__(self, candidates=None, error: Exception | None = None):
        self.candidates = list(candidates or [])
        self.error = error
        self.calls = 0

    def search_candidates(self, query):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.candidates


def _keyword_retriever(config: MemoryConfig, candidate_store=None) -> RagRetriever:
    return RagRetriever(
        memory_store=None,
        candidate_store=candidate_store,
        embedder=None,
        vector_store=None,
        retriever=MemoryRetriever(),
        reranker=NoOpReranker(),
        config=config,
    )


def test_rag_keyword_recall_auto_uses_fts_candidates_first():
    matched = MemoryItem(id="mem_match", name="pytest", description="pytest fixture", type="tool", content="pytest fixture details")
    unrelated = MemoryItem(id="mem_other", name="other", description="other", type="tool", content="other")
    retriever = _keyword_retriever(MemoryConfig(keyword_recall="auto"), _FakeCandidateStore([unrelated, matched]))

    recalled = retriever._keyword_recall(MemoryQuery(query="pytest fixture"), {matched.id: matched, unrelated.id: unrelated})

    assert [item.id for item in recalled] == [unrelated.id, matched.id]


def test_rag_keyword_recall_auto_falls_back_to_scored_scan_without_noise():
    matched = MemoryItem(id="mem_match", name="pytest", description="pytest fixture", type="tool", content="pytest fixture details")
    unrelated = MemoryItem(id="mem_other", name="other", description="other", type="tool", content="other")
    retriever = _keyword_retriever(MemoryConfig(keyword_recall="auto", keyword_candidate_limit=10), _FakeCandidateStore([]))

    recalled = retriever._keyword_recall(MemoryQuery(query="pytest fixture"), {unrelated.id: unrelated, matched.id: matched})

    assert [item.id for item in recalled] == [matched.id]


def test_rag_keyword_recall_fts_does_not_scan_on_empty_or_error():
    matched = MemoryItem(id="mem_match", name="pytest", description="pytest fixture", type="tool", content="pytest fixture details")
    allowed = {matched.id: matched}

    empty = _keyword_retriever(MemoryConfig(keyword_recall="fts"), _FakeCandidateStore([]))
    broken = _keyword_retriever(MemoryConfig(keyword_recall="fts"), _FakeCandidateStore(error=RuntimeError("fts broken")))

    assert empty._keyword_recall(MemoryQuery(query="pytest fixture"), allowed) == []
    assert broken._keyword_recall(MemoryQuery(query="pytest fixture"), allowed) == []


def test_rag_keyword_recall_scan_ignores_fts_candidates():
    matched = MemoryItem(id="mem_match", name="pytest", description="pytest fixture", type="tool", content="pytest fixture details")
    unrelated = MemoryItem(id="mem_other", name="other", description="other", type="tool", content="other")
    candidate_store = _FakeCandidateStore([unrelated])
    retriever = _keyword_retriever(MemoryConfig(keyword_recall="scan"), candidate_store)

    recalled = retriever._keyword_recall(MemoryQuery(query="pytest fixture"), {matched.id: matched, unrelated.id: unrelated})

    assert candidate_store.calls == 0
    assert [item.id for item in recalled] == [matched.id]


def test_rag_keyword_recall_off_returns_no_keyword_candidates():
    matched = MemoryItem(id="mem_match", name="pytest", description="pytest fixture", type="tool", content="pytest fixture details")
    retriever = _keyword_retriever(MemoryConfig(keyword_recall="off"), _FakeCandidateStore([matched]))

    recalled = retriever._keyword_recall(MemoryQuery(query="pytest fixture"), {matched.id: matched})

    assert recalled == []
