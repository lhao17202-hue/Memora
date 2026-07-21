from datetime import datetime, timedelta, timezone

import pytest

from memora.config import MemoryConfig
from memora.errors import MemoryValidationError
from memora.manager import MemoryManager
from memora.rag import build_embedding_provider, build_reranker, build_vector_metadata, build_vector_store
from memora.reranker import DeterministicReranker, NoOpReranker
from memora.schema import MemoryItem, MemoryQuery, MemorySearchResult


def test_rag_factories_support_only_v1_values(tmp_path):
    config = MemoryConfig(root_dir=tmp_path / ".memora", rag_enabled=True)

    assert build_embedding_provider(config).name == "hash"
    assert build_vector_store(config).name == "sqlite"
    assert build_reranker(config).name == "deterministic"

    with pytest.raises(MemoryValidationError, match="reserved but not implemented"):
        build_embedding_provider(MemoryConfig(rag_enabled=True, embedding_provider="openai"))
    with pytest.raises(MemoryValidationError, match="reserved but not implemented"):
        build_vector_store(MemoryConfig(rag_enabled=True, vector_store="qdrant"))
    with pytest.raises(MemoryValidationError, match="reserved but not implemented"):
        build_reranker(MemoryConfig(rag_enabled=True, reranker="llm"))
    with pytest.raises(MemoryValidationError, match="embedding_provider"):
        build_embedding_provider(MemoryConfig(rag_enabled=True, embedding_provider="unknown"))


def test_rerankers_are_deterministic():
    item = MemoryItem(id="mem_1", name="one", description="desc", type="user", content="content")
    low = MemorySearchResult(item, 0.1, 0.1, 0.1, 0.0, 0.1)
    high = MemorySearchResult(item, 0.9, 0.1, 0.1, 0.0, 0.9)
    query = MemoryQuery(query="content")

    assert NoOpReranker().rank(query, [low, high]) == [low, high]
    assert DeterministicReranker().rank(query, [low, high]) == [high, low]


def test_build_vector_metadata_tracks_hash_and_provider(tmp_path):
    manager = MemoryManager(MemoryConfig(root_dir=tmp_path / ".memora", rag_enabled=True))
    item = manager.save_memory("user", "用户偏好中文回答。", "用户偏好中文。", name="language")
    text = "first text"
    changed = "changed text"

    metadata = build_vector_metadata(item, build_embedding_provider(manager.config), text)
    changed_metadata = build_vector_metadata(item, build_embedding_provider(manager.config), changed)

    assert metadata["memory_id"] == item.id
    assert metadata["provider"] == "hash"
    assert metadata["model"] == "memora-hash-v1"
    assert metadata["dimension"] == 384
    assert metadata["text_hash"] != changed_metadata["text_hash"]


def test_rag_enabled_manager_retrieves_vector_synced_memory(tmp_path):
    manager = MemoryManager(MemoryConfig(root_dir=tmp_path / ".memora", rag_enabled=True))
    manager.init_storage()
    item = manager.save_memory("user", "alpha retrieval marker", "stored vector memory", name="vector-memory")

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
    alice = manager.save_memory("user", "shared retrieval marker", "Alice memory", name="alice-memory", user_id="alice")
    bob = manager.save_memory("user", "shared retrieval marker", "Bob memory", name="bob-memory", user_id="bob")

    alice_results = manager.retrieve_memory("shared retrieval marker", user_id="alice")
    bob_results = manager.retrieve_memory("shared retrieval marker", user_id="bob")

    assert {result.memory.id for result in alice_results} == {alice.id}
    assert {result.memory.id for result in bob_results} == {bob.id}


def test_rag_archive_delete_restore_and_rebuild_vector_index(tmp_path):
    manager = MemoryManager(MemoryConfig(root_dir=tmp_path / ".memora", rag_enabled=True))
    manager.init_storage()
    item = manager.save_memory("user", "restore retrieval marker", "restorable", name="restorable")

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
    source.save_memory("user", "import retrieval marker", "imported memory", name="imported")
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
    source.save_memory("user", "import failure marker", "imported memory", name="imported")
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

    item = manager.save_memory("user", "content", "description", name="safe-write")
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
