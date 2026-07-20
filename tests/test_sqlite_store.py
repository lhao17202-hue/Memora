import shutil
import sqlite3
from pathlib import Path

from memora.config import MemoryConfig
from memora.manager import MemoryManager
from memora.schema import MemoryItem, MemoryQuery
from memora.sqlite_store import SQLiteMemoryStore


def sqlite_store(tmp_path: Path) -> SQLiteMemoryStore:
    store = SQLiteMemoryStore(MemoryConfig(root_dir=tmp_path / ".memora", memory_backend="sqlite"))
    store.init_storage()
    return store


def test_sqlite_store_creates_default_database_and_schema(tmp_path: Path):
    store = sqlite_store(tmp_path)

    assert store.db_path == tmp_path / ".memora" / "memora.sqlite3"
    assert store.db_path.exists()
    with sqlite3.connect(store.db_path) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual')")}

    assert "schema_meta" in tables
    assert "memories" in tables
    assert "memory_fts" in tables


def test_sqlite_store_supports_custom_database_path(tmp_path: Path):
    db_path = tmp_path / "custom.sqlite3"
    store = SQLiteMemoryStore(MemoryConfig(root_dir=tmp_path / ".memora", memory_backend="sqlite", sqlite_path=db_path))

    store.init_storage()

    assert db_path.exists()


def test_sqlite_search_candidates_match_fields_and_filters(tmp_path: Path):
    store = sqlite_store(tmp_path)
    language = store.save_memory(
        MemoryItem(
            id="mem_language",
            name="language",
            description="用户偏好中文。",
            type="user",
            content="User wants pytest examples.",
            tags=["preference"],
        )
    )
    other = MemoryItem(
        id="mem_other",
        name="other",
        description="other desc",
        type="project",
        content="pytest project docs",
        tags=["docs"],
    )
    store.save_memory(other)

    by_content = store.search_candidates(MemoryQuery(query="pytest", top_k=8))
    by_tag = store.search_candidates(MemoryQuery(query="preference", tags=["preference"], top_k=8))
    by_type = store.search_candidates(MemoryQuery(query="pytest", memory_types=["user"], top_k=8))

    assert {item.id for item in by_content} == {language.id, other.id}
    assert [item.id for item in by_tag] == [language.id]
    assert [item.id for item in by_type] == [language.id]


def test_sqlite_rebuild_index_repairs_missing_fts_rows(tmp_path: Path):
    store = sqlite_store(tmp_path)
    store.save_memory(MemoryItem(id="mem_language", name="language", description="desc", type="user", content="pytest content"))

    with sqlite3.connect(store.db_path) as connection:
        connection.execute("DELETE FROM memory_fts")

    broken = store.verify()
    assert broken["checked"] == 1
    assert broken["index_ok"] is False

    store.rebuild_index()

    repaired = store.verify()
    assert repaired["index_ok"] is True


def test_sqlite_store_keeps_same_name_in_different_user_scopes(tmp_path: Path):
    store = sqlite_store(tmp_path)

    store.save_memory(
        MemoryItem(
            id="mem_alice",
            name="language",
            description="Alice language.",
            type="user",
            content="Alice prefers Chinese.",
            user_id="alice",
        )
    )
    store.save_memory(
        MemoryItem(
            id="mem_bob",
            name="language",
            description="Bob language.",
            type="user",
            content="Bob prefers English.",
            user_id="bob",
        )
    )

    items = store.list_memories(include_archived=True)

    assert {item.id for item in items} == {"mem_alice", "mem_bob"}


def test_sqlite_store_replaces_same_name_only_within_same_scope(tmp_path: Path):
    store = sqlite_store(tmp_path)

    store.save_memory(
        MemoryItem(
            id="mem_original",
            name="language",
            description="Original.",
            type="user",
            content="Original content.",
            user_id="alice",
        )
    )
    store.save_memory(
        MemoryItem(
            id="mem_replacement",
            name="language",
            description="Replacement.",
            type="user",
            content="Replacement content.",
            user_id="alice",
        )
    )

    items = store.list_memories(include_archived=True)

    assert len(items) == 1
    assert items[0].id == "mem_replacement"
    assert items[0].content == "Replacement content."


def test_sqlite_store_closes_connections_after_operations(tmp_path: Path):
    store = sqlite_store(tmp_path)
    store.save_memory(MemoryItem(id="mem_language", name="language", description="desc", type="user", content="pytest content"))
    store.list_memories()
    store.get_memory("language")
    store.search_candidates(MemoryQuery(query="pytest", top_k=8))
    store.verify()
    store.rebuild_index()

    shutil.rmtree(tmp_path / ".memora")

    assert not (tmp_path / ".memora").exists()


def test_manager_sqlite_backend_merges_fts_with_chinese_fallback_for_mixed_query(tmp_path: Path):
    manager = MemoryManager(MemoryConfig(root_dir=tmp_path / ".memora", memory_backend="sqlite"))
    manager.init_storage()
    manager.save_memory("user", "用户偏好中文回答。", "用户偏好中文。", name="language")
    manager.save_memory("project", "pytest docs", "pytest docs", name="pytest-docs")

    results = manager.retrieve_memory("中文 pytest", include_archived=False)

    assert {result.memory.name for result in results} == {"language", "pytest-docs"}


def test_manager_sqlite_backend_preserves_chinese_fallback(tmp_path: Path):
    manager = MemoryManager(MemoryConfig(root_dir=tmp_path / ".memora", memory_backend="sqlite"))
    manager.init_storage()
    manager.save_memory("user", "用户偏好中文回答。", "用户偏好中文。", name="language")

    for query in ["中文", "偏好", "回答"]:
        results = manager.retrieve_memory(query)
        assert len(results) == 1
        assert results[0].memory.name == "language"
