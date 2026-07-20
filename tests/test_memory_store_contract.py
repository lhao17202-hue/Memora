from pathlib import Path

import pytest

from memora.config import MemoryConfig
from memora.errors import MemoryNotFoundError
from memora.schema import MemoryItem
from memora.sqlite_store import SQLiteMemoryStore
from memora.stores import FileMemoryStore


def file_store(tmp_path: Path):
    return FileMemoryStore(MemoryConfig(root_dir=tmp_path / ".memora-file"))


def sqlite_store(tmp_path: Path):
    return SQLiteMemoryStore(MemoryConfig(root_dir=tmp_path / ".memora-sqlite", memory_backend="sqlite"))


@pytest.fixture(params=[file_store, sqlite_store], ids=["file", "sqlite"])
def memory_store(request, tmp_path: Path):
    store = request.param(tmp_path)
    store.init_storage()
    return store


def memory_item(name: str = "Language Preference") -> MemoryItem:
    return MemoryItem(
        id=f"mem_{name.lower().replace(' ', '_')}",
        name=name,
        description="用户偏好中文。",
        type="user",
        content="用户偏好使用中文回答。",
        tags=["language", "preference"],
        source="test",
        weight=8,
        confidence=0.7,
        supersedes=["old_memory"],
        related=["project_language"],
    )


def test_memory_store_save_list_get_and_round_trip(memory_store):
    saved = memory_store.save_memory(memory_item())

    listed = memory_store.list_memories()
    found_by_id = memory_store.get_memory(saved.id)
    found_by_name = memory_store.get_memory("language-preference")

    assert saved.name == "language-preference"
    assert len(listed) == 1
    assert found_by_id is not None
    assert found_by_id.content == "用户偏好使用中文回答。"
    assert found_by_id.tags == ["language", "preference"]
    assert found_by_id.supersedes == ["old_memory"]
    assert found_by_id.related == ["project_language"]
    assert found_by_name is not None
    assert found_by_name.id == saved.id


def test_memory_store_status_and_delete_contract(memory_store):
    saved = memory_store.save_memory(memory_item())

    archived = memory_store.set_memory_status(saved.id, "archived")
    assert archived.status == "archived"
    assert memory_store.list_memories() == []
    assert len(memory_store.list_memories(include_archived=True)) == 1

    restored = memory_store.set_memory_status("language-preference", "active")
    assert restored.status == "active"
    assert len(memory_store.list_memories()) == 1

    memory_store.delete_memory(saved.id, soft_delete=True)
    soft_deleted = memory_store.get_memory(saved.id)
    assert soft_deleted is not None
    assert soft_deleted.status == "archived"

    memory_store.hard_delete_memory(saved.id)
    assert memory_store.get_memory(saved.id) is None
    assert memory_store.list_memories(include_archived=True) == []


def test_memory_store_missing_status_and_hard_delete_raise(memory_store):
    with pytest.raises(MemoryNotFoundError, match="memory not found"):
        memory_store.set_memory_status("missing", "archived")

    with pytest.raises(MemoryNotFoundError, match="memory not found"):
        memory_store.hard_delete_memory("missing")
