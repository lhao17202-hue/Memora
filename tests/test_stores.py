from pathlib import Path

import pytest

from memora.config import MemoryConfig
from memora.errors import MemoryNotFoundError, MemoryValidationError
from memora.schema import MemoryItem, SessionMessage, WorkingMemoryState
from memora.stores import FileMemoryStore, FileSessionStore


def config_for(tmp_path: Path) -> MemoryConfig:
    return MemoryConfig(root_dir=str(tmp_path / ".memora"))


def test_memory_store_init_creates_layout(tmp_path: Path):
    store = FileMemoryStore(config_for(tmp_path))
    store.init_storage()

    root = tmp_path / ".memora"
    assert (root / "MEMORY.md").exists()
    assert (root / "memories").is_dir()
    assert (root / "sessions").is_dir()
    assert (root / "summaries").is_dir()
    assert (root / "archive").is_dir()


def test_save_list_get_and_rebuild_index(tmp_path: Path):
    store = FileMemoryStore(config_for(tmp_path))
    item = MemoryItem(
        id="mem_1",
        name="User Language Preference",
        description="用户偏好中文。",
        type="user",
        content="用户偏好使用中文讨论技术问题。",
        tags=["language"],
    )

    saved = store.save_memory(item)
    listed = store.list_memories()
    found = store.get_memory("mem_1")
    found_by_name = store.get_memory("user-language-preference")
    index = (tmp_path / ".memora" / "MEMORY.md").read_text(encoding="utf-8")

    assert saved.name == "user-language-preference"
    assert len(listed) == 1
    assert found is not None
    assert found.content == "用户偏好使用中文讨论技术问题。"
    assert found_by_name is not None
    assert "user-language-preference.md" in index
    assert "用户偏好中文。" in index


def test_soft_delete_archives_memory_by_default(tmp_path: Path):
    store = FileMemoryStore(config_for(tmp_path))
    store.save_memory(MemoryItem(id="mem_1", name="keep", description="desc", type="project", content="body"))

    store.delete_memory("mem_1", soft_delete=True)

    assert store.list_memories() == []
    archived = store.list_memories(include_archived=True)
    assert len(archived) == 1
    assert archived[0].status == "archived"


def test_session_store_save_load_append(tmp_path: Path):
    store = FileSessionStore(config_for(tmp_path))
    session = {
        "id": "session_1",
        "user_id": "default",
        "working_memory": WorkingMemoryState().__dict__,
        "history": [],
    }
    store.save_session(session)
    store.append_message("default", "session_1", SessionMessage(role="user", content="hello"))

    loaded = store.load_session("default", "session_1")

    assert loaded is not None
    assert loaded["id"] == "session_1"
    assert loaded["history"][0]["role"] == "user"
    assert loaded["history"][0]["content"] == "hello"


def test_memory_store_rejects_invalid_datetime_frontmatter(tmp_path: Path):
    store = FileMemoryStore(MemoryConfig(root_dir=str(tmp_path / ".memora")))
    store.init_storage()
    path = store.memories_dir / "bad.md"
    path.write_text(
        "---\n"
        "name: bad\n"
        "description: bad datetime\n"
        "metadata:\n"
        "  id: mem_bad\n"
        "  type: project\n"
        "  status: active\n"
        "  weight: 5\n"
        "  confidence: 1.0\n"
        "  created_at: not-a-date\n"
        "---\n\n"
        "content\n",
        encoding="utf-8",
    )

    with pytest.raises(MemoryValidationError, match="invalid datetime"):
        store.list_memories()


def test_session_store_rejects_invalid_session_id(tmp_path: Path):
    store = FileSessionStore(MemoryConfig(root_dir=str(tmp_path / ".memora")))

    with pytest.raises(MemoryValidationError, match="session_id"):
        store.append_message("default", "../bad", SessionMessage(role="user", content="hello"))


def test_session_store_rejects_invalid_message_role(tmp_path: Path):
    store = FileSessionStore(MemoryConfig(root_dir=str(tmp_path / ".memora")))

    with pytest.raises(MemoryValidationError, match="session role"):
        store.append_message("default", "session_1", SessionMessage(role="bad", content="hello"))


def test_set_memory_status_archives_and_restores(tmp_path: Path):
    store = FileMemoryStore(config_for(tmp_path))
    store.save_memory(MemoryItem(id="mem_1", name="language", description="desc", type="user", content="body"))

    archived = store.set_memory_status("language", "archived")
    assert archived.status == "archived"
    assert store.list_memories() == []

    restored = store.set_memory_status("mem_1", "active")
    assert restored.status == "active"
    assert len(store.list_memories()) == 1


def test_set_memory_status_missing_raises_not_found(tmp_path: Path):
    store = FileMemoryStore(config_for(tmp_path))

    with pytest.raises(MemoryNotFoundError, match="memory not found"):
        store.set_memory_status("missing", "archived")


def test_hard_delete_memory_removes_file(tmp_path: Path):
    store = FileMemoryStore(config_for(tmp_path))
    store.save_memory(MemoryItem(id="mem_1", name="language", description="desc", type="user", content="body"))

    store.hard_delete_memory("language")

    assert store.get_memory("language") is None
    assert store.list_memories(include_archived=True) == []


def test_hard_delete_memory_missing_raises_not_found(tmp_path: Path):
    store = FileMemoryStore(config_for(tmp_path))

    with pytest.raises(MemoryNotFoundError, match="memory not found"):
        store.hard_delete_memory("missing")
