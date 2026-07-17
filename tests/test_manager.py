from datetime import datetime, timedelta, timezone
from pathlib import Path

from memora.config import MemoryConfig
from memora.manager import MemoryManager
from memora.schema import SessionMessage


def manager_for(tmp_path: Path) -> MemoryManager:
    return MemoryManager(MemoryConfig(root_dir=str(tmp_path / ".memora")))


def test_save_retrieve_and_format_memory(tmp_path: Path):
    manager = manager_for(tmp_path)
    manager.init_storage()
    manager.save_memory(
        memory_type="user",
        content="用户偏好使用中文回答。",
        description="用户偏好中文。",
        name="user-language-preference",
    )

    results = manager.retrieve_memory(query="中文回答")
    formatted = manager.format_memories_for_prompt(results=results)

    assert len(results) == 1
    assert results[0].memory.name == "user-language-preference"
    assert "用户偏好使用中文回答。" in formatted


def test_policy_rejects_unsafe_save(tmp_path: Path):
    manager = manager_for(tmp_path)
    manager.init_storage()

    try:
        manager.save_memory(
            memory_type="user",
            content="api_key = sk-abcdef123456",
            description="secret",
            name="secret",
        )
    except ValueError as exc:
        assert "contains_secret" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_session_append_and_get_messages(tmp_path: Path):
    manager = manager_for(tmp_path)
    manager.append_message("default", "session_1", SessionMessage(role="user", content="hello"))

    messages = manager.get_messages("default", "session_1")

    assert len(messages) == 1
    assert messages[0].content == "hello"


def test_mark_memories_used_updates_access_stats(tmp_path: Path):
    manager = manager_for(tmp_path)
    manager.save_memory("user", "用户偏好中文。", "用户偏好中文。", name="language")
    results = manager.retrieve_memory(query="中文")

    manager.mark_memories_used(results)
    updated = manager.memory_store.get_memory(results[0].memory.id)

    assert updated is not None
    assert updated.access_count == 1
    assert updated.last_accessed_at is not None


def test_clean_expired_memory_archives_expired(tmp_path: Path):
    manager = manager_for(tmp_path)
    expired = manager.save_memory("project", "old", "old", name="old")
    expired.expires_at = datetime.now(timezone.utc) - timedelta(days=1)
    manager.memory_store.update_memory(expired)

    report = manager.clean_expired_memory()

    assert report["archived"] == 1
    assert manager.retrieve_memory(query="old") == []
