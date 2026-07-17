from pathlib import Path

import pytest

from memora.config import MemoryConfig
from memora.manager import MemoryManager
from memora.runtime import MemoryRuntime


def make_runtime(tmp_path: Path) -> MemoryRuntime:
    runtime = MemoryRuntime(config=MemoryConfig(root_dir=tmp_path / ".memora"))
    runtime.init_storage()
    return runtime


def test_build_context_returns_formatted_memory(tmp_path: Path):
    runtime = make_runtime(tmp_path)
    runtime.manager.save_memory(
        memory_type="user",
        name="language",
        description="用户偏好中文。",
        content="用户偏好使用中文回答。",
    )

    context = runtime.build_context("中文回答")

    assert "用户偏好使用中文回答。" in context


def test_retrieve_context_and_mark_context_used_updates_access_count(tmp_path: Path):
    runtime = make_runtime(tmp_path)
    item = runtime.manager.save_memory(
        memory_type="user",
        name="language",
        description="用户偏好中文。",
        content="用户偏好使用中文回答。",
    )

    results = runtime.retrieve_context("中文回答")
    runtime.mark_context_used(results)
    reloaded = runtime.manager.memory_store.get_memory(item.id)

    assert reloaded is not None
    assert reloaded.access_count == 1
    assert reloaded.last_accessed_at is not None


def test_remember_message_appends_session_message(tmp_path: Path):
    runtime = make_runtime(tmp_path)

    runtime.remember_message("session_1", "user", "hello")
    messages = runtime.manager.get_messages("default", "session_1")

    assert len(messages) == 1
    assert messages[0].role == "user"
    assert messages[0].content == "hello"


def test_remember_summary_saves_session_summary_memory(tmp_path: Path):
    runtime = make_runtime(tmp_path)

    item = runtime.remember_summary("session_1", "summary text")

    assert item.type == "session_summary"
    assert item.source == "runtime"
    assert item.content == "summary text"


def test_constructor_rejects_manager_and_config_together(tmp_path: Path):
    manager = MemoryManager(MemoryConfig(root_dir=tmp_path / ".memora"))

    with pytest.raises(ValueError, match="manager and config cannot both be provided"):
        MemoryRuntime(manager=manager, config=MemoryConfig(root_dir=tmp_path / "other"))
