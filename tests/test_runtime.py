from pathlib import Path

import pytest

from memora.config import MemoryConfig
from memora.manager import MemoryManager
from memora.runtime import MemoryRuntime


def make_runtime(tmp_path: Path) -> MemoryRuntime:
    runtime = MemoryRuntime(config=MemoryConfig(root_dir=tmp_path / ".memora"))
    runtime.init_storage()
    return runtime


def make_sqlite_runtime(tmp_path: Path) -> MemoryRuntime:
    runtime = MemoryRuntime(config=MemoryConfig(root_dir=tmp_path / ".memora", memory_backend="sqlite"))
    runtime.init_storage()
    return runtime


def make_rag_runtime(tmp_path: Path) -> MemoryRuntime:
    runtime = MemoryRuntime(config=MemoryConfig(root_dir=tmp_path / ".memora", memory_backend="sqlite", rag_enabled=True))
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


def test_remember_summary_uses_config_default_summary_weight(tmp_path: Path):
    runtime = MemoryRuntime(config=MemoryConfig(root_dir=tmp_path / ".memora", default_summary_weight=7))
    runtime.init_storage()

    item = runtime.remember_summary("session_1", "summary text")

    assert item.weight == 7


def test_constructor_rejects_manager_and_config_together(tmp_path: Path):
    manager = MemoryManager(MemoryConfig(root_dir=tmp_path / ".memora"))

    with pytest.raises(ValueError, match="manager and config cannot both be provided"):
        MemoryRuntime(manager=manager, config=MemoryConfig(root_dir=tmp_path / "other"))


def test_remember_extracted_creates_memory(tmp_path: Path):
    runtime = make_runtime(tmp_path)

    result = runtime.remember_extracted(
        memory_type="user",
        name="language",
        description="用户偏好中文。",
        content="用户偏好使用中文回答。",
    )

    assert result.action == "created"
    assert result.memory is not None
    assert result.memory.name == "language"
    assert result.memory.source == "runtime_extraction"


def test_remember_extracted_with_session_id_records_session_source(tmp_path: Path):
    runtime = make_runtime(tmp_path)

    result = runtime.remember_extracted(
        memory_type="user",
        name="language",
        description="用户偏好中文。",
        content="用户偏好使用中文回答。",
        session_id="session_1",
        tags=["preference"],
    )

    assert result.action == "created"
    assert result.memory is not None
    assert result.memory.source == "session_extraction"
    assert result.memory.tags == ["preference", "session:session_1"]


def test_sqlite_runtime_save_retrieve_remember_and_mark_used(tmp_path: Path):
    runtime = make_sqlite_runtime(tmp_path)

    created = runtime.remember_extracted(
        memory_type="user",
        name="language",
        description="用户偏好中文。",
        content="用户偏好使用中文回答。",
    )
    context = runtime.build_context("中文回答")
    results = runtime.retrieve_context("中文回答")
    runtime.mark_context_used(results)
    reloaded = runtime.manager.get_memory("language")

    assert created.action == "created"
    assert "用户偏好使用中文回答。" in context
    assert len(results) == 1
    assert reloaded is not None
    assert reloaded.access_count == 1
    assert reloaded.last_accessed_at is not None


def test_rag_runtime_uses_existing_top_level_api(tmp_path: Path):
    runtime = make_rag_runtime(tmp_path)

    created = runtime.remember_extracted(
        memory_type="user",
        name="language",
        description="用户偏好中文。",
        content="用户偏好使用中文回答。",
    )
    results = runtime.retrieve_context("中文回答")
    context = runtime.build_context("中文回答")
    runtime.mark_context_used(results)
    reloaded = runtime.manager.get_memory("language")

    assert created.action == "created"
    assert len(results) == 1
    assert results[0].semantic_score > 0
    assert "用户偏好使用中文回答。" in context
    assert reloaded is not None
    assert reloaded.access_count == 1


def test_runtime_remember_extracted_respects_disabled_auto_save(tmp_path: Path):
    runtime = MemoryRuntime(config=MemoryConfig(root_dir=tmp_path / ".memora", allow_auto_save_user_preferences=False))
    runtime.init_storage()

    result = runtime.remember_extracted(
        memory_type="user",
        name="language",
        description="用户偏好中文。",
        content="用户偏好中文回答。",
    )

    assert result.action == "requires_confirmation"
    assert result.reason == "auto_save_user_preferences_disabled"
    assert result.memory is None
    assert result.candidate is not None
    assert result.candidate.content == "用户偏好中文回答。"
    assert result.candidate.suggested_action == "create"


def test_runtime_confirm_memory_candidate_persists_pending_candidate(tmp_path: Path):
    runtime = MemoryRuntime(config=MemoryConfig(root_dir=tmp_path / ".memora", allow_auto_save_user_preferences=False))
    runtime.init_storage()
    pending = runtime.remember_extracted(
        memory_type="user",
        name="language",
        description="用户偏好中文。",
        content="用户偏好中文回答。",
    )

    confirmed = runtime.confirm_memory_candidate(pending.candidate)
    results = runtime.retrieve_context("中文回答")

    assert confirmed.action == "created"
    assert confirmed.memory is not None
    assert confirmed.memory.name == "language"
    assert len(results) == 1
    assert results[0].memory.id == confirmed.memory.id


def test_remember_extracted_rejects_secret_without_policy_exception(tmp_path: Path):
    runtime = make_runtime(tmp_path)

    result = runtime.remember_extracted(
        memory_type="user",
        name="secret",
        description="secret",
        content="api_key = sk-abcdef123456",
    )

    assert result.action == "rejected"
    assert result.reason == "contains_secret"
    assert result.memory is None
