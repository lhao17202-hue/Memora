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
        memory_type="preference",
        name="language",
        description="用户偏好中文。",
        content="用户偏好使用中文回答。",
    )

    context = runtime.build_context("中文回答")

    assert "用户偏好使用中文回答。" in context


def test_build_pinned_context_returns_preference_and_project_without_query(tmp_path: Path):
    runtime = make_runtime(tmp_path)
    runtime.manager.save_memory("preference", "Prefer concise answers.", "response style", name="response-style")
    runtime.manager.save_memory("project", "Project uses pytest.", "test framework", name="test-framework")
    runtime.manager.save_memory("tool", "Use pytest -q.", "tool lesson", name="tool-lesson")

    results = runtime.retrieve_pinned_context(top_k=10)
    context = runtime.build_pinned_context(top_k=10)

    assert [result.memory.type for result in results] == ["preference", "project"]
    assert "Prefer concise answers." in context
    assert "Project uses pytest." in context
    assert "Use pytest -q." not in context


def test_retrieve_task_context_combines_pinned_and_typed_on_demand_memories(tmp_path: Path):
    runtime = make_runtime(tmp_path)
    preference = runtime.manager.save_memory("preference", "Prefer concise answers.", "response style", name="response-style")
    project = runtime.manager.save_memory("project", "Project uses pytest.", "test framework", name="test-framework")
    tool = runtime.manager.save_memory("tool", "Use pytest -q after changes.", "pytest command", name="pytest-command")
    runtime.manager.save_memory("knowledge", "pytest fixture docs.", "pytest docs", name="pytest-docs")

    results = runtime.retrieve_task_context("pytest command", memory_types=["tool"], top_k=5, pinned_top_k=5)

    assert [result.memory.id for result in results] == [preference.id, project.id, tool.id]
    assert [result.reason for result in results[:2]] == ["pinned_context", "pinned_context"]


def test_retrieve_task_context_can_skip_pinned_memories(tmp_path: Path):
    runtime = make_runtime(tmp_path)
    runtime.manager.save_memory("preference", "Prefer concise answers.", "response style", name="response-style")
    tool = runtime.manager.save_memory("tool", "Use pytest -q after changes.", "pytest command", name="pytest-command")

    results = runtime.retrieve_task_context("pytest command", memory_types=["tool"], include_pinned=False)

    assert [result.memory.id for result in results] == [tool.id]


def test_retrieve_context_and_mark_context_used_updates_access_count(tmp_path: Path):
    runtime = make_runtime(tmp_path)
    item = runtime.manager.save_memory(
        memory_type="preference",
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


def test_remember_summary_saves_episodic_memory(tmp_path: Path):
    runtime = make_runtime(tmp_path)

    item = runtime.remember_summary("session_1", "summary text")

    assert item.type == "episodic"
    assert item.source == "runtime"
    assert item.content == "summary text"


def test_remember_summary_uses_config_default_episodic_weight(tmp_path: Path):
    runtime = MemoryRuntime(config=MemoryConfig(root_dir=tmp_path / ".memora", default_episodic_weight=7))
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
        memory_type="preference",
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
        memory_type="preference",
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
        memory_type="preference",
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
        memory_type="preference",
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


def test_rag_task_context_respects_typed_on_demand_filter(tmp_path: Path):
    runtime = make_rag_runtime(tmp_path)
    tool = runtime.manager.save_memory("tool", "shared retrieval marker", "tool lesson", name="tool-lesson")
    runtime.manager.save_memory("knowledge", "shared retrieval marker", "knowledge note", name="knowledge-note")

    results = runtime.retrieve_task_context("shared retrieval marker", memory_types=["tool"], include_pinned=False)

    assert [result.memory.id for result in results] == [tool.id]
    assert results[0].semantic_score > 0


def test_runtime_remember_extracted_respects_disabled_auto_save(tmp_path: Path):
    runtime = MemoryRuntime(config=MemoryConfig(root_dir=tmp_path / ".memora", allow_auto_save_user_preferences=False))
    runtime.init_storage()

    result = runtime.remember_extracted(
        memory_type="preference",
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
        memory_type="preference",
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
        memory_type="preference",
        name="secret",
        description="secret",
        content="api_key = sk-abcdef123456",
    )

    assert result.action == "rejected"
    assert result.reason == "contains_secret"
    assert result.memory is None
