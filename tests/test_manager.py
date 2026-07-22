from datetime import datetime, timedelta, timezone
from pathlib import Path

from memora.config import MemoryConfig
from memora.errors import MemoryNotFoundError, MemoryPolicyError, MemoryValidationError
from memora.manager import MemoryManager
from memora.schema import MemoryCandidate, SessionMessage


def manager_for(tmp_path: Path) -> MemoryManager:
    return MemoryManager(MemoryConfig(root_dir=str(tmp_path / ".memora")))


def test_memory_search_result_rag_score_defaults():
    from memora.schema import MemoryItem, MemorySearchResult

    item = MemoryItem(id="mem_1", name="language", description="desc", type="user", content="content")
    result = MemorySearchResult(item, 0.5, 0.5, 0.5, 0.0, 0.5)

    assert result.semantic_score == 0.0
    assert result.keyword_score == 0.0
    assert result.rerank_score is None


def test_save_memory_uses_config_default_weight_when_omitted(tmp_path: Path):
    manager = MemoryManager(
        MemoryConfig(
            root_dir=tmp_path / ".memora",
            default_user_weight=10,
            default_feedback_weight=8,
            default_project_weight=6,
            default_summary_weight=4,
            default_tool_experience_weight=3,
        )
    )

    user = manager.save_memory("user", "prefers Chinese", "language", name="user-language")
    feedback = manager.save_memory("feedback", "likes concise answers", "style", name="feedback-style")
    project = manager.save_memory("project", "uses pytest", "tests", name="project-tests")
    summary = manager.save_memory("session_summary", "session summary", "summary", name="session-summary")
    tool = manager.save_memory("tool_experience", "pytest worked", "tool", name="tool-pytest")

    assert user.weight == 10
    assert feedback.weight == 8
    assert project.weight == 6
    assert summary.weight == 4
    assert tool.weight == 3


def test_explicit_weight_is_preserved_over_config_default(tmp_path: Path):
    manager = MemoryManager(MemoryConfig(root_dir=tmp_path / ".memora", default_user_weight=10))

    item = manager.save_memory("user", "prefers Chinese", "language", name="language", weight=5)

    assert item.weight == 5


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
    except MemoryPolicyError as exc:
        assert "contains_secret" in str(exc)
    else:
        raise AssertionError("expected MemoryPolicyError")


def test_save_memory_rejects_invalid_memory_type(tmp_path: Path):
    manager = manager_for(tmp_path)

    try:
        manager.save_memory("invalid", "content", "description", name="bad")
    except MemoryValidationError as exc:
        assert "memory type" in str(exc)
    else:
        raise AssertionError("expected MemoryValidationError")


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


def test_update_memory_changes_selected_fields(tmp_path: Path):
    manager = manager_for(tmp_path)
    manager.save_memory("user", "old content", "old desc", name="language", tags=["old"], weight=5)

    updated = manager.update_memory(
        "language",
        description="new desc",
        content="new content",
        tags=["new"],
        weight=8,
        confidence=0.7,
    )

    assert updated.description == "new desc"
    assert updated.content == "new content"
    assert updated.tags == ["new"]
    assert updated.weight == 8
    assert updated.confidence == 0.7
    assert updated.updated_at is not None


def test_update_memory_missing_raises_not_found(tmp_path: Path):
    manager = manager_for(tmp_path)

    try:
        manager.update_memory("missing", content="new")
    except MemoryNotFoundError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("expected MemoryNotFoundError")


def test_archive_and_restore_memory_control_retrieval(tmp_path: Path):
    manager = manager_for(tmp_path)
    manager.save_memory("user", "用户偏好中文。", "用户偏好中文。", name="language")

    archived = manager.archive_memory("language")
    assert archived.status == "archived"
    assert manager.retrieve_memory("中文") == []

    restored = manager.restore_memory("language")
    assert restored.status == "active"
    assert len(manager.retrieve_memory("中文")) == 1


def test_delete_memory_marks_deleted_by_default(tmp_path: Path):
    manager = manager_for(tmp_path)
    manager.save_memory("user", "用户偏好中文。", "用户偏好中文。", name="language")

    manager.delete_memory("language")
    deleted = manager.memory_store.get_memory("language")

    assert deleted is not None
    assert deleted.status == "deleted"
    assert manager.retrieve_memory("中文") == []


def test_delete_memory_hard_removes_file(tmp_path: Path):
    manager = manager_for(tmp_path)
    manager.save_memory("user", "用户偏好中文。", "用户偏好中文。", name="language")

    manager.delete_memory("language", hard=True)

    assert manager.memory_store.get_memory("language") is None


def test_evaluate_memory_candidate_returns_decision_without_writing(tmp_path: Path):
    manager = manager_for(tmp_path)
    manager.init_storage()
    candidate = MemoryCandidate(
        action="create",
        type="user",
        name="language",
        description="用户偏好中文。",
        content="用户偏好使用中文回答。",
    )

    result = manager.evaluate_memory_candidate(candidate)

    assert result.action == "created"
    assert result.memory is None
    assert result.candidate is not None
    assert result.candidate.name == "language"
    assert result.reason == "accepted"
    assert manager.memory_store.list_memories() == []


def test_remember_candidate_creates_memory(tmp_path: Path):
    manager = manager_for(tmp_path)
    manager.init_storage()
    candidate = MemoryCandidate(
        action="create",
        type="user",
        name="language",
        description="用户偏好中文。",
        content="用户偏好使用中文回答。",
    )

    result = manager.remember_candidate(candidate)

    assert result.action == "created"
    assert result.memory is not None
    assert result.memory.name == "language"
    assert result.reason == "accepted"
    assert manager.memory_store.get_memory("language") is not None


def test_remember_candidate_updates_duplicate_memory(tmp_path: Path):
    manager = manager_for(tmp_path)
    manager.init_storage()
    original = manager.save_memory("user", "old content", "old desc", name="language")
    candidate = MemoryCandidate(
        action="create",
        type="user",
        name="language",
        description="new desc",
        content="new content",
        tags=["preference"],
        weight=8,
        confidence=0.7,
        source="session_extraction",
    )

    result = manager.remember_candidate(candidate)

    assert result.action == "updated"
    assert result.memory is not None
    assert result.memory.id == original.id
    assert result.memory.description == "new desc"
    assert result.memory.content == "new content"
    assert result.memory.tags == ["preference"]
    assert result.memory.weight == 8
    assert result.memory.confidence == 0.7
    assert result.memory.source == "session_extraction"
    assert result.reason == "duplicate_or_same_key"
    assert result.target_memory_id == original.id


def test_same_memory_name_is_isolated_across_users(tmp_path: Path):
    manager = manager_for(tmp_path)
    manager.init_storage()

    alice = manager.save_memory("user", "Alice prefers Chinese.", "Alice language.", name="language", user_id="alice")
    bob = manager.save_memory("user", "Bob prefers English.", "Bob language.", name="language", user_id="bob")

    alice_results = manager.retrieve_memory("Chinese", user_id="alice")
    bob_results = manager.retrieve_memory("English", user_id="bob")

    assert alice.id != bob.id
    assert {item.id for item in manager.list_memories(include_archived=True)} == {alice.id, bob.id}
    assert [result.memory.id for result in alice_results] == [alice.id]
    assert [result.memory.id for result in bob_results] == [bob.id]


def test_same_memory_name_is_isolated_across_explicit_projects(tmp_path: Path):
    manager = manager_for(tmp_path)
    manager.init_storage()

    project_a = manager.save_memory(
        "project",
        "Project A uses pytest.",
        "Project A test framework.",
        name="test-framework",
        user_id="default",
        project_id="project-a",
    )
    project_b = manager.save_memory(
        "project",
        "Project B uses unittest.",
        "Project B test framework.",
        name="test-framework",
        user_id="default",
        project_id="project-b",
    )

    project_a_results = manager.retrieve_memory("pytest", project_id="project-a")
    project_b_results = manager.retrieve_memory("unittest", project_id="project-b")

    assert project_a.id != project_b.id
    assert {item.id for item in manager.list_memories(include_archived=True)} == {project_a.id, project_b.id}
    assert [result.memory.id for result in project_a_results] == [project_a.id]
    assert [result.memory.id for result in project_b_results] == [project_b.id]


def test_remember_candidate_rejects_secret_without_raising_policy_error(tmp_path: Path):
    manager = manager_for(tmp_path)
    manager.init_storage()
    candidate = MemoryCandidate(
        action="create",
        type="user",
        name="secret",
        description="secret",
        content="api_key = sk-abcdef123456",
    )

    result = manager.remember_candidate(candidate)

    assert result.action == "rejected"
    assert result.memory is None
    assert result.reason == "contains_secret"
    assert manager.memory_store.get_memory("secret") is None


def test_remember_candidate_reports_conflict_without_writing(tmp_path: Path):
    manager = manager_for(tmp_path)
    manager.init_storage()
    existing = manager.save_memory("user", "用户偏好英文回答。", "用户偏好英文。", name="language-en")
    candidate = MemoryCandidate(
        action="create",
        type="user",
        name="language-zh",
        description="用户偏好中文。",
        content="用户偏好中文回答。",
    )

    result = manager.remember_candidate(candidate)

    assert result.action == "requires_confirmation"
    assert result.memory is None
    assert result.reason == "conflict_requires_confirmation"
    assert result.target_memory_id == existing.id
    assert manager.memory_store.get_memory("language-zh") is None
