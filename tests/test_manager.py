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

    item = MemoryItem(id="mem_1", name="language", description="desc", type="preference", content="content")
    result = MemorySearchResult(item, 0.5, 0.5, 0.5, 0.0, 0.5)

    assert result.semantic_score == 0.0
    assert result.keyword_score == 0.0
    assert result.rerank_score is None


def test_save_memory_uses_config_default_weight_when_omitted(tmp_path: Path):
    manager = MemoryManager(
        MemoryConfig(
            root_dir=tmp_path / ".memora",
            default_preference_weight=10,
            default_reflective_weight=8,
            default_project_weight=6,
            default_episodic_weight=4,
            default_tool_weight=3,
        )
    )

    user = manager.save_memory("preference", "prefers Chinese", "language", name="user-language")
    feedback = manager.save_memory("reflective", "likes concise answers", "style", name="feedback-style")
    project = manager.save_memory("project", "uses pytest", "tests", name="project-tests")
    summary = manager.save_memory("episodic", "session summary", "summary", name="session-summary")
    tool = manager.save_memory("tool", "pytest worked", "tool", name="tool-pytest")

    assert user.weight == 10
    assert feedback.weight == 8
    assert project.weight == 6
    assert summary.weight == 4
    assert tool.weight == 3


def test_explicit_weight_is_preserved_over_config_default(tmp_path: Path):
    manager = MemoryManager(MemoryConfig(root_dir=tmp_path / ".memora", default_preference_weight=10))

    item = manager.save_memory("preference", "prefers Chinese", "language", name="language", weight=5)

    assert item.weight == 5


def test_save_retrieve_and_format_memory(tmp_path: Path):
    manager = manager_for(tmp_path)
    manager.init_storage()
    manager.save_memory(
        memory_type="preference",
        content="用户偏好使用中文回答。",
        description="用户偏好中文。",
        name="user-language-preference",
    )

    results = manager.retrieve_memory(query="中文回答")
    formatted = manager.format_memories_for_prompt(results=results)

    assert len(results) == 1
    assert results[0].memory.name == "user-language-preference"
    assert "用户偏好使用中文回答。" in formatted


def test_retrieve_pinned_memories_returns_preference_and_project_without_query(tmp_path: Path):
    manager = manager_for(tmp_path)
    manager.init_storage()
    manager.save_memory("tool", "Use pytest -q for verification.", "tool lesson", name="tool-lesson", weight=10)
    project = manager.save_memory("project", "Project uses Python.", "project stack", name="project-stack", weight=8)
    preference = manager.save_memory("preference", "Prefer concise answers.", "response style", name="response-style", weight=9)

    results = manager.retrieve_pinned_memories(top_k=10)

    assert [result.memory.id for result in results] == [preference.id, project.id]
    assert {result.reason for result in results} == {"pinned_context"}


def test_retrieve_pinned_memories_respects_scope_and_status(tmp_path: Path):
    manager = manager_for(tmp_path)
    manager.init_storage()
    alice = manager.save_memory("preference", "Alice prefers Chinese.", "Alice language", name="language", user_id="alice")
    manager.save_memory("preference", "Bob prefers English.", "Bob language", name="language", user_id="bob")
    archived = manager.save_memory("project", "Archived project fact.", "old project", name="old-project", user_id="alice")
    manager.archive_memory(archived.id)

    active_results = manager.retrieve_pinned_memories(user_id="alice", top_k=10)
    archived_results = manager.retrieve_pinned_memories(user_id="alice", include_archived=True, top_k=10)

    assert [result.memory.id for result in active_results] == [alice.id]
    assert {result.memory.id for result in archived_results} == {alice.id, archived.id}


def test_retrieve_pinned_memories_works_with_sqlite_backend(tmp_path: Path):
    manager = MemoryManager(MemoryConfig(root_dir=tmp_path / ".memora", memory_backend="sqlite"))
    manager.init_storage()
    preference = manager.save_memory("preference", "Prefer Chinese.", "language", name="language")
    project = manager.save_memory("project", "Project uses pytest.", "tests", name="tests")
    manager.save_memory("knowledge", "Python docs note.", "python docs", name="python-docs")

    results = manager.retrieve_pinned_memories(top_k=10)

    assert {result.memory.id for result in results} == {preference.id, project.id}


def test_policy_rejects_unsafe_save(tmp_path: Path):
    manager = manager_for(tmp_path)
    manager.init_storage()

    try:
        manager.save_memory(
            memory_type="preference",
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
    manager.save_memory("preference", "用户偏好中文。", "用户偏好中文。", name="language")
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
    manager.save_memory("preference", "old content", "old desc", name="language", tags=["old"], weight=5)

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
    manager.save_memory("preference", "用户偏好中文。", "用户偏好中文。", name="language")

    archived = manager.archive_memory("language")
    assert archived.status == "archived"
    assert manager.retrieve_memory("中文") == []

    restored = manager.restore_memory("language")
    assert restored.status == "active"
    assert len(manager.retrieve_memory("中文")) == 1


def test_delete_memory_marks_deleted_by_default(tmp_path: Path):
    manager = manager_for(tmp_path)
    manager.save_memory("preference", "用户偏好中文。", "用户偏好中文。", name="language")

    manager.delete_memory("language")
    deleted = manager.memory_store.get_memory("language")

    assert deleted is not None
    assert deleted.status == "deleted"
    assert manager.retrieve_memory("中文") == []


def test_delete_memory_hard_removes_file(tmp_path: Path):
    manager = manager_for(tmp_path)
    manager.save_memory("preference", "用户偏好中文。", "用户偏好中文。", name="language")

    manager.delete_memory("language", hard=True)

    assert manager.memory_store.get_memory("language") is None


def test_evaluate_memory_candidate_returns_decision_without_writing(tmp_path: Path):
    manager = manager_for(tmp_path)
    manager.init_storage()
    candidate = MemoryCandidate(
        action="create",
        type="preference",
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
        type="preference",
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
    original = manager.save_memory("preference", "old content", "old desc", name="language")
    candidate = MemoryCandidate(
        action="create",
        type="preference",
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


def test_disabled_auto_save_user_returns_confirmation_without_writing(tmp_path: Path):
    manager = MemoryManager(MemoryConfig(root_dir=tmp_path / ".memora", allow_auto_save_user_preferences=False))
    manager.init_storage()
    candidate = MemoryCandidate(
        action="create",
        type="preference",
        name="language",
        description="用户偏好中文。",
        content="用户偏好中文回答。",
        source="runtime_extraction",
    )

    result = manager.remember_candidate(candidate)

    assert result.action == "requires_confirmation"
    assert result.reason == "auto_save_user_preferences_disabled"
    assert result.memory is None
    assert manager.memory_store.list_memories() == []


def test_manual_save_ignores_auto_save_disabled(tmp_path: Path):
    manager = MemoryManager(
        MemoryConfig(
            root_dir=tmp_path / ".memora",
            allow_auto_save_user_preferences=False,
            allow_auto_save_project_facts=False,
        )
    )

    item = manager.save_memory("preference", "用户偏好中文回答。", "用户偏好中文。", name="language", source="manual")

    assert item.name == "language"
    assert item.source == "manual"


def test_same_memory_name_is_isolated_across_users(tmp_path: Path):
    manager = manager_for(tmp_path)
    manager.init_storage()

    alice = manager.save_memory("preference", "Alice prefers Chinese.", "Alice language.", name="language", user_id="alice")
    bob = manager.save_memory("preference", "Bob prefers English.", "Bob language.", name="language", user_id="bob")

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


def test_conflict_confirmation_disabled_creates_new_memory(tmp_path: Path):
    manager = MemoryManager(MemoryConfig(root_dir=tmp_path / ".memora", require_confirmation_for_conflicts=False))
    manager.init_storage()
    existing = manager.save_memory("preference", "用户偏好英文回答。", "用户偏好英文。", name="language-en")
    candidate = MemoryCandidate(
        action="create",
        type="preference",
        name="language-zh",
        description="用户偏好中文。",
        content="用户偏好中文回答。",
        source="runtime_extraction",
    )

    result = manager.remember_candidate(candidate)

    assert result.action == "created"
    assert result.memory is not None
    assert result.memory.id != existing.id
    assert result.reason == "accepted"


def test_remember_candidate_rejects_secret_without_raising_policy_error(tmp_path: Path):
    manager = manager_for(tmp_path)
    manager.init_storage()
    candidate = MemoryCandidate(
        action="create",
        type="preference",
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
    existing = manager.save_memory("preference", "用户偏好英文回答。", "用户偏好英文。", name="language-en")
    candidate = MemoryCandidate(
        action="create",
        type="preference",
        name="language-zh",
        description="用户偏好中文。",
        content="用户偏好中文回答。",
    )

    result = manager.remember_candidate(candidate)

    assert result.action == "requires_confirmation"
    assert result.memory is None
    assert result.reason == "conflict_requires_confirmation"
    assert result.target_memory_id == existing.id
    assert result.candidate is not None
    assert result.candidate.suggested_action == "update"
    assert manager.memory_store.get_memory("language-zh") is None


def test_evaluate_memory_candidate_returns_enriched_confirmation(tmp_path: Path):
    manager = MemoryManager(
        MemoryConfig(root_dir=tmp_path / ".memora", allow_auto_save_user_preferences=False, default_preference_weight=10)
    )
    manager.init_storage()
    candidate = MemoryCandidate(
        action="create",
        type="preference",
        name="language",
        description="用户偏好中文。",
        content="用户偏好中文回答。",
        source="runtime_extraction",
    )

    result = manager.evaluate_memory_candidate(candidate)

    assert result.action == "requires_confirmation"
    assert result.memory is None
    assert result.reason == "auto_save_user_preferences_disabled"
    assert result.target_memory_id is None
    assert result.candidate is not None
    assert result.candidate.content == "用户偏好中文回答。"
    assert result.candidate.weight == 10
    assert result.candidate.suggested_action == "create"


def test_disabled_auto_save_duplicate_requires_confirmation_with_target(tmp_path: Path):
    manager = MemoryManager(MemoryConfig(root_dir=tmp_path / ".memora", allow_auto_save_user_preferences=False))
    manager.init_storage()
    existing = manager.save_memory("preference", "old content", "old desc", name="language", source="manual")
    candidate = MemoryCandidate(
        action="create",
        type="preference",
        name="language",
        description="new desc",
        content="new content",
        source="runtime_extraction",
    )

    result = manager.remember_candidate(candidate)

    assert result.action == "requires_confirmation"
    assert result.memory is None
    assert result.target_memory_id == existing.id
    assert result.candidate is not None
    assert result.candidate.target_memory_id == existing.id
    assert result.candidate.suggested_action == "update"
    assert result.candidate.content == "new content"
    assert manager.memory_store.get_memory(existing.id).content == "old content"


def test_confirm_memory_candidate_creates_after_confirmation(tmp_path: Path):
    manager = MemoryManager(MemoryConfig(root_dir=tmp_path / ".memora", allow_auto_save_user_preferences=False))
    manager.init_storage()
    pending = manager.remember_candidate(
        MemoryCandidate(
            action="create",
            type="preference",
            name="language",
            description="用户偏好中文。",
            content="用户偏好中文回答。",
            source="runtime_extraction",
        )
    )

    confirmed = manager.confirm_memory_candidate(pending.candidate)

    assert confirmed.action == "created"
    assert confirmed.reason == "confirmed:auto_save_user_preferences_disabled"
    assert confirmed.memory is not None
    assert confirmed.memory.name == "language"
    assert manager.memory_store.get_memory("language") is not None


def test_confirm_memory_candidate_updates_duplicate_target(tmp_path: Path):
    manager = MemoryManager(MemoryConfig(root_dir=tmp_path / ".memora", allow_auto_save_user_preferences=False))
    manager.init_storage()
    existing = manager.save_memory("preference", "old content", "old desc", name="language", source="manual")
    pending = manager.remember_candidate(
        MemoryCandidate(
            action="create",
            type="preference",
            name="language",
            description="new desc",
            content="new content",
            source="runtime_extraction",
        )
    )

    confirmed = manager.confirm_memory_candidate(pending.candidate)

    assert confirmed.action == "updated"
    assert confirmed.memory is not None
    assert confirmed.memory.id == existing.id
    assert confirmed.memory.content == "new content"
    assert len(manager.memory_store.list_memories(include_archived=True)) == 1


def test_confirm_memory_candidate_updates_conflict_target(tmp_path: Path):
    manager = manager_for(tmp_path)
    manager.init_storage()
    existing = manager.save_memory("preference", "用户偏好英文回答。", "用户偏好英文。", name="language-en")
    pending = manager.remember_candidate(
        MemoryCandidate(
            action="create",
            type="preference",
            name="language-zh",
            description="用户偏好中文。",
            content="用户偏好中文回答。",
        )
    )

    confirmed = manager.confirm_memory_candidate(pending.candidate)

    assert confirmed.action == "updated"
    assert confirmed.memory is not None
    assert confirmed.memory.id == existing.id
    assert confirmed.memory.content == "用户偏好中文回答。"


def test_confirm_memory_candidate_detects_stale_create_candidate_after_duplicate_appears(tmp_path: Path):
    manager = MemoryManager(MemoryConfig(root_dir=tmp_path / ".memora", allow_auto_save_user_preferences=False))
    manager.init_storage()
    pending = manager.remember_candidate(
        MemoryCandidate(
            action="create",
            type="preference",
            name="language",
            description="first desc",
            content="first content",
            source="runtime_extraction",
        )
    )
    newer = manager.save_memory("preference", "second content", "second desc", name="language", source="manual")

    result = manager.confirm_memory_candidate(pending.candidate)

    assert result.action == "requires_confirmation"
    assert result.reason == "confirmation_state_changed"
    assert result.target_memory_id == newer.id
    assert result.candidate is not None
    assert result.candidate.suggested_action == "update"
    assert manager.get_memory(newer.id).content == "second content"


def test_confirm_memory_candidate_detects_stale_update_target(tmp_path: Path):
    manager = MemoryManager(MemoryConfig(root_dir=tmp_path / ".memora", allow_auto_save_user_preferences=False))
    manager.init_storage()
    existing = manager.save_memory("preference", "old content", "old desc", name="language", source="manual")
    pending = manager.remember_candidate(
        MemoryCandidate(
            action="create",
            type="preference",
            name="language",
            description="first desc",
            content="first content",
            source="runtime_extraction",
        )
    )
    manager.update_memory(existing.id, content="second content")

    result = manager.confirm_memory_candidate(pending.candidate)

    assert result.action == "requires_confirmation"
    assert result.reason == "confirmation_state_changed"
    assert result.target_memory_id == existing.id
    assert result.candidate is not None
    assert result.candidate.suggested_action == "update"
    assert manager.get_memory(existing.id).content == "second content"


def test_confirm_memory_candidate_rejects_create_override_when_current_policy_suggests_update(tmp_path: Path):
    manager = MemoryManager(MemoryConfig(root_dir=tmp_path / ".memora", allow_auto_save_user_preferences=False))
    manager.init_storage()
    existing = manager.save_memory("preference", "old content", "old desc", name="language", source="manual")
    pending = manager.remember_candidate(
        MemoryCandidate(
            action="create",
            type="preference",
            name="language",
            description="new desc",
            content="new content",
            source="runtime_extraction",
        )
    )

    try:
        manager.confirm_memory_candidate(pending.candidate, action="create")
    except MemoryValidationError as exc:
        assert "confirmed action" in str(exc)
    else:
        raise AssertionError("expected MemoryValidationError")

    assert manager.get_memory(existing.id).content == "old content"
    assert len(manager.memory_store.list_memories(include_archived=True)) == 1


def test_confirm_memory_candidate_rejects_target_override_that_differs_from_current_policy_target(tmp_path: Path):
    manager = MemoryManager(MemoryConfig(root_dir=tmp_path / ".memora", allow_auto_save_user_preferences=False))
    manager.init_storage()
    target_a = manager.save_memory("preference", "a old content", "a old desc", name="a", source="manual")
    target_b = manager.save_memory("preference", "b old content", "b old desc", name="b", source="manual")
    pending = manager.remember_candidate(
        MemoryCandidate(
            action="create",
            type="preference",
            name="a",
            description="a new desc",
            content="a new content",
            source="runtime_extraction",
        )
    )

    try:
        manager.confirm_memory_candidate(pending.candidate, target_memory_id=target_b.id)
    except MemoryValidationError as exc:
        assert "target_memory_id" in str(exc)
    else:
        raise AssertionError("expected MemoryValidationError")

    assert manager.get_memory(target_a.id).content == "a old content"
    assert manager.get_memory(target_b.id).content == "b old content"


def test_confirm_memory_candidate_rejects_non_confirmation_candidate(tmp_path: Path):
    manager = manager_for(tmp_path)
    candidate = MemoryCandidate(action="create", type="preference", name="language", description="desc", content="content")

    try:
        manager.confirm_memory_candidate(candidate)
    except MemoryPolicyError as exc:
        assert "does not require confirmation" in str(exc)
    else:
        raise AssertionError("expected MemoryPolicyError")


def test_confirm_memory_candidate_rejects_secret_candidate_without_writing(tmp_path: Path):
    manager = manager_for(tmp_path)
    manager.init_storage()
    candidate = MemoryCandidate(
        action="ask_user",
        suggested_action="create",
        type="preference",
        name="secret",
        description="secret",
        content="api_key = sk-abcdef123456",
        reason="auto_save_user_preferences_disabled",
    )

    try:
        manager.confirm_memory_candidate(candidate)
    except MemoryPolicyError as exc:
        assert "contains_secret" in str(exc)
    else:
        raise AssertionError("expected MemoryPolicyError")

    assert manager.memory_store.get_memory("secret") is None


def test_confirm_memory_candidate_rejects_transient_candidate_without_writing(tmp_path: Path):
    manager = manager_for(tmp_path)
    manager.init_storage()
    candidate = MemoryCandidate(
        action="ask_user",
        suggested_action="create",
        type="preference",
        name="transient",
        description="transient",
        content="下一步：实现 CLI",
        reason="auto_save_user_preferences_disabled",
    )

    try:
        manager.confirm_memory_candidate(candidate)
    except MemoryPolicyError as exc:
        assert "transient_task_state" in str(exc)
    else:
        raise AssertionError("expected MemoryPolicyError")

    assert manager.memory_store.get_memory("transient") is None


def test_confirm_memory_candidate_rejects_noisy_candidate_without_writing(tmp_path: Path):
    manager = MemoryManager(MemoryConfig(root_dir=tmp_path / ".memora", max_memory_content_chars=10))
    manager.init_storage()
    candidate = MemoryCandidate(
        action="ask_user",
        suggested_action="create",
        type="preference",
        name="noisy",
        description="noisy",
        content="x" * 11,
        reason="auto_save_user_preferences_disabled",
    )

    try:
        manager.confirm_memory_candidate(candidate)
    except MemoryPolicyError as exc:
        assert "noisy_output" in str(exc)
    else:
        raise AssertionError("expected MemoryPolicyError")

    assert manager.memory_store.get_memory("noisy") is None


def test_confirm_memory_candidate_rejects_invalid_suggested_action(tmp_path: Path):
    manager = manager_for(tmp_path)
    candidate = MemoryCandidate(
        action="ask_user",
        suggested_action="delete",
        type="preference",
        name="language",
        description="desc",
        content="content",
        reason="auto_save_user_preferences_disabled",
    )

    try:
        manager.confirm_memory_candidate(candidate)
    except MemoryValidationError as exc:
        assert "suggested action" in str(exc)
    else:
        raise AssertionError("expected MemoryValidationError")


def test_confirm_memory_candidate_update_requires_target(tmp_path: Path):
    manager = manager_for(tmp_path)
    candidate = MemoryCandidate(
        action="ask_user",
        suggested_action="update",
        type="preference",
        name="language",
        description="desc",
        content="content",
        reason="auto_save_user_preferences_disabled",
    )

    try:
        manager.confirm_memory_candidate(candidate)
    except MemoryValidationError as exc:
        assert "target_memory_id" in str(exc)
    else:
        raise AssertionError("expected MemoryValidationError")


def test_confirm_memory_candidate_missing_target_raises_not_found(tmp_path: Path):
    manager = manager_for(tmp_path)
    candidate = MemoryCandidate(
        action="ask_user",
        suggested_action="update",
        target_memory_id="missing",
        type="preference",
        name="language",
        description="desc",
        content="content",
        reason="auto_save_user_preferences_disabled",
    )

    try:
        manager.confirm_memory_candidate(candidate)
    except MemoryNotFoundError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("expected MemoryNotFoundError")


def test_confirm_memory_candidate_syncs_rag_index(tmp_path: Path):
    manager = MemoryManager(
        MemoryConfig(
            root_dir=tmp_path / ".memora",
            memory_backend="sqlite",
            rag_enabled=True,
            allow_auto_save_user_preferences=False,
        )
    )
    manager.init_storage()
    pending = manager.remember_candidate(
        MemoryCandidate(
            action="create",
            type="preference",
            name="language",
            description="用户偏好中文。",
            content="用户偏好中文回答。",
            source="runtime_extraction",
        )
    )

    confirmed = manager.confirm_memory_candidate(pending.candidate)
    results = manager.retrieve_memory("中文回答")

    assert confirmed.action == "created"
    assert len(results) == 1
    assert results[0].memory.id == confirmed.memory.id
    assert results[0].semantic_score > 0
