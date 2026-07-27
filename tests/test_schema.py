from datetime import datetime, timezone

import pytest

from memora.config import MemoryConfig
from memora.errors import MemoryValidationError
from memora.schema import (
    MemoryCandidate,
    MemoryItem,
    MemoryQuery,
    SessionMessage,
    WorkingMemoryState,
    validate_memory_candidate,
    validate_memory_item,
    validate_memory_query,
    validate_session_message,
)


def test_memory_item_defaults_are_safe():
    item = MemoryItem(
        id="mem_1",
        name="user-language-preference",
        description="User prefers Chinese.",
        type="preference",
        content="用户偏好使用中文。",
    )

    assert item.user_id == "default"
    assert item.project_id is None
    assert item.workspace_id is None
    assert item.tags == []
    assert item.source == "unknown"
    assert item.confidence == 1.0
    assert item.weight == 5
    assert item.status == "active"
    assert item.access_count == 0
    assert item.supersedes == []
    assert item.related == []


def test_memory_query_defaults():
    query = MemoryQuery(query="中文偏好")

    assert query.user_id == "default"
    assert query.top_k == 8
    assert query.max_tokens == 2000
    assert query.include_archived is False
    assert query.include_knowledge is True


def test_working_memory_defaults():
    state = WorkingMemoryState()

    assert state.task_summary == ""
    assert state.open_questions == []
    assert state.file_summaries == {}


def test_session_message_accepts_metadata():
    created = datetime(2026, 7, 16, tzinfo=timezone.utc)
    message = SessionMessage(role="user", content="hello", created_at=created)

    assert message.role == "user"
    assert message.content == "hello"
    assert message.created_at == created


def test_memory_config_defaults():
    config = MemoryConfig()

    assert config.root_dir == ".memora"
    assert config.memory_backend == "file"
    assert config.sqlite_path is None
    assert config.fts_enabled is True
    assert config.fts_candidate_limit == 100
    assert config.max_retrieved_memories == 8
    assert config.max_memory_prompt_tokens == 2000
    assert config.default_preference_weight == 9
    assert config.default_project_weight == 8
    assert config.default_episodic_weight == 5
    assert config.default_reflective_weight == 7
    assert config.default_tool_weight == 6
    assert config.default_knowledge_weight == 6
    assert config.default_general_weight == 4
    assert config.archive_cold_days == 180
    assert config.require_confirmation_for_conflicts is True


def test_validate_memory_item_rejects_invalid_type():
    item = MemoryItem(id="mem_1", name="language", description="desc", type="invalid", content="content")

    with pytest.raises(MemoryValidationError, match="memory type"):
        validate_memory_item(item)


def test_validate_memory_item_rejects_invalid_status():
    item = MemoryItem(id="mem_1", name="language", description="desc", type="preference", content="content", status="bad")

    with pytest.raises(MemoryValidationError, match="memory status"):
        validate_memory_item(item)


def test_validate_memory_item_rejects_invalid_weight_and_confidence():
    overweight = MemoryItem(id="mem_1", name="language", description="desc", type="preference", content="content", weight=11)
    overconfident = MemoryItem(id="mem_2", name="style", description="desc", type="preference", content="content", confidence=1.1)

    with pytest.raises(MemoryValidationError, match="weight"):
        validate_memory_item(overweight)
    with pytest.raises(MemoryValidationError, match="confidence"):
        validate_memory_item(overconfident)


def test_validate_memory_candidate_rejects_invalid_action():
    candidate = MemoryCandidate(action="bad", name="language", description="desc", type="preference", content="content")

    with pytest.raises(MemoryValidationError, match="candidate action"):
        validate_memory_candidate(candidate)


def test_validate_memory_query_rejects_invalid_limits():
    with pytest.raises(MemoryValidationError, match="top_k"):
        validate_memory_query(MemoryQuery(query="中文", top_k=0))
    with pytest.raises(MemoryValidationError, match="max_tokens"):
        validate_memory_query(MemoryQuery(query="中文", max_tokens=0))


def test_validate_session_message_rejects_invalid_role():
    message = SessionMessage(role="invalid", content="hello")

    with pytest.raises(MemoryValidationError, match="session role"):
        validate_session_message(message)
