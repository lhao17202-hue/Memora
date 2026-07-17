from datetime import datetime, timezone

from memora.config import MemoryConfig
from memora.schema import MemoryItem, MemoryQuery, SessionMessage, WorkingMemoryState


def test_memory_item_defaults_are_safe():
    item = MemoryItem(
        id="mem_1",
        name="user-language-preference",
        description="User prefers Chinese.",
        type="user",
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
    assert config.max_retrieved_memories == 8
    assert config.max_memory_prompt_tokens == 2000
    assert config.archive_cold_days == 180
    assert config.require_confirmation_for_conflicts is True
