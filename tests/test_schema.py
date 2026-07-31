from datetime import datetime, timezone

import pytest

from memora.config import MemoryConfig
from memora.errors import MemoryValidationError
from memora.schema import (
    MemoryCandidate,
    MemoryItem,
    MemoryQuery,
    MemoryRelation,
    MemoryRelationDecision,
    SessionMessage,
    WorkingMemoryState,
    validate_memory_candidate,
    validate_memory_item,
    validate_memory_query,
    validate_memory_relation,
    validate_memory_relation_decision,
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
    assert config.keyword_recall == "auto"
    assert config.vector_store == "sqlite"
    assert config.vector_store_options == {}
    assert not hasattr(config, "qdrant_url")
    assert not hasattr(config, "qdrant_host")
    assert not hasattr(config, "qdrant_port")
    assert not hasattr(config, "qdrant_api_key")
    assert not hasattr(config, "qdrant_collection")
    assert not hasattr(config, "qdrant_timeout")
    assert not hasattr(config, "qdrant_prefer_grpc")
    assert not hasattr(config, "qdrant_recreate_collection")
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
    boolean_weight = MemoryItem(id="mem_3", name="weight", description="desc", type="preference", content="content", weight=True)
    boolean_confidence = MemoryItem(id="mem_4", name="confidence", description="desc", type="preference", content="content", confidence=True)

    with pytest.raises(MemoryValidationError, match="weight"):
        validate_memory_item(overweight)
    with pytest.raises(MemoryValidationError, match="confidence"):
        validate_memory_item(overconfident)
    with pytest.raises(MemoryValidationError, match="weight"):
        validate_memory_item(boolean_weight)
    with pytest.raises(MemoryValidationError, match="confidence"):
        validate_memory_item(boolean_confidence)


def test_validate_memory_item_rejects_invalid_list_fields_and_access_count():
    bad_tags = MemoryItem(id="mem_1", name="tags", description="desc", type="preference", content="content", tags="tag")
    bad_supersedes = MemoryItem(id="mem_2", name="supersedes", description="desc", type="preference", content="content", supersedes=[1])
    bad_related = MemoryItem(id="mem_3", name="related", description="desc", type="preference", content="content", related="mem_1")
    bad_access_count = MemoryItem(id="mem_4", name="access", description="desc", type="preference", content="content", access_count=True)

    with pytest.raises(MemoryValidationError, match="tags"):
        validate_memory_item(bad_tags)
    with pytest.raises(MemoryValidationError, match="supersedes"):
        validate_memory_item(bad_supersedes)
    with pytest.raises(MemoryValidationError, match="related"):
        validate_memory_item(bad_related)
    with pytest.raises(MemoryValidationError, match="access_count"):
        validate_memory_item(bad_access_count)


def test_validate_memory_candidate_and_query_reject_invalid_tags():
    candidate = MemoryCandidate(action="create", name="language", description="desc", type="preference", content="content", tags="tag")

    with pytest.raises(MemoryValidationError, match="tags"):
        validate_memory_candidate(candidate)
    with pytest.raises(MemoryValidationError, match="tags"):
        validate_memory_query(MemoryQuery(query="content", tags="tag"))


def test_validate_memory_candidate_rejects_invalid_action():
    candidate = MemoryCandidate(action="bad", name="language", description="desc", type="preference", content="content")

    with pytest.raises(MemoryValidationError, match="candidate action"):
        validate_memory_candidate(candidate)


def test_validate_memory_query_rejects_invalid_limits():
    with pytest.raises(MemoryValidationError, match="top_k"):
        validate_memory_query(MemoryQuery(query="中文", top_k=0))
    with pytest.raises(MemoryValidationError, match="max_tokens"):
        validate_memory_query(MemoryQuery(query="中文", max_tokens=0))
    with pytest.raises(MemoryValidationError, match="top_k"):
        validate_memory_query(MemoryQuery(query="中文", top_k=True))
    with pytest.raises(MemoryValidationError, match="max_tokens"):
        validate_memory_query(MemoryQuery(query="中文", max_tokens="100"))


def test_validate_memory_relation_accepts_none_and_targeted_relations():
    validate_memory_relation(MemoryRelation(kind="none", similarity_score=0.42, reason="below threshold"))
    validate_memory_relation(MemoryRelation(kind="merge", target_memory_id="mem_1", similarity_score=0.92))
    validate_memory_relation(MemoryRelation(kind="supersede", target_memory_id="mem_1", similarity_score=1.0))


def test_validate_memory_relation_rejects_invalid_relation_data():
    with pytest.raises(MemoryValidationError, match="relation kind"):
        validate_memory_relation(MemoryRelation(kind="bad", similarity_score=0.5))
    with pytest.raises(MemoryValidationError, match="similarity_score"):
        validate_memory_relation(MemoryRelation(kind="none", similarity_score=1.1))
    with pytest.raises(MemoryValidationError, match="target_memory_id"):
        validate_memory_relation(MemoryRelation(kind="conflict", similarity_score=0.95))
    with pytest.raises(MemoryValidationError, match="target_memory_id"):
        validate_memory_relation(MemoryRelation(kind="merge", target_memory_id="", similarity_score=0.95))


def test_validate_memory_relation_decision_accepts_valid_decisions():
    validate_memory_relation_decision(MemoryRelationDecision(kind="none", confidence=0.9, reason="Different fact."))
    validate_memory_relation_decision(
        MemoryRelationDecision(
            kind="merge",
            confidence=0.9,
            reason="Refinement.",
            merged_description="Response style.",
            merged_content="Prefer concise answers with summaries.",
            merged_tags=["style"],
        )
    )
    validate_memory_relation_decision(MemoryRelationDecision(kind="supersede", confidence=0.95, reason="Preference changed."))


def test_validate_memory_relation_decision_rejects_invalid_decisions():
    with pytest.raises(MemoryValidationError, match="relation kind"):
        validate_memory_relation_decision(MemoryRelationDecision(kind="bad", confidence=0.9))
    with pytest.raises(MemoryValidationError, match="confidence"):
        validate_memory_relation_decision(MemoryRelationDecision(kind="none", confidence=True))
    with pytest.raises(MemoryValidationError, match="merge relation decision"):
        validate_memory_relation_decision(MemoryRelationDecision(kind="merge", confidence=0.9, merged_description="desc"))
    with pytest.raises(MemoryValidationError, match="merged_tags"):
        validate_memory_relation_decision(
            MemoryRelationDecision(
                kind="merge",
                confidence=0.9,
                merged_description="desc",
                merged_content="content",
                merged_tags="tag",
            )
        )


def test_validate_session_message_rejects_invalid_role():
    message = SessionMessage(role="invalid", content="hello")

    with pytest.raises(MemoryValidationError, match="session role"):
        validate_session_message(message)
