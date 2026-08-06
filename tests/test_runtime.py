from pathlib import Path

import pytest

from memora.config import MemoryConfig
from memora.extraction import ExtractionArtifact, ExtractedMemory, LLMMemoryExtractor, parse_extraction_json
from memora.manager import MemoryManager
from memora.relations import LLMMemoryRelationJudge
from memora.runtime import MemoryRuntime
from memora.schema import MemoryCandidate, MemoryRelation, SessionMessage, WorkingMemoryState


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


class FakeLLMClient:
    def __init__(self, response: str):
        self.response = response

    def complete(self, messages):
        return self.response


class RecordingExtractor:
    def __init__(self, artifact: ExtractionArtifact):
        self.artifact = artifact
        self.messages = None
        self.working_memory = None

    def extract(self, messages, working_memory=None):
        self.messages = messages
        self.working_memory = working_memory
        return self.artifact


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


def test_constructor_rejects_manager_and_relation_judge_together(tmp_path: Path):
    manager = MemoryManager(MemoryConfig(root_dir=tmp_path / ".memora"))
    judge = LLMMemoryRelationJudge(FakeLLMClient('{"kind":"none","confidence":1,"reason":"unused"}'))

    with pytest.raises(ValueError, match="manager and relation_judge cannot both be provided"):
        MemoryRuntime(manager=manager, relation_judge=judge)


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


def test_extract_memories_without_configured_extractor_does_not_write(tmp_path: Path):
    runtime = make_runtime(tmp_path)

    artifact = runtime.extract_memories([{"role": "user", "content": "Remember I prefer concise answers."}])
    results = runtime.remember_extraction_artifact(artifact)

    assert artifact.errors == ["memory_extractor_not_configured"]
    assert results == []
    assert runtime.manager.list_memories() == []


def test_extract_memories_forwards_working_memory_to_extractor(tmp_path):
    state = WorkingMemoryState(process_notes=["Use working memory as extraction evidence."])
    artifact = ExtractionArtifact(should_remember=False, memories=[])
    extractor = RecordingExtractor(artifact)
    runtime = MemoryRuntime(config=MemoryConfig(root_dir=str(tmp_path / ".memora")), extractor=extractor)
    messages = [SessionMessage(role="assistant", content="Finished extraction design.")]

    returned = runtime.extract_memories(messages, working_memory=state)

    assert returned is artifact
    assert extractor.messages == messages
    assert extractor.working_memory is state



def test_extract_and_remember_forwards_working_memory(tmp_path):
    state = WorkingMemoryState(process_notes=["Working memory can produce reflective memories."])
    artifact = ExtractionArtifact(
        should_remember=True,
        memories=[
            ExtractedMemory(
                type="reflective",
                name="working-memory-evidence",
                description="Working memory extraction evidence.",
                content="Treat working memory as an extraction evidence source.",
            )
        ],
    )
    extractor = RecordingExtractor(artifact)
    runtime = MemoryRuntime(config=MemoryConfig(root_dir=str(tmp_path / ".memora")), extractor=extractor)
    runtime.init_storage()
    messages = [SessionMessage(role="assistant", content="Prepared extraction improvement.")]

    returned_artifact, results = runtime.extract_and_remember(messages, working_memory=state)

    assert returned_artifact is artifact
    assert extractor.working_memory is state
    assert [result.action for result in results] == ["created"]



def test_extract_and_remember_uses_injected_llm_extractor(tmp_path: Path):
    runtime = make_runtime(tmp_path)
    extractor = LLMMemoryExtractor(
        FakeLLMClient(
            '{"should_remember": true, "memories": ['
            '{"type": "preference", "name": "response-style", '
            '"description": "Response style preference.", '
            '"content": "Prefer concise answers."},'
            '{"type": "tool", "name": "pytest-command", '
            '"description": "Verification command.", '
            '"content": "Use pytest -q after changes."}'
            "]}"
        )
    )

    artifact, results = runtime.extract_and_remember(
        [{"role": "user", "content": "Please be concise and use pytest -q."}],
        extractor=extractor,
    )

    assert artifact.ok is True
    assert [result.action for result in results] == ["created", "created"]
    assert {memory.type for memory in runtime.manager.list_memories()} == {"preference", "tool"}


def test_runtime_llm_relation_judge_merges_and_keeps_rag_index_current(tmp_path: Path):
    relation_client = FakeLLMClient(
        '{"kind":"merge","confidence":0.91,"reason":"Candidate refines the response style.",'
        '"merged":{"name":"response-style","description":"Response style preference.",'
        '"content":"Prefer concise answers with short summaries.","tags":["style","summary"]}}'
    )
    runtime = MemoryRuntime(
        config=MemoryConfig(
            root_dir=tmp_path / ".memora",
            memory_backend="sqlite",
            rag_enabled=True,
            llm_relation_judge_enabled=True,
            semantic_relation_threshold=0.10,
            semantic_merge_threshold=0.10,
            semantic_conflict_threshold=0.95,
        ),
        relation_judge=LLMMemoryRelationJudge(relation_client),
    )
    runtime.init_storage()
    created = runtime.remember_extracted(
        memory_type="preference",
        name="response-style",
        description="Response style preference.",
        content="Prefer concise answers.",
    )

    updated = runtime.remember_extracted(
        memory_type="preference",
        name="short-summary-style",
        description="Short summary preference.",
        content="Prefer concise answers with short summaries.",
    )
    results = runtime.retrieve_context("short summaries", memory_types=["preference"])
    verify = runtime.manager.verify_memories()

    assert created.action == "created"
    assert updated.action == "updated"
    assert updated.reason == "llm_semantic_merge"
    assert updated.memory is not None
    assert updated.memory.content == "Prefer concise answers with short summaries."
    assert updated.memory.tags == ["style", "summary"]
    assert [result.memory.id for result in results] == [updated.memory.id]
    assert verify["vector_ok"] is True
    assert verify["embedding_mismatches"] == []


def test_remember_extraction_artifact_returns_confirmation_for_low_confidence(tmp_path: Path):
    runtime = make_runtime(tmp_path)
    existing = runtime.manager.save_memory("preference", "Prefer long answers.", "response style", name="long-style")

    class FixedRelationResolver:
        def resolve(self, candidate, existing_memories):
            return MemoryRelation(
                kind="conflict",
                target_memory_id=existing.id,
                target_updated_at=existing.updated_at,
                similarity_score=0.91,
                reason="semantic_conflict",
            )

    runtime.manager.relation_resolver = FixedRelationResolver()
    artifact = parse_extraction_json(
        '{"should_remember": true, "memories": ['
        '{"type": "preference", "name": "tentative-style", '
        '"description": "Tentative response style.", '
        '"content": "Maybe prefer concise answers.", '
        '"confidence": 0.4}'
        "]}"
    )

    results = runtime.remember_extraction_artifact(artifact)

    assert [result.action for result in results] == ["requires_confirmation"]
    assert results[0].reason == "low_confidence_extraction"
    assert isinstance(results[0].candidate, MemoryCandidate)
    assert results[0].target_memory_id == existing.id
    assert results[0].relation_kind == "conflict"
    assert results[0].relation_confidence == 0.91
    assert results[0].relation_reason == "semantic_conflict"
    assert [memory.id for memory in runtime.manager.list_memories()] == [existing.id]


def test_low_confidence_extraction_still_respects_policy_rejection(tmp_path: Path):
    runtime = make_runtime(tmp_path)
    artifact = parse_extraction_json(
        '{"should_remember": true, "memories": ['
        '{"type": "preference", "name": "secret", '
        '"description": "Secret.", '
        '"content": "api_key = sk-abcdef123456", '
        '"confidence": 0.4}'
        "]}"
    )

    results = runtime.remember_extraction_artifact(artifact)

    assert [result.action for result in results] == ["rejected"]
    assert results[0].reason == "contains_secret"
    assert runtime.manager.list_memories() == []


def test_invalid_extraction_artifact_does_not_write_memory(tmp_path: Path):
    runtime = make_runtime(tmp_path)
    artifact = parse_extraction_json(
        '{"should_remember": true, "memories": ['
        '{"type": "user", "name": "old", "description": "old", "content": "old"}'
        "]}"
    )

    results = runtime.remember_extraction_artifact(artifact)

    assert artifact.errors == ["memories[0].invalid_type:user"]
    assert results == []
    assert runtime.manager.list_memories() == []
