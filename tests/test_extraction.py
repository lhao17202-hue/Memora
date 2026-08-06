import json

from memora.extraction import (
    EXTRACTION_SYSTEM_PROMPT,
    LLMMemoryExtractor,
    extraction_prompt_messages,
    parse_extraction_json,
)
from memora.schema import SessionMessage, WorkingMemoryState


class FakeLLMClient:
    def __init__(self, response: str):
        self.response = response
        self.messages = None

    def complete(self, messages):
        self.messages = messages
        return self.response


def test_parse_extraction_json_returns_no_memory_artifact():
    artifact = parse_extraction_json('{"should_remember": false, "memories": []}')

    assert artifact.should_remember is False
    assert artifact.memories == []
    assert artifact.errors == []


def test_parse_extraction_json_extracts_multiple_memories():
    artifact = parse_extraction_json(
        json.dumps(
            {
                "should_remember": True,
                "memories": [
                    {
                        "type": "preference",
                        "name": "response-style",
                        "description": "Response style preference.",
                        "content": "Prefer concise answers.",
                        "tags": ["style"],
                        "confidence": 0.9,
                    },
                    {
                        "type": "tool",
                        "name": "pytest-command",
                        "description": "Verification command.",
                        "content": "Use pytest -q after changes.",
                    },
                ],
            }
        )
    )

    assert artifact.ok is True
    assert artifact.should_remember is True
    assert [memory.type for memory in artifact.memories] == ["preference", "tool"]
    assert artifact.memories[0].tags == ["style"]
    assert artifact.memories[0].confidence == 0.9


def test_parse_extraction_json_rejects_invalid_json_without_memories():
    artifact = parse_extraction_json("not json")

    assert artifact.should_remember is False
    assert artifact.memories == []
    assert artifact.errors[0].startswith("invalid_json:")


def test_parse_extraction_json_rejects_old_memory_type():
    artifact = parse_extraction_json(
        json.dumps(
            {
                "should_remember": True,
                "memories": [
                    {
                        "type": "user",
                        "name": "language",
                        "description": "Old type.",
                        "content": "Prefer Chinese.",
                    }
                ],
            }
        )
    )

    assert artifact.should_remember is False
    assert artifact.memories == []
    assert artifact.errors == ["memories[0].invalid_type:user"]


def test_low_confidence_extracted_memory_requires_confirmation():
    artifact = parse_extraction_json(
        json.dumps(
            {
                "should_remember": True,
                "memories": [
                    {
                        "type": "preference",
                        "name": "tentative-style",
                        "description": "Tentative style.",
                        "content": "Maybe prefer concise answers.",
                        "confidence": 0.4,
                    }
                ],
            }
        )
    )

    assert artifact.memories[0].requires_confirmation is True


def test_parse_extraction_json_rejects_boolean_numeric_fields():
    artifact = parse_extraction_json(
        json.dumps(
            {
                "should_remember": True,
                "memories": [
                    {
                        "type": "preference",
                        "name": "bad-fields",
                        "description": "Bad numeric fields.",
                        "content": "Bad numeric fields.",
                        "confidence": True,
                        "weight": False,
                    }
                ],
            }
        )
    )

    assert artifact.memories == []
    assert artifact.errors == [
        "memories[0].confidence_must_be_0_to_1",
        "memories[0].weight_must_be_1_to_10",
    ]


def test_parse_extraction_json_requires_boolean_confirmation_flag():
    artifact = parse_extraction_json(
        json.dumps(
            {
                "should_remember": True,
                "memories": [
                    {
                        "type": "preference",
                        "name": "bad-confirmation",
                        "description": "Bad confirmation flag.",
                        "content": "Bad confirmation flag.",
                        "requires_confirmation": "false",
                    }
                ],
            }
        )
    )

    assert artifact.memories == []
    assert artifact.errors == ["memories[0].requires_confirmation_must_be_boolean"]


def test_extracted_memory_converts_to_scoped_candidate_with_session_tag():
    artifact = parse_extraction_json(
        json.dumps(
            {
                "should_remember": True,
                "memories": [
                    {
                        "type": "tool",
                        "name": "pytest-command",
                        "description": "Verification command.",
                        "content": "Use pytest -q after changes.",
                        "tags": ["python"],
                    }
                ],
            }
        )
    )

    candidate = artifact.to_candidates(user_id="alice", project_id="project-a", session_id="session_1")[0]

    assert candidate.type == "tool"
    assert candidate.user_id == "alice"
    assert candidate.project_id == "project-a"
    assert candidate.source == "session_extraction"
    assert candidate.tags == ["python", "session:session_1"]


def test_llm_memory_extractor_uses_json_only_prompt_and_parses_response():
    client = FakeLLMClient(
        json.dumps(
            {
                "should_remember": True,
                "memories": [
                    {
                        "type": "preference",
                        "name": "response-style",
                        "description": "Response style preference.",
                        "content": "Prefer concise answers.",
                    }
                ],
            }
        )
    )

    artifact = LLMMemoryExtractor(client).extract([SessionMessage(role="user", content="Please be concise.")])

    assert artifact.ok is True
    assert artifact.memories[0].name == "response-style"
    assert client.messages[0]["role"] == "system"
    assert EXTRACTION_SYSTEM_PROMPT in client.messages[0]["content"]
    assert client.messages[1] == {"role": "user", "content": "Please be concise."}


def test_extraction_prompt_describes_memora_memory_boundary():
    assert "session or task end" in EXTRACTION_SYSTEM_PROMPT
    assert "MemoryCandidate" in EXTRACTION_SYSTEM_PROMPT
    assert "not final MemoryItem" in EXTRACTION_SYSTEM_PROMPT
    assert "tool: durable tool-use lesson summarized from traces" in EXTRACTION_SYSTEM_PROMPT
    assert "Current task state belongs to short-term memory" in EXTRACTION_SYSTEM_PROMPT



def test_extraction_prompt_includes_json_output_schema_and_example():
    assert "Output JSON schema" in EXTRACTION_SYSTEM_PROMPT
    assert '"should_remember": true' in EXTRACTION_SYSTEM_PROMPT
    assert '"memories": [' in EXTRACTION_SYSTEM_PROMPT
    assert '"type": "preference"' in EXTRACTION_SYSTEM_PROMPT
    assert '"name": "response-style"' in EXTRACTION_SYSTEM_PROMPT
    assert '"requires_confirmation": false' in EXTRACTION_SYSTEM_PROMPT
    assert "Return exactly one top-level JSON object" in EXTRACTION_SYSTEM_PROMPT



def test_extraction_prompt_guides_retrieval_fields():
    assert "Retrieval field guidance" in EXTRACTION_SYSTEM_PROMPT
    assert "type controls policy, default weight, pinning, and type-filtered retrieval" in EXTRACTION_SYSTEM_PROMPT
    assert "name is a stable canonical key" in EXTRACTION_SYSTEM_PROMPT
    assert "description is a search-facing summary" in EXTRACTION_SYSTEM_PROMPT
    assert "tags are exact-match retrieval facets" in EXTRACTION_SYSTEM_PROMPT
    assert "Prefer canonical tags from this vocabulary" in EXTRACTION_SYSTEM_PROMPT
    assert "response-style" in EXTRACTION_SYSTEM_PROMPT
    assert "test-command" in EXTRACTION_SYSTEM_PROMPT



def test_extraction_prompt_describes_working_memory_source_rules():
    assert "conversation_messages" in EXTRACTION_SYSTEM_PROMPT
    assert "working_memory_snapshot" in EXTRACTION_SYSTEM_PROMPT
    assert "agent-maintained short-term state" in EXTRACTION_SYSTEM_PROMPT
    assert "Do not directly memorize task, trace, or recent_files" in EXTRACTION_SYSTEM_PROMPT
    assert "preference: explicit stable user preference" in EXTRACTION_SYSTEM_PROMPT
    assert "general: fallback only" in EXTRACTION_SYSTEM_PROMPT



def test_extraction_prompt_messages_include_working_memory_snapshot():
    state = WorkingMemoryState(
        task="Add working memory as extraction evidence.",
        tool_notes=["pytest tests/test_extraction.py verifies extraction prompt behavior."],
        recent_files=["memora/extraction.py"],
        file_summaries={"memora/extraction.py": "Defines extraction prompt and parser."},
        notes=["Working memory should be evidence, not direct long-term memory."],
        trace="User requested extraction prompt support, agent implemented tests and runtime forwarding.",
    )

    messages = extraction_prompt_messages(
        [SessionMessage(role="user", content="Use working memory for extraction too.")],
        working_memory=state,
    )

    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "Use working memory for extraction too."}
    assert messages[2]["role"] == "user"
    assert "<working_memory_snapshot>" in messages[2]["content"]
    assert "task: Add working memory as extraction evidence." in messages[2]["content"]
    assert "tool_notes:" in messages[2]["content"]
    assert "- pytest tests/test_extraction.py verifies extraction prompt behavior." in messages[2]["content"]
    assert "recent_files:" in messages[2]["content"]
    assert "- memora/extraction.py" in messages[2]["content"]
    assert "file_summaries:" in messages[2]["content"]
    assert "memora/extraction.py: Defines extraction prompt and parser." in messages[2]["content"]
    assert "notes:" in messages[2]["content"]
    assert "trace: User requested extraction prompt support" in messages[2]["content"]
    assert "</working_memory_snapshot>" in messages[2]["content"]



def test_extraction_prompt_messages_omit_working_memory_when_not_provided():
    messages = extraction_prompt_messages([SessionMessage(role="user", content="Remember this preference.")])

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "Remember this preference."}



def test_llm_memory_extractor_accepts_working_memory():
    client = FakeLLMClient(
        json.dumps(
            {
                "should_remember": True,
                "memories": [
                    {
                        "type": "reflective",
                        "name": "working-memory-source-boundary",
                        "description": "Working memory extraction source boundary.",
                        "content": "Treat working memory as extraction evidence, not direct long-term memory.",
                    }
                ],
            }
        )
    )
    state = WorkingMemoryState(notes=["Working memory should be evidence, not direct long-term memory."])

    artifact = LLMMemoryExtractor(client).extract(
        [SessionMessage(role="assistant", content="Updated the extraction design.")],
        working_memory=state,
    )

    assert artifact.ok is True
    assert artifact.memories[0].type == "reflective"
    assert client.messages[0]["role"] == "system"
    assert client.messages[1] == {"role": "assistant", "content": "Updated the extraction design."}
    assert client.messages[2]["role"] == "user"
    assert "<working_memory_snapshot>" in client.messages[2]["content"]
    assert "Working memory should be evidence" in client.messages[2]["content"]



def test_extraction_prompt_messages_accept_mapping_messages():
    messages = extraction_prompt_messages([{"role": "assistant", "content": "Done."}])

    assert messages[1] == {"role": "assistant", "content": "Done."}
