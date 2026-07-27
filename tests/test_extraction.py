import json

from memora.extraction import (
    EXTRACTION_SYSTEM_PROMPT,
    LLMMemoryExtractor,
    extraction_prompt_messages,
    parse_extraction_json,
)
from memora.schema import SessionMessage


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
    assert "tool: durable tool-use lessons summarized from traces" in EXTRACTION_SYSTEM_PROMPT
    assert "Current task state belongs to short-term memory" in EXTRACTION_SYSTEM_PROMPT


def test_extraction_prompt_messages_accept_mapping_messages():
    messages = extraction_prompt_messages([{"role": "assistant", "content": "Done."}])

    assert messages[1] == {"role": "assistant", "content": "Done."}
