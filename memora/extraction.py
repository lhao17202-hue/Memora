"""LLM memory extraction contract and JSON parsing."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence

from .schema import MemoryCandidate, SessionMessage, validate_memory_candidate, validate_memory_type

LOW_CONFIDENCE_THRESHOLD = 0.5

EXTRACTION_SYSTEM_PROMPT = """Extract durable long-term memories from the conversation.
Return JSON only. Do not include markdown.
Use only these memory types: preference, project, episodic, reflective, tool, knowledge, general.
If nothing should be remembered, return {"should_remember": false, "memories": []}.
If something should be remembered, return {"should_remember": true, "memories": [...]}.
Each memory must include: type, name, description, content.
Optional fields: tags, confidence, weight, requires_confirmation, reason."""


class LLMClient(Protocol):
    def complete(self, messages: Sequence[Mapping[str, str]]) -> str:
        """Return a JSON-only extraction response for prompt messages."""


class MemoryExtractor(Protocol):
    def extract(self, messages: Sequence[SessionMessage | Mapping[str, str]]) -> "ExtractionArtifact":
        """Extract candidate memories from conversation messages."""


@dataclass
class ExtractedMemory:
    type: str
    name: str
    description: str
    content: str
    tags: list[str] = field(default_factory=list)
    confidence: float = 1.0
    weight: int | None = None
    requires_confirmation: bool = False
    reason: str = ""

    def to_candidate(
        self,
        user_id: str = "default",
        project_id: str | None = None,
        workspace_id: str | None = None,
        session_id: str | None = None,
        source: str = "runtime_extraction",
    ) -> MemoryCandidate:
        tags = list(self.tags)
        if session_id is not None:
            session_tag = f"session:{session_id}"
            if session_tag not in tags:
                tags.append(session_tag)
            source = "session_extraction"
        candidate = MemoryCandidate(
            action="create",
            name=self.name,
            description=self.description,
            type=self.type,
            content=self.content,
            user_id=user_id,
            project_id=project_id,
            workspace_id=workspace_id,
            tags=tags,
            source=source,
            confidence=self.confidence,
            weight=self.weight,
        )
        validate_memory_candidate(candidate)
        return candidate


@dataclass
class ExtractionArtifact:
    should_remember: bool
    memories: list[ExtractedMemory] = field(default_factory=list)
    raw_text: str = ""
    errors: list[str] = field(default_factory=list)
    source: str = "llm"

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_candidates(
        self,
        user_id: str = "default",
        project_id: str | None = None,
        workspace_id: str | None = None,
        session_id: str | None = None,
        source: str = "runtime_extraction",
    ) -> list[MemoryCandidate]:
        return [
            memory.to_candidate(
                user_id=user_id,
                project_id=project_id,
                workspace_id=workspace_id,
                session_id=session_id,
                source=source,
            )
            for memory in self.memories
        ]


def parse_extraction_json(raw_text: str, source: str = "llm") -> ExtractionArtifact:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return ExtractionArtifact(
            should_remember=False,
            raw_text=raw_text,
            errors=[f"invalid_json:{exc.msg}"],
            source=source,
        )
    if not isinstance(payload, dict):
        return ExtractionArtifact(
            should_remember=False,
            raw_text=raw_text,
            errors=["extraction_payload_must_be_object"],
            source=source,
        )

    should_remember = payload.get("should_remember")
    if not isinstance(should_remember, bool):
        return ExtractionArtifact(
            should_remember=False,
            raw_text=raw_text,
            errors=["should_remember_must_be_boolean"],
            source=source,
        )
    if not should_remember:
        return ExtractionArtifact(should_remember=False, memories=[], raw_text=raw_text, source=source)

    raw_memories = payload.get("memories")
    if raw_memories is None and _looks_like_memory(payload):
        raw_memories = [payload]
    if not isinstance(raw_memories, list):
        return ExtractionArtifact(
            should_remember=True,
            raw_text=raw_text,
            errors=["memories_must_be_list"],
            source=source,
        )

    memories = []
    errors = []
    for index, raw_memory in enumerate(raw_memories):
        memory, item_errors = _parse_extracted_memory(raw_memory)
        if item_errors:
            errors.extend(f"memories[{index}].{error}" for error in item_errors)
            continue
        memories.append(memory)
    return ExtractionArtifact(
        should_remember=bool(memories),
        memories=memories,
        raw_text=raw_text,
        errors=errors,
        source=source,
    )


def extraction_prompt_messages(messages: Sequence[SessionMessage | Mapping[str, str]]) -> list[dict[str, str]]:
    normalized = [{"role": "system", "content": EXTRACTION_SYSTEM_PROMPT}]
    for message in messages:
        if isinstance(message, SessionMessage):
            role = message.role
            content = message.content
        else:
            role = str(message.get("role") or "user")
            content = str(message.get("content") or "")
        normalized.append({"role": role, "content": content})
    return normalized


class LLMMemoryExtractor:
    def __init__(self, client: LLMClient):
        self.client = client

    def extract(self, messages: Sequence[SessionMessage | Mapping[str, str]]) -> ExtractionArtifact:
        raw_text = self.client.complete(extraction_prompt_messages(messages))
        return parse_extraction_json(raw_text)


def _looks_like_memory(payload: dict) -> bool:
    return any(key in payload for key in ("type", "name", "description", "content"))


def _parse_extracted_memory(raw_memory: object) -> tuple[ExtractedMemory | None, list[str]]:
    if not isinstance(raw_memory, dict):
        return None, ["must_be_object"]
    errors = []
    memory_type = _string_field(raw_memory, "type", errors)
    name = _string_field(raw_memory, "name", errors)
    description = _string_field(raw_memory, "description", errors)
    content = _string_field(raw_memory, "content", errors)
    if memory_type:
        try:
            validate_memory_type(memory_type)
        except Exception:
            errors.append(f"invalid_type:{memory_type}")

    tags = raw_memory.get("tags", [])
    if tags is None:
        tags = []
    if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
        errors.append("tags_must_be_string_list")
        tags = []

    confidence = raw_memory.get("confidence", 1.0)
    if isinstance(confidence, bool) or not isinstance(confidence, int | float) or confidence < 0.0 or confidence > 1.0:
        errors.append("confidence_must_be_0_to_1")
        confidence = 1.0
    confidence = float(confidence)

    weight = raw_memory.get("weight")
    if weight is not None and (isinstance(weight, bool) or not isinstance(weight, int) or weight < 1 or weight > 10):
        errors.append("weight_must_be_1_to_10")
        weight = None

    requires_confirmation = raw_memory.get("requires_confirmation", False)
    if not isinstance(requires_confirmation, bool):
        errors.append("requires_confirmation_must_be_boolean")
        requires_confirmation = False
    requires_confirmation = requires_confirmation or confidence < LOW_CONFIDENCE_THRESHOLD
    reason = raw_memory.get("reason", "")
    if reason is None:
        reason = ""
    if not isinstance(reason, str):
        errors.append("reason_must_be_string")
        reason = ""

    if errors:
        return None, errors
    return (
        ExtractedMemory(
            type=memory_type,
            name=name,
            description=description,
            content=content,
            tags=tags,
            confidence=confidence,
            weight=weight,
            requires_confirmation=requires_confirmation,
            reason=reason,
        ),
        [],
    )


def _string_field(data: dict, field_name: str, errors: list[str]) -> str:
    value = data.get(field_name)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{field_name}_must_be_non_empty_string")
        return ""
    return value
