"""LLM memory extraction contract and JSON parsing."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence

from .schema import MemoryCandidate, SessionMessage, WorkingMemoryState, validate_memory_candidate, validate_memory_type

LOW_CONFIDENCE_THRESHOLD = 0.5

EXTRACTION_SYSTEM_PROMPT = """Extract durable long-term memory candidates at session or task end.
Return JSON only. Do not include markdown.

Memora stores MemoryCandidate objects first, not final MemoryItem records. The
runtime will validate candidates, apply safety policy, resolve relations, ask
for confirmation when needed, and then write to the local backend.

The input may contain two evidence sections:
- conversation_messages: direct user/assistant interaction evidence.
- working_memory_snapshot: agent-maintained short-term state and summaries.

Use conversation_messages to extract explicit user preferences, durable project
facts, important decisions, and intentionally imported knowledge. Treat these
messages as stronger evidence than working_memory_snapshot.

Use working_memory_snapshot conservatively. It is evidence for durable
conclusions, reusable lessons, tool-use lessons, and important decisions; it is
not itself long-term memory. Do not directly memorize current_goal, next_step, open_questions, or recent_files
unless they capture a durable project direction or important decision. Never
memorize raw logs, raw stdout/stderr, stack traces, or transient task progress
from working_memory_snapshot.

Use only these memory types:
- preference: explicit stable user preference, user identity, answer style, or personal constraint.
- project: durable project requirement, tech stack, architecture, repo convention, or business rule.
- episodic: important dated interaction event or decision worth recalling later.
- reflective: reusable lesson from successes, failures, reviews, implementation, or debugging.
- tool: durable tool-use lesson summarized from traces, not raw tool logs.
- knowledge: stable external/reference knowledge that was intentionally imported.
- general: fallback only for durable memory that is useful but does not fit the other types.

Type routing guidance:
- Prefer preference only when the user expresses a stable preference or constraint.
- Prefer project for durable repo facts, architecture boundaries, conventions, and accepted design decisions.
- Prefer episodic for dated session decisions or milestones, not ordinary progress updates.
- Prefer reflective for reusable lessons about how to work or debug better next time.
- Prefer tool for commands, tool behavior, verification lessons, or tool-failure lessons after summarization.
- Prefer knowledge for stable imported references, not speculation.
- Use general sparingly.

Remember only durable information. Do not remember secrets, raw credentials,
full transcripts, raw stdout/stderr, stack traces, temporary task progress,
speculation, or one-turn plans. Current task state belongs to short-term memory.

Use stable short kebab-case names. Keep description and content concise,
auditable, and evidence-backed. Prefer fewer high-quality memories over many
small fragments. Set requires_confirmation=true for low confidence, sensitive
user preferences, or uncertain facts. Set confidence below 0.5 when the memory
is plausible but weakly supported.

If nothing should be remembered, return {"should_remember": false, "memories": []}.
If something should be remembered, return {"should_remember": true, "memories": [...]}.
Each memory must include: type, name, description, content.
Optional fields: tags, confidence, weight, requires_confirmation, reason."""


class LLMClient(Protocol):
    def complete(self, messages: Sequence[Mapping[str, str]]) -> str:
        """Return a JSON-only extraction response for prompt messages."""


class MemoryExtractor(Protocol):
    def extract(
        self,
        messages: Sequence[SessionMessage | Mapping[str, str]],
        working_memory: WorkingMemoryState | Mapping[str, object] | None = None,
    ) -> "ExtractionArtifact":
        """Extract candidate memories from conversation messages and optional working memory."""


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


def _format_working_memory_snapshot(working_memory: WorkingMemoryState | Mapping[str, object]) -> str:
    if isinstance(working_memory, WorkingMemoryState):
        data: Mapping[str, object] = {
            "task_summary": working_memory.task_summary,
            "current_goal": working_memory.current_goal,
            "open_questions": working_memory.open_questions,
            "recent_files": working_memory.recent_files,
            "file_summaries": working_memory.file_summaries,
            "process_notes": working_memory.process_notes,
            "tool_failures": working_memory.tool_failures,
            "next_step": working_memory.next_step,
        }
    else:
        data = working_memory

    lines = ["<working_memory_snapshot>"]
    for field_name in (
        "task_summary",
        "current_goal",
        "open_questions",
        "recent_files",
        "file_summaries",
        "process_notes",
        "tool_failures",
        "next_step",
    ):
        value = data.get(field_name)
        if value in (None, "", [], {}):
            continue
        lines.extend(_format_snapshot_field(field_name, value))
    lines.append("</working_memory_snapshot>")
    return "\n".join(lines)



def _format_snapshot_field(field_name: str, value: object) -> list[str]:
    if isinstance(value, list):
        lines = [f"{field_name}:"]
        lines.extend(f"- {item}" for item in value if str(item).strip())
        return lines
    if isinstance(value, dict):
        lines = [f"{field_name}:"]
        lines.extend(f"- {key}: {item}" for key, item in value.items() if str(item).strip())
        return lines
    return [f"{field_name}: {value}"]



def extraction_prompt_messages(
    messages: Sequence[SessionMessage | Mapping[str, str]],
    working_memory: WorkingMemoryState | Mapping[str, object] | None = None,
) -> list[dict[str, str]]:
    normalized = [{"role": "system", "content": EXTRACTION_SYSTEM_PROMPT}]
    for message in messages:
        if isinstance(message, SessionMessage):
            role = message.role
            content = message.content
        else:
            role = str(message.get("role") or "user")
            content = str(message.get("content") or "")
        normalized.append({"role": role, "content": content})
    if working_memory is not None:
        normalized.append({"role": "user", "content": _format_working_memory_snapshot(working_memory)})
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
