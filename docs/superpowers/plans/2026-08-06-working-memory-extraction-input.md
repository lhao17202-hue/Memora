# Working Memory Extraction Input Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional `WorkingMemoryState` input to Memora memory extraction so LLM extraction can use both conversation messages and session working-memory summaries while preserving the existing extraction artifact and write pipeline.

**Architecture:** Extend the extraction prompt boundary only: `extraction_prompt_messages(...)`, `LLMMemoryExtractor.extract(...)`, and `MemoryRuntime.extract_memories/extract_and_remember(...)` accept an optional `working_memory` argument. The prompt renders working memory as a clearly separated snapshot, not as a normal chat message, and existing calls without working memory remain compatible.

**Tech Stack:** Python dataclasses and protocols, existing Memora runtime/extraction modules, pytest.

## Global Constraints

- Do not change `ExtractedMemory`, `ExtractionArtifact`, `MemoryCandidate`, `MemoryPolicy.evaluate`, relation judging, manager write actions, or storage behavior.
- `working_memory` input must be optional and backward compatible.
- Do not directly persist `working_memory`; treat it only as an extraction evidence source.
- Do not add dependencies.
- Preserve existing JSON output contract: `{"should_remember": bool, "memories": [...]}`.
- The prompt must distinguish raw conversation evidence from agent-maintained working-memory summary.
- The prompt must warn against memorizing transient task state, raw logs, stack traces, or next-step/open-question noise.

---

## File Structure

- Modify `memora/extraction.py`
  - Extend extraction prompt instructions with source-specific and type-routing rules.
  - Add optional `working_memory` support to `extraction_prompt_messages(...)`.
  - Add optional `working_memory` support to `LLMMemoryExtractor.extract(...)` and the `MemoryExtractor` protocol.
  - Add a private formatting helper for `WorkingMemoryState`/mapping snapshots.
- Modify `memora/runtime.py`
  - Thread optional `working_memory` through `MemoryRuntime.extract_memories(...)` and `MemoryRuntime.extract_and_remember(...)`.
- Modify `tests/test_extraction.py`
  - Cover prompt wording and working-memory snapshot rendering.
  - Cover backward compatibility when no working memory is passed.
  - Cover `LLMMemoryExtractor.extract(..., working_memory=...)` passes snapshot into client messages.
- Modify `tests/test_runtime.py`
  - Cover runtime forwarding optional working memory to extractors that accept it.
  - Cover compatibility with older/fake extractors that only accept `messages` if needed by current tests.

---

### Task 1: Prompt Rules and Working-Memory Snapshot Rendering

**Files:**
- Modify: `memora/extraction.py`
- Test: `tests/test_extraction.py`

**Interfaces:**
- Consumes: existing `SessionMessage` and `WorkingMemoryState` dataclasses from `memora.schema`.
- Produces:
  - `extraction_prompt_messages(messages: Sequence[SessionMessage | Mapping[str, str]], working_memory: WorkingMemoryState | Mapping[str, object] | None = None) -> list[dict[str, str]]`
  - Private helper `_format_working_memory_snapshot(working_memory: WorkingMemoryState | Mapping[str, object]) -> str`

- [ ] **Step 1: Write failing tests for source-specific prompt rules**

Add these tests to `tests/test_extraction.py`:

```python
from memora.schema import SessionMessage, WorkingMemoryState
```

If `SessionMessage` is already imported, extend the existing import instead of adding a duplicate import.

Add this test:

```python
def test_extraction_prompt_describes_working_memory_source_rules():
    assert "conversation_messages" in EXTRACTION_SYSTEM_PROMPT
    assert "working_memory_snapshot" in EXTRACTION_SYSTEM_PROMPT
    assert "agent-maintained short-term state" in EXTRACTION_SYSTEM_PROMPT
    assert "Do not directly memorize current_goal, next_step, open_questions, or recent_files" in EXTRACTION_SYSTEM_PROMPT
    assert "preference: explicit stable user preference" in EXTRACTION_SYSTEM_PROMPT
    assert "general: fallback only" in EXTRACTION_SYSTEM_PROMPT
```

- [ ] **Step 2: Write failing tests for optional working-memory prompt rendering**

Add these tests to `tests/test_extraction.py`:

```python
def test_extraction_prompt_messages_include_working_memory_snapshot():
    state = WorkingMemoryState(
        task_summary="Reviewed Memora extraction design.",
        current_goal="Add working memory as extraction evidence.",
        open_questions=["Should runtime auto-load sessions?"],
        recent_files=["memora/extraction.py"],
        file_summaries={"memora/extraction.py": "Defines extraction prompt and parser."},
        process_notes=["Working memory should be evidence, not direct long-term memory."],
        tool_failures=["A raw pytest traceback should not be memorized."],
        next_step="Update extraction tests.",
    )

    messages = extraction_prompt_messages(
        [SessionMessage(role="user", content="Use working memory for extraction too.")],
        working_memory=state,
    )

    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "Use working memory for extraction too."}
    assert messages[2]["role"] == "user"
    assert "<working_memory_snapshot>" in messages[2]["content"]
    assert "task_summary: Reviewed Memora extraction design." in messages[2]["content"]
    assert "current_goal: Add working memory as extraction evidence." in messages[2]["content"]
    assert "open_questions:" in messages[2]["content"]
    assert "- Should runtime auto-load sessions?" in messages[2]["content"]
    assert "file_summaries:" in messages[2]["content"]
    assert "memora/extraction.py: Defines extraction prompt and parser." in messages[2]["content"]
    assert "</working_memory_snapshot>" in messages[2]["content"]
```

Add this compatibility test:

```python
def test_extraction_prompt_messages_omit_working_memory_when_not_provided():
    messages = extraction_prompt_messages([SessionMessage(role="user", content="Remember this preference.")])

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "Remember this preference."}
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
pytest tests/test_extraction.py::test_extraction_prompt_describes_working_memory_source_rules tests/test_extraction.py::test_extraction_prompt_messages_include_working_memory_snapshot tests/test_extraction.py::test_extraction_prompt_messages_omit_working_memory_when_not_provided -q
```

Expected: failures because `WorkingMemoryState` is not imported in `extraction.py`, prompt rules are missing, and `extraction_prompt_messages` does not accept `working_memory` yet.

- [ ] **Step 4: Update imports and protocol signature**

In `memora/extraction.py`, change the schema import from:

```python
from .schema import MemoryCandidate, SessionMessage, validate_memory_candidate, validate_memory_type
```

to:

```python
from .schema import MemoryCandidate, SessionMessage, WorkingMemoryState, validate_memory_candidate, validate_memory_type
```

Change the `MemoryExtractor` protocol from:

```python
class MemoryExtractor(Protocol):
    def extract(self, messages: Sequence[SessionMessage | Mapping[str, str]]) -> "ExtractionArtifact":
        """Extract candidate memories from conversation messages."""
```

to:

```python
class MemoryExtractor(Protocol):
    def extract(
        self,
        messages: Sequence[SessionMessage | Mapping[str, str]],
        working_memory: WorkingMemoryState | Mapping[str, object] | None = None,
    ) -> "ExtractionArtifact":
        """Extract candidate memories from conversation messages and optional working memory."""
```

- [ ] **Step 5: Replace `EXTRACTION_SYSTEM_PROMPT` with expanded rules**

In `memora/extraction.py`, replace the current `EXTRACTION_SYSTEM_PROMPT` string with this content:

```python
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
not itself long-term memory. Do not directly memorize current_goal, next_step,
open_questions, or recent_files unless they capture a durable project direction
or important decision. Never memorize raw logs, raw stdout/stderr, stack traces,
or transient task progress from working_memory_snapshot.

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
```

- [ ] **Step 6: Add private formatting helpers**

In `memora/extraction.py`, add these helpers above `extraction_prompt_messages(...)`:

```python
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
```

- [ ] **Step 7: Extend `extraction_prompt_messages(...)`**

Replace the existing function signature and body with:

```python
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
```

- [ ] **Step 8: Run tests to verify Task 1 passes**

Run:

```bash
pytest tests/test_extraction.py -q
```

Expected: all extraction tests pass, except failures related to `LLMMemoryExtractor.extract(..., working_memory=...)` not yet implemented if Task 2 tests have already been added.

- [ ] **Step 9: Commit Task 1**

```bash
git add memora/extraction.py tests/test_extraction.py
git commit -m "feat: render working memory in extraction prompts"
```

---

### Task 2: Extractor API Compatibility

**Files:**
- Modify: `memora/extraction.py`
- Test: `tests/test_extraction.py`

**Interfaces:**
- Consumes:
  - `extraction_prompt_messages(messages, working_memory=None)` from Task 1.
- Produces:
  - `LLMMemoryExtractor.extract(messages: Sequence[SessionMessage | Mapping[str, str]], working_memory: WorkingMemoryState | Mapping[str, object] | None = None) -> ExtractionArtifact`

- [ ] **Step 1: Write failing test for `LLMMemoryExtractor.extract` working-memory forwarding**

Add this test to `tests/test_extraction.py`:

```python
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
    state = WorkingMemoryState(process_notes=["Working memory should be evidence, not direct long-term memory."])

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
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_extraction.py::test_llm_memory_extractor_accepts_working_memory -q
```

Expected: FAIL with `TypeError` because `LLMMemoryExtractor.extract()` does not accept `working_memory` yet.

- [ ] **Step 3: Update `LLMMemoryExtractor.extract` signature and call**

In `memora/extraction.py`, replace:

```python
class LLMMemoryExtractor:
    def __init__(self, client: LLMClient):
        self.client = client

    def extract(self, messages: Sequence[SessionMessage | Mapping[str, str]]) -> ExtractionArtifact:
        raw_text = self.client.complete(extraction_prompt_messages(messages))
        return parse_extraction_json(raw_text)
```

with:

```python
class LLMMemoryExtractor:
    def __init__(self, client: LLMClient):
        self.client = client

    def extract(
        self,
        messages: Sequence[SessionMessage | Mapping[str, str]],
        working_memory: WorkingMemoryState | Mapping[str, object] | None = None,
    ) -> ExtractionArtifact:
        raw_text = self.client.complete(extraction_prompt_messages(messages, working_memory=working_memory))
        return parse_extraction_json(raw_text)
```

- [ ] **Step 4: Run tests to verify Task 2 passes**

Run:

```bash
pytest tests/test_extraction.py -q
```

Expected: all extraction tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add memora/extraction.py tests/test_extraction.py
git commit -m "feat: pass working memory to LLM extractor"
```

---

### Task 3: Runtime Forwarding Without Write-Pipeline Changes

**Files:**
- Modify: `memora/runtime.py`
- Test: `tests/test_runtime.py`

**Interfaces:**
- Consumes:
  - `MemoryExtractor.extract(messages, working_memory=None)` protocol from Task 1.
  - `LLMMemoryExtractor.extract(messages, working_memory=None)` from Task 2.
- Produces:
  - `MemoryRuntime.extract_memories(messages, extractor=None, working_memory=None) -> ExtractionArtifact`
  - `MemoryRuntime.extract_and_remember(..., extractor=None, working_memory=None) -> tuple[ExtractionArtifact, list[MemoryWriteResult]]`

- [ ] **Step 1: Write fake extractor test double that records working memory**

In `tests/test_runtime.py`, add or update a fake extractor class:

```python
class RecordingExtractor:
    def __init__(self, artifact: ExtractionArtifact):
        self.artifact = artifact
        self.messages = None
        self.working_memory = None

    def extract(self, messages, working_memory=None):
        self.messages = messages
        self.working_memory = working_memory
        return self.artifact
```

If `tests/test_runtime.py` already has fake extractors, prefer adding this focused class instead of changing all existing fakes.

- [ ] **Step 2: Write failing test for `extract_memories(..., working_memory=...)` forwarding**

Add this test to `tests/test_runtime.py`:

```python
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
```

Ensure these imports exist in `tests/test_runtime.py`:

```python
from memora.extraction import ExtractionArtifact, ExtractedMemory
from memora.schema import SessionMessage, WorkingMemoryState
```

If `ExtractionArtifact`, `ExtractedMemory`, or `SessionMessage` are already imported, extend the existing import lines.

- [ ] **Step 3: Write failing test for `extract_and_remember(..., working_memory=...)` forwarding**

Add this test to `tests/test_runtime.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they fail**

Run:

```bash
pytest tests/test_runtime.py::test_extract_memories_forwards_working_memory_to_extractor tests/test_runtime.py::test_extract_and_remember_forwards_working_memory -q
```

Expected: FAIL with `TypeError` because runtime methods do not accept `working_memory` yet.

- [ ] **Step 5: Update imports in `memora/runtime.py`**

Change:

```python
from .schema import MemoryCandidate, MemoryItem, MemorySearchResult, MemoryWriteResult, SessionMessage
```

to:

```python
from .schema import MemoryCandidate, MemoryItem, MemorySearchResult, MemoryWriteResult, SessionMessage, WorkingMemoryState
```

- [ ] **Step 6: Update `MemoryRuntime.extract_memories`**

Replace the method with:

```python
    def extract_memories(
        self,
        messages: list[SessionMessage | dict[str, str]],
        extractor: MemoryExtractor | None = None,
        working_memory: WorkingMemoryState | dict[str, object] | None = None,
    ) -> ExtractionArtifact:
        selected_extractor = extractor or self.extractor
        if selected_extractor is None:
            return ExtractionArtifact(
                should_remember=False,
                memories=[],
                errors=["memory_extractor_not_configured"],
                source="not_configured",
            )
        return selected_extractor.extract(messages, working_memory=working_memory)
```

- [ ] **Step 7: Update `MemoryRuntime.extract_and_remember`**

Add a parameter before `extractor` or after it. Prefer after `extractor` to avoid disturbing existing positional calls:

```python
    def extract_and_remember(
        self,
        messages: list[SessionMessage | dict[str, str]],
        user_id: str = "default",
        project_id: str | None = None,
        workspace_id: str | None = None,
        session_id: str | None = None,
        extractor: MemoryExtractor | None = None,
        working_memory: WorkingMemoryState | dict[str, object] | None = None,
    ) -> tuple[ExtractionArtifact, list[MemoryWriteResult]]:
        artifact = self.extract_memories(messages, extractor=extractor, working_memory=working_memory)
        results = self.remember_extraction_artifact(
            artifact,
            user_id=user_id,
            project_id=project_id,
            workspace_id=workspace_id,
            session_id=session_id,
        )
        return artifact, results
```

- [ ] **Step 8: Run runtime tests**

Run:

```bash
pytest tests/test_runtime.py -q
```

Expected: all runtime tests pass.

- [ ] **Step 9: Run extraction + runtime tests together**

Run:

```bash
pytest tests/test_extraction.py tests/test_runtime.py -q
```

Expected: all tests pass.

- [ ] **Step 10: Commit Task 3**

```bash
git add memora/runtime.py tests/test_runtime.py
git commit -m "feat: forward working memory during extraction"
```

---

### Task 4: Full Regression and Documentation Touch-Up

**Files:**
- Modify: `README.md` if it documents extraction input boundaries.
- Test: full test suite.

**Interfaces:**
- Consumes:
  - Optional working-memory extraction input from Tasks 1-3.
- Produces:
  - Passing test suite.
  - README documentation if existing docs mention session extraction input.

- [ ] **Step 1: Check README for extraction boundary wording**

Read the README sections around memory extraction and write-time relation flow. Look for current wording that says extraction only uses messages. If no such wording exists, do not edit README.

- [ ] **Step 2: If README needs an update, add this wording near extraction documentation**

Use this wording if the README has an extraction input section:

```markdown
Extraction can receive conversation messages plus an optional `WorkingMemoryState` snapshot. Conversation messages are treated as direct evidence; working memory is treated as agent-maintained short-term state that may contain durable conclusions, reusable lessons, tool-use lessons, or important decisions. The extractor still returns the same `ExtractionArtifact` JSON contract, and policy/write behavior is unchanged.
```

- [ ] **Step 3: Run targeted tests**

Run:

```bash
pytest tests/test_extraction.py tests/test_runtime.py -q
```

Expected: PASS.

- [ ] **Step 4: Run full test suite**

Run:

```bash
pytest -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 4 if README changed**

If README changed, run:

```bash
git add README.md
git commit -m "docs: document working memory extraction input"
```

If README did not change, do not create an empty commit.

---

## Self-Review

**Spec coverage:**
- Optional working-memory input is covered by Tasks 1-3.
- Prompt routing rules for seven memory types and working-memory fields are covered by Task 1.
- No write-pipeline changes are covered by Global Constraints and Task 3 scope.
- Backward compatibility is covered by Task 1 no-working-memory test and optional parameters.
- Runtime forwarding is covered by Task 3.
- Documentation is covered by Task 4.

**Placeholder scan:**
- No `TBD`, `TODO`, "similar to", or undefined implementation placeholders remain.
- Code snippets include exact signatures and expected assertions.

**Type consistency:**
- `WorkingMemoryState | Mapping[str, object] | None` is used in `memora/extraction.py`.
- `WorkingMemoryState | dict[str, object] | None` is used in `memora/runtime.py` because runtime public input commonly receives dicts.
- `ExtractionArtifact`, `ExtractedMemory`, `SessionMessage`, and `WorkingMemoryState` names match existing modules.
