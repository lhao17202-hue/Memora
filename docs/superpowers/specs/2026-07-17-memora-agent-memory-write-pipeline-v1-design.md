# Memora Agent Memory Write Pipeline v1 Design

## Goal

Build the first agent-facing memory write pipeline for Memora: external RAG / LLM agents extract candidate memories, then Memora deterministically validates, evaluates, writes, updates, rejects, or asks for confirmation through a structured result API.

This moves Memora from a manual save/search library toward an agent memory subsystem while preserving the project boundary: Memora does not call an LLM internally.

## Context

Memora already has:

- Markdown memory files with YAML frontmatter.
- `MemoryCandidate`, `MemoryItem`, and validation helpers.
- `MemoryPolicy` for deterministic reject/update/ask/create decisions.
- `MemoryManager.save_memory()` for direct manual memory creation.
- `MemoryRuntime` for external runtime integration.
- Session history as source material for external agents.
- Deterministic retrieval and prompt formatting.

The missing piece is a standard write contract for external agents. Today an agent must call `save_memory(...)` directly and handle policy rejections through exceptions. That is too low-level for a runtime integration where policy decisions are expected outcomes, not exceptional failures.

## Design Principles

- Memora remains independent, local, deterministic, and testable.
- Memora does not perform LLM extraction.
- External agents may use RAG / LLM logic to produce candidate memories.
- Memora owns validation, policy evaluation, persistence decisions, and result reporting.
- Policy rejection and confirmation-required outcomes are normal structured results for agent workflows.
- Existing `save_memory()` behavior remains intact for manual/direct usage.
- No new dependencies.
- No embeddings, vector database, cloud sync, encryption, or automatic summarization in this feature.

## User-Facing Behavior

External runtime flow:

```text
Agent / LLM extracts candidate memory
→ Agent passes candidate to Memora
→ Memora validates candidate
→ Memora evaluates policy
→ Memora creates, updates, rejects, or requires confirmation
→ Agent receives structured result
```

CLI debugging flow:

```bash
python -m memora --root .memora remember \
  --type user \
  --name language \
  --description "用户偏好中文。" \
  --content "用户偏好使用中文回答。" \
  --session session_1 \
  --tag preference
```

Expected outputs:

```text
created mem_xxx language accepted
updated mem_xxx language duplicate_or_same_key
rejected contains_secret
requires_confirmation mem_xxx conflict_requires_confirmation
```

The `remember` command represents an agent candidate-memory write. The existing `save` command remains a direct manual save command.

## Data Model

Add a structured result dataclass in `memora/schema.py`:

```python
@dataclass
class MemoryWriteResult:
    action: str
    memory: MemoryItem | None = None
    candidate: MemoryCandidate | None = None
    reason: str = ""
    target_memory_id: str | None = None
```

Allowed `action` values for this feature:

- `created`
- `updated`
- `rejected`
- `requires_confirmation`

`memory` is populated for `created` and `updated`.

`candidate` is populated for outcomes where useful to the caller, especially `rejected` and `requires_confirmation`.

`reason` mirrors the deterministic policy reason such as:

- `accepted`
- `duplicate_or_same_key`
- `contains_secret`
- `transient_task_state`
- `noisy_output`
- `conflict_requires_confirmation`

`target_memory_id` is populated when policy identified an existing related memory, such as duplicate update or conflict confirmation.

## Manager API

Add two public methods to `MemoryManager`.

### `evaluate_memory_candidate(candidate)`

Signature:

```python
def evaluate_memory_candidate(self, candidate: MemoryCandidate) -> MemoryWriteResult:
    ...
```

Behavior:

1. Validate the candidate with `validate_memory_candidate(candidate)`.
2. Evaluate it with `MemoryPolicy.evaluate(...)` against active existing memories.
3. Do not write any files.
4. Return `MemoryWriteResult`:
   - policy `create` → `action="created"`, no memory yet, `reason="accepted"`
   - policy `update` → `action="updated"`, no memory yet, `target_memory_id` populated
   - policy `reject` → `action="rejected"`
   - policy `ask_user` → `action="requires_confirmation"`, `target_memory_id` populated

The `created` / `updated` names are used because they describe the action that would happen if the candidate is committed. The method itself remains read-only.

Validation failures still raise `MemoryValidationError`, because malformed input is not a policy decision.

### `remember_candidate(candidate)`

Signature:

```python
def remember_candidate(self, candidate: MemoryCandidate) -> MemoryWriteResult:
    ...
```

Behavior:

1. Validate candidate.
2. Evaluate policy.
3. If policy returns `create`:
   - Create a new `MemoryItem` using the same field mapping as `save_memory()`.
   - Persist with `memory_store.save_memory(...)`.
   - Return `MemoryWriteResult(action="created", memory=item, candidate=decision, reason=decision.reason)`.
4. If policy returns `update`:
   - Load `target_memory_id`.
   - Update description, content, tags, weight, confidence, source, and `updated_at`.
   - Persist with `memory_store.update_memory(...)`.
   - Return `MemoryWriteResult(action="updated", memory=item, candidate=decision, reason=decision.reason, target_memory_id=item.id)`.
5. If policy returns `reject`:
   - Do not write.
   - Return `MemoryWriteResult(action="rejected", candidate=decision, reason=decision.reason)`.
6. If policy returns `ask_user`:
   - Do not write.
   - Return `MemoryWriteResult(action="requires_confirmation", candidate=decision, reason=decision.reason, target_memory_id=decision.target_memory_id)`.

Policy rejection does not raise `MemoryPolicyError` in this method. It is a normal agent-facing result.

Implementation should avoid duplicating large blocks from `save_memory()` where practical. If helper extraction is small and clear, introduce a private helper for applying create/update decisions.

## Runtime API

Add `remember_extracted(...)` to `MemoryRuntime`:

```python
def remember_extracted(
    self,
    memory_type: str,
    name: str,
    description: str,
    content: str,
    user_id: str = "default",
    project_id: str | None = None,
    workspace_id: str | None = None,
    session_id: str | None = None,
    tags: list[str] | None = None,
    weight: int = 5,
    confidence: float = 1.0,
) -> MemoryWriteResult:
    ...
```

Behavior:

- Build a `MemoryCandidate` with `action="create"`.
- Use `source="session_extraction"` when `session_id` is provided.
- Use `source="runtime_extraction"` when `session_id` is not provided.
- If `session_id` is provided, append deterministic tag `session:<session_id>` unless already present.
- Call `manager.remember_candidate(candidate)`.
- Return the resulting `MemoryWriteResult`.

This method is the primary simple integration point for external RAG / LLM agents.

## CLI API

Add top-level command `remember` to `memora/cli.py`.

Arguments:

```text
--type          required memory type
--name          required candidate name
--description   required candidate description
--content       required candidate content
--source        optional source, default runtime_extraction
--session       optional session id; when present, source defaults to session_extraction unless --source is explicitly set
--tag           repeatable tag
--weight        optional integer weight
--confidence    optional float confidence
```

Output format:

```text
created <memory_id> <memory_name> <reason>
updated <memory_id> <memory_name> <reason>
rejected <reason>
requires_confirmation <target_memory_id> <reason>
```

Exit behavior:

- Policy outcomes `created`, `updated`, `rejected`, and `requires_confirmation` exit `0`.
- Malformed candidate validation exits `1` through existing `MemoraError` handling.

## Session Relationship

This feature does not add full session CRUD.

Session is only used as source context for candidate memories:

- `session_id` can be encoded as tag `session:<session_id>`.
- `source` can be `session_extraction`.
- Actual extraction is external to Memora.

This keeps session scoped to the memory system purpose instead of turning Memora into a chat log product.

## README Updates

Document a new section explaining:

- Memora does not call LLMs to extract memory.
- External agents extract candidate memories.
- Memora deterministically validates and writes candidates.
- `remember` is the CLI debug path for this pipeline.
- `MemoryRuntime.remember_extracted(...)` is the Python runtime integration path.

Example:

```python
from memora.runtime import MemoryRuntime

runtime = MemoryRuntime()
runtime.init_storage()

result = runtime.remember_extracted(
    memory_type="user",
    name="language",
    description="用户偏好中文。",
    content="用户偏好使用中文回答。",
    session_id="session_1",
)
print(result.action, result.reason)
```

## Tests

Add or update tests for:

### Manager

- Accepted candidate returns `created` and persists memory.
- Duplicate candidate returns `updated` and updates existing memory.
- Secret candidate returns `rejected` and does not persist memory.
- Conflict candidate returns `requires_confirmation` and does not write.
- `evaluate_memory_candidate(...)` returns a decision without creating files.

### Runtime

- `remember_extracted(...)` creates memory.
- `remember_extracted(..., session_id="session_1")` uses `source="session_extraction"` and tag `session:session_1`.
- Secret extracted memory returns `rejected` without raising `MemoryPolicyError`.

### CLI

- `remember` creates memory and prints `created`.
- Repeating `remember` with the same name prints `updated`.
- Secret candidate prints `rejected` and exits `0`.
- Invalid memory type exits `1` and prints `error:`.

### Full Suite

Run:

```bash
pytest -v
```

## Non-Goals

This feature does not implement:

- LLM extraction inside Memora.
- Embeddings.
- Vector databases.
- Hybrid semantic retrieval.
- Full session CRUD.
- Full backup including sessions.
- Automatic user confirmation.
- Automatic summarization.
- New dependencies.
- Changes to existing `save_memory()` policy-error semantics.
