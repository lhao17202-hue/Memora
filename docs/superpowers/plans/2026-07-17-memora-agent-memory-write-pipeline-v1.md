# Memora Agent Memory Write Pipeline v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an agent-facing memory write pipeline so external RAG / LLM agents can submit candidate memories and receive deterministic structured write decisions.

**Architecture:** Add `MemoryWriteResult` to the schema, then expose non-throwing candidate evaluation/write methods on `MemoryManager`. `MemoryRuntime.remember_extracted(...)` becomes the simple external-agent integration point, and a top-level CLI `remember` command provides local debugging for the same pipeline.

**Tech Stack:** Python standard library, dataclasses, existing Memora file store, existing `MemoryPolicy`, pytest, no new dependencies.

## Global Constraints

- Memora remains independent, local, deterministic, and testable.
- Memora does not perform LLM extraction.
- External agents may use RAG / LLM logic to produce candidate memories.
- Memora owns validation, policy evaluation, persistence decisions, and result reporting.
- Policy rejection and confirmation-required outcomes are normal structured results for agent workflows.
- Existing `save_memory()` behavior remains intact for manual/direct usage.
- No new dependencies.
- No embeddings, vector database, cloud sync, encryption, or automatic summarization in this feature.
- This feature does not add full session CRUD.
- Session is only used as source context for candidate memories through `source="session_extraction"` and tag `session:<session_id>`.

---

## File Structure

- Modify `memora/schema.py`: add `MemoryWriteResult` dataclass near `MemorySearchResult`.
- Modify `memora/manager.py`: import `MemoryWriteResult`; add private candidate decision helpers; add `evaluate_memory_candidate(...)` and `remember_candidate(...)`.
- Modify `memora/runtime.py`: import `MemoryCandidate` and `MemoryWriteResult`; add `remember_extracted(...)`.
- Modify `memora/cli.py`: add top-level `remember` parser and handler.
- Modify `tests/test_manager.py`: add manager-level pipeline tests.
- Modify `tests/test_runtime.py`: add runtime-level pipeline tests.
- Modify `tests/test_cli.py`: add CLI `remember` tests.
- Modify `README.md`: document the agent memory write pipeline.

---

### Task 1: Schema and Manager Candidate Pipeline

**Files:**
- Modify: `memora/schema.py`
- Modify: `memora/manager.py`
- Test: `tests/test_manager.py`

**Interfaces:**
- Consumes existing:
  - `MemoryCandidate`
  - `MemoryItem`
  - `MemoryPolicy.evaluate(candidate, existing)`
  - `validate_memory_candidate(candidate)`
  - `validate_memory_item(item)`
  - `FileMemoryStore.save_memory(item)`
  - `FileMemoryStore.update_memory(item)`
- Produces:
  - `MemoryWriteResult(action: str, memory: MemoryItem | None = None, candidate: MemoryCandidate | None = None, reason: str = "", target_memory_id: str | None = None)`
  - `MemoryManager.evaluate_memory_candidate(candidate: MemoryCandidate) -> MemoryWriteResult`
  - `MemoryManager.remember_candidate(candidate: MemoryCandidate) -> MemoryWriteResult`

- [ ] **Step 1: Add failing manager tests**

Append these tests to `tests/test_manager.py`:

```python
from memora.schema import MemoryCandidate
```

If the import section already has `from memora.schema import SessionMessage`, replace it with:

```python
from memora.schema import MemoryCandidate, SessionMessage
```

Then append:

```python
def test_evaluate_memory_candidate_returns_decision_without_writing(tmp_path: Path):
    manager = manager_for(tmp_path)
    manager.init_storage()
    candidate = MemoryCandidate(
        action="create",
        type="user",
        name="language",
        description="用户偏好中文。",
        content="用户偏好使用中文回答。",
    )

    result = manager.evaluate_memory_candidate(candidate)

    assert result.action == "created"
    assert result.memory is None
    assert result.candidate is not None
    assert result.candidate.name == "language"
    assert result.reason == "accepted"
    assert manager.memory_store.list_memories() == []


def test_remember_candidate_creates_memory(tmp_path: Path):
    manager = manager_for(tmp_path)
    manager.init_storage()
    candidate = MemoryCandidate(
        action="create",
        type="user",
        name="language",
        description="用户偏好中文。",
        content="用户偏好使用中文回答。",
    )

    result = manager.remember_candidate(candidate)

    assert result.action == "created"
    assert result.memory is not None
    assert result.memory.name == "language"
    assert result.reason == "accepted"
    assert manager.memory_store.get_memory("language") is not None


def test_remember_candidate_updates_duplicate_memory(tmp_path: Path):
    manager = manager_for(tmp_path)
    manager.init_storage()
    original = manager.save_memory("user", "old content", "old desc", name="language")
    candidate = MemoryCandidate(
        action="create",
        type="user",
        name="language",
        description="new desc",
        content="new content",
        tags=["preference"],
        weight=8,
        confidence=0.7,
        source="session_extraction",
    )

    result = manager.remember_candidate(candidate)

    assert result.action == "updated"
    assert result.memory is not None
    assert result.memory.id == original.id
    assert result.memory.description == "new desc"
    assert result.memory.content == "new content"
    assert result.memory.tags == ["preference"]
    assert result.memory.weight == 8
    assert result.memory.confidence == 0.7
    assert result.memory.source == "session_extraction"
    assert result.reason == "duplicate_or_same_key"
    assert result.target_memory_id == original.id


def test_remember_candidate_rejects_secret_without_raising_policy_error(tmp_path: Path):
    manager = manager_for(tmp_path)
    manager.init_storage()
    candidate = MemoryCandidate(
        action="create",
        type="user",
        name="secret",
        description="secret",
        content="api_key = sk-abcdef123456",
    )

    result = manager.remember_candidate(candidate)

    assert result.action == "rejected"
    assert result.memory is None
    assert result.reason == "contains_secret"
    assert manager.memory_store.get_memory("secret") is None


def test_remember_candidate_reports_conflict_without_writing(tmp_path: Path):
    manager = manager_for(tmp_path)
    manager.init_storage()
    existing = manager.save_memory("user", "用户偏好英文回答。", "用户偏好英文。", name="language-en")
    candidate = MemoryCandidate(
        action="create",
        type="user",
        name="language-zh",
        description="用户偏好中文。",
        content="用户偏好中文回答。",
    )

    result = manager.remember_candidate(candidate)

    assert result.action == "requires_confirmation"
    assert result.memory is None
    assert result.reason == "conflict_requires_confirmation"
    assert result.target_memory_id == existing.id
    assert manager.memory_store.get_memory("language-zh") is None
```

- [ ] **Step 2: Run manager tests to verify they fail**

Run:

```bash
pytest tests/test_manager.py -v
```

Expected: FAIL because `MemoryWriteResult`, `evaluate_memory_candidate`, and `remember_candidate` do not exist yet.

- [ ] **Step 3: Add `MemoryWriteResult` to schema**

In `memora/schema.py`, after `MemorySearchResult`, add:

```python
@dataclass
class MemoryWriteResult:
    action: str
    memory: MemoryItem | None = None
    candidate: MemoryCandidate | None = None
    reason: str = ""
    target_memory_id: str | None = None
```

- [ ] **Step 4: Update manager imports**

In `memora/manager.py`, update the schema import block from:

```python
from .schema import (
    MemoryCandidate,
    MemoryItem,
    MemoryQuery,
    MemorySearchResult,
    SessionMessage,
    validate_memory_candidate,
    validate_memory_item,
    validate_memory_query,
)
```

to:

```python
from .schema import (
    MemoryCandidate,
    MemoryItem,
    MemoryQuery,
    MemorySearchResult,
    MemoryWriteResult,
    SessionMessage,
    validate_memory_candidate,
    validate_memory_item,
    validate_memory_query,
)
```

- [ ] **Step 5: Add manager helper methods and public pipeline methods**

In `memora/manager.py`, inside `class MemoryManager`, after `init_storage()` and before `save_memory(...)`, add:

```python
    def _write_result_from_decision(self, decision: MemoryCandidate, memory: MemoryItem | None = None) -> MemoryWriteResult:
        if decision.action == "create":
            action = "created"
        elif decision.action == "update":
            action = "updated"
        elif decision.action == "reject":
            action = "rejected"
        elif decision.action == "ask_user":
            action = "requires_confirmation"
        else:
            action = decision.action
        return MemoryWriteResult(
            action=action,
            memory=memory,
            candidate=decision,
            reason=decision.reason,
            target_memory_id=decision.target_memory_id,
        )

    def _new_memory_from_candidate(self, decision: MemoryCandidate) -> MemoryItem:
        now = now_utc()
        item = MemoryItem(
            id=f"mem_{uuid.uuid4().hex[:12]}",
            name=slugify(decision.name),
            description=decision.description,
            type=decision.type,
            content=decision.content,
            user_id=decision.user_id,
            project_id=decision.project_id,
            workspace_id=decision.workspace_id,
            tags=decision.tags,
            source=decision.source,
            confidence=decision.confidence,
            weight=decision.weight,
            created_at=now,
            updated_at=now,
        )
        validate_memory_item(item)
        return item

    def _apply_candidate_update(self, existing: MemoryItem, decision: MemoryCandidate) -> MemoryItem:
        existing.description = decision.description
        existing.content = decision.content
        existing.tags = decision.tags
        existing.weight = decision.weight
        existing.confidence = decision.confidence
        existing.source = decision.source
        existing.updated_at = now_utc()
        validate_memory_item(existing)
        return self.memory_store.update_memory(existing)

    def evaluate_memory_candidate(self, candidate: MemoryCandidate) -> MemoryWriteResult:
        validate_memory_candidate(candidate)
        decision = self.policy.evaluate(candidate, self.memory_store.list_memories(include_archived=False))
        return self._write_result_from_decision(decision)

    def remember_candidate(self, candidate: MemoryCandidate) -> MemoryWriteResult:
        validate_memory_candidate(candidate)
        decision = self.policy.evaluate(candidate, self.memory_store.list_memories(include_archived=False))
        if decision.action == "create":
            item = self._new_memory_from_candidate(decision)
            saved = self.memory_store.save_memory(item)
            return self._write_result_from_decision(decision, memory=saved)
        if decision.action == "update" and decision.target_memory_id:
            existing = self.memory_store.get_memory(decision.target_memory_id)
            if existing is None:
                raise MemoryNotFoundError(f"memory not found: {decision.target_memory_id}")
            updated = self._apply_candidate_update(existing, decision)
            return self._write_result_from_decision(decision, memory=updated)
        return self._write_result_from_decision(decision)
```

- [ ] **Step 6: Refactor `save_memory()` to reuse helper creation/update code**

In `memora/manager.py`, inside `save_memory(...)`, replace the duplicate update block:

```python
            existing.description = description
            existing.content = content
            existing.tags = tags or []
            existing.weight = weight
            existing.confidence = confidence
            existing.source = source
            existing.updated_at = now
            validate_memory_item(existing)
            return self.memory_store.update_memory(existing)
```

with:

```python
            return self._apply_candidate_update(existing, candidate)
```

Then replace the new `MemoryItem(...)` construction block:

```python
        item = MemoryItem(
            id=f"mem_{uuid.uuid4().hex[:12]}",
            name=slugify(decision.name),
            description=description,
            type=memory_type,
            content=content,
            user_id=user_id,
            project_id=project_id,
            workspace_id=workspace_id,
            tags=tags or [],
            source=source,
            confidence=confidence,
            weight=weight,
            created_at=now,
            updated_at=now,
        )
        validate_memory_item(item)
        return self.memory_store.save_memory(item)
```

with:

```python
        item = self._new_memory_from_candidate(decision)
        return self.memory_store.save_memory(item)
```

Also remove the now-unused local line:

```python
        now = now_utc()
```

from `save_memory(...)` if it is no longer referenced.

- [ ] **Step 7: Run manager tests to verify they pass**

Run:

```bash
pytest tests/test_manager.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit Task 1**

Run:

```bash
git add memora/schema.py memora/manager.py tests/test_manager.py
git commit -m "feat: add agent memory write manager API" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Runtime Integration API

**Files:**
- Modify: `memora/runtime.py`
- Test: `tests/test_runtime.py`

**Interfaces:**
- Consumes from Task 1:
  - `MemoryWriteResult`
  - `MemoryCandidate`
  - `MemoryManager.remember_candidate(candidate: MemoryCandidate) -> MemoryWriteResult`
- Produces:
  - `MemoryRuntime.remember_extracted(...) -> MemoryWriteResult`

- [ ] **Step 1: Add failing runtime tests**

Append these tests to `tests/test_runtime.py`:

```python
def test_remember_extracted_creates_memory(tmp_path: Path):
    runtime = make_runtime(tmp_path)

    result = runtime.remember_extracted(
        memory_type="user",
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
        memory_type="user",
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


def test_remember_extracted_rejects_secret_without_policy_exception(tmp_path: Path):
    runtime = make_runtime(tmp_path)

    result = runtime.remember_extracted(
        memory_type="user",
        name="secret",
        description="secret",
        content="api_key = sk-abcdef123456",
    )

    assert result.action == "rejected"
    assert result.reason == "contains_secret"
    assert result.memory is None
```

- [ ] **Step 2: Run runtime tests to verify they fail**

Run:

```bash
pytest tests/test_runtime.py -v
```

Expected: FAIL because `MemoryRuntime.remember_extracted` does not exist yet.

- [ ] **Step 3: Update runtime imports**

In `memora/runtime.py`, change:

```python
from .schema import MemoryItem, MemorySearchResult, SessionMessage
```

to:

```python
from .schema import MemoryCandidate, MemoryItem, MemorySearchResult, MemoryWriteResult, SessionMessage
```

- [ ] **Step 4: Add `remember_extracted(...)` to `MemoryRuntime`**

In `memora/runtime.py`, inside `class MemoryRuntime`, after `remember_summary(...)` and before `mark_context_used(...)`, add:

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
        candidate_tags = list(tags or [])
        source = "runtime_extraction"
        if session_id is not None:
            source = "session_extraction"
            session_tag = f"session:{session_id}"
            if session_tag not in candidate_tags:
                candidate_tags.append(session_tag)
        candidate = MemoryCandidate(
            action="create",
            name=name,
            description=description,
            type=memory_type,
            content=content,
            user_id=user_id,
            project_id=project_id,
            workspace_id=workspace_id,
            tags=candidate_tags,
            source=source,
            confidence=confidence,
            weight=weight,
        )
        return self.manager.remember_candidate(candidate)
```

- [ ] **Step 5: Run runtime tests to verify they pass**

Run:

```bash
pytest tests/test_runtime.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit Task 2**

Run:

```bash
git add memora/runtime.py tests/test_runtime.py
git commit -m "feat: add runtime extracted memory writes" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: CLI `remember` Command

**Files:**
- Modify: `memora/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes from Task 1:
  - `MemoryCandidate`
  - `MemoryManager.remember_candidate(candidate: MemoryCandidate) -> MemoryWriteResult`
- Produces:
  - CLI command `python -m memora --root .memora remember ...`

- [ ] **Step 1: Add failing CLI tests**

Append these tests to `tests/test_cli.py`:

```python
def test_remember_command_creates_and_updates_candidate_memory(tmp_path: Path):
    root = tmp_path / ".memora"

    created = run_cli(
        root,
        "remember",
        "--type",
        "user",
        "--name",
        "language",
        "--description",
        "用户偏好中文。",
        "--content",
        "用户偏好使用中文回答。",
        "--session",
        "session_1",
        "--tag",
        "preference",
    )
    updated = run_cli(
        root,
        "remember",
        "--type",
        "user",
        "--name",
        "language",
        "--description",
        "updated desc",
        "--content",
        "updated content",
    )
    shown = run_cli(root, "show", "language")

    assert created.returncode == 0
    assert "created" in created.stdout
    assert "accepted" in created.stdout
    assert updated.returncode == 0
    assert "updated" in updated.stdout
    assert "duplicate_or_same_key" in updated.stdout
    assert "updated content" in shown.stdout


def test_remember_command_rejects_secret_as_normal_policy_result(tmp_path: Path):
    root = tmp_path / ".memora"

    result = run_cli(
        root,
        "remember",
        "--type",
        "user",
        "--name",
        "secret",
        "--description",
        "secret",
        "--content",
        "api_key = sk-abcdef123456",
    )
    listed = run_cli(root, "list", "--all")

    assert result.returncode == 0
    assert "rejected" in result.stdout
    assert "contains_secret" in result.stdout
    assert "secret" not in listed.stdout


def test_remember_command_invalid_type_reports_cli_error(tmp_path: Path):
    root = tmp_path / ".memora"

    result = run_cli(
        root,
        "remember",
        "--type",
        "invalid",
        "--name",
        "bad",
        "--description",
        "bad",
        "--content",
        "bad",
    )

    assert result.returncode == 1
    assert "error:" in result.stderr
    assert "memory type" in result.stderr
```

- [ ] **Step 2: Run CLI tests to verify they fail**

Run:

```bash
pytest tests/test_cli.py -v
```

Expected: FAIL because `remember` is not a recognized command.

- [ ] **Step 3: Update CLI imports**

In `memora/cli.py`, change:

```python
from .schema import SessionMessage
```

to:

```python
from .schema import MemoryCandidate, SessionMessage
```

- [ ] **Step 4: Add parser for top-level `remember` command**

In `memora/cli.py`, inside `build_parser()`, after the `save_parser` block and before `list_parser`, add:

```python
    remember_parser = subparsers.add_parser("remember", help="Evaluate and write an agent-extracted candidate memory.")
    remember_parser.add_argument("--type", required=True)
    remember_parser.add_argument("--name", required=True)
    remember_parser.add_argument("--description", required=True)
    remember_parser.add_argument("--content", required=True)
    remember_parser.add_argument("--source")
    remember_parser.add_argument("--session", dest="session_id")
    remember_parser.add_argument("--tag", action="append", dest="tags")
    remember_parser.add_argument("--weight", type=int, default=5)
    remember_parser.add_argument("--confidence", type=float, default=1.0)
```

- [ ] **Step 5: Add CLI output helper**

In `memora/cli.py`, after `main(...)` and before `_run_command(...)`, add:

```python
def _print_write_result(result) -> None:
    if result.action in {"created", "updated"} and result.memory is not None:
        print(f"{result.action} {result.memory.id} {result.memory.name} {result.reason}")
        return
    if result.action == "requires_confirmation":
        print(f"requires_confirmation {result.target_memory_id} {result.reason}")
        return
    print(f"{result.action} {result.reason}")
```

- [ ] **Step 6: Add remember command handler**

In `memora/cli.py`, inside `_run_command(...)`, after the existing `if args.command == "save":` block and before `if args.command == "list":`, add:

```python
    if args.command == "remember":
        tags = list(args.tags or [])
        source = args.source
        if args.session_id is not None:
            session_tag = f"session:{args.session_id}"
            if session_tag not in tags:
                tags.append(session_tag)
            if source is None:
                source = "session_extraction"
        if source is None:
            source = "runtime_extraction"
        candidate = MemoryCandidate(
            action="create",
            name=args.name,
            description=args.description,
            type=args.type,
            content=args.content,
            tags=tags,
            source=source,
            weight=args.weight,
            confidence=args.confidence,
        )
        result = manager.remember_candidate(candidate)
        _print_write_result(result)
        return 0
```

- [ ] **Step 7: Run CLI tests to verify they pass**

Run:

```bash
pytest tests/test_cli.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit Task 3**

Run:

```bash
git add memora/cli.py tests/test_cli.py
git commit -m "feat: add remember CLI command" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Documentation and Full Verification

**Files:**
- Modify: `README.md`
- Test: full pytest suite

**Interfaces:**
- Consumes from Tasks 1-3:
  - `MemoryRuntime.remember_extracted(...)`
  - CLI `remember`
  - `MemoryWriteResult.action` and `MemoryWriteResult.reason`
- Produces:
  - README docs for agent memory write pipeline.

- [ ] **Step 1: Update README CLI quickstart**

In `README.md`, in the CLI quickstart block, after the existing `save` example line:

```bash
python -m memora --root .memora save --type user --name language --description "用户偏好中文。" --content "用户偏好使用中文回答。"
```

add:

```bash
python -m memora --root .memora remember --type user --name language --description "用户偏好中文。" --content "用户偏好使用中文回答。" --session session_1 --tag preference
```

- [ ] **Step 2: Add README section for agent memory write pipeline**

In `README.md`, after the `## Runtime integration` section code block and before `## Agent runtime demo`, add:

````markdown
## Agent memory write pipeline

Memora does not call an LLM to extract memories. External RAG or LLM agent runtimes can extract candidate memories, then pass those candidates to Memora for deterministic validation, policy evaluation, writing, updating, rejection, or confirmation handling.

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

For CLI debugging, use `remember` to simulate an agent-extracted candidate memory:

```bash
python -m memora --root .memora remember --type user --name language --description "用户偏好中文。" --content "用户偏好使用中文回答。" --session session_1
```

Policy outcomes such as `rejected` and `requires_confirmation` are returned as normal write results for agent workflows.
````

- [ ] **Step 3: Run focused tests**

Run:

```bash
pytest tests/test_manager.py tests/test_runtime.py tests/test_cli.py -v
```

Expected: PASS.

- [ ] **Step 4: Run full test suite**

Run:

```bash
pytest -v
```

Expected: PASS with all tests.

- [ ] **Step 5: Commit Task 4**

Run:

```bash
git add README.md
git commit -m "docs: document agent memory write pipeline" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Final Verification

After all tasks are complete, run:

```bash
pytest -v
```

Expected: all tests pass.

Then run:

```bash
python -m memora --root .memora-demo-agent-write remember --type user --name language --description "用户偏好中文。" --content "用户偏好使用中文回答。" --session session_1
python -m memora --root .memora-demo-agent-write remember --type user --name secret --description "secret" --content "api_key = sk-abcdef123456"
```

Expected first command prints `created ... accepted` and second command prints `rejected contains_secret` with exit code `0`.
