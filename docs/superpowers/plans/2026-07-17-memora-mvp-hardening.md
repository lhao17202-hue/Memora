# Memora MVP Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the existing Memora deterministic MVP by adding runtime schema validation, typed project errors, robust store boundary handling, and reliable CLI error reporting.

**Architecture:** Keep the current lightweight dataclass-based core. Add explicit validation functions in `memora/schema.py`, use existing exceptions from `memora/errors.py`, validate store inputs and parsed records at file boundaries, and wrap CLI command execution with Memora-specific error handling.

**Tech Stack:** Python 3.11+, dataclasses, PyYAML, argparse, pytest.

## Global Constraints

- Do not introduce Pydantic or any new validation dependency.
- Preserve the existing Markdown/YAML memory file format.
- Preserve existing public APIs unless this plan explicitly changes an error type.
- Keep `delete_memory(identifier, soft_delete=True)` backward-compatible: soft delete archives by setting `status = "archived"`.
- This round does not add JSON CLI output, update/delete/archive/restore CLI commands, LLM extraction, SQLite storage, vector retrieval, Agent runtime integration, or automatic corrupted-file repair.
- All new validation failures must raise Memora project exceptions, not raw `ValueError`.
- Final verification command is `pytest -v`.

---

## File Structure

- Modify `memora/schema.py`: add supported value constants and validation functions for memory items, candidates, queries, and session messages.
- Modify `memora/manager.py`: call schema validation and replace policy `ValueError` with `MemoryPolicyError`.
- Modify `memora/stores.py`: validate parsed/written memory records, validate session IDs, validate session messages, and raise `MemoryValidationError` for invalid datetime metadata.
- Modify `memora/cli.py`: add `MemoraError` handling that prints clear errors to stderr and returns exit code `1`.
- Modify `tests/test_schema.py`: add validation tests.
- Modify `tests/test_manager.py`: assert typed errors from manager.
- Modify `tests/test_stores.py`: assert store robustness boundaries.
- Modify `tests/test_cli.py`: assert CLI error reporting.

---

### Task 1: Schema Validation

**Files:**
- Modify: `memora/schema.py`
- Modify: `tests/test_schema.py`

**Interfaces:**
- Consumes: existing dataclasses `MemoryItem`, `MemoryCandidate`, `MemoryQuery`, `SessionMessage` and exception `MemoryValidationError` from `memora.errors`.
- Produces:
  - `VALID_MEMORY_TYPES: tuple[str, ...]`
  - `VALID_MEMORY_STATUSES: tuple[str, ...]`
  - `VALID_CANDIDATE_ACTIONS: tuple[str, ...]`
  - `VALID_SESSION_ROLES: tuple[str, ...]`
  - `validate_memory_type(value: str) -> None`
  - `validate_memory_status(value: str) -> None`
  - `validate_candidate_action(value: str) -> None`
  - `validate_memory_item(item: MemoryItem) -> None`
  - `validate_memory_candidate(candidate: MemoryCandidate) -> None`
  - `validate_memory_query(query: MemoryQuery) -> None`
  - `validate_session_message(message: SessionMessage) -> None`

- [ ] **Step 1: Add failing schema validation tests**

Append these tests to `tests/test_schema.py`:

```python
import pytest

from memora.errors import MemoryValidationError
from memora.schema import (
    MemoryCandidate,
    MemoryItem,
    MemoryQuery,
    SessionMessage,
    validate_memory_candidate,
    validate_memory_item,
    validate_memory_query,
    validate_session_message,
)


def test_validate_memory_item_rejects_invalid_type():
    item = MemoryItem(id="mem_1", name="language", description="desc", type="invalid", content="content")

    with pytest.raises(MemoryValidationError, match="memory type"):
        validate_memory_item(item)


def test_validate_memory_item_rejects_invalid_status():
    item = MemoryItem(id="mem_1", name="language", description="desc", type="user", content="content", status="bad")

    with pytest.raises(MemoryValidationError, match="memory status"):
        validate_memory_item(item)


def test_validate_memory_item_rejects_invalid_weight_and_confidence():
    overweight = MemoryItem(id="mem_1", name="language", description="desc", type="user", content="content", weight=11)
    overconfident = MemoryItem(id="mem_2", name="style", description="desc", type="user", content="content", confidence=1.1)

    with pytest.raises(MemoryValidationError, match="weight"):
        validate_memory_item(overweight)
    with pytest.raises(MemoryValidationError, match="confidence"):
        validate_memory_item(overconfident)


def test_validate_memory_candidate_rejects_invalid_action():
    candidate = MemoryCandidate(action="bad", name="language", description="desc", type="user", content="content")

    with pytest.raises(MemoryValidationError, match="candidate action"):
        validate_memory_candidate(candidate)


def test_validate_memory_query_rejects_invalid_limits():
    with pytest.raises(MemoryValidationError, match="top_k"):
        validate_memory_query(MemoryQuery(query="中文", top_k=0))
    with pytest.raises(MemoryValidationError, match="max_tokens"):
        validate_memory_query(MemoryQuery(query="中文", max_tokens=0))


def test_validate_session_message_rejects_invalid_role():
    message = SessionMessage(role="invalid", content="hello")

    with pytest.raises(MemoryValidationError, match="session role"):
        validate_session_message(message)
```

- [ ] **Step 2: Run schema tests to verify failure**

Run:

```bash
pytest tests/test_schema.py -v
```

Expected: FAIL because validation functions/constants do not exist yet.

- [ ] **Step 3: Implement validation functions in `memora/schema.py`**

Add the import near the top:

```python
from .errors import MemoryValidationError
```

Add constants after the `Literal` declarations:

```python
VALID_MEMORY_TYPES = (
    "user",
    "feedback",
    "project",
    "decision",
    "entity",
    "session_summary",
    "tool_experience",
    "reference",
    "knowledge",
)

VALID_MEMORY_STATUSES = ("active", "archived", "deleted")

VALID_CANDIDATE_ACTIONS = ("create", "update", "archive", "delete", "reject", "ask_user")

VALID_SESSION_ROLES = ("user", "assistant", "system", "tool")
```

Add helper and validation functions after the dataclasses:

```python
def _require_non_empty_string(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise MemoryValidationError(f"{field_name} must be a non-empty string")


def validate_memory_type(value: str) -> None:
    if value not in VALID_MEMORY_TYPES:
        raise MemoryValidationError(f"invalid memory type: {value}")


def validate_memory_status(value: str) -> None:
    if value not in VALID_MEMORY_STATUSES:
        raise MemoryValidationError(f"invalid memory status: {value}")


def validate_candidate_action(value: str) -> None:
    if value not in VALID_CANDIDATE_ACTIONS:
        raise MemoryValidationError(f"invalid candidate action: {value}")


def _validate_weight(weight: int) -> None:
    if not isinstance(weight, int) or weight < 1 or weight > 10:
        raise MemoryValidationError("weight must be an integer from 1 to 10")


def _validate_confidence(confidence: float) -> None:
    if not isinstance(confidence, int | float) or confidence < 0.0 or confidence > 1.0:
        raise MemoryValidationError("confidence must be from 0.0 to 1.0")


def validate_memory_item(item: MemoryItem) -> None:
    _require_non_empty_string(item.id, "memory id")
    _require_non_empty_string(item.name, "memory name")
    _require_non_empty_string(item.description, "memory description")
    _require_non_empty_string(item.content, "memory content")
    validate_memory_type(item.type)
    validate_memory_status(item.status)
    _validate_weight(item.weight)
    _validate_confidence(item.confidence)
    if item.access_count < 0:
        raise MemoryValidationError("access_count must be >= 0")


def validate_memory_candidate(candidate: MemoryCandidate) -> None:
    validate_candidate_action(candidate.action)
    _require_non_empty_string(candidate.name, "candidate name")
    _require_non_empty_string(candidate.description, "candidate description")
    _require_non_empty_string(candidate.content, "candidate content")
    validate_memory_type(candidate.type)
    _validate_weight(candidate.weight)
    _validate_confidence(candidate.confidence)


def validate_memory_query(query: MemoryQuery) -> None:
    if not isinstance(query.query, str):
        raise MemoryValidationError("query must be a string")
    if query.top_k <= 0:
        raise MemoryValidationError("top_k must be > 0")
    if query.max_tokens <= 0:
        raise MemoryValidationError("max_tokens must be > 0")
    if query.memory_types:
        for memory_type in query.memory_types:
            validate_memory_type(memory_type)


def validate_session_message(message: SessionMessage) -> None:
    if message.role not in VALID_SESSION_ROLES:
        raise MemoryValidationError(f"invalid session role: {message.role}")
    if not isinstance(message.content, str):
        raise MemoryValidationError("session message content must be a string")
```

- [ ] **Step 4: Run schema tests to verify pass**

Run:

```bash
pytest tests/test_schema.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit schema validation**

Run:

```bash
git add memora/schema.py tests/test_schema.py
git commit -m "feat: add schema validation"
```

---

### Task 2: Typed Manager Errors

**Files:**
- Modify: `memora/manager.py`
- Modify: `tests/test_manager.py`

**Interfaces:**
- Consumes: validation functions from Task 1 and existing `MemoryPolicyError`, `MemoryValidationError`.
- Produces: `MemoryManager.save_memory()` raises `MemoryPolicyError` for policy rejections/confirmation-required saves and `MemoryValidationError` for invalid schema input.

- [ ] **Step 1: Add failing manager error tests**

Modify imports in `tests/test_manager.py`:

```python
from memora.errors import MemoryPolicyError, MemoryValidationError
```

Replace `test_policy_rejects_unsafe_save` with:

```python
def test_policy_rejects_unsafe_save(tmp_path: Path):
    manager = manager_for(tmp_path)
    manager.init_storage()

    try:
        manager.save_memory(
            memory_type="user",
            content="api_key = sk-abcdef123456",
            description="secret",
            name="secret",
        )
    except MemoryPolicyError as exc:
        assert "contains_secret" in str(exc)
    else:
        raise AssertionError("expected MemoryPolicyError")
```

Append:

```python
def test_save_memory_rejects_invalid_memory_type(tmp_path: Path):
    manager = manager_for(tmp_path)

    try:
        manager.save_memory("invalid", "content", "description", name="bad")
    except MemoryValidationError as exc:
        assert "memory type" in str(exc)
    else:
        raise AssertionError("expected MemoryValidationError")
```

- [ ] **Step 2: Run manager tests to verify failure**

Run:

```bash
pytest tests/test_manager.py -v
```

Expected: FAIL because manager still raises `ValueError` and does not validate invalid memory types early.

- [ ] **Step 3: Update manager imports and validation calls**

In `memora/manager.py`, add imports:

```python
from .errors import MemoryPolicyError
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

Replace the existing `.schema import ...` line with the expanded import above.

Inside `save_memory()`, after constructing `candidate`, add:

```python
validate_memory_candidate(candidate)
```

Replace policy `ValueError` branches with:

```python
if decision.action == "reject":
    raise MemoryPolicyError(f"memory rejected: {decision.reason}")
if decision.action == "ask_user":
    raise MemoryPolicyError(f"memory requires confirmation: {decision.reason}")
```

Before saving a newly-created `MemoryItem`, add:

```python
validate_memory_item(item)
```

Inside the update branch, before `return self.memory_store.update_memory(existing)`, add:

```python
validate_memory_item(existing)
```

Inside `retrieve_memory()`, after constructing `memory_query`, add:

```python
validate_memory_query(memory_query)
```

- [ ] **Step 4: Run manager tests to verify pass**

Run:

```bash
pytest tests/test_manager.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit typed manager errors**

Run:

```bash
git add memora/manager.py tests/test_manager.py
git commit -m "feat: use typed manager errors"
```

---

### Task 3: Store Robustness

**Files:**
- Modify: `memora/stores.py`
- Modify: `tests/test_stores.py`

**Interfaces:**
- Consumes: `MemoryValidationError`, `validate_memory_item`, `validate_session_message`.
- Produces:
  - Invalid datetimes raise `MemoryValidationError`.
  - Invalid memory records are rejected at read/write boundaries.
  - Invalid session IDs are rejected before path construction.
  - Invalid session messages are rejected before append.

- [ ] **Step 1: Add failing store robustness tests**

Append to `tests/test_stores.py`:

```python
import pytest

from memora.errors import MemoryValidationError
```

If these imports already exist, do not duplicate them.

Append tests:

```python
def test_memory_store_rejects_invalid_datetime_frontmatter(tmp_path: Path):
    store = FileMemoryStore(MemoryConfig(root_dir=str(tmp_path / ".memora")))
    store.init_storage()
    path = store.memories_dir / "bad.md"
    path.write_text(
        "---\n"
        "name: bad\n"
        "description: bad datetime\n"
        "metadata:\n"
        "  id: mem_bad\n"
        "  type: project\n"
        "  status: active\n"
        "  weight: 5\n"
        "  confidence: 1.0\n"
        "  created_at: not-a-date\n"
        "---\n\n"
        "content\n",
        encoding="utf-8",
    )

    with pytest.raises(MemoryValidationError, match="invalid datetime"):
        store.list_memories()


def test_session_store_rejects_invalid_session_id(tmp_path: Path):
    store = FileSessionStore(MemoryConfig(root_dir=str(tmp_path / ".memora")))

    with pytest.raises(MemoryValidationError, match="session_id"):
        store.append_message("default", "../bad", SessionMessage(role="user", content="hello"))


def test_session_store_rejects_invalid_message_role(tmp_path: Path):
    store = FileSessionStore(MemoryConfig(root_dir=str(tmp_path / ".memora")))

    with pytest.raises(MemoryValidationError, match="session role"):
        store.append_message("default", "session_1", SessionMessage(role="bad", content="hello"))
```

- [ ] **Step 2: Run store tests to verify failure**

Run:

```bash
pytest tests/test_stores.py -v
```

Expected: FAIL because store does not yet wrap invalid datetime or validate session IDs/messages.

- [ ] **Step 3: Implement store validation imports and helpers**

In `memora/stores.py`, add imports:

```python
from .errors import MemoryValidationError
from .schema import MemoryItem, SessionMessage, validate_memory_item, validate_session_message
```

Replace the existing schema import with the expanded import above.

Replace `_dt_from_text()` with:

```python
def _dt_from_text(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise MemoryValidationError(f"invalid datetime value: {value}") from exc
```

Add session ID helper near `_clean_dict()`:

```python
def _validate_session_id(session_id: str) -> None:
    if not isinstance(session_id, str) or not session_id.strip():
        raise MemoryValidationError("session_id must be a non-empty string")
    if "/" in session_id or "\\" in session_id or ".." in session_id:
        raise MemoryValidationError("session_id must not contain path separators or '..'")
```

- [ ] **Step 4: Validate memory records at read/write boundaries**

In `FileMemoryStore._item_from_text()`, assign the constructed object to `item`, validate it, then return it:

```python
item = MemoryItem(
    id=str(meta.get("id") or ""),
    name=slugify(str(frontmatter.get("name") or "memory")),
    description=str(frontmatter.get("description") or ""),
    type=meta.get("type") or "project",
    content=body,
    user_id=meta.get("user_id") or "default",
    project_id=meta.get("project_id"),
    workspace_id=meta.get("workspace_id"),
    tags=list(meta.get("tags") or []),
    source=meta.get("source") or "unknown",
    confidence=float(meta.get("confidence") if meta.get("confidence") is not None else 1.0),
    weight=int(meta.get("weight") if meta.get("weight") is not None else 5),
    status=meta.get("status") or "active",
    created_at=_dt_from_text(meta.get("created_at")),
    updated_at=_dt_from_text(meta.get("updated_at")),
    last_accessed_at=_dt_from_text(meta.get("last_accessed_at")),
    access_count=int(meta.get("access_count") or 0),
    expires_at=_dt_from_text(meta.get("expires_at")),
    supersedes=list(meta.get("supersedes") or []),
    related=list(meta.get("related") or []),
)
validate_memory_item(item)
return item
```

In `FileMemoryStore.save_memory()`, before writing, add:

```python
validate_memory_item(item)
```

Place it after `created_at` and `updated_at` are populated.

- [ ] **Step 5: Validate session IDs and messages**

In `FileSessionStore._path()`, add:

```python
_validate_session_id(session_id)
```

before returning the path.

In `FileSessionStore.load_session()`, the call to `_path(session_id)` will validate the ID.

In `FileSessionStore.save_session()`, before writing, validate the session ID:

```python
_validate_session_id(str(session["id"]))
```

In `FileSessionStore.append_message()`, add at the start:

```python
_validate_session_id(session_id)
validate_session_message(message)
```

- [ ] **Step 6: Run store tests to verify pass**

Run:

```bash
pytest tests/test_stores.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit store robustness**

Run:

```bash
git add memora/stores.py tests/test_stores.py
git commit -m "feat: harden file stores"
```

---

### Task 4: CLI Error Reporting

**Files:**
- Modify: `memora/cli.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: `MemoraError` base exception from `memora.errors`.
- Produces: CLI commands catch Memora project exceptions, print `error: <message>` to stderr, and return exit code `1`.

- [ ] **Step 1: Add failing CLI error test**

Append to `tests/test_cli.py`:

```python
def test_save_secret_reports_clear_error(tmp_path: Path):
    root = tmp_path / ".memora"

    result = run_cli(
        root,
        "save",
        "--type",
        "user",
        "--name",
        "secret",
        "--description",
        "secret",
        "--content",
        "api_key = sk-abcdef123456",
    )

    assert result.returncode == 1
    assert "error:" in result.stderr
    assert "contains_secret" in result.stderr
```

- [ ] **Step 2: Run CLI tests to verify failure**

Run:

```bash
pytest tests/test_cli.py -v
```

Expected: FAIL because CLI does not catch Memora project exceptions yet.

- [ ] **Step 3: Refactor CLI to wrap command handling**

In `memora/cli.py`, add imports:

```python
import sys

from .errors import MemoraError
```

Refactor `main()` so parsing remains outside the `try`, but command execution is wrapped:

```python
def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    manager = MemoryManager(MemoryConfig(root_dir=args.root))

    try:
        return _run_command(args, manager, parser)
    except MemoraError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
```

Create `_run_command(args, manager: MemoryManager, parser: argparse.ArgumentParser) -> int` and move the existing command `if` blocks into it unchanged except for indentation.

The final fallback remains:

```python
parser.print_help()
return 0
```

- [ ] **Step 4: Run CLI tests to verify pass**

Run:

```bash
pytest tests/test_cli.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit CLI reliability**

Run:

```bash
git add memora/cli.py tests/test_cli.py
git commit -m "feat: report CLI errors clearly"
```

---

### Task 5: Final Regression and README Note

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: all hardening changes from Tasks 1-4.
- Produces: README documents that policy/validation errors are reported by the CLI as clear non-zero failures.

- [ ] **Step 1: Update README with error behavior**

In `README.md`, after the CLI quickstart block and before `## Python usage`, add:

```markdown
## CLI error behavior

Validation and policy failures are reported to stderr and return a non-zero exit code:

```bash
python -m memora --root .memora save --type user --name secret --description "secret" --content "api_key = sk-abcdef123456"
# stderr: error: memory rejected: contains_secret
```
```

- [ ] **Step 2: Run the full test suite**

Run:

```bash
pytest -v
```

Expected: all tests pass.

- [ ] **Step 3: Commit README and final verification**

Run:

```bash
git add README.md
git commit -m "docs: document CLI error behavior"
```

- [ ] **Step 4: Self-review final diff**

Run:

```bash
git status --short
git log --oneline -5
```

Expected:

- Only intentionally untracked design/plan documents may remain if they were not committed.
- Recent commits include the hardening commits.

---

## Plan Self-Review

- Spec coverage: covered schema validation, typed errors, store robustness, CLI reliability, and final regression.
- Placeholder scan: no placeholders, TODOs, or vague implementation steps remain.
- Type consistency: all referenced validation functions and exception classes are defined in earlier tasks before being consumed later.
- Scope control: excluded JSON output, memory management CLI commands, LLM extraction, SQLite/vector storage, Agent integration, and automatic corrupted-file repair.
