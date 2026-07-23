# Memora Memory Management Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add explicit memory update, archive, restore, delete, and filtered list/search operations to Memora.

**Architecture:** Keep storage responsibilities in `FileMemoryStore`, expose user-facing semantics through `MemoryManager`, and keep the CLI as a thin argument-parsing wrapper around the manager. Existing validation and typed errors remain the boundary for invalid data and missing targets.

**Tech Stack:** Python 3.11+, dataclasses, argparse, pytest, PyYAML.

## Global Constraints

- Preserve the existing Markdown/YAML memory file format.
- Do not introduce new dependencies.
- Keep `delete_memory(identifier, soft_delete=True)` backward-compatible: `soft_delete=True` still archives by setting `status = "archived"`.
- New management APIs must use typed Memora exceptions.
- CLI errors must continue flowing through the existing `MemoraError` handler.
- Default retrieval and default list must exclude archived and deleted memories.
- `list --archived` shows archived memories only.
- `list --all` shows active, archived, and deleted memories; if passed with `--archived`, `--all` wins.
- `search --archived` includes archived memories but not deleted memories.
- This round does not add LLM extraction, JSON CLI output, SQLite/vector storage, Agent runtime integration, batch editing, or automatic corrupted-file repair.
- Final verification command is `pytest -v`.

---

## File Structure

- Modify `memora/stores.py`: add `set_memory_status()` and `hard_delete_memory()`.
- Modify `memora/manager.py`: add `update_memory()`, `archive_memory()`, `restore_memory()`, and `delete_memory()`.
- Modify `memora/cli.py`: add `update`, `archive`, `restore`, `delete`, enhanced `list`, and enhanced `search` commands.
- Modify `tests/test_stores.py`: test new store state/hard-delete operations.
- Modify `tests/test_manager.py`: test new manager APIs.
- Modify `tests/test_cli.py`: test new CLI commands and filters.
- Modify `README.md`: document memory management commands.

---

### Task 1: Store Status Operations

**Files:**
- Modify: `memora/stores.py`
- Modify: `tests/test_stores.py`

**Interfaces:**
- Consumes: `MemoryNotFoundError`, `validate_memory_status()`, existing `FileMemoryStore.get_memory()` and `update_memory()`.
- Produces:
  - `FileMemoryStore.set_memory_status(identifier: str, status: str) -> MemoryItem`
  - `FileMemoryStore.hard_delete_memory(identifier: str) -> None`

- [ ] **Step 1: Add failing store tests**

Append to `tests/test_stores.py`:

```python
from memora.errors import MemoryNotFoundError
```

If `MemoryValidationError` is already imported, combine the imports:

```python
from memora.errors import MemoryNotFoundError, MemoryValidationError
```

Append these tests:

```python
def test_set_memory_status_archives_and_restores(tmp_path: Path):
    store = FileMemoryStore(config_for(tmp_path))
    store.save_memory(MemoryItem(id="mem_1", name="language", description="desc", type="user", content="body"))

    archived = store.set_memory_status("language", "archived")
    assert archived.status == "archived"
    assert store.list_memories() == []

    restored = store.set_memory_status("mem_1", "active")
    assert restored.status == "active"
    assert len(store.list_memories()) == 1


def test_set_memory_status_missing_raises_not_found(tmp_path: Path):
    store = FileMemoryStore(config_for(tmp_path))

    with pytest.raises(MemoryNotFoundError, match="memory not found"):
        store.set_memory_status("missing", "archived")


def test_hard_delete_memory_removes_file(tmp_path: Path):
    store = FileMemoryStore(config_for(tmp_path))
    store.save_memory(MemoryItem(id="mem_1", name="language", description="desc", type="user", content="body"))

    store.hard_delete_memory("language")

    assert store.get_memory("language") is None
    assert store.list_memories(include_archived=True) == []


def test_hard_delete_memory_missing_raises_not_found(tmp_path: Path):
    store = FileMemoryStore(config_for(tmp_path))

    with pytest.raises(MemoryNotFoundError, match="memory not found"):
        store.hard_delete_memory("missing")
```

- [ ] **Step 2: Run store tests to verify failure**

Run:

```bash
pytest tests/test_stores.py -v
```

Expected: FAIL because `set_memory_status` and `hard_delete_memory` do not exist yet.

- [ ] **Step 3: Update store imports**

In `memora/stores.py`, update imports to include:

```python
from .errors import MemoryNotFoundError, MemoryValidationError
from .schema import (
    MemoryItem,
    SessionMessage,
    validate_memory_item,
    validate_memory_status,
    validate_session_message,
)
```

Preserve existing imported names.

- [ ] **Step 4: Implement `set_memory_status`**

Add this method inside `FileMemoryStore`, after `update_memory()`:

```python
    def set_memory_status(self, identifier: str, status: str) -> MemoryItem:
        validate_memory_status(status)
        item = self.get_memory(identifier)
        if item is None:
            raise MemoryNotFoundError(f"memory not found: {identifier}")
        item.status = status
        return self.update_memory(item)
```

- [ ] **Step 5: Implement `hard_delete_memory`**

Add this method inside `FileMemoryStore`, after `set_memory_status()`:

```python
    def hard_delete_memory(self, identifier: str) -> None:
        item = self.get_memory(identifier)
        if item is None:
            raise MemoryNotFoundError(f"memory not found: {identifier}")
        path = self._path_for_name(item.name)
        if path.exists():
            path.unlink()
        self.rebuild_index()
```

- [ ] **Step 6: Run store tests to verify pass**

Run:

```bash
pytest tests/test_stores.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit store status operations**

Run:

```bash
git add memora/stores.py tests/test_stores.py
git commit -m "feat: add memory store status operations" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Manager Memory Management API

**Files:**
- Modify: `memora/manager.py`
- Modify: `tests/test_manager.py`

**Interfaces:**
- Consumes: `FileMemoryStore.set_memory_status()`, `FileMemoryStore.hard_delete_memory()`, `MemoryNotFoundError`, `validate_memory_item()`.
- Produces:
  - `MemoryManager.update_memory(...) -> MemoryItem`
  - `MemoryManager.archive_memory(identifier: str) -> MemoryItem`
  - `MemoryManager.restore_memory(identifier: str) -> MemoryItem`
  - `MemoryManager.delete_memory(identifier: str, hard: bool = False) -> None`

- [ ] **Step 1: Add failing manager tests**

Update imports in `tests/test_manager.py`:

```python
from memora.errors import MemoryNotFoundError, MemoryPolicyError, MemoryValidationError
```

Append these tests:

```python
def test_update_memory_changes_selected_fields(tmp_path: Path):
    manager = manager_for(tmp_path)
    manager.save_memory("user", "old content", "old desc", name="language", tags=["old"], weight=5)

    updated = manager.update_memory(
        "language",
        description="new desc",
        content="new content",
        tags=["new"],
        weight=8,
        confidence=0.7,
    )

    assert updated.description == "new desc"
    assert updated.content == "new content"
    assert updated.tags == ["new"]
    assert updated.weight == 8
    assert updated.confidence == 0.7
    assert updated.updated_at is not None


def test_update_memory_missing_raises_not_found(tmp_path: Path):
    manager = manager_for(tmp_path)

    try:
        manager.update_memory("missing", content="new")
    except MemoryNotFoundError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("expected MemoryNotFoundError")


def test_archive_and_restore_memory_control_retrieval(tmp_path: Path):
    manager = manager_for(tmp_path)
    manager.save_memory("user", "用户偏好中文。", "用户偏好中文。", name="language")

    archived = manager.archive_memory("language")
    assert archived.status == "archived"
    assert manager.retrieve_memory("中文") == []

    restored = manager.restore_memory("language")
    assert restored.status == "active"
    assert len(manager.retrieve_memory("中文")) == 1


def test_delete_memory_marks_deleted_by_default(tmp_path: Path):
    manager = manager_for(tmp_path)
    manager.save_memory("user", "用户偏好中文。", "用户偏好中文。", name="language")

    manager.delete_memory("language")
    deleted = manager.memory_store.get_memory("language")

    assert deleted is not None
    assert deleted.status == "deleted"
    assert manager.retrieve_memory("中文") == []


def test_delete_memory_hard_removes_file(tmp_path: Path):
    manager = manager_for(tmp_path)
    manager.save_memory("user", "用户偏好中文。", "用户偏好中文。", name="language")

    manager.delete_memory("language", hard=True)

    assert manager.memory_store.get_memory("language") is None
```

- [ ] **Step 2: Run manager tests to verify failure**

Run:

```bash
pytest tests/test_manager.py -v
```

Expected: FAIL because new manager methods do not exist yet.

- [ ] **Step 3: Update manager imports**

In `memora/manager.py`, update error imports:

```python
from .errors import MemoryNotFoundError, MemoryPolicyError
```

`validate_memory_item` should already be imported from the prior hardening pass.

- [ ] **Step 4: Implement `update_memory`**

Add this method to `MemoryManager`, after `retrieve_memory()` or before session methods:

```python
    def update_memory(
        self,
        identifier: str,
        description: str | None = None,
        content: str | None = None,
        tags: list[str] | None = None,
        weight: int | None = None,
        confidence: float | None = None,
    ) -> MemoryItem:
        memory = self.memory_store.get_memory(identifier)
        if memory is None:
            raise MemoryNotFoundError(f"memory not found: {identifier}")
        if description is not None:
            memory.description = description
        if content is not None:
            memory.content = content
        if tags is not None:
            memory.tags = tags
        if weight is not None:
            memory.weight = weight
        if confidence is not None:
            memory.confidence = confidence
        memory.updated_at = now_utc()
        validate_memory_item(memory)
        return self.memory_store.update_memory(memory)
```

- [ ] **Step 5: Implement archive/restore/delete methods**

Add these methods to `MemoryManager` after `update_memory()`:

```python
    def archive_memory(self, identifier: str) -> MemoryItem:
        return self.memory_store.set_memory_status(identifier, "archived")

    def restore_memory(self, identifier: str) -> MemoryItem:
        return self.memory_store.set_memory_status(identifier, "active")

    def delete_memory(self, identifier: str, hard: bool = False) -> None:
        if hard:
            self.memory_store.hard_delete_memory(identifier)
            return
        self.memory_store.set_memory_status(identifier, "deleted")
```

- [ ] **Step 6: Run manager tests to verify pass**

Run:

```bash
pytest tests/test_manager.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit manager API**

Run:

```bash
git add memora/manager.py tests/test_manager.py
git commit -m "feat: add memory management manager API" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: CLI Management Commands

**Files:**
- Modify: `memora/cli.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: manager methods from Task 2.
- Produces CLI commands:
  - `memora update <identifier> ...`
  - `memora archive <identifier>`
  - `memora restore <identifier>`
  - `memora delete <identifier> [--hard]`

- [ ] **Step 1: Add failing CLI management tests**

Append these helpers and tests to `tests/test_cli.py`:

```python
def save_language(root: Path):
    return run_cli(
        root,
        "save",
        "--type",
        "user",
        "--name",
        "language",
        "--description",
        "用户偏好中文。",
        "--content",
        "用户偏好使用中文回答。",
    )


def test_update_command_changes_memory(tmp_path: Path):
    root = tmp_path / ".memora"
    assert save_language(root).returncode == 0

    updated = run_cli(root, "update", "language", "--description", "updated desc", "--content", "updated content", "--tag", "language", "--weight", "8", "--confidence", "0.7")
    shown = run_cli(root, "show", "language")

    assert updated.returncode == 0
    assert "updated" in updated.stdout
    assert "updated desc" in shown.stdout
    assert "updated content" in shown.stdout


def test_archive_and_restore_commands(tmp_path: Path):
    root = tmp_path / ".memora"
    assert save_language(root).returncode == 0

    archived = run_cli(root, "archive", "language")
    listed = run_cli(root, "list")
    archived_list = run_cli(root, "list", "--archived")
    restored = run_cli(root, "restore", "language")
    listed_again = run_cli(root, "list")

    assert archived.returncode == 0
    assert "archived" in archived.stdout
    assert "language" not in listed.stdout
    assert "language" in archived_list.stdout
    assert restored.returncode == 0
    assert "restored" in restored.stdout
    assert "language" in listed_again.stdout


def test_delete_command_marks_deleted_and_hard_delete_removes(tmp_path: Path):
    root = tmp_path / ".memora"
    assert save_language(root).returncode == 0

    deleted = run_cli(root, "delete", "language")
    listed = run_cli(root, "list")
    all_list = run_cli(root, "list", "--all")
    hard_deleted = run_cli(root, "delete", "language", "--hard")
    all_after_hard_delete = run_cli(root, "list", "--all")

    assert deleted.returncode == 0
    assert "deleted" in deleted.stdout
    assert "language" not in listed.stdout
    assert "language" in all_list.stdout
    assert hard_deleted.returncode == 0
    assert "hard deleted" in hard_deleted.stdout
    assert "language" not in all_after_hard_delete.stdout
```

- [ ] **Step 2: Run CLI tests to verify failure**

Run:

```bash
pytest tests/test_cli.py -v
```

Expected: FAIL because new CLI commands do not exist yet.

- [ ] **Step 3: Add CLI parsers**

In `memora/cli.py`, inside `build_parser()`, after the `show` parser, add:

```python
    update_parser = subparsers.add_parser("update", help="Update one memory.")
    update_parser.add_argument("identifier")
    update_parser.add_argument("--description")
    update_parser.add_argument("--content")
    update_parser.add_argument("--tag", action="append", dest="tags")
    update_parser.add_argument("--weight", type=int)
    update_parser.add_argument("--confidence", type=float)

    archive_parser = subparsers.add_parser("archive", help="Archive one memory.")
    archive_parser.add_argument("identifier")

    restore_parser = subparsers.add_parser("restore", help="Restore one archived or deleted memory.")
    restore_parser.add_argument("identifier")

    delete_parser = subparsers.add_parser("delete", help="Delete one memory.")
    delete_parser.add_argument("identifier")
    delete_parser.add_argument("--hard", action="store_true")
```

- [ ] **Step 4: Add CLI command handlers**

In `_run_command()`, after the `show` block and before `search`, add:

```python
    if args.command == "update":
        item = manager.update_memory(
            args.identifier,
            description=args.description,
            content=args.content,
            tags=args.tags,
            weight=args.weight,
            confidence=args.confidence,
        )
        print(f"updated {item.id} {item.name}")
        return 0

    if args.command == "archive":
        item = manager.archive_memory(args.identifier)
        print(f"archived {item.id} {item.name}")
        return 0

    if args.command == "restore":
        item = manager.restore_memory(args.identifier)
        print(f"restored {item.id} {item.name}")
        return 0

    if args.command == "delete":
        manager.delete_memory(args.identifier, hard=args.hard)
        if args.hard:
            print(f"hard deleted {args.identifier}")
        else:
            print(f"deleted {args.identifier}")
        return 0
```

- [ ] **Step 5: Run CLI tests to verify management commands**

Run:

```bash
pytest tests/test_cli.py -v
```

Expected: management command tests still may fail until list filters are implemented in Task 4. The `update`, command parser, and direct command handling failures should be resolved.

- [ ] **Step 6: Commit CLI management commands**

Run only if CLI tests failures are limited to list/search filter behavior scheduled for Task 4. If unrelated failures remain, fix them first.

```bash
git add memora/cli.py tests/test_cli.py
git commit -m "feat: add memory management CLI commands" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: List and Search Filters

**Files:**
- Modify: `memora/cli.py`
- Modify: `memora/manager.py`
- Modify: `memora/retriever.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: existing `MemoryManager.retrieve_memory()` filters.
- Produces:
  - `memora list --archived`
  - `memora list --all`
  - `memora search QUERY --type TYPE ... --tag TAG ... --top-k N --archived`
  - Deleted memories remain excluded from search even when archived results are included.

- [ ] **Step 1: Add failing search filter test**

Append to `tests/test_cli.py`:

```python
def test_search_filters_type_tag_and_top_k(tmp_path: Path):
    root = tmp_path / ".memora"
    assert run_cli(root, "save", "--type", "user", "--name", "language", "--description", "用户偏好中文。", "--content", "用户偏好中文回答。",).returncode == 0
    assert run_cli(root, "save", "--type", "project", "--name", "project-language", "--description", "项目使用中文。", "--content", "项目中文文档。",).returncode == 0
    assert run_cli(root, "update", "language", "--tag", "language").returncode == 0

    result = run_cli(root, "search", "中文", "--type", "user", "--tag", "language", "--top-k", "1")

    assert result.returncode == 0
    assert "language" in result.stdout
    assert "project-language" not in result.stdout
```

- [ ] **Step 2: Run CLI tests to verify filter failure**

Run:

```bash
pytest tests/test_cli.py -v
```

Expected: FAIL because list/search parsers do not yet expose these filters.

- [ ] **Step 3: Update list parser**

Replace current list parser line in `build_parser()`:

```python
    subparsers.add_parser("list", help="List memories.")
```

with:

```python
    list_parser = subparsers.add_parser("list", help="List memories.")
    list_parser.add_argument("--archived", action="store_true", help="List archived memories only.")
    list_parser.add_argument("--all", action="store_true", help="List active, archived, and deleted memories.")
```

- [ ] **Step 4: Update search parser**

In `build_parser()`, extend the search parser:

```python
    search_parser.add_argument("--type", action="append", dest="memory_types")
    search_parser.add_argument("--tag", action="append", dest="tags")
    search_parser.add_argument("--top-k", type=int)
    search_parser.add_argument("--archived", action="store_true", help="Include archived memories.")
```

- [ ] **Step 5: Update list handler**

Replace the `list` command handler in `_run_command()` with:

```python
    if args.command == "list":
        items = manager.memory_store.list_memories(include_archived=args.archived or args.all)
        if args.archived and not args.all:
            items = [item for item in items if item.status == "archived"]
        if not args.all:
            items = [item for item in items if item.status == "active" or (args.archived and item.status == "archived")]
        for item in items:
            print(f"{item.id}\t{item.name}\t{item.type}\t{item.status}\t{item.description}")
        return 0
```

- [ ] **Step 6: Ensure deleted memories are excluded from search**

In `memora/retriever.py`, update the status filter at the start of `score()`.

Replace:

```python
        if memory.status != "active" and not query.include_archived:
            return None
```

with:

```python
        if memory.status == "deleted":
            return None
        if memory.status != "active" and not query.include_archived:
            return None
```

- [ ] **Step 7: Update search handler**

Replace the `search` command handler in `_run_command()` with:

```python
    if args.command == "search":
        results = manager.retrieve_memory(
            args.query,
            memory_types=args.memory_types,
            tags=args.tags,
            top_k=args.top_k,
            include_archived=args.archived,
        )
        for result in results:
            print(f"{result.final_score:.3f}\t{result.memory.id}\t{result.memory.name}\t{result.memory.description}")
        return 0
```

- [ ] **Step 8: Run CLI tests to verify pass**

Run:

```bash
pytest tests/test_cli.py -v
```

Expected: PASS.

- [ ] **Step 9: Commit list/search filters**

Run:

```bash
git add memora/cli.py memora/retriever.py tests/test_cli.py
git commit -m "feat: add memory list and search filters" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: README and Final Verification

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: all memory management features from Tasks 1-4.
- Produces: README examples for update/archive/restore/delete and filtered list/search.

- [ ] **Step 1: Update README quickstart**

In `README.md`, inside the CLI quickstart block, after the `show language` line, add:

```bash
python -m memora --root .memora update language --tag language --weight 8
python -m memora --root .memora archive language
python -m memora --root .memora list --archived
python -m memora --root .memora restore language
python -m memora --root .memora search "中文回答" --type user --tag language --top-k 5
python -m memora --root .memora delete language
python -m memora --root .memora list --all
```

- [ ] **Step 2: Run full test suite**

Run:

```bash
pytest -v
```

Expected: all tests pass.

- [ ] **Step 3: Commit README update**

Run:

```bash
git add README.md
git commit -m "docs: document memory management commands" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

- [ ] **Step 4: Self-review final state**

Run:

```bash
git status --short
git log --oneline -6
```

Expected:

- Recent commits include store operations, manager API, CLI commands, list/search filters, and README docs.
- Only intentional untracked design/plan/editor files remain.

---

## Plan Self-Review

- Spec coverage: covers store operations, manager API, CLI commands, list/search filters, errors, tests, README, and final verification.
- Placeholder scan: no TBD, TODO, or vague steps remain.
- Type consistency: store methods are introduced before manager methods consume them; manager methods are introduced before CLI consumes them.
- Scope control: excludes LLM extraction, JSON output, SQLite/vector backends, Agent integration, batch editing, and automatic repair.
