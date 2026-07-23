# Memora Memory Management Design

**Goal:** Complete the manual memory-management loop for Memora by adding Manager APIs and CLI commands for update, archive, restore, delete, and filtered list/search operations.

**Scope:** This builds on the existing deterministic MVP and hardening pass. It keeps the local Markdown/YAML + JSON storage model and does not introduce LLM extraction, JSON CLI output, SQLite, vector search, or Agent runtime integration.

## Current State

Memora currently supports initialization, saving memories, listing active memories, showing one memory, keyword search, session append/show, lifecycle cleanup, schema validation, typed project errors, store robustness, and CLI error reporting.

The missing management loop is: after a memory is saved, users and future Agent runtimes need a safe way to modify, archive, restore, and delete it. The current store has `delete_memory(identifier, soft_delete=True)`, but soft delete currently means archive. This pass adds explicit semantics so the old method can remain compatible while new APIs are clear.

## Design

### 1. Store operations

Add explicit state operations to `memora/stores.py`:

- `set_memory_status(identifier: str, status: str) -> MemoryItem`
- `hard_delete_memory(identifier: str) -> None`

Semantics:

- `set_memory_status` finds a memory by id or slugified name, validates the requested status, sets `MemoryItem.status`, persists the item, rebuilds `MEMORY.md`, and returns the updated item.
- Missing memory raises `MemoryNotFoundError`.
- Invalid status raises `MemoryValidationError` through schema validation.
- `hard_delete_memory` removes the backing Markdown file and rebuilds the index.
- Missing memory raises `MemoryNotFoundError`.
- Existing `delete_memory(identifier, soft_delete=True)` remains backward-compatible. With `soft_delete=True`, it still archives by setting `status="archived"`. New code should prefer `set_memory_status` and `hard_delete_memory`.

### 2. Manager API

Add public methods to `memora/manager.py`:

```python
def update_memory(
    self,
    identifier: str,
    description: str | None = None,
    content: str | None = None,
    tags: list[str] | None = None,
    weight: int | None = None,
    confidence: float | None = None,
) -> MemoryItem: ...

def archive_memory(self, identifier: str) -> MemoryItem: ...

def restore_memory(self, identifier: str) -> MemoryItem: ...

def delete_memory(self, identifier: str, hard: bool = False) -> None: ...
```

Semantics:

- `update_memory` loads a memory by id or name, raises `MemoryNotFoundError` if absent, updates only supplied fields, refreshes `updated_at`, validates the item, persists it, and returns it.
- `archive_memory` sets `status="archived"` via the store and returns the updated item.
- `restore_memory` sets `status="active"` via the store and returns the updated item.
- `delete_memory(hard=False)` sets `status="deleted"` via the store.
- `delete_memory(hard=True)` physically removes the file via the store.

Retrieval behavior:

- Existing `retrieve_memory(..., include_archived=False)` continues to return active memories only by default.
- This pass does not add a separate `include_deleted` option to retrieval. Deleted memories are only visible through `list --all` or direct show if the store includes archived/deleted lookup internally.

### 3. CLI commands

Add commands to `memora/cli.py`:

```bash
memora update <identifier> [--description TEXT] [--content TEXT] [--tag TAG ...] [--weight N] [--confidence F]
memora archive <identifier>
memora restore <identifier>
memora delete <identifier> [--hard]
```

Output:

```text
updated <id> <name>
archived <id> <name>
restored <id> <name>
deleted <identifier>
hard deleted <identifier>
```

The CLI already catches `MemoraError`, prints `error: <message>` to stderr, and exits with code `1`.

### 4. List and search filters

Enhance existing commands:

```bash
memora list [--archived] [--all]
memora search QUERY [--type TYPE ...] [--tag TAG ...] [--top-k N] [--archived]
```

List semantics:

- Default `list` shows active memories only.
- `list --archived` shows archived memories only.
- `list --all` shows active, archived, and deleted memories.
- If both `--archived` and `--all` are passed, `--all` wins.

Search semantics:

- `--type` can be repeated and maps to `MemoryManager.retrieve_memory(memory_types=[...])`.
- `--tag` can be repeated and maps to `MemoryManager.retrieve_memory(tags=[...])`.
- `--top-k` maps to `top_k`.
- `--archived` sets `include_archived=True` so archived memories can be searched. Deleted memories remain excluded from search.

### 5. Error behavior

Use existing typed errors:

- `MemoryNotFoundError` for missing target memories.
- `MemoryValidationError` for invalid statuses, weights, confidence values, or search filters.
- `MemoryPolicyError` remains reserved for save-policy rejection.

The CLI converts these into clear stderr output and non-zero exit codes.

## Tests

Add or update tests in:

- `tests/test_stores.py`
  - `set_memory_status` archives and restores memory.
  - `hard_delete_memory` removes the memory file.
  - missing status/hard delete targets raise `MemoryNotFoundError`.

- `tests/test_manager.py`
  - `update_memory` changes selected fields.
  - missing update target raises `MemoryNotFoundError`.
  - archive hides memory from default retrieval.
  - restore makes memory retrievable again.
  - default delete marks status as `deleted`.
  - hard delete removes the memory from store lookup.

- `tests/test_cli.py`
  - `update` changes content or description.
  - `archive` hides memory from default list/search.
  - `restore` returns memory to default list/search.
  - `delete` marks memory deleted and `list --all` shows it.
  - `delete --hard` removes memory from `list --all`.
  - `list --archived` and `list --all` work.
  - `search --type --tag --top-k` filters results.

Final verification command:

```bash
pytest -v
```

## Non-goals

This design intentionally excludes:

- LLM-based memory extraction
- interactive conflict confirmation
- batch editing
- JSON CLI output
- SQLite storage
- vector retrieval
- Agent runtime integration
- automatic repair of corrupted memory files

## Self-review

- Placeholder scan: no placeholders or TBDs remain.
- Internal consistency: store status operations, Manager API semantics, CLI commands, and retrieval/list behavior align.
- Scope check: this is a focused management loop over the existing MVP, not a new storage or extraction subsystem.
- Ambiguity check: archive, restore, soft delete, hard delete, list visibility, and search visibility are explicitly defined.
