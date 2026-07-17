# Memora Data Portability and Integrity Design

## Goal

Add practical data portability and integrity workflows for Memora memory files. Users should be able to export, import, verify, rebuild, and back up local memories without adding databases, cloud services, compression, encryption, or new dependencies.

## User Direction

The user selected the Import / Export / Verify direction and approved a slightly expanded scope. This round should do more than the smallest possible export/import pair, but remain deterministic and local.

Approved scope:

- Export memories to JSON.
- Import memories from JSON.
- Verify memory store health.
- Rebuild the memory index manually.
- Add backup as a semantic alias for memory export.
- Cover memories only, not sessions.

## Architecture

Add a focused portability module:

- `memora/portable.py`

Responsibilities:

- Convert `MemoryItem` objects to JSON-safe dictionaries.
- Convert dictionaries back to validated `MemoryItem` objects.
- Export all memories to a versioned JSON document.
- Import memories from a versioned JSON document.
- Verify memory markdown files and index consistency.
- Delegate index rebuild to `FileMemoryStore.rebuild_index()`.

Keep responsibilities separated:

- `stores.py`: Markdown file storage and index rebuild implementation.
- `portable.py`: JSON portability and integrity workflows.
- `manager.py`: public facade methods.
- `cli.py`: user-facing command parsing and output.

No new dependency is introduced.

## Export Format

Use readable JSON:

```json
{
  "format": "memora.memories.v1",
  "memories": [
    {
      "id": "mem_abc123",
      "name": "language",
      "description": "用户偏好中文。",
      "type": "user",
      "content": "用户偏好使用中文回答。",
      "user_id": "default",
      "project_id": null,
      "workspace_id": null,
      "tags": ["language"],
      "source": "manual",
      "confidence": 1.0,
      "weight": 5,
      "status": "active",
      "created_at": "2026-07-17T00:00:00+00:00",
      "updated_at": "2026-07-17T00:00:00+00:00",
      "last_accessed_at": null,
      "access_count": 0,
      "expires_at": null,
      "supersedes": [],
      "related": []
    }
  ]
}
```

Rules:

- Export all memories returned by `list_memories(include_archived=True)`, including `active`, `archived`, and `deleted`.
- Preserve IDs, names, statuses, timestamps, tags, related links, access stats, confidence, and weight.
- Use ISO strings for datetimes.
- Write UTF-8 JSON with `ensure_ascii=False` and indentation.

## Import Behavior

Import from `memora.memories.v1` JSON.

Rules:

- Validate the top-level `format` field.
- Validate `memories` is a list.
- Convert each dictionary to `MemoryItem`.
- Validate each imported item with existing schema validation.
- Skip duplicates by existing `id` or slugified `name`.
- Save non-duplicates with `FileMemoryStore.save_memory()`.
- Do not overwrite existing memories.
- Do not merge conflicts.
- Do not prompt interactively.
- Continue importing remaining memories if one item fails.

Return a summary:

```python
{
    "imported": 3,
    "skipped": 1,
    "errors": [],
}
```

Errors should include enough detail to debug the item, such as item index and exception message.

## Verify Behavior

`verify_memories()` should check the current memory store and return:

```python
{
    "checked": 12,
    "errors": [],
    "index_ok": True,
}
```

Checks:

1. Every `*.md` file under `.memora/memories` can be read.
2. Frontmatter can be parsed.
3. Datetimes are valid.
4. Parsed items pass `validate_memory_item()`.
5. The current `MEMORY.md` content matches the index that would be generated for active memories.

Verification should not mutate storage. In particular, it should not rebuild the index automatically.

Index comparison should use the same line format as `FileMemoryStore.rebuild_index()`:

```text
- [<name>](memories/<name>.md) — <description>
```

Only active memories appear in the expected index.

## Rebuild Index Behavior

Expose a manual rebuild operation:

```python
manager.rebuild_index()
```

CLI:

```bash
python -m memora --root .memora rebuild-index
```

This should call `FileMemoryStore.rebuild_index()` and print a short success message.

## Backup Behavior

Backup is currently a semantic alias for export:

```python
manager.backup(path)
```

CLI:

```bash
python -m memora --root .memora backup backup.json
```

This writes the same `memora.memories.v1` format as export. The name exists because users think in terms of backup, and the method can later expand to include sessions without changing the command surface.

## Manager API

Add these methods to `MemoryManager`:

```python
def export_memories(self, path: str | Path) -> dict:
    ...

def import_memories(self, path: str | Path) -> dict:
    ...

def verify_memories(self) -> dict:
    ...

def rebuild_index(self) -> None:
    ...

def backup(self, path: str | Path) -> dict:
    ...
```

Manager methods should delegate to `memora.portable.MemoryPortable` or simple module-level functions from `portable.py`.

## CLI

Add commands:

```bash
python -m memora --root .memora export memories.json
python -m memora --root .memora import memories.json
python -m memora --root .memora verify
python -m memora --root .memora rebuild-index
python -m memora --root .memora backup backup.json
```

Output should be simple text summaries. Example:

```text
exported 3 memories to memories.json
imported 2 skipped 1 errors 0
verified 3 memories index_ok=True errors=0
rebuilt index
backed up 3 memories to backup.json
```

If import or verify reports errors, the command should still return `0` when it completed and produced a report. Existing `MemoraError` handling remains responsible for malformed command-level failures.

## Tests

Add tests for:

- Export writes `format == "memora.memories.v1"` and all memory statuses.
- Import imports new memories and skips duplicates by name or id.
- Import reports per-item errors and continues.
- Verify returns `index_ok=True` for a fresh store.
- Verify returns `index_ok=False` after manually corrupting `MEMORY.md`.
- Rebuild index repairs the corrupted index.
- Backup writes the same format as export.
- CLI commands print expected summaries.

## Non-Goals

This round does not add:

- session export/import
- automatic repair during verify
- conflict merging
- overwrite mode
- interactive prompts
- zip compression
- encryption
- cloud sync
- database backends
- new dependencies

## Success Criteria

- Users can export memories to JSON and import them into another root.
- Duplicate imports are skipped safely.
- Users can verify memory file health and index consistency.
- Users can rebuild the index explicitly.
- Users can create a backup JSON using the backup command.
- All behavior is covered by tests.
- Full test suite passes with `pytest -v`.
