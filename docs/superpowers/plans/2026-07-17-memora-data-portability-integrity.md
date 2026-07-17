# Memora Data Portability and Integrity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add export, import, verify, rebuild-index, and backup workflows for local Memora memory files.

**Architecture:** Add `memora/portable.py` for JSON portability and verification logic, expose it through `MemoryManager`, then add CLI commands and README usage. Storage remains Markdown/YAML; portability JSON is a separate versioned interchange format.

**Tech Stack:** Python standard library (`json`, `pathlib`, `datetime`), existing Memora modules, pytest.

## Global Constraints

- Export memories to JSON.
- Import memories from JSON.
- Verify memory store health.
- Rebuild the memory index manually.
- Add backup as a semantic alias for memory export.
- Cover memories only, not sessions.
- Do not add session export/import.
- Do not add automatic repair during verify.
- Do not add conflict merging.
- Do not add overwrite mode.
- Do not add interactive prompts.
- Do not add zip compression.
- Do not add encryption.
- Do not add cloud sync.
- Do not add database backends.
- Do not add new dependencies.

---

## File Structure

- Create `memora/portable.py`: conversion, export/import, verify, backup helpers.
- Modify `memora/manager.py`: add public facade methods.
- Modify `memora/cli.py`: add `export`, `import`, `verify`, `rebuild-index`, `backup` commands.
- Create `tests/test_portable.py`: unit tests for portability helpers through manager/store.
- Modify `tests/test_manager.py`: manager facade tests.
- Modify `tests/test_cli.py`: CLI command tests.
- Modify `README.md`: short CLI usage examples.

---

### Task 1: Portable Module

**Files:**
- Create: `memora/portable.py`
- Test: `tests/test_portable.py`

**Interfaces:**
- Consumes: `FileMemoryStore`, `MemoryItem`, `validate_memory_item`, `safe_json_load`, `safe_json_write`, `slugify`
- Produces:
  - `EXPORT_FORMAT = "memora.memories.v1"`
  - `memory_to_dict(item: MemoryItem) -> dict`
  - `memory_from_dict(data: dict) -> MemoryItem`
  - `export_memories(store: FileMemoryStore, path: str | Path) -> dict`
  - `import_memories(store: FileMemoryStore, path: str | Path) -> dict`
  - `verify_memories(store: FileMemoryStore) -> dict`
  - `rebuild_index(store: FileMemoryStore) -> None`
  - `backup_memories(store: FileMemoryStore, path: str | Path) -> dict`

- [ ] **Step 1: Write failing portable tests**

Create `tests/test_portable.py` with tests for export, import, verify, rebuild, and backup:

```python
import json
from pathlib import Path

from memora.config import MemoryConfig
from memora.manager import MemoryManager
from memora.portable import EXPORT_FORMAT


def manager_for(tmp_path: Path) -> MemoryManager:
    manager = MemoryManager(MemoryConfig(root_dir=tmp_path / ".memora"))
    manager.init_storage()
    return manager


def test_export_memories_writes_versioned_json_with_all_statuses(tmp_path: Path):
    manager = manager_for(tmp_path)
    active = manager.save_memory("user", "active content", "active desc", name="active")
    archived = manager.archive_memory(active.id)
    deleted = manager.save_memory("project", "deleted content", "deleted desc", name="deleted")
    manager.delete_memory(deleted.id)
    path = tmp_path / "memories.json"

    report = manager.export_memories(path)
    data = json.loads(path.read_text(encoding="utf-8"))

    assert report == {"exported": 2, "path": str(path)}
    assert data["format"] == EXPORT_FORMAT
    assert {item["status"] for item in data["memories"]} == {"archived", "deleted"}
    assert any(item["id"] == archived.id for item in data["memories"])


def test_import_memories_imports_new_and_skips_duplicates(tmp_path: Path):
    source = manager_for(tmp_path / "source")
    source.save_memory("user", "用户偏好中文回答。", "用户偏好中文。", name="language")
    export_path = tmp_path / "memories.json"
    source.export_memories(export_path)

    target = manager_for(tmp_path / "target")
    target.save_memory("user", "existing", "existing", name="language")

    report = target.import_memories(export_path)

    assert report["imported"] == 0
    assert report["skipped"] == 1
    assert report["errors"] == []


def test_import_memories_reports_item_errors_and_continues(tmp_path: Path):
    manager = manager_for(tmp_path)
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "format": EXPORT_FORMAT,
                "memories": [
                    {
                        "id": "mem_good",
                        "name": "good",
                        "description": "good desc",
                        "type": "user",
                        "content": "good content",
                        "user_id": "default",
                        "project_id": None,
                        "workspace_id": None,
                        "tags": [],
                        "source": "test",
                        "confidence": 1.0,
                        "weight": 5,
                        "status": "active",
                        "created_at": None,
                        "updated_at": None,
                        "last_accessed_at": None,
                        "access_count": 0,
                        "expires_at": None,
                        "supersedes": [],
                        "related": [],
                    },
                    {"id": "bad"},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = manager.import_memories(path)

    assert report["imported"] == 1
    assert report["skipped"] == 0
    assert len(report["errors"]) == 1
    assert manager.memory_store.get_memory("good") is not None


def test_verify_memories_reports_index_health_and_rebuild_repairs(tmp_path: Path):
    manager = manager_for(tmp_path)
    manager.save_memory("user", "content", "description", name="language")

    healthy = manager.verify_memories()
    assert healthy["checked"] == 1
    assert healthy["errors"] == []
    assert healthy["index_ok"] is True

    manager.memory_store.index_path.write_text("broken\n", encoding="utf-8")
    broken = manager.verify_memories()
    assert broken["checked"] == 1
    assert broken["index_ok"] is False

    manager.rebuild_index()
    repaired = manager.verify_memories()
    assert repaired["index_ok"] is True


def test_backup_writes_same_format_as_export(tmp_path: Path):
    manager = manager_for(tmp_path)
    manager.save_memory("user", "content", "description", name="language")
    path = tmp_path / "backup.json"

    report = manager.backup(path)
    data = json.loads(path.read_text(encoding="utf-8"))

    assert report == {"exported": 1, "path": str(path)}
    assert data["format"] == EXPORT_FORMAT
    assert len(data["memories"]) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_portable.py -v
```

Expected: FAIL because `memora.portable` and manager facade methods do not exist.

- [ ] **Step 3: Implement `memora/portable.py`**

Create `memora/portable.py`:

```python
"""Memory import, export, backup, and verification helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from .schema import MemoryItem, validate_memory_item
from .stores import FileMemoryStore
from .utils import safe_json_load, safe_json_write, slugify

EXPORT_FORMAT = "memora.memories.v1"


def _dt_to_text(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _dt_from_text(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def memory_to_dict(item: MemoryItem) -> dict[str, Any]:
    return {
        "id": item.id,
        "name": item.name,
        "description": item.description,
        "type": item.type,
        "content": item.content,
        "user_id": item.user_id,
        "project_id": item.project_id,
        "workspace_id": item.workspace_id,
        "tags": item.tags,
        "source": item.source,
        "confidence": item.confidence,
        "weight": item.weight,
        "status": item.status,
        "created_at": _dt_to_text(item.created_at),
        "updated_at": _dt_to_text(item.updated_at),
        "last_accessed_at": _dt_to_text(item.last_accessed_at),
        "access_count": item.access_count,
        "expires_at": _dt_to_text(item.expires_at),
        "supersedes": item.supersedes,
        "related": item.related,
    }


def memory_from_dict(data: dict[str, Any]) -> MemoryItem:
    item = MemoryItem(
        id=str(data["id"]),
        name=slugify(str(data["name"])),
        description=str(data["description"]),
        type=data["type"],
        content=str(data["content"]),
        user_id=data.get("user_id") or "default",
        project_id=data.get("project_id"),
        workspace_id=data.get("workspace_id"),
        tags=list(data.get("tags") or []),
        source=data.get("source") or "unknown",
        confidence=float(data.get("confidence") if data.get("confidence") is not None else 1.0),
        weight=int(data.get("weight") if data.get("weight") is not None else 5),
        status=data.get("status") or "active",
        created_at=_dt_from_text(data.get("created_at")),
        updated_at=_dt_from_text(data.get("updated_at")),
        last_accessed_at=_dt_from_text(data.get("last_accessed_at")),
        access_count=int(data.get("access_count") or 0),
        expires_at=_dt_from_text(data.get("expires_at")),
        supersedes=list(data.get("supersedes") or []),
        related=list(data.get("related") or []),
    )
    validate_memory_item(item)
    return item


def export_memories(store: FileMemoryStore, path: str | Path) -> dict[str, Any]:
    items = store.list_memories(include_archived=True)
    output = {"format": EXPORT_FORMAT, "memories": [memory_to_dict(item) for item in items]}
    safe_json_write(Path(path), output)
    return {"exported": len(items), "path": str(path)}


def import_memories(store: FileMemoryStore, path: str | Path) -> dict[str, Any]:
    data = safe_json_load(Path(path), default={})
    if data.get("format") != EXPORT_FORMAT:
        raise ValueError(f"unsupported import format: {data.get('format')}")
    memories = data.get("memories")
    if not isinstance(memories, list):
        raise ValueError("memories must be a list")

    existing = store.list_memories(include_archived=True)
    existing_ids = {item.id for item in existing}
    existing_names = {slugify(item.name) for item in existing}
    report = {"imported": 0, "skipped": 0, "errors": []}

    for index, raw_item in enumerate(memories):
        try:
            if not isinstance(raw_item, dict):
                raise ValueError("memory entry must be an object")
            item = memory_from_dict(raw_item)
            if item.id in existing_ids or slugify(item.name) in existing_names:
                report["skipped"] += 1
                continue
            store.save_memory(item)
            existing_ids.add(item.id)
            existing_names.add(slugify(item.name))
            report["imported"] += 1
        except Exception as exc:  # noqa: BLE001 - reports per-item import errors and continues
            report["errors"].append({"index": index, "error": str(exc)})
    return report


def _expected_index(store: FileMemoryStore, items: list[MemoryItem]) -> str:
    lines = [
        f"- [{item.name}](memories/{item.name}.md) — {item.description}"
        for item in sorted(items, key=lambda memory: memory.name)
        if item.status == "active"
    ]
    return ("\n".join(lines) + "\n") if lines else ""


def verify_memories(store: FileMemoryStore) -> dict[str, Any]:
    store.init_storage()
    report = {"checked": 0, "errors": [], "index_ok": True}
    items = []
    for path in sorted(store.memories_dir.glob("*.md")):
        try:
            item = store._item_from_text(path.read_text(encoding="utf-8"))
            items.append(item)
            report["checked"] += 1
        except Exception as exc:  # noqa: BLE001 - verification reports file errors
            report["errors"].append({"path": str(path), "error": str(exc)})
    expected = _expected_index(store, items)
    actual = store.index_path.read_text(encoding="utf-8") if store.index_path.exists() else ""
    report["index_ok"] = actual == expected
    return report


def rebuild_index(store: FileMemoryStore) -> None:
    store.rebuild_index()


def backup_memories(store: FileMemoryStore, path: str | Path) -> dict[str, Any]:
    return export_memories(store, path)
```

- [ ] **Step 4: Implement manager facade methods**

Modify `memora/manager.py`:

1. Add imports:

```python
from pathlib import Path

from .portable import backup_memories, export_memories, import_memories, rebuild_index, verify_memories
```

2. Add methods after `delete_memory()`:

```python
    def export_memories(self, path: str | Path) -> dict:
        return export_memories(self.memory_store, path)

    def import_memories(self, path: str | Path) -> dict:
        return import_memories(self.memory_store, path)

    def verify_memories(self) -> dict:
        return verify_memories(self.memory_store)

    def rebuild_index(self) -> None:
        rebuild_index(self.memory_store)

    def backup(self, path: str | Path) -> dict:
        return backup_memories(self.memory_store, path)
```

- [ ] **Step 5: Run portable tests**

Run:

```bash
pytest tests/test_portable.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit portable module and manager API**

Run:

```bash
git add memora/portable.py memora/manager.py tests/test_portable.py
git commit -m "feat: add memory portability helpers" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: CLI Commands

**Files:**
- Modify: `memora/cli.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes manager methods from Task 1.
- Produces CLI commands: `export`, `import`, `verify`, `rebuild-index`, `backup`.

- [ ] **Step 1: Add failing CLI tests**

Append to `tests/test_cli.py`:

```python

def test_export_import_verify_rebuild_and_backup_commands(tmp_path: Path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    export_path = tmp_path / "memories.json"
    backup_path = tmp_path / "backup.json"

    assert save_language(source).returncode == 0

    exported = run_cli(source, "export", str(export_path))
    assert exported.returncode == 0
    assert "exported 1 memories" in exported.stdout
    assert export_path.exists()

    imported = run_cli(target, "import", str(export_path))
    assert imported.returncode == 0
    assert "imported 1 skipped 0 errors 0" in imported.stdout

    duplicate = run_cli(target, "import", str(export_path))
    assert duplicate.returncode == 0
    assert "imported 0 skipped 1 errors 0" in duplicate.stdout

    verified = run_cli(target, "verify")
    assert verified.returncode == 0
    assert "verified 1 memories" in verified.stdout
    assert "index_ok=True" in verified.stdout
    assert "errors=0" in verified.stdout

    rebuilt = run_cli(target, "rebuild-index")
    assert rebuilt.returncode == 0
    assert "rebuilt index" in rebuilt.stdout

    backed_up = run_cli(target, "backup", str(backup_path))
    assert backed_up.returncode == 0
    assert "backed up 1 memories" in backed_up.stdout
    assert backup_path.exists()
```

- [ ] **Step 2: Run CLI test to verify it fails**

Run:

```bash
pytest tests/test_cli.py::test_export_import_verify_rebuild_and_backup_commands -v
```

Expected: FAIL because commands do not exist.

- [ ] **Step 3: Add CLI parsers**

Modify `build_parser()` in `memora/cli.py` after the `delete` parser:

```python
    export_parser = subparsers.add_parser("export", help="Export memories to JSON.")
    export_parser.add_argument("path")

    import_parser = subparsers.add_parser("import", help="Import memories from JSON.")
    import_parser.add_argument("path")

    subparsers.add_parser("verify", help="Verify memory store health.")
    subparsers.add_parser("rebuild-index", help="Rebuild the memory index.")

    backup_parser = subparsers.add_parser("backup", help="Back up memories to JSON.")
    backup_parser.add_argument("path")
```

- [ ] **Step 4: Add CLI command handlers**

Modify `_run_command()` in `memora/cli.py` after the `delete` handler:

```python
    if args.command == "export":
        report = manager.export_memories(args.path)
        print(f"exported {report['exported']} memories to {args.path}")
        return 0

    if args.command == "import":
        report = manager.import_memories(args.path)
        print(f"imported {report['imported']} skipped {report['skipped']} errors {len(report['errors'])}")
        return 0

    if args.command == "verify":
        report = manager.verify_memories()
        print(f"verified {report['checked']} memories index_ok={report['index_ok']} errors={len(report['errors'])}")
        for error in report["errors"]:
            print(f"error: {error}")
        return 0

    if args.command == "rebuild-index":
        manager.rebuild_index()
        print("rebuilt index")
        return 0

    if args.command == "backup":
        report = manager.backup(args.path)
        print(f"backed up {report['exported']} memories to {args.path}")
        return 0
```

- [ ] **Step 5: Run CLI tests**

Run:

```bash
pytest tests/test_cli.py::test_export_import_verify_rebuild_and_backup_commands -v
```

Expected: PASS.

- [ ] **Step 6: Commit CLI commands**

Run:

```bash
git add memora/cli.py tests/test_cli.py
git commit -m "feat: add portability CLI commands" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: README and Final Verification

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes CLI commands from Task 2.
- Produces README portability examples.

- [ ] **Step 1: Add README examples**

In `README.md`, add these commands to the CLI quickstart before `clean`:

```bash
python -m memora --root .memora export memories.json
python -m memora --root .memora import memories.json
python -m memora --root .memora verify
python -m memora --root .memora rebuild-index
python -m memora --root .memora backup backup.json
```

Add a short section after CLI error behavior:

```markdown
## Data portability and integrity

Memora can export, import, verify, rebuild, and back up memory files:

```bash
python -m memora --root .memora export memories.json
python -m memora --root .memora import memories.json
python -m memora --root .memora verify
python -m memora --root .memora rebuild-index
python -m memora --root .memora backup backup.json
```

These commands cover memories only, not session history.
```

- [ ] **Step 2: Run full test suite**

Run:

```bash
pytest -v
```

Expected: PASS.

- [ ] **Step 3: Commit README update**

Run:

```bash
git add README.md
git commit -m "docs: document data portability commands" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Self-Review

Spec coverage:
- Export: Task 1 and Task 2.
- Import: Task 1 and Task 2.
- Verify: Task 1 and Task 2.
- Rebuild index: Task 1 and Task 2.
- Backup alias: Task 1 and Task 2.
- README documentation: Task 3.
- No sessions/compression/encryption/cloud/new deps: Global Constraints.

Placeholder scan:
- No TBD, TODO, or incomplete implementation steps.

Type consistency:
- Manager method names match the approved design.
- CLI command names match the approved design.
- Export format string is consistent: `memora.memories.v1`.
