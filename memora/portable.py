"""Memory import, export, backup, and verification helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from .errors import MemoryValidationError
from .schema import MemoryItem, validate_memory_item
from .stores import MemoryStore
from .utils import safe_json_load, safe_json_write, slugify

EXPORT_FORMAT = "memora.memories.v1"


def _dt_to_text(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _dt_from_text(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise MemoryValidationError(f"invalid datetime value: {value}") from exc


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


def export_memories(store: MemoryStore, path: str | Path) -> dict[str, Any]:
    items = store.list_memories(include_archived=True)
    output = {"format": EXPORT_FORMAT, "memories": [memory_to_dict(item) for item in items]}
    safe_json_write(Path(path), output)
    return {"exported": len(items), "path": str(path)}


def import_memories(store: MemoryStore, path: str | Path) -> dict[str, Any]:
    data = safe_json_load(Path(path), default=None)
    if data is None:
        raise MemoryValidationError(f"import file not found: {path}")
    if not isinstance(data, dict):
        raise MemoryValidationError("import file must contain a JSON object")
    if data.get("format") != EXPORT_FORMAT:
        raise MemoryValidationError(f"unsupported import format: {data.get('format')}")
    memories = data.get("memories")
    if not isinstance(memories, list):
        raise MemoryValidationError("memories must be a list")

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


def _expected_index(store, entries: list[tuple[Path, MemoryItem]]) -> str:
    lines = []
    for path, item in sorted(entries, key=lambda entry: entry[1].name):
        if item.status == "active":
            relative = path.relative_to(store.root).as_posix()
            lines.append(f"- [{item.name}]({relative}) — {item.description}")
    return ("\n".join(lines) + "\n") if lines else ""


def verify_memories(store: MemoryStore) -> dict[str, Any]:
    if hasattr(store, "verify"):
        return store.verify()
    store.init_storage()
    report = {"checked": 0, "errors": [], "index_ok": True}
    entries = []
    for path in sorted(store.memories_dir.rglob("*.md")):
        try:
            item = store._item_from_text(path.read_text(encoding="utf-8"))
            entries.append((path, item))
            report["checked"] += 1
        except Exception as exc:  # noqa: BLE001 - verification reports file errors
            report["errors"].append({"path": str(path), "error": str(exc)})
    expected = _expected_index(store, entries)
    actual = store.index_path.read_text(encoding="utf-8") if store.index_path.exists() else ""
    report["index_ok"] = actual == expected
    return report


def rebuild_index(store: MemoryStore) -> None:
    store.rebuild_index()


def backup_memories(store: MemoryStore, path: str | Path) -> dict[str, Any]:
    return export_memories(store, path)
