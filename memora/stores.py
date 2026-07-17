"""Local file stores for Memora memories and sessions."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import MemoryConfig
from .errors import MemoryValidationError
from .schema import MemoryItem, SessionMessage, validate_memory_item, validate_session_message
from .utils import dump_frontmatter, now_utc, parse_frontmatter, safe_json_load, safe_json_write, slugify


def _dt_to_text(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _dt_from_text(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise MemoryValidationError(f"invalid datetime value: {value}") from exc


def _clean_dict(data: dict[str, Any]) -> dict[str, Any]:
    cleaned = {}
    for key, value in data.items():
        if isinstance(value, datetime):
            cleaned[key] = value.isoformat()
        elif is_dataclass(value):
            cleaned[key] = _clean_dict(asdict(value))
        elif isinstance(value, dict):
            cleaned[key] = _clean_dict(value)
        elif isinstance(value, list):
            cleaned[key] = [
                item.isoformat() if isinstance(item, datetime) else item
                for item in value
            ]
        else:
            cleaned[key] = value
    return cleaned


def _validate_session_id(session_id: str) -> None:
    if not isinstance(session_id, str) or not session_id.strip():
        raise MemoryValidationError("session_id must be a non-empty string")
    if "/" in session_id or "\\" in session_id or ".." in session_id:
        raise MemoryValidationError("session_id must not contain path separators or '..'")


class FileMemoryStore:
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.root = Path(config.root_dir)
        self.index_path = self.root / "MEMORY.md"
        self.memories_dir = self.root / "memories"
        self.sessions_dir = self.root / "sessions"
        self.summaries_dir = self.root / "summaries"
        self.archive_dir = self.root / "archive"

    def init_storage(self) -> None:
        self.memories_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.summaries_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        if not self.index_path.exists():
            self.index_path.write_text("", encoding="utf-8")

    def _path_for_name(self, name: str) -> Path:
        return self.memories_dir / f"{slugify(name)}.md"

    def _item_to_text(self, item: MemoryItem) -> str:
        metadata = {
            "name": item.name,
            "description": item.description,
            "metadata": {
                "id": item.id,
                "type": item.type,
                "user_id": item.user_id,
                "project_id": item.project_id,
                "workspace_id": item.workspace_id,
                "source": item.source,
                "confidence": item.confidence,
                "weight": item.weight,
                "status": item.status,
                "created_at": _dt_to_text(item.created_at),
                "updated_at": _dt_to_text(item.updated_at),
                "last_accessed_at": _dt_to_text(item.last_accessed_at),
                "access_count": item.access_count,
                "expires_at": _dt_to_text(item.expires_at),
                "tags": item.tags,
                "supersedes": item.supersedes,
                "related": item.related,
            },
        }
        return dump_frontmatter(metadata, item.content)

    def _item_from_text(self, text: str) -> MemoryItem:
        frontmatter, body = parse_frontmatter(text)
        meta = frontmatter.get("metadata") or {}
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

    def save_memory(self, item: MemoryItem) -> MemoryItem:
        self.init_storage()
        now = now_utc()
        item.name = slugify(item.name)
        item.created_at = item.created_at or now
        item.updated_at = item.updated_at or now
        validate_memory_item(item)
        self._path_for_name(item.name).write_text(self._item_to_text(item), encoding="utf-8")
        self.rebuild_index()
        return item

    def list_memories(self, include_archived: bool = False) -> list[MemoryItem]:
        self.init_storage()
        items = []
        for path in sorted(self.memories_dir.glob("*.md")):
            item = self._item_from_text(path.read_text(encoding="utf-8"))
            if item.status == "active" or include_archived:
                items.append(item)
        return items

    def get_memory(self, identifier: str) -> MemoryItem | None:
        self.init_storage()
        wanted = slugify(identifier)
        for item in self.list_memories(include_archived=True):
            if item.id == identifier or item.name == wanted:
                return item
        return None

    def update_memory(self, item: MemoryItem) -> MemoryItem:
        item.updated_at = now_utc()
        return self.save_memory(item)

    def delete_memory(self, identifier: str, soft_delete: bool = True) -> None:
        item = self.get_memory(identifier)
        if item is None:
            return
        path = self._path_for_name(item.name)
        if soft_delete:
            item.status = "archived"
            self.update_memory(item)
        elif path.exists():
            path.unlink()
            self.rebuild_index()

    def rebuild_index(self) -> None:
        self.memories_dir.mkdir(parents=True, exist_ok=True)
        lines = []
        for path in sorted(self.memories_dir.glob("*.md")):
            item = self._item_from_text(path.read_text(encoding="utf-8"))
            if item.status == "active":
                lines.append(f"- [{item.name}](memories/{item.name}.md) — {item.description}")
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")


class FileSessionStore:
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.root = Path(config.root_dir)
        self.sessions_dir = self.root / "sessions"

    def init_storage(self) -> None:
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        _validate_session_id(session_id)
        return self.sessions_dir / f"{session_id}.json"

    def save_session(self, session: dict[str, Any]) -> None:
        self.init_storage()
        _validate_session_id(str(session["id"]))
        safe_json_write(self._path(str(session["id"])), _clean_dict(session))

    def load_session(self, user_id: str, session_id: str) -> dict[str, Any] | None:
        self.init_storage()
        session = safe_json_load(self._path(session_id), default=None)
        if session is None or session.get("user_id") != user_id:
            return None
        return session

    def append_message(self, user_id: str, session_id: str, message: SessionMessage) -> None:
        _validate_session_id(session_id)
        validate_session_message(message)
        session = self.load_session(user_id, session_id) or {
            "id": session_id,
            "user_id": user_id,
            "created_at": now_utc().isoformat(),
            "updated_at": now_utc().isoformat(),
            "working_memory": {},
            "history": [],
        }
        message.created_at = message.created_at or now_utc()
        session.setdefault("history", []).append(_clean_dict(asdict(message)))
        session["updated_at"] = now_utc().isoformat()
        self.save_session(session)
