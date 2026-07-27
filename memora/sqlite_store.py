"""SQLite memory store for Memora."""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .config import MemoryConfig
from .errors import MemoryNotFoundError, MemoryValidationError
from .schema import MemoryItem, MemoryQuery, validate_memory_item, validate_memory_status
from .stores import MemoryCandidateStore, _dt_from_text, _dt_to_text
from .utils import slugify

SCHEMA_VERSION = "1"


class SQLiteMemoryStore(MemoryCandidateStore):
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.root = Path(config.root_dir)
        self.db_path = Path(config.sqlite_path) if config.sqlite_path is not None else self.root / "memora.sqlite3"

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def init_storage(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    user_id TEXT NOT NULL DEFAULT 'default',
                    project_id TEXT,
                    workspace_id TEXT,
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    source TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    weight INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT,
                    updated_at TEXT,
                    last_accessed_at TEXT,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    expires_at TEXT,
                    supersedes_json TEXT NOT NULL DEFAULT '[]',
                    related_json TEXT NOT NULL DEFAULT '[]'
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(user_id, project_id, workspace_id)")
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_scope_name_unique
                ON memories(name, user_id, COALESCE(project_id, ''), COALESCE(workspace_id, ''))
                """
            )
            connection.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                    memory_id UNINDEXED,
                    name,
                    description,
                    tags,
                    content,
                    tokenize='unicode61'
                )
                """
            )
            connection.execute(
                "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('schema_version', ?)",
                (SCHEMA_VERSION,),
            )

    def _row_to_item(self, row: sqlite3.Row) -> MemoryItem:
        item = MemoryItem(
            id=str(row["id"]),
            name=slugify(str(row["name"])),
            description=str(row["description"]),
            type=row["type"],
            content=str(row["content"]),
            user_id=row["user_id"] or "default",
            project_id=row["project_id"],
            workspace_id=row["workspace_id"],
            tags=json.loads(row["tags_json"] or "[]"),
            source=row["source"] or "unknown",
            confidence=float(row["confidence"]),
            weight=int(row["weight"]),
            status=row["status"] or "active",
            created_at=_dt_from_text(row["created_at"]),
            updated_at=_dt_from_text(row["updated_at"]),
            last_accessed_at=_dt_from_text(row["last_accessed_at"]),
            access_count=int(row["access_count"] or 0),
            expires_at=_dt_from_text(row["expires_at"]),
            supersedes=json.loads(row["supersedes_json"] or "[]"),
            related=json.loads(row["related_json"] or "[]"),
        )
        validate_memory_item(item)
        return item

    def _item_values(self, item: MemoryItem) -> tuple[Any, ...]:
        return (
            item.id,
            item.name,
            item.description,
            item.type,
            item.content,
            item.user_id,
            item.project_id,
            item.workspace_id,
            json.dumps(item.tags, ensure_ascii=False),
            item.source,
            item.confidence,
            item.weight,
            item.status,
            _dt_to_text(item.created_at),
            _dt_to_text(item.updated_at),
            _dt_to_text(item.last_accessed_at),
            item.access_count,
            _dt_to_text(item.expires_at),
            json.dumps(item.supersedes, ensure_ascii=False),
            json.dumps(item.related, ensure_ascii=False),
        )

    def _sync_fts(self, connection: sqlite3.Connection, item: MemoryItem) -> None:
        connection.execute("DELETE FROM memory_fts WHERE memory_id = ?", (item.id,))
        connection.execute(
            "INSERT INTO memory_fts(memory_id, name, description, tags, content) VALUES (?, ?, ?, ?, ?)",
            (item.id, item.name, item.description, " ".join(item.tags), item.content),
        )

    def save_memory(self, item: MemoryItem) -> MemoryItem:
        self.init_storage()
        from .utils import now_utc

        now = now_utc()
        item.name = slugify(item.name)
        item.created_at = item.created_at or now
        item.updated_at = item.updated_at or now
        validate_memory_item(item)
        with self._connection() as connection:
            existing_ids = [
                row["id"]
                for row in connection.execute(
                    """
                    SELECT id FROM memories
                    WHERE id = ?
                       OR (
                           name = ?
                           AND user_id = ?
                           AND COALESCE(project_id, '') = COALESCE(?, '')
                           AND COALESCE(workspace_id, '') = COALESCE(?, '')
                       )
                    """,
                    (item.id, item.name, item.user_id, item.project_id, item.workspace_id),
                )
            ]
            for existing_id in existing_ids:
                connection.execute("DELETE FROM memory_fts WHERE memory_id = ?", (existing_id,))
            connection.execute(
                """
                INSERT OR REPLACE INTO memories(
                    id, name, description, type, content, user_id, project_id, workspace_id,
                    tags_json, source, confidence, weight, status, created_at, updated_at,
                    last_accessed_at, access_count, expires_at, supersedes_json, related_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                self._item_values(item),
            )
            self._sync_fts(connection, item)
        return item

    def list_memories(self, include_archived: bool = False) -> list[MemoryItem]:
        self.init_storage()
        query = "SELECT * FROM memories"
        params: tuple[Any, ...] = ()
        if not include_archived:
            query += " WHERE status = ?"
            params = ("active",)
        query += " ORDER BY name"
        with self._connection() as connection:
            return [self._row_to_item(row) for row in connection.execute(query, params)]

    def get_memory(self, identifier: str) -> MemoryItem | None:
        self.init_storage()
        wanted = slugify(identifier)
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM memories
                WHERE id = ? OR name = ?
                ORDER BY user_id, COALESCE(project_id, ''), COALESCE(workspace_id, ''), name
                LIMIT 1
                """,
                (identifier, wanted),
            ).fetchone()
        return self._row_to_item(row) if row is not None else None

    def update_memory(self, item: MemoryItem) -> MemoryItem:
        from .utils import now_utc

        item.updated_at = now_utc()
        return self.save_memory(item)

    def set_memory_status(self, identifier: str, status: str) -> MemoryItem:
        validate_memory_status(status)
        item = self.get_memory(identifier)
        if item is None:
            raise MemoryNotFoundError(f"memory not found: {identifier}")
        item.status = status
        return self.update_memory(item)

    def hard_delete_memory(self, identifier: str) -> None:
        item = self.get_memory(identifier)
        if item is None:
            raise MemoryNotFoundError(f"memory not found: {identifier}")
        with self._connection() as connection:
            connection.execute("DELETE FROM memory_fts WHERE memory_id = ?", (item.id,))
            connection.execute("DELETE FROM memories WHERE id = ?", (item.id,))

    def delete_memory(self, identifier: str, soft_delete: bool = True) -> None:
        item = self.get_memory(identifier)
        if item is None:
            return
        if soft_delete:
            item.status = "archived"
            self.update_memory(item)
            return
        self.hard_delete_memory(item.id)

    def rebuild_index(self) -> None:
        self.init_storage()
        with self._connection() as connection:
            connection.execute("DELETE FROM memory_fts")
            for row in connection.execute("SELECT * FROM memories ORDER BY name"):
                self._sync_fts(connection, self._row_to_item(row))

    def _fts_query(self, query: str) -> str:
        terms = re.findall(r"[a-z0-9_]+", (query or "").lower())
        if not terms:
            return ""
        return " OR ".join(f'"{term}"' for term in terms)

    def search_candidates(self, query: MemoryQuery) -> list[MemoryItem]:
        if not self.config.fts_enabled:
            return []
        self.init_storage()
        fts_query = self._fts_query(query.query)
        if not fts_query:
            return []
        limit = max(self.config.fts_candidate_limit, query.top_k * 5)
        clauses = ["memory_fts MATCH ?", "memories.user_id = ?"]
        params: list[Any] = [fts_query, query.user_id]
        if query.project_id is not None:
            clauses.append("memories.project_id = ?")
            params.append(query.project_id)
        if query.workspace_id is not None:
            clauses.append("memories.workspace_id = ?")
            params.append(query.workspace_id)
        if not query.include_archived:
            clauses.append("memories.status = 'active'")
        if query.memory_types:
            placeholders = ", ".join("?" for _ in query.memory_types)
            clauses.append(f"memories.type IN ({placeholders})")
            params.extend(query.memory_types)
        if query.tags:
            for tag in query.tags:
                clauses.append("memories.tags_json LIKE ?")
                params.append(f'%"{tag}"%')
        params.append(limit)
        try:
            with self._connection() as connection:
                rows = connection.execute(
                    f"""
                    SELECT memories.*
                    FROM memory_fts
                    JOIN memories ON memories.id = memory_fts.memory_id
                    WHERE {' AND '.join(clauses)}
                    ORDER BY rank
                    LIMIT ?
                    """,
                    params,
                ).fetchall()
        except sqlite3.Error:
            return []
        return [self._row_to_item(row) for row in rows]

    def verify(self) -> dict[str, Any]:
        self.init_storage()
        report = {"checked": 0, "errors": [], "index_ok": True}
        with self._connection() as connection:
            rows = connection.execute("SELECT * FROM memories ORDER BY name").fetchall()
            memory_ids = set()
            for row in rows:
                try:
                    item = self._row_to_item(row)
                    memory_ids.add(item.id)
                    report["checked"] += 1
                except Exception as exc:  # noqa: BLE001 - verification reports row errors
                    report["errors"].append({"id": row["id"], "error": str(exc)})
            fts_ids = {row["memory_id"] for row in connection.execute("SELECT memory_id FROM memory_fts")}
        report["index_ok"] = memory_ids == fts_ids
        return report
