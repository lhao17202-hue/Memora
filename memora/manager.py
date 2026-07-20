"""Public facade for Memora."""

from __future__ import annotations

import re
import uuid
from pathlib import Path

from .config import MemoryConfig
from .errors import MemoryNotFoundError, MemoryPolicyError, MemoryValidationError
from .formatter import MemoryFormatter
from .lifecycle import LifecycleManager
from .policy import MemoryPolicy
from .portable import backup_memories, export_memories, import_memories, rebuild_index, verify_memories
from .retriever import MemoryRetriever
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
from .session import SessionService
from .sqlite_store import SQLiteMemoryStore
from .stores import FileMemoryStore, FileSessionStore, MemoryCandidateStore, MemoryStore, SessionStore
from .utils import now_utc, slugify


class MemoryManager:
    def __init__(
        self,
        config: MemoryConfig | None = None,
        memory_store: MemoryStore | None = None,
        session_store: SessionStore | None = None,
    ):
        self.config = config or MemoryConfig()
        self.memory_store = memory_store or self._build_memory_store()
        self.session_store = session_store or FileSessionStore(self.config)
        self.session_service = SessionService(self.session_store)
        self.policy = MemoryPolicy()
        self.retriever = MemoryRetriever()
        self.formatter = MemoryFormatter()
        self.lifecycle = LifecycleManager(self.config)

    def _build_memory_store(self) -> MemoryStore:
        if self.config.memory_backend == "file":
            return FileMemoryStore(self.config)
        if self.config.memory_backend == "sqlite":
            return SQLiteMemoryStore(self.config)
        raise MemoryValidationError(f"unsupported memory backend: {self.config.memory_backend}")

    def init_storage(self) -> None:
        self.memory_store.init_storage()
        self.session_store.init_storage()

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
        decision = self.policy.evaluate(
            candidate,
            self._scoped_memories(candidate.user_id, candidate.project_id, candidate.workspace_id, include_archived=False),
        )
        return self._write_result_from_decision(decision)

    def remember_candidate(self, candidate: MemoryCandidate) -> MemoryWriteResult:
        validate_memory_candidate(candidate)
        decision = self.policy.evaluate(
            candidate,
            self._scoped_memories(candidate.user_id, candidate.project_id, candidate.workspace_id, include_archived=False),
        )
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

    def save_memory(
        self,
        memory_type: str,
        content: str,
        description: str,
        name: str | None = None,
        user_id: str = "default",
        project_id: str | None = None,
        workspace_id: str | None = None,
        tags: list[str] | None = None,
        weight: int = 5,
        confidence: float = 1.0,
        source: str = "manual",
    ) -> MemoryItem:
        candidate = MemoryCandidate(
            action="create",
            name=name or description,
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
        )
        validate_memory_candidate(candidate)
        decision = self.policy.evaluate(
            candidate,
            self._scoped_memories(candidate.user_id, candidate.project_id, candidate.workspace_id, include_archived=False),
        )
        if decision.action == "reject":
            raise MemoryPolicyError(f"memory rejected: {decision.reason}")
        if decision.action == "ask_user":
            raise MemoryPolicyError(f"memory requires confirmation: {decision.reason}")

        if decision.action == "update" and decision.target_memory_id:
            existing = self.memory_store.get_memory(decision.target_memory_id)
            if existing is None:
                raise ValueError("target memory missing for update")
            return self._apply_candidate_update(existing, candidate)

        item = self._new_memory_from_candidate(decision)
        return self.memory_store.save_memory(item)

    def list_memories(self, include_archived: bool = False) -> list[MemoryItem]:
        return self.memory_store.list_memories(include_archived=include_archived)

    def get_memory(self, identifier: str) -> MemoryItem | None:
        return self.memory_store.get_memory(identifier)

    def _scoped_memories(self, user_id: str, project_id: str | None, workspace_id: str | None, include_archived: bool) -> list[MemoryItem]:
        return [
            memory
            for memory in self.memory_store.list_memories(include_archived=include_archived)
            if memory.user_id == user_id
            and (project_id is None or memory.project_id == project_id)
            and (workspace_id is None or memory.workspace_id == workspace_id)
        ]

    def _query_needs_full_scan_fallback(self, query: MemoryQuery) -> bool:
        return bool(re.search(r"[一-鿿]", query.query))

    def _candidate_memories(self, query: MemoryQuery) -> list[MemoryItem]:
        scoped_memories = self._scoped_memories(query.user_id, query.project_id, query.workspace_id, query.include_archived)
        if isinstance(self.memory_store, MemoryCandidateStore):
            candidates = self.memory_store.search_candidates(query)
            if candidates and not self._query_needs_full_scan_fallback(query):
                return candidates
            if candidates:
                merged = {memory.id: memory for memory in scoped_memories}
                merged.update({memory.id: memory for memory in candidates})
                return list(merged.values())
        return scoped_memories

    def retrieve_memory(
        self,
        query: str,
        user_id: str = "default",
        project_id: str | None = None,
        workspace_id: str | None = None,
        memory_types: list[str] | None = None,
        tags: list[str] | None = None,
        top_k: int | None = None,
        include_archived: bool = False,
        include_knowledge: bool = True,
    ) -> list[MemorySearchResult]:
        memory_query = MemoryQuery(
            query=query,
            user_id=user_id,
            project_id=project_id,
            workspace_id=workspace_id,
            memory_types=memory_types,
            tags=tags,
            top_k=top_k or self.config.max_retrieved_memories,
            max_tokens=self.config.max_memory_prompt_tokens,
            include_archived=include_archived,
            include_knowledge=include_knowledge,
        )
        validate_memory_query(memory_query)
        return self.retriever.retrieve(self._candidate_memories(memory_query), memory_query)

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

    def archive_memory(self, identifier: str) -> MemoryItem:
        return self.memory_store.set_memory_status(identifier, "archived")

    def restore_memory(self, identifier: str) -> MemoryItem:
        return self.memory_store.set_memory_status(identifier, "active")

    def delete_memory(self, identifier: str, hard: bool = False) -> None:
        if hard:
            self.memory_store.hard_delete_memory(identifier)
            return
        self.memory_store.set_memory_status(identifier, "deleted")

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

    def mark_memories_used(self, results: list[MemorySearchResult]) -> None:
        now = now_utc()
        for result in results:
            memory = self.memory_store.get_memory(result.memory.id)
            if memory is None:
                continue
            memory.access_count += 1
            memory.last_accessed_at = now
            self.memory_store.update_memory(memory)

    def format_memories_for_prompt(
        self,
        results: list[MemorySearchResult] | None = None,
        query: str | None = None,
        **kwargs,
    ) -> str:
        if results is None:
            if query is None:
                results = []
            else:
                results = self.retrieve_memory(query=query, **kwargs)
        return self.formatter.format_results(results, max_tokens=self.config.max_memory_prompt_tokens)

    def append_message(self, user_id: str, session_id: str, message: SessionMessage) -> None:
        self.session_service.append_message(user_id, session_id, message)

    def get_messages(self, user_id: str, session_id: str, limit: int | None = None) -> list[SessionMessage]:
        return self.session_service.get_messages(user_id, session_id, limit=limit)

    def clean_expired_memory(self, user_id: str | None = None) -> dict:
        report = {"archived": 0, "deleted": 0, "kept": 0, "errors": []}
        for memory in self.memory_store.list_memories(include_archived=False):
            if user_id is not None and memory.user_id != user_id:
                continue
            decision = self.lifecycle.decide(memory)
            if decision == "archive":
                memory.status = "archived"
                self.memory_store.update_memory(memory)
                report["archived"] += 1
            else:
                report["kept"] += 1
        return report
