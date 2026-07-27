"""Thin runtime integration helpers for external agent runtimes."""

from __future__ import annotations

from .config import MemoryConfig
from .manager import MemoryManager
from .schema import MemoryCandidate, MemoryItem, MemorySearchResult, MemoryWriteResult, SessionMessage


class MemoryRuntime:
    def __init__(self, manager: MemoryManager | None = None, config: MemoryConfig | None = None):
        if manager is not None and config is not None:
            raise ValueError("manager and config cannot both be provided")
        self.manager = manager or MemoryManager(config)

    def init_storage(self) -> None:
        self.manager.init_storage()

    def retrieve_context(self, query: str, **kwargs) -> list[MemorySearchResult]:
        return self.manager.retrieve_memory(query, **kwargs)

    def build_context(self, query: str, **kwargs) -> str:
        results = self.retrieve_context(query, **kwargs)
        return self.manager.format_memories_for_prompt(results=results)

    def remember_message(self, session_id: str, role: str, content: str, user_id: str = "default") -> None:
        self.manager.append_message(user_id, session_id, SessionMessage(role=role, content=content))

    def remember_summary(
        self,
        session_id: str,
        content: str,
        user_id: str = "default",
        project_id: str | None = None,
        workspace_id: str | None = None,
    ) -> MemoryItem:
        return self.manager.save_memory(
            memory_type="episodic",
            name=f"{session_id}-summary",
            description=f"Summary for session {session_id}",
            content=content,
            user_id=user_id,
            project_id=project_id,
            workspace_id=workspace_id,
            source="runtime",
        )

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
        weight: int | None = None,
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

    def confirm_memory_candidate(
        self,
        candidate: MemoryCandidate,
        action: str | None = None,
        target_memory_id: str | None = None,
    ) -> MemoryWriteResult:
        return self.manager.confirm_memory_candidate(candidate, action=action, target_memory_id=target_memory_id)

    def mark_context_used(self, results: list[MemorySearchResult]) -> None:
        self.manager.mark_memories_used(results)
