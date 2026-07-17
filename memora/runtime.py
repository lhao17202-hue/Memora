"""Thin runtime integration helpers for external agent runtimes."""

from __future__ import annotations

from .config import MemoryConfig
from .manager import MemoryManager
from .schema import MemoryItem, MemorySearchResult, SessionMessage


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
            memory_type="session_summary",
            name=f"{session_id}-summary",
            description=f"Summary for session {session_id}",
            content=content,
            user_id=user_id,
            project_id=project_id,
            workspace_id=workspace_id,
            source="runtime",
        )

    def mark_context_used(self, results: list[MemorySearchResult]) -> None:
        self.manager.mark_memories_used(results)
