"""Thin runtime integration helpers for external agent runtimes."""

from __future__ import annotations

from .config import MemoryConfig
from .extraction import ExtractionArtifact, MemoryExtractor
from .manager import MemoryManager
from .schema import MemoryCandidate, MemoryItem, MemorySearchResult, MemoryWriteResult, SessionMessage
from .taxonomy import ON_DEMAND_CONTEXT_TYPES


class MemoryRuntime:
    def __init__(
        self,
        manager: MemoryManager | None = None,
        config: MemoryConfig | None = None,
        extractor: MemoryExtractor | None = None,
    ):
        if manager is not None and config is not None:
            raise ValueError("manager and config cannot both be provided")
        self.manager = manager or MemoryManager(config)
        self.extractor = extractor

    def init_storage(self) -> None:
        self.manager.init_storage()

    def retrieve_context(self, query: str, **kwargs) -> list[MemorySearchResult]:
        return self.manager.retrieve_memory(query, **kwargs)

    def build_context(self, query: str, **kwargs) -> str:
        results = self.retrieve_context(query, **kwargs)
        return self.manager.format_memories_for_prompt(results=results)

    def retrieve_pinned_context(
        self,
        user_id: str = "default",
        project_id: str | None = None,
        workspace_id: str | None = None,
        top_k: int | None = None,
        include_archived: bool = False,
    ) -> list[MemorySearchResult]:
        return self.manager.retrieve_pinned_memories(
            user_id=user_id,
            project_id=project_id,
            workspace_id=workspace_id,
            top_k=top_k,
            include_archived=include_archived,
        )

    def build_pinned_context(self, **kwargs) -> str:
        results = self.retrieve_pinned_context(**kwargs)
        return self.manager.format_memories_for_prompt(results=results)

    def retrieve_task_context(
        self,
        query: str,
        user_id: str = "default",
        project_id: str | None = None,
        workspace_id: str | None = None,
        memory_types: list[str] | None = None,
        tags: list[str] | None = None,
        top_k: int | None = None,
        pinned_top_k: int | None = None,
        include_pinned: bool = True,
        include_archived: bool = False,
        include_knowledge: bool = True,
    ) -> list[MemorySearchResult]:
        pinned_results = []
        if include_pinned:
            pinned_results = self.manager.retrieve_pinned_memories(
                user_id=user_id,
                project_id=project_id,
                workspace_id=workspace_id,
                top_k=pinned_top_k,
                include_archived=include_archived,
            )
        retrieved_results = self.manager.retrieve_memory(
            query=query,
            user_id=user_id,
            project_id=project_id,
            workspace_id=workspace_id,
            memory_types=memory_types or list(ON_DEMAND_CONTEXT_TYPES),
            tags=tags,
            top_k=top_k,
            include_archived=include_archived,
            include_knowledge=include_knowledge,
        )
        return self._merge_context_results(pinned_results, retrieved_results)

    def build_task_context(self, query: str, **kwargs) -> str:
        results = self.retrieve_task_context(query, **kwargs)
        return self.manager.format_memories_for_prompt(results=results)

    def _merge_context_results(
        self,
        pinned_results: list[MemorySearchResult],
        retrieved_results: list[MemorySearchResult],
    ) -> list[MemorySearchResult]:
        merged = []
        seen_ids = set()
        for result in pinned_results + retrieved_results:
            if result.memory.id in seen_ids:
                continue
            merged.append(result)
            seen_ids.add(result.memory.id)
        return merged

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

    def extract_memories(
        self,
        messages: list[SessionMessage | dict[str, str]],
        extractor: MemoryExtractor | None = None,
    ) -> ExtractionArtifact:
        selected_extractor = extractor or self.extractor
        if selected_extractor is None:
            return ExtractionArtifact(
                should_remember=False,
                memories=[],
                errors=["memory_extractor_not_configured"],
                source="not_configured",
            )
        return selected_extractor.extract(messages)

    def remember_extraction_artifact(
        self,
        artifact: ExtractionArtifact,
        user_id: str = "default",
        project_id: str | None = None,
        workspace_id: str | None = None,
        session_id: str | None = None,
    ) -> list[MemoryWriteResult]:
        results = []
        for extracted in artifact.memories:
            candidate = extracted.to_candidate(
                user_id=user_id,
                project_id=project_id,
                workspace_id=workspace_id,
                session_id=session_id,
            )
            if extracted.requires_confirmation:
                evaluated = self.manager.evaluate_memory_candidate(candidate)
                if evaluated.action == "rejected":
                    results.append(evaluated)
                    continue
                pending = evaluated.candidate
                if pending is None:
                    results.append(evaluated)
                    continue
                pending.action = "ask_user"
                pending.reason = "low_confidence_extraction"
                results.append(
                    MemoryWriteResult(
                        action="requires_confirmation",
                        candidate=pending,
                        reason=pending.reason,
                        target_memory_id=pending.target_memory_id,
                    )
                )
                continue
            results.append(self.manager.remember_candidate(candidate))
        return results

    def extract_and_remember(
        self,
        messages: list[SessionMessage | dict[str, str]],
        user_id: str = "default",
        project_id: str | None = None,
        workspace_id: str | None = None,
        session_id: str | None = None,
        extractor: MemoryExtractor | None = None,
    ) -> tuple[ExtractionArtifact, list[MemoryWriteResult]]:
        artifact = self.extract_memories(messages, extractor=extractor)
        results = self.remember_extraction_artifact(
            artifact,
            user_id=user_id,
            project_id=project_id,
            workspace_id=workspace_id,
            session_id=session_id,
        )
        return artifact, results

    def confirm_memory_candidate(
        self,
        candidate: MemoryCandidate,
        action: str | None = None,
        target_memory_id: str | None = None,
    ) -> MemoryWriteResult:
        return self.manager.confirm_memory_candidate(candidate, action=action, target_memory_id=target_memory_id)

    def mark_context_used(self, results: list[MemorySearchResult]) -> None:
        self.manager.mark_memories_used(results)
