"""Public facade for Memora."""

from __future__ import annotations

import math
import re
import uuid
from dataclasses import replace
from pathlib import Path

from .config import MemoryConfig
from .errors import MemoryNotFoundError, MemoryPolicyError, MemoryValidationError
from .formatter import MemoryFormatter
from .lifecycle import LifecycleManager
from .policy import MemoryPolicy
from .portable import backup_memories, export_memories, import_memories, rebuild_index, verify_memories
from .rag import RagIndex, RagRetriever, build_embedding_provider, build_reranker, build_vector_store
from .relations import MemoryRelationJudge, SemanticMemoryRelationResolver
from .retriever import MemoryRetriever
from .schema import (
    MemoryCandidate,
    MemoryItem,
    MemoryQuery,
    MemoryRelation,
    MemoryRelationDecision,
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
from .taxonomy import PINNED_CONTEXT_TYPES, configured_default_weight
from .utils import estimate_tokens, now_utc, slugify


class MemoryManager:
    def __init__(
        self,
        config: MemoryConfig | None = None,
        memory_store: MemoryStore | None = None,
        session_store: SessionStore | None = None,
        relation_judge: MemoryRelationJudge | None = None,
    ):
        self.config = config or MemoryConfig()
        self.memory_store = memory_store or self._build_memory_store()
        self.session_store = session_store or FileSessionStore(self.config)
        self.session_service = SessionService(self.session_store)
        self.policy = MemoryPolicy(self.config)
        self.retriever = MemoryRetriever()
        self.formatter = MemoryFormatter()
        self.lifecycle = LifecycleManager(self.config)
        self.rag_index: RagIndex | None = None
        self.rag_retriever: RagRetriever | None = None
        self.relation_resolver: SemanticMemoryRelationResolver | None = None
        self.relation_judge = relation_judge
        self._rag_sync_errors: list[dict[str, str]] = []
        embedder = None
        relation_resolution_enabled = self.config.semantic_write_relations_enabled or self.config.llm_relation_judge_enabled
        if self.config.rag_enabled or relation_resolution_enabled:
            embedder = build_embedding_provider(self.config)
        if relation_resolution_enabled and embedder is not None:
            self.relation_resolver = SemanticMemoryRelationResolver(embedder, self.config)
        if self.config.rag_enabled and embedder is not None:
            vector_store = build_vector_store(self.config)
            reranker = build_reranker(self.config)
            self.rag_index = RagIndex(self.memory_store, embedder, vector_store)
            candidate_store = self.memory_store if isinstance(self.memory_store, MemoryCandidateStore) else None
            self.rag_retriever = RagRetriever(
                self.memory_store,
                candidate_store,
                embedder,
                vector_store,
                self.retriever,
                reranker,
                self.config,
            )

    def _build_memory_store(self) -> MemoryStore:
        if self.config.memory_backend == "file":
            return FileMemoryStore(self.config)
        if self.config.memory_backend == "sqlite":
            return SQLiteMemoryStore(self.config)
        raise MemoryValidationError(f"unsupported memory backend: {self.config.memory_backend}")

    def init_storage(self) -> None:
        self.memory_store.init_storage()
        self.session_store.init_storage()
        if self.rag_index is not None:
            self.rag_index.init_storage()

    def _record_rag_sync_error(self, operation: str, memory_id: str, exc: Exception) -> None:
        self._rag_sync_errors.append({"operation": operation, "memory_id": memory_id, "error": str(exc)})

    def _sync_rag_memory(self, item: MemoryItem) -> None:
        if self.rag_index is None:
            return
        try:
            self.rag_index.sync_memory(item)
        except Exception as exc:  # noqa: BLE001 - memory writes should degrade when RAG sync fails
            self._record_rag_sync_error("sync", item.id, exc)
            return

    def _delete_rag_memory(self, memory_id: str) -> None:
        if self.rag_index is None:
            return
        try:
            self.rag_index.delete_memory(memory_id)
        except Exception as exc:  # noqa: BLE001 - lifecycle writes should degrade when RAG cleanup fails
            self._record_rag_sync_error("delete", memory_id, exc)
            return

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

    def _default_weight_for_type(self, memory_type: str) -> int:
        return configured_default_weight(memory_type, self.config)

    def _resolve_candidate_defaults(self, candidate: MemoryCandidate) -> MemoryCandidate:
        if candidate.weight is None:
            candidate.weight = self._default_weight_for_type(candidate.type)
        return candidate

    def _evaluate_candidate_decision(self, candidate: MemoryCandidate) -> MemoryCandidate:
        scoped = self._scoped_memories(candidate.user_id, candidate.project_id, candidate.workspace_id, include_archived=False)
        relation = None
        relation_decision = None
        if self.relation_resolver is not None and self.policy.rejection_reason(candidate) is None:
            relation = self.relation_resolver.resolve(candidate, scoped)
            relation_decision = self._judge_relation(candidate, scoped, relation)
        decision = self.policy.evaluate(candidate, scoped, relation=relation, relation_decision=relation_decision)
        if decision.target_memory_id is not None:
            self._apply_relation_decision(decision, relation_decision)
        return decision

    def _judge_relation(
        self,
        candidate: MemoryCandidate,
        scoped: list[MemoryItem],
        relation: MemoryRelation,
    ) -> MemoryRelationDecision | None:
        target = self._relation_target(scoped, relation)
        if target is None:
            return None
        if not self.config.llm_relation_judge_enabled:
            return None
        if self.relation_judge is None:
            return None
        try:
            decision = self.relation_judge.judge(candidate, target, relation)
        except Exception:  # noqa: BLE001 - invalid LLM decisions fall back to deterministic relation behavior
            return None
        return self._accepted_relation_decision(decision)

    def _accepted_relation_decision(self, decision: MemoryRelationDecision) -> MemoryRelationDecision:
        if decision.kind == "merge" and decision.confidence < self.config.llm_merge_confidence_threshold:
            return MemoryRelationDecision(kind="none", confidence=decision.confidence, reason="llm_merge_below_confidence_threshold")
        if decision.kind in {"duplicate", "none"} and decision.confidence < self.config.llm_relation_confidence_threshold:
            return MemoryRelationDecision(kind="none", confidence=decision.confidence, reason="llm_relation_below_confidence_threshold")
        return decision

    def _apply_relation_decision(self, candidate: MemoryCandidate, decision: MemoryRelationDecision | None) -> None:
        if decision is None or decision.kind != "merge":
            return
        if decision.merged_name is not None:
            candidate.name = decision.merged_name
            candidate.target_name = slugify(decision.merged_name)
        if decision.merged_description is not None:
            candidate.description = decision.merged_description
        if decision.merged_content is not None:
            candidate.content = decision.merged_content
        if decision.merged_tags is not None:
            candidate.tags = decision.merged_tags

    def _relation_target(self, scoped: list[MemoryItem], relation: MemoryRelation) -> MemoryItem | None:
        if relation.kind == "none" or relation.target_memory_id is None:
            return None
        for item in scoped:
            if item.id == relation.target_memory_id and item.status == "active":
                return item
        return None

    def _new_memory_from_candidate(self, decision: MemoryCandidate) -> MemoryItem:
        if decision.weight is None:
            decision.weight = self._default_weight_for_type(decision.type)
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
        if decision.target_name is not None:
            existing.name = decision.target_name
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
        candidate = self._resolve_candidate_defaults(candidate)
        validate_memory_candidate(candidate)
        decision = self._evaluate_candidate_decision(candidate)
        return self._write_result_from_decision(decision)

    def remember_candidate(self, candidate: MemoryCandidate) -> MemoryWriteResult:
        candidate = self._resolve_candidate_defaults(candidate)
        validate_memory_candidate(candidate)
        decision = self._evaluate_candidate_decision(candidate)
        if decision.action == "create":
            item = self._new_memory_from_candidate(decision)
            saved = self.memory_store.save_memory(item)
            self._sync_rag_memory(saved)
            return self._write_result_from_decision(decision, memory=saved)
        if decision.action == "update" and decision.target_memory_id:
            existing = self.memory_store.get_memory(decision.target_memory_id)
            if existing is None:
                raise MemoryNotFoundError(f"memory not found: {decision.target_memory_id}")
            updated = self._apply_candidate_update(existing, decision)
            self._sync_rag_memory(updated)
            return self._write_result_from_decision(decision, memory=updated)
        return self._write_result_from_decision(decision)

    def confirm_memory_candidate(
        self,
        candidate: MemoryCandidate,
        action: str | None = None,
        target_memory_id: str | None = None,
    ) -> MemoryWriteResult:
        if candidate.action != "ask_user":
            raise MemoryPolicyError("candidate does not require confirmation")
        candidate = self._resolve_candidate_defaults(candidate)
        validate_memory_candidate(candidate)
        if self.policy.contains_secret(candidate.content):
            raise MemoryPolicyError("memory rejected: contains_secret")
        if self.policy.is_transient_task_state(candidate.content):
            raise MemoryPolicyError("memory rejected: transient_task_state")
        if self.policy.is_noisy_output(candidate.content):
            raise MemoryPolicyError("memory rejected: noisy_output")
        confirmed_action = action or candidate.suggested_action or ("update" if (target_memory_id or candidate.target_memory_id) else "create")
        if confirmed_action == "update" and (target_memory_id or candidate.target_memory_id) is None:
            raise MemoryValidationError("target_memory_id is required for confirmed update")
        if confirmed_action == "update" and self.memory_store.get_memory(target_memory_id or candidate.target_memory_id) is None:
            raise MemoryNotFoundError(f"memory not found: {target_memory_id or candidate.target_memory_id}")
        fresh = replace(candidate, action="create", target_memory_id=None, target_updated_at=None, suggested_action=None, reason="")
        fresh = self._evaluate_candidate_decision(fresh)
        if fresh.action == "reject":
            raise MemoryPolicyError(f"memory rejected: {fresh.reason}")
        if fresh.action == "ask_user":
            if fresh.suggested_action != candidate.suggested_action or fresh.target_memory_id != candidate.target_memory_id or fresh.target_updated_at != candidate.target_updated_at:
                fresh.reason = "confirmation_state_changed"
                return self._write_result_from_decision(fresh)
        elif fresh.action == "update":
            if candidate.suggested_action != "update" or fresh.target_memory_id != candidate.target_memory_id or fresh.target_updated_at != candidate.target_updated_at:
                fresh.action = "ask_user"
                fresh.reason = "confirmation_state_changed"
                return self._write_result_from_decision(fresh)
        elif fresh.action == "create":
            if candidate.suggested_action != "create" or candidate.target_memory_id is not None:
                fresh.action = "ask_user"
                fresh.reason = "confirmation_state_changed"
                return self._write_result_from_decision(fresh)
        if confirmed_action not in {"create", "update"}:
            raise MemoryValidationError(f"unsupported confirmation action: {confirmed_action}")
        if confirmed_action != fresh.suggested_action:
            raise MemoryValidationError(f"confirmed action must match current suggested action: {fresh.suggested_action}")
        if target_memory_id is not None and target_memory_id != fresh.target_memory_id:
            raise MemoryValidationError(f"target_memory_id must match current suggested target: {fresh.target_memory_id}")
        confirmed = replace(candidate)
        confirmed.action = confirmed_action
        confirmed.target_memory_id = target_memory_id or candidate.target_memory_id
        confirmed.reason = f"confirmed:{candidate.reason}" if candidate.reason else "confirmed"
        if confirmed_action == "create":
            confirmed.target_memory_id = None
            item = self._new_memory_from_candidate(confirmed)
            saved = self.memory_store.save_memory(item)
            self._sync_rag_memory(saved)
            return self._write_result_from_decision(confirmed, memory=saved)
        if confirmed.target_memory_id is None:
            raise MemoryValidationError("target_memory_id is required for confirmed update")
        existing = self.memory_store.get_memory(confirmed.target_memory_id)
        if existing is None:
            raise MemoryNotFoundError(f"memory not found: {confirmed.target_memory_id}")
        updated = self._apply_candidate_update(existing, confirmed)
        self._sync_rag_memory(updated)
        return self._write_result_from_decision(confirmed, memory=updated)

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
        weight: int | None = None,
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
        candidate = self._resolve_candidate_defaults(candidate)
        validate_memory_candidate(candidate)
        decision = self._evaluate_candidate_decision(candidate)
        if decision.action == "reject":
            raise MemoryPolicyError(f"memory rejected: {decision.reason}")
        if decision.action == "ask_user":
            raise MemoryPolicyError(f"memory requires confirmation: {decision.reason}")

        if decision.action == "update" and decision.target_memory_id:
            existing = self.memory_store.get_memory(decision.target_memory_id)
            if existing is None:
                raise ValueError("target memory missing for update")
            updated = self._apply_candidate_update(existing, candidate)
            self._sync_rag_memory(updated)
            return updated

        item = self._new_memory_from_candidate(decision)
        saved = self.memory_store.save_memory(item)
        self._sync_rag_memory(saved)
        return saved

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

    def _memory_context_result(self, memory: MemoryItem, reason: str) -> MemorySearchResult:
        importance_score = min(max(memory.weight, 1), 10) / 10
        recency_score = self.retriever._recency_score(memory)
        access_score = min(math.log1p(memory.access_count) / math.log1p(20), 1.0)
        final_score = importance_score * 0.45 + recency_score * 0.35 + access_score * 0.20
        return MemorySearchResult(
            memory=memory,
            similarity_score=0.0,
            importance_score=importance_score,
            recency_score=recency_score,
            access_score=access_score,
            final_score=final_score,
            reason=reason,
        )

    def _limit_results_by_tokens(self, results: list[MemorySearchResult], max_tokens: int) -> list[MemorySearchResult]:
        limited = []
        used_tokens = 0
        for result in results:
            memory = result.memory
            block_tokens = estimate_tokens(f"{memory.id} {memory.type} {memory.description} {memory.content}")
            if used_tokens + block_tokens > max_tokens:
                break
            limited.append(result)
            used_tokens += block_tokens
        return limited

    def retrieve_pinned_memories(
        self,
        user_id: str = "default",
        project_id: str | None = None,
        workspace_id: str | None = None,
        top_k: int | None = None,
        max_tokens: int | None = None,
        include_archived: bool = False,
    ) -> list[MemorySearchResult]:
        scoped_memories = self._scoped_memories(user_id, project_id, workspace_id, include_archived=include_archived)
        results = [
            self._memory_context_result(memory, reason="pinned_context")
            for memory in scoped_memories
            if memory.type in PINNED_CONTEXT_TYPES
            and memory.status != "deleted"
            and (include_archived or memory.status == "active")
        ]
        results.sort(key=lambda result: result.final_score, reverse=True)
        results = results[: top_k or self.config.max_retrieved_memories]
        return self._limit_results_by_tokens(results, max_tokens or self.config.max_memory_prompt_tokens)

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
        if self.rag_retriever is not None:
            return self.rag_retriever.retrieve(memory_query)
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
        updated = self.memory_store.update_memory(memory)
        self._sync_rag_memory(updated)
        return updated

    def archive_memory(self, identifier: str) -> MemoryItem:
        archived = self.memory_store.set_memory_status(identifier, "archived")
        self._delete_rag_memory(archived.id)
        return archived

    def restore_memory(self, identifier: str) -> MemoryItem:
        restored = self.memory_store.set_memory_status(identifier, "active")
        self._sync_rag_memory(restored)
        return restored

    def delete_memory(self, identifier: str, hard: bool = False) -> None:
        if hard:
            memory = self.memory_store.get_memory(identifier)
            memory_id = memory.id if memory is not None else identifier
            self.memory_store.hard_delete_memory(identifier)
            self._delete_rag_memory(memory_id)
            return
        deleted = self.memory_store.set_memory_status(identifier, "deleted")
        self._delete_rag_memory(deleted.id)

    def export_memories(self, path: str | Path) -> dict:
        return export_memories(self.memory_store, path)

    def import_memories(self, path: str | Path) -> dict:
        report = import_memories(self.memory_store, path)
        if self.rag_index is not None and report.get("imported", 0) > 0:
            try:
                self.rag_index.rebuild()
            except Exception as exc:  # noqa: BLE001 - import should preserve authoritative writes when RAG rebuild fails
                self._record_rag_sync_error("rebuild", "*", exc)
        return report

    def verify_memories(self) -> dict:
        report = verify_memories(self.memory_store)
        if self.rag_index is not None:
            report.update(self.rag_index.verify())
            report["rag_sync_errors"] = list(self._rag_sync_errors)
            report["vector_ok"] = report["vector_ok"] and not report["rag_sync_errors"]
        return report

    def rebuild_index(self) -> None:
        rebuild_index(self.memory_store)
        if self.rag_index is not None:
            self.rag_index.rebuild()
            self._rag_sync_errors.clear()

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
                archived = self.memory_store.update_memory(memory)
                self._delete_rag_memory(archived.id)
                report["archived"] += 1
            else:
                report["kept"] += 1
        return report
