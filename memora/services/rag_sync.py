"""RAG index synchronization with write-path error isolation."""

from __future__ import annotations

from typing import Any

from ..schema import MemoryItem


class RagSync:
    def __init__(self, index: Any | None):
        self.index = index
        self.errors: list[dict[str, str]] = []

    def init_storage(self) -> None:
        if self.index is not None:
            self.index.init_storage()

    def record_error(self, operation: str, memory_id: str, exc: Exception) -> None:
        self.errors.append({"operation": operation, "memory_id": memory_id, "error": str(exc)})

    def mark(self) -> int:
        return len(self.errors)

    def errors_since(self, start_index: int) -> list[dict[str, str]]:
        return list(self.errors[start_index:])

    def sync_memory(self, item: MemoryItem) -> None:
        if self.index is None:
            return
        try:
            self.index.sync_memory(item)
        except Exception as exc:  # noqa: BLE001 - authoritative memory writes must survive RAG sync failures
            self.record_error("sync", item.id, exc)

    def delete_memory(self, memory_id: str) -> None:
        if self.index is None:
            return
        try:
            self.index.delete_memory(memory_id)
        except Exception as exc:  # noqa: BLE001 - authoritative lifecycle writes must survive RAG cleanup failures
            self.record_error("delete", memory_id, exc)

    def rebuild(self) -> None:
        if self.index is None:
            return
        self.index.rebuild()
        self.errors.clear()

    def rebuild_for_import(self) -> None:
        if self.index is None:
            return
        try:
            self.index.rebuild()
        except Exception as exc:  # noqa: BLE001 - imports should preserve authoritative writes when RAG rebuild fails
            self.record_error("rebuild", "*", exc)

    def add_verify_report(self, report: dict[str, Any]) -> dict[str, Any]:
        if self.index is None:
            return report
        report.update(self.index.verify())
        report["rag_sync_errors"] = list(self.errors)
        report["vector_ok"] = report["vector_ok"] and not report["rag_sync_errors"]
        return report

