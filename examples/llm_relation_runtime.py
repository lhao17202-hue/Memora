"""Offline demo for LLM-assisted write-time memory relations.

Memora does not bundle a hosted LLM client. This demo shows the integration
shape an external agent can use:

1. The agent extracts structured candidate memories.
2. Memora uses embeddings to find a possible existing relation.
3. An injected relation judge can refine that hit into none/duplicate/merge/conflict.
4. Memora writes the local store and syncs the RAG index when enabled.

Run:
    python examples/llm_relation_runtime.py
"""

from __future__ import annotations

from pathlib import Path
import sys
from tempfile import mkdtemp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memora.config import MemoryConfig
from memora.relations import LLMMemoryRelationJudge
from memora.runtime import MemoryRuntime


class ScriptedRelationClient:
    """Tiny stand-in for an external LLM provider."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)

    def complete(self, messages):
        if not self.responses:
            raise RuntimeError("no scripted relation response left")
        return self.responses.pop(0)


class BrokenRelationClient:
    def complete(self, messages):
        raise RuntimeError("simulated provider outage")


def runtime_with_relation_client(client) -> MemoryRuntime:
    return MemoryRuntime(
        config=MemoryConfig(
            root_dir=Path(mkdtemp(prefix="memora-llm-relation-demo-")),
            memory_backend="sqlite",
            rag_enabled=True,
            llm_relation_judge_enabled=True,
            semantic_relation_threshold=0.10,
            semantic_merge_threshold=0.10,
            semantic_conflict_threshold=0.95,
        ),
        relation_judge=LLMMemoryRelationJudge(client),
    )


def run_merge_demo() -> None:
    runtime = runtime_with_relation_client(
        ScriptedRelationClient(
            [
                '{"kind":"merge","confidence":0.91,"reason":"Candidate refines response style.",'
                '"merged":{"name":"response-style","description":"Response style preference.",'
                '"content":"Prefer concise answers with short summaries.","tags":["style","summary"]}}'
            ]
        )
    )
    runtime.init_storage()
    runtime.remember_extracted(
        memory_type="preference",
        name="response-style",
        description="Response style preference.",
        content="Prefer concise answers.",
        project_id="merge-demo",
    )

    result = runtime.remember_extracted(
        memory_type="preference",
        name="short-summary-style",
        description="Short summary preference.",
        content="Prefer concise answers with short summaries.",
        project_id="merge-demo",
    )
    context = runtime.build_context("short summaries", project_id="merge-demo", memory_types=["preference"])

    print("=== LLM Merge ===")
    print(f"action={result.action} reason={result.reason}")
    print(context)


def run_conflict_demo() -> None:
    runtime = runtime_with_relation_client(
        ScriptedRelationClient(
            [
                '{"kind":"conflict","confidence":0.95,'
                '"reason":"Candidate changes the response language preference."}'
            ]
        )
    )
    runtime.init_storage()
    runtime.remember_extracted(
        memory_type="preference",
        name="language",
        description="Response language preference.",
        content="Prefer English responses.",
        project_id="conflict-demo",
    )

    result = runtime.remember_extracted(
        memory_type="preference",
        name="language-zh",
        description="Response language preference.",
        content="Prefer Chinese responses.",
        project_id="conflict-demo",
    )

    print("=== LLM Conflict ===")
    print(f"action={result.action} reason={result.reason}")
    print(result.memory.content if result.memory else "(no write)")


def run_fallback_demo() -> None:
    runtime = runtime_with_relation_client(BrokenRelationClient())
    runtime.init_storage()
    runtime.remember_extracted(
        memory_type="preference",
        name="fallback-style",
        description="Fallback style preference.",
        content="Prefer terse answers.",
        project_id="fallback-demo",
    )

    result = runtime.remember_extracted(
        memory_type="preference",
        name="fallback-style-update",
        description="Fallback style preference update.",
        content="Prefer terse answers with bullets.",
        project_id="fallback-demo",
    )

    print("=== Fallback ===")
    print(f"action={result.action} reason={result.reason}")
    print(result.memory.content if result.memory else "(no write)")


def main() -> None:
    run_merge_demo()
    print()
    run_conflict_demo()
    print()
    run_fallback_demo()


if __name__ == "__main__":
    main()
