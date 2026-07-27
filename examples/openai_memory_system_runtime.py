"""OpenAI-backed end-to-end Memora memory-system demo.

This example follows the current Memora architecture:

1. Build typed task context before the agent turn.
2. Store short-term session messages in the runtime.
3. At task/session end, ask an LLM to extract durable MemoryCandidate data.
4. Let Memora validate, classify relations, write the local backend, and sync RAG.
5. Build final typed context from pinned preference/project memories plus on-demand recall.

Run:
    pip install openai
    set OPENAI_API_KEY=your-key
    python examples/openai_memory_system_runtime.py

Optional:
    set OPENAI_MODEL=your-model
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from tempfile import mkdtemp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from openai_memory_clients import OpenAIExtractionClient, OpenAIRelationClient

from memora.config import MemoryConfig
from memora.extraction import LLMMemoryExtractor
from memora.relations import LLMMemoryRelationJudge
from memora.runtime import MemoryRuntime
from memora.schema import MemoryWriteResult


MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6")
USER_ID = "demo-user"
PROJECT_ID = "memora-system-demo"
SESSION_ID = "task-end-memory-extraction"


def make_openai_client():
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running this example.")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Install the OpenAI Python SDK first: pip install openai") from exc
    return OpenAI()


def make_runtime(client) -> MemoryRuntime:
    return MemoryRuntime(
        config=MemoryConfig(
            root_dir=Path(mkdtemp(prefix="memora-openai-system-demo-")),
            memory_backend="sqlite",
            rag_enabled=True,
            llm_relation_judge_enabled=True,
            semantic_relation_threshold=0.10,
            semantic_merge_threshold=0.10,
            semantic_conflict_threshold=0.95,
            llm_conflict_auto_replace_threshold=0.90,
        ),
        extractor=LLMMemoryExtractor(OpenAIExtractionClient(client, MODEL)),
        relation_judge=LLMMemoryRelationJudge(OpenAIRelationClient(client, MODEL)),
    )


def seed_existing_memory(runtime: MemoryRuntime) -> None:
    runtime.manager.save_memory(
        "preference",
        "Prefer English responses.",
        "Response language preference.",
        name="response-language",
        user_id=USER_ID,
        project_id=PROJECT_ID,
    )
    runtime.manager.save_memory(
        "project",
        "Memora uses SQLite or Markdown as the local source of truth. RAG is a rebuildable retrieval index.",
        "Memora storage architecture.",
        name="storage-architecture",
        user_id=USER_ID,
        project_id=PROJECT_ID,
    )
    runtime.manager.save_memory(
        "tool",
        "Use python -m pytest -q for fast local verification.",
        "Python verification command.",
        name="pytest-verification",
        user_id=USER_ID,
        project_id=PROJECT_ID,
    )


def print_context(runtime: MemoryRuntime, title: str, query: str) -> None:
    print(title)
    context = runtime.build_task_context(
        query,
        user_id=USER_ID,
        project_id=PROJECT_ID,
        memory_types=["episodic", "reflective", "tool", "knowledge", "general"],
        top_k=5,
        pinned_top_k=5,
    )
    print(context or "(no memory)")


def print_write_result(result: MemoryWriteResult) -> None:
    memory_name = result.memory.name if result.memory is not None else "-"
    supersedes = result.memory.supersedes if result.memory is not None else []
    print(
        "write "
        f"action={result.action} "
        f"reason={result.reason} "
        f"memory={memory_name} "
        f"relation={result.relation_kind}:{result.relation_confidence} "
        f"judge={result.relation_judge_status} "
        f"target={result.target_memory_id} "
        f"supersedes={supersedes}"
    )
    if result.action == "requires_confirmation":
        print("ask_user=true")


def run_task_end_memory_flow(runtime: MemoryRuntime) -> None:
    user_message = (
        "From now on, answer me in Chinese. For this Memora project, keep using "
        "python -m pytest -q as the quick verification command. Do not store raw "
        "tool logs as long-term memory; only keep stable tool lessons."
    )
    assistant_message = (
        "Understood. I will answer in Chinese, keep pytest as the quick verification "
        "command, and only store durable tool lessons instead of raw logs."
    )
    trace_summary = (
        "Task summary: the user changed a response-language preference, reaffirmed "
        "the project verification command, and clarified that raw tool logs are "
        "short-term trace data rather than long-term tool memory."
    )

    print_context(runtime, "=== Context Before Task ===", user_message)

    runtime.remember_message(SESSION_ID, "user", user_message, user_id=USER_ID)
    runtime.remember_message(SESSION_ID, "assistant", assistant_message, user_id=USER_ID)

    artifact, results = runtime.extract_and_remember(
        [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
            {"role": "system", "content": trace_summary},
        ],
        user_id=USER_ID,
        project_id=PROJECT_ID,
        session_id=SESSION_ID,
    )

    print("=== Extraction Artifact ===")
    print(f"ok={artifact.ok} should_remember={artifact.should_remember} errors={artifact.errors}")
    for extracted in artifact.memories:
        print(
            "candidate "
            f"type={extracted.type} "
            f"name={extracted.name} "
            f"confidence={extracted.confidence} "
            f"requires_confirmation={extracted.requires_confirmation}"
        )

    print("=== Write Results ===")
    for result in results:
        print_write_result(result)

    print_context(runtime, "=== Context After Task ===", "How should you answer me, and how should this project be verified?")


def main() -> None:
    client = make_openai_client()
    runtime = make_runtime(client)
    runtime.init_storage()
    seed_existing_memory(runtime)
    run_task_end_memory_flow(runtime)


if __name__ == "__main__":
    main()
