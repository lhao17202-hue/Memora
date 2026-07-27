"""Full OpenAI-backed memory turn demo for Memora.

This example uses OpenAI twice:

1. As an extractor that turns a conversation turn into candidate memories.
2. As a relation judge that decides whether a new candidate should merge with,
   conflict with, duplicate, or remain separate from an existing memory.

Run:
    pip install openai
    set OPENAI_API_KEY=your-key
    python examples/openai_full_memory_turn_runtime.py
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


MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6")
USER_ID = "demo-user"
PROJECT_ID = "openai-full-turn-demo"
SESSION_ID = "openai_full_turn_session"


def make_openai_client():
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running this example.")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Install the OpenAI Python SDK first: pip install openai") from exc
    return OpenAI()


def remember_turn(runtime: MemoryRuntime, user_message: str, assistant_message: str) -> None:
    context = runtime.build_task_context(
        user_message,
        user_id=USER_ID,
        project_id=PROJECT_ID,
        memory_types=["preference", "tool", "knowledge"],
    )
    print("=== Context ===")
    print(context or "(no memory)")

    runtime.remember_message(SESSION_ID, "user", user_message, user_id=USER_ID)
    runtime.remember_message(SESSION_ID, "assistant", assistant_message, user_id=USER_ID)
    artifact, results = runtime.extract_and_remember(
        [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ],
        user_id=USER_ID,
        project_id=PROJECT_ID,
        session_id=SESSION_ID,
    )

    print("=== Extraction ===")
    print(f"ok={artifact.ok} errors={artifact.errors}")
    print("=== Writes ===")
    for result in results:
        memory_name = result.memory.name if result.memory is not None else "-"
        print(f"action={result.action} reason={result.reason} memory={memory_name}")


def main() -> None:
    client = make_openai_client()
    runtime = MemoryRuntime(
        config=MemoryConfig(
            root_dir=Path(mkdtemp(prefix="memora-openai-full-turn-demo-")),
            memory_backend="sqlite",
            rag_enabled=True,
            llm_relation_judge_enabled=True,
            semantic_relation_threshold=0.10,
            semantic_merge_threshold=0.10,
            semantic_conflict_threshold=0.95,
        ),
        extractor=LLMMemoryExtractor(OpenAIExtractionClient(client, MODEL)),
        relation_judge=LLMMemoryRelationJudge(OpenAIRelationClient(client, MODEL)),
    )
    runtime.init_storage()

    remember_turn(
        runtime,
        "Please remember that I prefer concise answers.",
        "Got it. I will keep answers concise.",
    )
    print()
    remember_turn(
        runtime,
        "Update that preference: concise answers with a short summary first.",
        "Understood. I will start with a short summary and keep details concise.",
    )

    print()
    print("=== Final Context ===")
    print(runtime.build_task_context("How should you answer me?", user_id=USER_ID, project_id=PROJECT_ID))


if __name__ == "__main__":
    main()
