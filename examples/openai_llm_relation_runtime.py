"""OpenAI relation-judge demo for Memora.

Run:
    pip install openai
    set OPENAI_API_KEY=your-key
    python examples/openai_llm_relation_runtime.py
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from tempfile import mkdtemp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from openai_memory_clients import OpenAIRelationClient

from memora.config import MemoryConfig
from memora.relations import LLMMemoryRelationJudge
from memora.runtime import MemoryRuntime


MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6")


def make_openai_client():
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running this example.")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Install the OpenAI Python SDK first: pip install openai") from exc
    return OpenAI()


def main() -> None:
    client = make_openai_client()
    runtime = MemoryRuntime(
        config=MemoryConfig(
            root_dir=Path(mkdtemp(prefix="memora-openai-relation-demo-")),
            memory_backend="sqlite",
            rag_enabled=True,
            llm_relation_judge_enabled=True,
            semantic_relation_threshold=0.10,
            semantic_merge_threshold=0.10,
            semantic_conflict_threshold=0.95,
        ),
        relation_judge=LLMMemoryRelationJudge(OpenAIRelationClient(client, MODEL)),
    )
    runtime.init_storage()

    runtime.remember_extracted(
        memory_type="preference",
        name="response-style",
        description="Response style preference.",
        content="Prefer concise answers.",
        project_id="openai-relation-demo",
    )

    result = runtime.remember_extracted(
        memory_type="preference",
        name="short-summary-style",
        description="Short summary preference.",
        content="Prefer concise answers with short summaries.",
        project_id="openai-relation-demo",
    )

    print("=== OpenAI Relation Judge ===")
    print(f"model={MODEL}")
    print(f"action={result.action} reason={result.reason}")
    if result.memory is not None:
        print(f"memory={result.memory.name}: {result.memory.content}")


if __name__ == "__main__":
    main()
