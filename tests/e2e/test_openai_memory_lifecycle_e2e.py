import os
from pathlib import Path

import pytest

from examples.openai_memory_clients import OpenAIExtractionClient
from memora.config import MemoryConfig
from memora.extraction import LLMMemoryExtractor
from memora.runtime import MemoryRuntime
from memora.schema import WorkingMemoryState


RUN_TRUE_VALUES = {"1", "true", "yes", "on"}
USER_ID = "llm-e2e-user"
PROJECT_ID = "llm-memory-lifecycle-e2e"
SESSION_ID = "llm_memory_lifecycle_session"


def _llm_e2e_enabled() -> bool:
    return os.environ.get("RUN_MEMORA_LLM_E2E", "").strip().lower() in RUN_TRUE_VALUES


def _make_openai_client():
    if not _llm_e2e_enabled():
        pytest.skip("set RUN_MEMORA_LLM_E2E=1 to run real LLM lifecycle e2e")
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("set OPENAI_API_KEY to run real LLM lifecycle e2e")
    try:
        from openai import OpenAI
    except ImportError:
        pytest.skip("install openai to run real LLM lifecycle e2e")
    return OpenAI()


@pytest.mark.e2e
@pytest.mark.llm_e2e
def test_real_llm_extracts_writes_lists_and_retrieves_memory(tmp_path: Path):
    client = _make_openai_client()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    runtime = MemoryRuntime(
        config=MemoryConfig(
            root_dir=tmp_path / ".memora",
            memory_backend="sqlite",
            rag_enabled=True,
        ),
        extractor=LLMMemoryExtractor(OpenAIExtractionClient(client, model)),
    )
    runtime.init_storage()

    user_message = (
        "Please remember this durable preference for future coding help: "
        "I prefer Chinese answers with a short summary first."
    )
    assistant_message = "Understood. I will answer in Chinese and put a short summary first."
    working_memory = WorkingMemoryState(
        task="Remember the user's answer style preference.",
        notes=["The user explicitly stated a durable response style preference."],
        trace="The turn established a stable answer-language and summary-format preference.",
    )

    runtime.remember_message(SESSION_ID, "user", user_message, user_id=USER_ID)
    runtime.remember_message(SESSION_ID, "assistant", assistant_message, user_id=USER_ID)

    artifact, write_results = runtime.extract_and_remember(
        [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ],
        user_id=USER_ID,
        project_id=PROJECT_ID,
        session_id=SESSION_ID,
        working_memory=working_memory,
    )

    assert artifact.ok is True, artifact.errors
    assert artifact.should_remember is True
    assert artifact.memories, artifact.raw_text
    assert any(memory.type == "preference" for memory in artifact.memories)
    assert any(result.action == "created" for result in write_results)

    listed = runtime.manager.list_memories()
    listed_names = {memory.name for memory in listed}
    listed_contents = "\n".join(memory.content for memory in listed)

    assert listed, "expected real LLM extraction to write at least one memory"
    assert any(memory.type == "preference" for memory in listed)
    assert any(name.startswith("response") or "style" in name or "language" in name for name in listed_names)
    assert "Chinese" in listed_contents or "中文" in listed_contents
    assert "summary" in listed_contents.lower() or "总结" in listed_contents

    retrieved = runtime.retrieve_task_context(
        "How should you answer this user?",
        user_id=USER_ID,
        project_id=PROJECT_ID,
        memory_types=["preference"],
        include_pinned=False,
        top_k=5,
    )
    retrieved_contents = "\n".join(result.memory.content for result in retrieved)

    assert retrieved, "expected retrieval to find the LLM-written preference"
    assert any(result.memory.id in {memory.id for memory in listed} for result in retrieved)
    assert "Chinese" in retrieved_contents or "中文" in retrieved_contents
