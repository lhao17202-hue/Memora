"""Minimal OpenAI agent demo using Memora's runtime integration layer.

This example intentionally keeps the LLM side small. Memora does not call the
LLM internally: the external agent retrieves memory context, calls the LLM,
asks the LLM to extract a candidate memory, then passes that candidate back to
Memora for deterministic validation and writing.

Run:
    pip install openai
    set OPENAI_API_KEY=your-key
    python examples/openai_agent_runtime.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memora.config import MemoryConfig
from memora.runtime import MemoryRuntime
from memora.schema import MemorySearchResult


USER_ID = "demo-user"
PROJECT_ID = "openai-runtime-demo"
SESSION_ID = "openai_demo_session"
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def format_memory_context(results: list[MemorySearchResult]) -> str:
    if not results:
        return ""
    lines = []
    for result in results:
        memory = result.memory
        lines.append(
            f"- {memory.name}: {memory.description}\n"
            f"  type={memory.type}, tags={', '.join(memory.tags) or '-'}\n"
            f"  content={memory.content}"
        )
    return "\n".join(lines)


def chat_once(client: OpenAI, user_message: str, memory_context: str) -> str:
    system_prompt = (
        "You are a small assistant using external memory. "
        "Use the memory context when it is relevant. "
        "If there is no relevant memory, answer normally."
    )
    if memory_context:
        system_prompt += f"\n\nMemory context:\n{memory_context}"

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


def extract_memory_candidate(client: OpenAI, user_message: str, assistant_message: str) -> dict[str, Any] | None:
    """Ask the external LLM to extract one candidate memory.

    Memora still decides whether to create, update, reject, or require
    confirmation when this candidate is passed to runtime.remember_extracted().
    """

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract at most one durable memory from the conversation. "
                    "Return JSON only. If nothing should be remembered, return "
                    "{\"should_remember\": false}. If something should be remembered, "
                    "return an object with: should_remember=true, type, name, "
                    "description, content, tags. Use type=user for user preferences, "
                    "type=project for project facts, or type=tool_experience for reusable workflow lessons."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"User message:\n{user_message}\n\n"
                    f"Assistant message:\n{assistant_message}\n"
                ),
            },
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    raw = response.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not data.get("should_remember"):
        return None
    return data


def run_turn(runtime: MemoryRuntime, client: OpenAI, user_message: str) -> None:
    print(f"\n=== User ===\n{user_message}\n")

    # 1. Before calling the LLM, retrieve memory context for the current turn.
    results = runtime.retrieve_context(user_message, user_id=USER_ID, project_id=PROJECT_ID)
    memory_context = format_memory_context(results)
    print("=== Memory Context ===")
    print(memory_context or "(no memory)")
    print()

    # 2. Store the raw user message in session history.
    runtime.remember_message(SESSION_ID, "user", user_message, user_id=USER_ID)

    # 3. Call the LLM with the retrieved memory context.
    assistant_message = chat_once(client, user_message, memory_context)
    print("=== Assistant ===")
    print(assistant_message)
    print()

    # 4. Store the assistant message in session history.
    runtime.remember_message(SESSION_ID, "assistant", assistant_message, user_id=USER_ID)

    # 5. Ask the external LLM to extract a candidate memory from this turn.
    candidate = extract_memory_candidate(client, user_message, assistant_message)
    if candidate is None:
        print("=== Memory Write ===")
        print("no candidate memory extracted")
    else:
        result = runtime.remember_extracted(
            memory_type=str(candidate.get("type") or "project"),
            name=str(candidate.get("name") or "memory"),
            description=str(candidate.get("description") or "LLM-extracted memory."),
            content=str(candidate.get("content") or ""),
            user_id=USER_ID,
            project_id=PROJECT_ID,
            session_id=SESSION_ID,
            tags=list(candidate.get("tags") or []),
            confidence=0.8,
        )
        print("=== Memory Write ===")
        print(f"action={result.action} reason={result.reason}")

    # 6. Mark retrieved memories as used after the turn completes.
    runtime.mark_context_used(results)


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running this example.")

    runtime = MemoryRuntime(
        config=MemoryConfig(
            root_dir=Path(".memora-openai-demo"),
            memory_backend=os.getenv("MEMORA_BACKEND", "file"),
        )
    )
    runtime.init_storage()
    client = OpenAI()

    run_turn(runtime, client, "以后请尽量用中文回答我。")
    run_turn(runtime, client, "你还记得我的回答语言偏好吗？")

    runtime.remember_summary(
        SESSION_ID,
        "用户进行了 OpenAI runtime demo，并测试了语言偏好记忆的写入和召回。",
        user_id=USER_ID,
        project_id=PROJECT_ID,
    )


if __name__ == "__main__":
    main()
