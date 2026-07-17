"""Minimal fake-agent demo using Memora's runtime integration layer."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memora.config import MemoryConfig
from memora.runtime import MemoryRuntime


def fake_agent_response(user_message: str, memory_context: str) -> str:
    if memory_context:
        return f"我会参考已有记忆来回答：{user_message}"
    return f"没有找到相关记忆，但我会直接回答：{user_message}"


def main() -> None:
    runtime = MemoryRuntime(config=MemoryConfig(root_dir=Path(".memora-demo")))
    runtime.init_storage()

    runtime.manager.save_memory(
        memory_type="user",
        name="language",
        description="用户偏好中文。",
        content="用户偏好使用中文回答。",
    )

    session_id = "demo_session"
    user_message = "请用中文回答，下一步做什么？"

    results = runtime.retrieve_context(user_message)
    memory_context = runtime.manager.format_memories_for_prompt(results=results)

    print("=== Memory Context ===")
    print(memory_context or "(no memory)")
    print()

    runtime.remember_message(session_id, "user", user_message)

    assistant_message = fake_agent_response(user_message, memory_context)
    print("=== Assistant ===")
    print(assistant_message)

    runtime.remember_message(session_id, "assistant", assistant_message)
    runtime.remember_summary(session_id, "用户询问下一步，助手基于记忆建议继续推进。")
    runtime.mark_context_used(results)


if __name__ == "__main__":
    main()
