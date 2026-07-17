# Memora Agent Loop Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one minimal runnable fake-agent example that demonstrates how external agent runtime code uses `MemoryRuntime`.

**Architecture:** Create a single example script under `examples/` that imports `MemoryConfig` and `MemoryRuntime`, seeds one memory, retrieves context, records messages, saves a manual summary, and marks retrieved context as used. Add one subprocess test that runs the script and checks visible output. Add `.memora-demo/` to `.gitignore` so demo state is not committed.

**Tech Stack:** Python standard library, existing Memora modules, pytest.

## Global Constraints

- Use a simulated/fake agent response.
- Do not integrate Claude, OpenAI, or any other LLM SDK.
- Keep the code very simple.
- Use the demo to validate that `MemoryRuntime` feels natural from external agent code.
- Do not add real LLM calls.
- Do not add Claude SDK integration.
- Do not add OpenAI SDK integration.
- Do not add an interactive chat loop.
- Do not add new CLI commands.
- Do not add async APIs.
- Do not add prompt template systems.
- Do not add automatic summarization.
- Do not add automatic preference extraction.
- Do not add new dependencies.

---

## File Structure

- Create `examples/simple_agent_runtime.py`: runnable fake-agent demo.
- Create `tests/test_examples.py`: subprocess test for the demo.
- Modify `.gitignore`: add `.memora-demo/`.
- Modify `README.md`: add short demo run instructions.

---

### Task 1: Fake Agent Runtime Example

**Files:**
- Create: `examples/simple_agent_runtime.py`
- Create: `tests/test_examples.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes:
  - `MemoryConfig` from `memora.config`
  - `MemoryRuntime` from `memora.runtime`
- Produces:
  - `fake_agent_response(user_message: str, memory_context: str) -> str`
  - `main() -> None`
  - ignored local runtime root `.memora-demo/`

- [ ] **Step 1: Write failing example test**

Create `tests/test_examples.py` with exactly:

```python
import subprocess
import sys


def test_simple_agent_runtime_example_runs_successfully():
    result = subprocess.run(
        [sys.executable, "examples/simple_agent_runtime.py"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "=== Memory Context ===" in result.stdout
    assert "=== Assistant ===" in result.stdout
    assert "用户偏好使用中文回答。" in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_examples.py -v
```

Expected: FAIL because `examples/simple_agent_runtime.py` does not exist yet.

- [ ] **Step 3: Add `.memora-demo/` to `.gitignore`**

Edit `.gitignore` to exactly:

```gitignore
__pycache__/
*.py[cod]
.pytest_cache/
.memora/
.memora-demo/
```

- [ ] **Step 4: Create the example script**

Create `examples/simple_agent_runtime.py` with exactly:

```python
"""Minimal fake-agent demo using Memora's runtime integration layer."""

from __future__ import annotations

from pathlib import Path

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
    user_message = "下一步做什么？"

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
```

- [ ] **Step 5: Run the example test**

Run:

```bash
pytest tests/test_examples.py -v
```

Expected: PASS, `1 passed`.

- [ ] **Step 6: Run the example manually**

Run:

```bash
python examples/simple_agent_runtime.py
```

Expected: output includes `=== Memory Context ===`, `用户偏好使用中文回答。`, and `=== Assistant ===`.

- [ ] **Step 7: Commit the example**

Run:

```bash
git add .gitignore examples/simple_agent_runtime.py tests/test_examples.py
git commit -m "feat: add fake agent runtime demo" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: README Demo Instructions and Final Verification

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: `examples/simple_agent_runtime.py`
- Produces: README section `## Agent runtime demo`

- [ ] **Step 1: Add README demo instructions**

In `README.md`, add this section after `## Runtime integration` and its code example, before `## MVP boundaries`:

````markdown
## Agent runtime demo

Run the fake agent runtime example:

```bash
python examples/simple_agent_runtime.py
```

The demo uses `MemoryRuntime` with a local fake assistant response. It does not call an LLM.
````

- [ ] **Step 2: Run full test suite**

Run:

```bash
pytest -v
```

Expected: PASS with all existing tests plus the new example test.

- [ ] **Step 3: Commit README update**

Run:

```bash
git add README.md
git commit -m "docs: document agent runtime demo" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Self-Review

Spec coverage:
- Runnable fake-agent demo: Task 1.
- Uses `MemoryRuntime` as external caller code: Task 1.
- No real LLM calls or SDK integrations: Global Constraints and Task 1 code.
- `.memora-demo/` ignored: Task 1.
- Subprocess test validates visible output: Task 1.
- README run instructions: Task 2.
- Full suite verification: Task 2.

Placeholder scan:
- No TBD, TODO, or incomplete implementation steps.

Type consistency:
- `fake_agent_response(user_message: str, memory_context: str) -> str` matches the design.
- Example imports match existing runtime interfaces.
