# Memora Runtime Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the thinnest Python runtime wrapper that lets external agent runtimes use Memora before, during, and after conversations.

**Architecture:** Create one small `MemoryRuntime` wrapper around the existing `MemoryManager`. The wrapper delegates retrieval, formatting, session message storage, manual session summary storage, and usage marking to existing manager methods. No LLM calls, no automatic summarization, no new storage layer, and no new dependency.

**Tech Stack:** Python standard library, existing Memora modules, pytest.

## Global Constraints

- The LLM is only an external runtime caller, not an internal Memora dependency.
- The implementation should use the simplest possible code.
- No worktree is required; continue inline on `master`.
- Do not add LLM calls.
- Do not add automatic summarization.
- Do not add automatic preference extraction.
- Do not add embeddings.
- Do not add vector storage.
- Do not add async APIs.
- Do not add background services.
- Do not add plugin systems.
- Do not add hosted runtime behavior.
- Do not add new CLI commands.
- Do not add new dependencies.

---

## File Structure

- Create `memora/runtime.py`: one thin `MemoryRuntime` class that delegates to `MemoryManager`.
- Create `tests/test_runtime.py`: focused tests for the runtime wrapper.
- Modify `README.md`: add a short runtime integration example.

---

### Task 1: Runtime Wrapper

**Files:**
- Create: `memora/runtime.py`
- Test: `tests/test_runtime.py`

**Interfaces:**
- Consumes:
  - `MemoryConfig` from `memora.config`
  - `MemoryManager` from `memora.manager`
  - `MemoryItem`, `MemorySearchResult`, and `SessionMessage` from `memora.schema`
- Produces:
  - `MemoryRuntime.__init__(manager: MemoryManager | None = None, config: MemoryConfig | None = None)`
  - `MemoryRuntime.init_storage() -> None`
  - `MemoryRuntime.retrieve_context(query: str, **kwargs) -> list[MemorySearchResult]`
  - `MemoryRuntime.build_context(query: str, **kwargs) -> str`
  - `MemoryRuntime.remember_message(session_id: str, role: str, content: str, user_id: str = "default") -> None`
  - `MemoryRuntime.remember_summary(session_id: str, content: str, user_id: str = "default", project_id: str | None = None, workspace_id: str | None = None) -> MemoryItem`
  - `MemoryRuntime.mark_context_used(results: list[MemorySearchResult]) -> None`

- [ ] **Step 1: Write failing runtime tests**

Create `tests/test_runtime.py` with exactly:

```python
from pathlib import Path

import pytest

from memora.config import MemoryConfig
from memora.manager import MemoryManager
from memora.runtime import MemoryRuntime


def make_runtime(tmp_path: Path) -> MemoryRuntime:
    runtime = MemoryRuntime(config=MemoryConfig(root_dir=tmp_path / ".memora"))
    runtime.init_storage()
    return runtime


def test_build_context_returns_formatted_memory(tmp_path: Path):
    runtime = make_runtime(tmp_path)
    runtime.manager.save_memory(
        memory_type="user",
        name="language",
        description="用户偏好中文。",
        content="用户偏好使用中文回答。",
    )

    context = runtime.build_context("中文回答")

    assert "用户偏好使用中文回答。" in context


def test_retrieve_context_and_mark_context_used_updates_access_count(tmp_path: Path):
    runtime = make_runtime(tmp_path)
    item = runtime.manager.save_memory(
        memory_type="user",
        name="language",
        description="用户偏好中文。",
        content="用户偏好使用中文回答。",
    )

    results = runtime.retrieve_context("中文回答")
    runtime.mark_context_used(results)
    reloaded = runtime.manager.memory_store.get_memory(item.id)

    assert reloaded is not None
    assert reloaded.access_count == 1
    assert reloaded.last_accessed_at is not None


def test_remember_message_appends_session_message(tmp_path: Path):
    runtime = make_runtime(tmp_path)

    runtime.remember_message("session_1", "user", "hello")
    messages = runtime.manager.get_messages("default", "session_1")

    assert len(messages) == 1
    assert messages[0].role == "user"
    assert messages[0].content == "hello"


def test_remember_summary_saves_session_summary_memory(tmp_path: Path):
    runtime = make_runtime(tmp_path)

    item = runtime.remember_summary("session_1", "summary text")

    assert item.type == "session_summary"
    assert item.source == "runtime"
    assert item.content == "summary text"


def test_constructor_rejects_manager_and_config_together(tmp_path: Path):
    manager = MemoryManager(MemoryConfig(root_dir=tmp_path / ".memora"))

    with pytest.raises(ValueError, match="manager and config cannot both be provided"):
        MemoryRuntime(manager=manager, config=MemoryConfig(root_dir=tmp_path / "other"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_runtime.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'memora.runtime'` or equivalent import failure.

- [ ] **Step 3: Implement the runtime wrapper**

Create `memora/runtime.py` with exactly:

```python
"""Thin runtime integration helpers for external agent runtimes."""

from __future__ import annotations

from .config import MemoryConfig
from .manager import MemoryManager
from .schema import MemoryItem, MemorySearchResult, SessionMessage


class MemoryRuntime:
    def __init__(self, manager: MemoryManager | None = None, config: MemoryConfig | None = None):
        if manager is not None and config is not None:
            raise ValueError("manager and config cannot both be provided")
        self.manager = manager or MemoryManager(config)

    def init_storage(self) -> None:
        self.manager.init_storage()

    def retrieve_context(self, query: str, **kwargs) -> list[MemorySearchResult]:
        return self.manager.retrieve_memory(query, **kwargs)

    def build_context(self, query: str, **kwargs) -> str:
        results = self.retrieve_context(query, **kwargs)
        return self.manager.format_memories_for_prompt(results=results)

    def remember_message(self, session_id: str, role: str, content: str, user_id: str = "default") -> None:
        self.manager.append_message(user_id, session_id, SessionMessage(role=role, content=content))

    def remember_summary(
        self,
        session_id: str,
        content: str,
        user_id: str = "default",
        project_id: str | None = None,
        workspace_id: str | None = None,
    ) -> MemoryItem:
        return self.manager.save_memory(
            memory_type="session_summary",
            name=f"{session_id}-summary",
            description=f"Summary for session {session_id}",
            content=content,
            user_id=user_id,
            project_id=project_id,
            workspace_id=workspace_id,
            source="runtime",
        )

    def mark_context_used(self, results: list[MemorySearchResult]) -> None:
        self.manager.mark_memories_used(results)
```

- [ ] **Step 4: Run runtime tests**

Run:

```bash
pytest tests/test_runtime.py -v
```

Expected: PASS, `5 passed`.

- [ ] **Step 5: Commit runtime wrapper**

Run:

```bash
git add memora/runtime.py tests/test_runtime.py
git commit -m "feat: add runtime integration wrapper" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: README Runtime Example and Final Verification

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: `MemoryRuntime` from `memora.runtime`
- Produces: README section `## Runtime integration`

- [ ] **Step 1: Add README example**

In `README.md`, add this section after the Python usage section and before `## MVP boundaries`:

````markdown
## Runtime integration

External agent runtimes can use `MemoryRuntime` as a thin wrapper around the manager API:

```python
from memora.runtime import MemoryRuntime

runtime = MemoryRuntime()
runtime.init_storage()

context = runtime.build_context("用户偏好和当前项目")
runtime.remember_message("session_1", "user", "下一步做什么")
runtime.remember_message("session_1", "assistant", "建议做 runtime integration。")
runtime.remember_summary("session_1", "用户认可最简单的 runtime integration。")
```
````

- [ ] **Step 2: Run full test suite**

Run:

```bash
pytest -v
```

Expected: PASS with all existing tests plus the 5 new runtime tests.

- [ ] **Step 3: Commit README update**

Run:

```bash
git add README.md
git commit -m "docs: document runtime integration" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Self-Review

Spec coverage:
- `MemoryRuntime` wrapper: Task 1.
- No LLM dependency: Global Constraints and Task 1 implementation.
- `init_storage()`: Task 1.
- `retrieve_context()`: Task 1.
- `build_context()`: Task 1.
- `remember_message()`: Task 1.
- `remember_summary()`: Task 1.
- `mark_context_used()`: Task 1.
- README example: Task 2.
- Focused tests and full suite: Tasks 1 and 2.

Placeholder scan:
- No TBD, TODO, or incomplete implementation steps.

Type consistency:
- Method names and signatures match the approved design spec.
- Tests import and call the same method names implemented by Task 1.
