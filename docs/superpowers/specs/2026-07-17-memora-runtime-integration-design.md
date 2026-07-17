# Memora Runtime Integration Design

## Goal

Add the thinnest useful runtime integration layer for external agent runtimes. The layer must make Memora easy to call before, during, and after an agent conversation without adding any LLM dependency or new runtime system.

## User Direction

The user explicitly approved this scope:

- The LLM is only an external runtime caller, not an internal Memora dependency.
- The implementation should use the simplest possible code.
- No worktree is required; continue inline on `master`.

## Architecture

Add one small Python module, `memora/runtime.py`, containing a `MemoryRuntime` class. `MemoryRuntime` is a convenience wrapper around the existing `MemoryManager`; it does not own storage logic, retrieval logic, summarization logic, policy logic, or session logic.

The runtime layer delegates to existing manager methods:

- context retrieval: `MemoryManager.retrieve_memory()` and `MemoryManager.format_memories_for_prompt()`
- session message recording: `MemoryManager.append_message()`
- manual summary saving: `MemoryManager.save_memory(memory_type="session_summary", ...)`
- usage marking: `MemoryManager.mark_memories_used()`

No new dependency is introduced.

## Public API

Create `memora/runtime.py` with:

```python
from __future__ import annotations

from .config import MemoryConfig
from .manager import MemoryManager
from .schema import MemoryItem, MemorySearchResult, SessionMessage


class MemoryRuntime:
    def __init__(self, manager: MemoryManager | None = None, config: MemoryConfig | None = None):
        ...

    def init_storage(self) -> None:
        ...

    def retrieve_context(self, query: str, **kwargs) -> list[MemorySearchResult]:
        ...

    def build_context(self, query: str, **kwargs) -> str:
        ...

    def remember_message(self, session_id: str, role: str, content: str, user_id: str = "default") -> None:
        ...

    def remember_summary(
        self,
        session_id: str,
        content: str,
        user_id: str = "default",
        project_id: str | None = None,
        workspace_id: str | None = None,
    ) -> MemoryItem:
        ...

    def mark_context_used(self, results: list[MemorySearchResult]) -> None:
        ...
```

### Constructor behavior

`MemoryRuntime(manager=...)` uses the supplied manager.

`MemoryRuntime(config=...)` creates `MemoryManager(config)`.

If both `manager` and `config` are provided, raise `ValueError("manager and config cannot both be provided")`. This keeps ownership unambiguous while avoiding a new project-specific error for a programmer mistake.

### `init_storage()`

Thin pass-through to `self.manager.init_storage()`.

This exists so a minimal external runtime can use only `MemoryRuntime` without separately importing `MemoryManager`.

### `retrieve_context()`

Return raw search results for callers that want to inspect scores or mark usage later:

```python
return self.manager.retrieve_memory(query, **kwargs)
```

Supported keyword arguments are exactly those already accepted by `MemoryManager.retrieve_memory()`, such as `memory_types`, `tags`, `top_k`, `include_archived`, `project_id`, and `workspace_id`.

### `build_context()`

Retrieve matching memories and return the formatted prompt text:

```python
results = self.retrieve_context(query, **kwargs)
return self.manager.format_memories_for_prompt(results=results)
```

It should not automatically call `mark_context_used()`. Marking usage remains explicit so read-only preview flows do not mutate storage.

### `remember_message()`

Append a session message:

```python
self.manager.append_message(user_id, session_id, SessionMessage(role=role, content=content))
```

Validation remains in existing schema/store code.

### `remember_summary()`

Save a manual session summary as a normal memory item:

```python
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
```

This method does not summarize automatically. The caller supplies already-written summary text.

### `mark_context_used()`

Thin pass-through:

```python
self.manager.mark_memories_used(results)
```

## README Example

Add a short runtime example:

```python
from memora.runtime import MemoryRuntime

runtime = MemoryRuntime()
runtime.init_storage()

context = runtime.build_context("用户偏好和当前项目")
runtime.remember_message("session_1", "user", "下一步做什么")
runtime.remember_message("session_1", "assistant", "建议做 runtime integration。")
runtime.remember_summary("session_1", "用户认可最简单的 runtime integration。")
```

Keep the README example short. Do not mention LLM provider SDKs.

## Tests

Add `tests/test_runtime.py`.

Test cases:

1. `test_build_context_returns_formatted_memory`
   - create a temporary root config
   - initialize storage through `MemoryRuntime`
   - save a user memory through `runtime.manager.save_memory(...)`
   - call `runtime.build_context("中文回答")`
   - assert the saved memory content appears in the returned context

2. `test_retrieve_context_and_mark_context_used_updates_access_count`
   - save a matching memory
   - call `runtime.retrieve_context(...)`
   - call `runtime.mark_context_used(results)`
   - reload the memory from `runtime.manager.memory_store`
   - assert `access_count == 1`
   - assert `last_accessed_at is not None`

3. `test_remember_message_appends_session_message`
   - call `runtime.remember_message("session_1", "user", "hello")`
   - read messages through `runtime.manager.get_messages("default", "session_1")`
   - assert role and content match

4. `test_remember_summary_saves_session_summary_memory`
   - call `runtime.remember_summary("session_1", "summary text")`
   - assert returned item has `type == "session_summary"`
   - assert returned item has `source == "runtime"`
   - assert returned item content is `"summary text"`

5. `test_constructor_rejects_manager_and_config_together`
   - pass both a manager and config
   - assert `ValueError` with `manager and config cannot both be provided`

## Non-Goals

This round does not add:

- LLM calls
- automatic summarization
- automatic preference extraction
- embeddings
- vector storage
- async APIs
- background services
- plugin systems
- hosted runtime behavior
- new CLI commands unless needed for documentation examples
- new dependencies

## Success Criteria

- External agent code can use `MemoryRuntime` as the only Memora integration import for common runtime operations.
- Runtime implementation remains a thin wrapper over existing `MemoryManager` methods.
- All new behavior is covered by focused tests.
- Full test suite passes with `pytest -v`.
