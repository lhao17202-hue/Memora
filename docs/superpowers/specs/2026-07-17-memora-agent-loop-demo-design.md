# Memora Agent Loop Demo Design

## Goal

Add one minimal, runnable example that shows how an external agent runtime can use `MemoryRuntime` before, during, and after a conversation. The example must not call a real LLM or introduce any new dependency.

## User Direction

The user approved the smallest useful demo:

- Use a simulated/fake agent response.
- Do not integrate Claude, OpenAI, or any other LLM SDK.
- Keep the code very simple.
- Use the demo to validate that `MemoryRuntime` feels natural from external agent code.

## Recommended Approach

Create a single runnable example file:

- `examples/simple_agent_runtime.py`

The example should demonstrate the full runtime flow:

1. Create `MemoryRuntime` with an isolated demo root: `.memora-demo`.
2. Initialize storage.
3. Seed one user preference memory.
4. Retrieve memory context for a user message.
5. Print the formatted memory context.
6. Record the user message.
7. Generate a fake assistant response with a local helper function.
8. Print the assistant response.
9. Record the assistant message.
10. Save a manual session summary.
11. Mark retrieved memories as used.

This proves the integration path without expanding Memora's scope.

## Architecture

The demo is external caller code. It should import and use public Memora interfaces only:

```python
from memora.config import MemoryConfig
from memora.runtime import MemoryRuntime
```

The fake agent function is local to the example:

```python
def fake_agent_response(user_message: str, memory_context: str) -> str:
    if memory_context:
        return f"我会参考已有记忆来回答：{user_message}"
    return f"没有找到相关记忆，但我会直接回答：{user_message}"
```

No production Memora module should depend on this example.

## Example Behavior

Running:

```bash
python examples/simple_agent_runtime.py
```

Should print sections like:

```text
=== Memory Context ===
...
用户偏好使用中文回答。

=== Assistant ===
我会参考已有记忆来回答：下一步做什么？
```

The script may create `.memora-demo` in the repository root. That directory is already covered by the existing `.memora/`-style runtime-root pattern only if explicitly added; for this demo, add `.memora-demo/` to `.gitignore` so local demo state is not committed.

## Tests

Add `tests/test_examples.py` with one subprocess test that runs the script from a temporary working directory or with a temporary root-safe setup.

To keep the example itself simple, prefer running the script as-is from the repository root in the test and ensure `.memora-demo/` is ignored. The test should assert:

- process return code is `0`
- stdout contains `=== Memory Context ===`
- stdout contains `=== Assistant ===`
- stdout contains `用户偏好使用中文回答。`

The test should not inspect implementation details.

## README Update

Add a short section after Runtime integration:

```markdown
## Agent runtime demo

Run the fake agent runtime example:

```bash
python examples/simple_agent_runtime.py
```

The demo uses `MemoryRuntime` with a local fake assistant response. It does not call an LLM.
```

## Non-Goals

This round does not add:

- real LLM calls
- Claude SDK integration
- OpenAI SDK integration
- an interactive chat loop
- new CLI commands
- async APIs
- prompt template systems
- automatic summarization
- automatic preference extraction
- new dependencies

## Success Criteria

- `python examples/simple_agent_runtime.py` runs successfully.
- The output visibly shows memory context and a fake assistant response.
- The demo uses `MemoryRuntime` as an external runtime caller would.
- The demo does not add LLM dependencies or production runtime behavior.
- Full test suite passes with `pytest -v`.
