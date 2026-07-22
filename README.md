# Memora

Memora is a deterministic local memory system for agent runtimes.

It provides:

- Markdown memory files with YAML frontmatter
- SQLite memory storage with FTS candidate recall
- JSON session history
- Working memory state
- Deterministic safety policy
- Keyword retrieval and scoring
- Prompt formatting
- Lifecycle cleanup
- A thin CLI for debugging

## Install for development

```bash
pip install -e .[dev]
```

## Run tests

```bash
pytest
```

## CLI quickstart

```bash
python -m memora --root .memora init
python -m memora --root .memora save --type user --name language --description "用户偏好中文。" --content "用户偏好使用中文回答。"
python -m memora --root .memora remember --type user --name language --description "用户偏好中文。" --content "用户偏好使用中文回答。" --session session_1 --tag preference
python -m memora --root .memora list
python -m memora --root .memora search "中文回答"
python -m memora --root .memora show language
python -m memora --root .memora update language --tag language --weight 8
python -m memora --root .memora archive language
python -m memora --root .memora list --archived
python -m memora --root .memora restore language
python -m memora --root .memora search "中文回答" --type user --tag language --top-k 5
python -m memora --root .memora delete language
python -m memora --root .memora list --all
python -m memora --root .memora session append session_1 --role user --content "hello"
python -m memora --root .memora session show session_1
python -m memora --root .memora export memories.json
python -m memora --root .memora import memories.json
python -m memora --root .memora verify
python -m memora --root .memora rebuild-index
python -m memora --root .memora backup backup.json
python -m memora --root .memora clean
```

The default memory backend is the Markdown file store.

## SQLite backend

Use `--backend sqlite` to store memories in SQLite at `<root>/memora.sqlite3`:

```bash
python -m memora --root .memora --backend sqlite init
python -m memora --root .memora --backend sqlite save --type user --name language --description "用户偏好中文。" --content "用户偏好使用中文回答。"
python -m memora --root .memora --backend sqlite search "中文回答"
python -m memora --root .memora --backend sqlite verify
python -m memora --root .memora --backend sqlite rebuild-index
```

SQLite FTS is used for candidate recall only. Final ranking still uses Memora's deterministic scoring. Chinese short-query fallback is preserved.

## RAG v1

RAG is disabled by default. Enable the deterministic local RAG path with `--rag`:

```bash
python -m memora --root .memora --backend sqlite --rag init
python -m memora --root .memora --backend sqlite --rag save --type user --name language --description "用户偏好中文。" --content "用户偏好使用中文回答。"
python -m memora --root .memora --backend sqlite --rag search "中文回答"
python -m memora --root .memora --backend sqlite --rag verify
python -m memora --root .memora --backend sqlite --rag rebuild-index
```

RAG v1 supports only the local `hash` embedding provider, `sqlite` vector store, and `none` or `deterministic` rerankers. Future provider names such as `openai`, `qdrant`, and `chroma` are reserved in one registry but report `reserved but not implemented` if selected.

`verify` prints vector diagnostics when RAG is enabled:

```text
vector_ok=True missing=0 orphans=0 mismatches=0 sync_errors=0
```

`rebuild-index` rebuilds both the normal memory index and the RAG vector index when `--rag` is enabled. JSON `import` also syncs imported active memories into the vector index.

Memory storage is scoped by `user_id`, `project_id`, `workspace_id`, and name so different scopes can keep the same memory name independently. Name-based operations such as `show`, `update`, `archive`, and `delete` remain unscoped in the current CLI/API, so use memory IDs when duplicate names exist across scopes.

Session history remains JSON-file backed in this phase, even when `--backend sqlite` is used for memories.

To move memories between backends, use the existing JSON export/import format:

```bash
python -m memora --root .memora --backend file export memories.json
python -m memora --root .memora --backend sqlite import memories.json
```

## CLI error behavior

Validation and policy failures are reported to stderr and return a non-zero exit code:

```bash
python -m memora --root .memora save --type user --name secret --description "secret" --content "api_key = sk-abcdef123456"
# stderr: error: memory rejected: contains_secret
```

## Data portability and integrity

Memora can export, import, verify, rebuild, and back up memory files:

```bash
python -m memora --root .memora export memories.json
python -m memora --root .memora import memories.json
python -m memora --root .memora verify
python -m memora --root .memora rebuild-index
python -m memora --root .memora backup backup.json
```

These commands cover memories only, not session history.

## Python usage

```python
from memora.manager import MemoryManager

manager = MemoryManager()
manager.init_storage()
manager.save_memory(
    memory_type="user",
    name="language",
    description="用户偏好中文。",
    content="用户偏好使用中文回答。",
)
results = manager.retrieve_memory("中文回答")
print(manager.format_memories_for_prompt(results=results))
```

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

## Agent memory write pipeline

Memora does not call an LLM to extract memories. External RAG or LLM agent runtimes can extract candidate memories, then pass those candidates to Memora for deterministic validation, policy evaluation, writing, updating, rejection, or confirmation handling.

```python
from memora.runtime import MemoryRuntime

runtime = MemoryRuntime()
runtime.init_storage()

result = runtime.remember_extracted(
    memory_type="user",
    name="language",
    description="用户偏好中文。",
    content="用户偏好使用中文回答。",
    session_id="session_1",
)
print(result.action, result.reason)
```

For CLI debugging, use `remember` to simulate an agent-extracted candidate memory:

```bash
python -m memora --root .memora remember --type user --name language --description "用户偏好中文。" --content "用户偏好使用中文回答。" --session session_1
```

Policy outcomes such as `rejected` and `requires_confirmation` are returned as normal write results for agent workflows.

## Configuration behavior

Memora's policy-related configuration fields are active in the manager/runtime write paths:

- `max_memory_content_chars` controls the noisy-output content length limit.
- omitted write weights use type-specific defaults such as `default_user_weight`, `default_feedback_weight`, `default_project_weight`, `default_summary_weight`, and `default_tool_experience_weight`.
- explicit write weights are preserved.
- `allow_auto_save_user_preferences` and `allow_auto_save_project_facts` control whether automatic `runtime_extraction`, `session_extraction`, and `conversation` candidates can be written without confirmation.
- disabled auto-save returns `requires_confirmation`, not `rejected`.
- `require_confirmation_for_conflicts` controls whether simple deterministic conflicts require confirmation.

Manual writes remain allowed unless rejected by safety, transient-state, noisy-output, or conflict policy.

## Agent runtime demo

Run the fake agent runtime example:

```bash
python examples/simple_agent_runtime.py
```

The demo uses `MemoryRuntime` with a local fake assistant response. It does not call an LLM.

## MVP boundaries

This version includes a deterministic local RAG v1 path: hash embeddings, a SQLite-backed vector index, hybrid vector + keyword retrieval, and RAG verify/rebuild diagnostics. It does not include LLM-based extraction, external embedding providers, hosted vector databases, model rerankers, web UI, or hosted multi-tenant service.
