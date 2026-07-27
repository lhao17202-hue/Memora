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

## Release verification

CI runs the same lightweight checks expected before an alpha release:

```bash
python -m pip install -e ".[dev]"
pytest
python -m pip install build
python -m build
python -m pip install --force-reinstall dist/*.whl
python -c "import memora; print(memora.__version__)"
memora --help
```

The build step verifies both source and wheel distributions. Memora remains local-first: the core package has no hosted service, LLM, or external vector database dependency.

## CLI quickstart

```bash
python -m memora --root .memora init
python -m memora --root .memora save --type preference --name language --description "用户偏好中文。" --content "用户偏好使用中文回答。"
python -m memora --root .memora remember --type preference --name language --description "用户偏好中文。" --content "用户偏好使用中文回答。" --session session_1 --tag preference
python -m memora --root .memora remember --type preference --name language-json --description "用户偏好中文。" --content "用户偏好使用中文回答。" --json
python -m memora --root .memora list
python -m memora --root .memora search "中文回答"
python -m memora --root .memora show language
python -m memora --root .memora update language --tag language --weight 8
python -m memora --root .memora archive language
python -m memora --root .memora list --archived
python -m memora --root .memora restore language
python -m memora --root .memora search "中文回答" --type preference --tag language --top-k 5
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
python -m memora --root .memora --backend sqlite save --type preference --name language --description "用户偏好中文。" --content "用户偏好使用中文回答。"
python -m memora --root .memora --backend sqlite search "中文回答"
python -m memora --root .memora --backend sqlite verify
python -m memora --root .memora --backend sqlite rebuild-index
```

SQLite FTS is used for candidate recall only. Final ranking still uses Memora's deterministic scoring. Chinese short-query fallback is preserved.

## RAG v1

RAG is disabled by default. Enable the deterministic local RAG path with `--rag`:

```bash
python -m memora --root .memora --backend sqlite --rag init
python -m memora --root .memora --backend sqlite --rag save --type preference --name language --description "用户偏好中文。" --content "用户偏好使用中文回答。"
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

### Deterministic retrieval quality

Memora's local retrieval favors deterministic lexical evidence before broad semantic recall. Exact name and description matches rank highest, followed by adjacent content phrases, tag token matches, and partial content token matches. When RAG is enabled, the built-in hash-vector recall remains deterministic and local, but weak semantic-only candidates below `min_semantic_score` are filtered so exact and phrase keyword matches stay prominent.

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
python -m memora --root .memora save --type preference --name secret --description "secret" --content "api_key = sk-abcdef123456"
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
    memory_type="preference",
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

Recommended agent integration pattern:

1. Build task context before the external agent call with `runtime.build_task_context(...)`.
2. Memora automatically prepends pinned `preference` and `project` memories, then retrieves on-demand memories such as `tool` or `knowledge` by query and type.
3. Store user and assistant messages with `runtime.remember_message(...)`.
4. Let the external runtime, not Memora, extract candidate memories.
5. Pass candidates to `runtime.remember_extracted(...)` so Memora can deterministically create, update, reject, or require confirmation.
6. If `result.action == "requires_confirmation"`, ask the user before calling `runtime.confirm_memory_candidate(result.candidate)`.

```python
context = runtime.build_task_context(
    "pytest failure after refactor",
    memory_types=["tool", "knowledge"],
)
```

Use `runtime.build_context(...)` only when you want the older query-only retrieval behavior without pinned context.

On Windows PowerShell, if Chinese CLI output renders incorrectly, switch the console to UTF-8 before running examples:

```powershell
chcp 65001
$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
```

## Agent memory write pipeline

Memora does not call an LLM by default. Agent runtimes can either pass explicit candidate memories to Memora, or inject an `LLMMemoryExtractor`-compatible client that returns JSON-only extraction output. Extraction produces an auditable `ExtractionArtifact`; writing still goes through deterministic validation, policy evaluation, updating, rejection, or confirmation handling.

```python
from memora.runtime import MemoryRuntime

runtime = MemoryRuntime()
runtime.init_storage()

result = runtime.remember_extracted(
    memory_type="preference",
    name="language",
    description="用户偏好中文。",
    content="用户偏好使用中文回答。",
    session_id="session_1",
)
print(result.action, result.reason)
```

Optional LLM extraction keeps extraction and writing separate:

```python
from memora.extraction import LLMMemoryExtractor
from memora.runtime import MemoryRuntime


class MyLLMClient:
    def complete(self, messages):
        # Call your model here and return JSON text only.
        return '{"should_remember": false, "memories": []}'


runtime = MemoryRuntime(extractor=LLMMemoryExtractor(MyLLMClient()))
runtime.init_storage()

artifact, results = runtime.extract_and_remember(
    [{"role": "user", "content": "Please remember I prefer concise answers."}]
)
print(artifact.errors, [result.action for result in results])
```

If no extractor is configured, `runtime.extract_memories(...)` returns a `memory_extractor_not_configured` artifact and writes nothing.

For CLI debugging, use `remember --json` to simulate an agent-extracted candidate memory and `confirm` to write a returned pending candidate after user approval:

```bash
python -m memora --root .memora remember --type preference --name language --description "用户偏好中文。" --content "用户偏好使用中文回答。" --session session_1 --json > candidate.json
python -m memora --root .memora confirm --candidate candidate.json --json
```

On Windows PowerShell, prefer explicit UTF-8 output for candidate files:

```powershell
python -m memora --root .memora remember --type preference --name language --description "用户偏好中文。" --content "用户偏好使用中文回答。" --session session_1 --json | Set-Content -Encoding utf8 candidate.json
python -m memora --root .memora confirm --candidate candidate.json --json
```

Policy outcomes such as `rejected` and `requires_confirmation` are returned as normal write results for agent workflows. `MemoryPolicy` is decision-only: direct policy calls do not resolve omitted defaults such as `weight`. Use `MemoryManager` or `MemoryRuntime` APIs for persistence-ready decisions.

`requires_confirmation` results include the pending candidate, a `suggested_action`, and `target_memory_id` when confirmation would update an existing memory. Agent runtimes can pass the returned candidate back after user approval:

```python
result = runtime.remember_extracted(
    memory_type="preference",
    name="language",
    description="用户偏好中文。",
    content="用户偏好中文回答。",
)
if result.action == "requires_confirmation" and result.candidate is not None:
    confirmed = runtime.confirm_memory_candidate(result.candidate)
    print(confirmed.action, confirmed.memory.id if confirmed.memory else None)
```

CLI `remember --json` prints machine-readable write results with `action`, `reason`, `target_memory_id`, `memory`, and `candidate` fields for agent runtime debugging. If a pending candidate is later confirmed, Memora re-checks the current store state before writing; stale confirmations return a new `requires_confirmation` result instead of overwriting newer memory.

## Configuration behavior

Memora's policy-related configuration fields are active in the manager/runtime write paths:

- `max_memory_content_chars` controls the noisy-output content length limit.
- omitted write weights use type-specific defaults such as `default_preference_weight`, `default_project_weight`, `default_episodic_weight`, `default_reflective_weight`, `default_tool_weight`, `default_knowledge_weight`, and `default_general_weight`.
- explicit write weights are preserved.
- `allow_auto_save_user_preferences` and `allow_auto_save_project_facts` control whether automatic `runtime_extraction`, `session_extraction`, and `conversation` candidates can be written without confirmation.
- disabled auto-save returns `requires_confirmation`, not `rejected`.
- `semantic_write_relations_enabled` enables embedding-backed write-time duplicate, merge, and conflict relation detection. It is disabled by default.
- `semantic_relation_threshold`, `semantic_merge_threshold`, and `semantic_conflict_threshold` tune the similarity gates used before semantic write relations can affect policy decisions.
- `require_confirmation_for_conflicts` controls whether detected semantic conflicts require confirmation when high-confidence replacement is not allowed.
- `allow_high_confidence_conflict_replace` and `high_confidence_conflict_threshold` allow high-confidence semantic conflicts to update the target memory automatically.

Manual writes remain allowed unless rejected by safety, transient-state, noisy-output, or semantic conflict policy.

## Agent runtime demo

Run the fake agent runtime example:

```bash
python examples/simple_agent_runtime.py
```

The demo uses `MemoryRuntime` with a local fake assistant response. It does not call an LLM.

## MVP boundaries

This version includes a deterministic local RAG v1 path, an LLM extraction contract, embedding-backed write-time relation detection, hash embeddings, a SQLite-backed vector index, hybrid vector + keyword retrieval, and RAG verify/rebuild diagnostics. It does not include bundled external LLM clients, external embedding providers, hosted vector databases, model rerankers, web UI, or hosted multi-tenant service.
