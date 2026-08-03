# Memora

Memora is a deterministic, local-first long-term memory system for agent runtimes.

It provides:

- Markdown memory files with YAML frontmatter
- SQLite memory storage with FTS candidate recall
- JSON session history
- Working memory state helpers
- Deterministic safety and write policy
- Keyword retrieval and scoring
- Optional local RAG with hash or BGE-M3 embeddings and SQLite vector storage
- Embedding-backed write-time duplicate, merge, and conflict detection
- Optional injected LLM relation judging
- Prompt formatting for retrieved memories
- Lifecycle cleanup, export, import, verify, rebuild, and backup commands
- A thin CLI for local debugging

Memora does not bundle hosted LLM clients, hosted vector databases, or a hosted service. The default install is dependency-light and uses hash embeddings; real local embedding providers such as BGE-M3 are opt-in extras.

## Install For Development

```bash
pip install -e .[dev]
```

## Run Tests

```bash
pytest
```

## Release Verification

```bash
python -m pip install -e ".[dev]"
pytest
python -m pip install build
python -m build
python -m pip install --force-reinstall dist/*.whl
python -c "import memora; print(memora.__version__)"
memora --help
```

The build step verifies both source and wheel distributions. The source distribution includes docs, examples, and tests. The wheel installs the core `memora` package and CLI.

## CLI Quickstart

The default memory backend is the Markdown file store.

```bash
python -m memora --root .memora init
python -m memora --root .memora save --type preference --name response-style --description "Response style preference." --content "Prefer concise answers."
python -m memora --root .memora remember --type preference --name response-style --description "Response style preference." --content "Prefer concise answers." --session session_1 --tag preference
python -m memora --root .memora remember --type preference --name response-style-json --description "Response style preference." --content "Prefer concise answers." --json
python -m memora --root .memora list
python -m memora --root .memora search "concise answers"
python -m memora --root .memora context "pytest verification"
python -m memora --root .memora show response-style
python -m memora --root .memora update response-style --tag style --weight 8
python -m memora --root .memora archive response-style
python -m memora --root .memora list --archived
python -m memora --root .memora restore response-style
python -m memora --root .memora search "concise answers" --type preference --tag style --top-k 5
python -m memora --root .memora delete response-style
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

Validation and policy failures are reported to stderr and return a non-zero exit code:

```bash
python -m memora --root .memora save --type preference --name secret --description "secret" --content "api_key = sk-abcdef123456"
# stderr: error: memory rejected: contains_secret
```

## SQLite Backend

Use `--backend sqlite` to store memories in SQLite at `<root>/memora.sqlite3`.

```bash
python -m memora --root .memora --backend sqlite init
python -m memora --root .memora --backend sqlite save --type preference --name response-style --description "Response style preference." --content "Prefer concise answers."
python -m memora --root .memora --backend sqlite search "concise answers"
python -m memora --root .memora --backend sqlite verify
python -m memora --root .memora --backend sqlite rebuild-index
```

SQLite FTS is used for candidate recall only. Final ranking still uses Memora's deterministic scoring. Session history remains JSON-file backed in this phase, even when SQLite is used for memories.

## RAG V1

RAG is disabled by default. Enable the deterministic local RAG path with `--rag`.

```bash
python -m memora --root .memora --backend sqlite --rag init
python -m memora --root .memora --backend sqlite --rag save --type preference --name response-style --description "Response style preference." --content "Prefer concise answers."
python -m memora --root .memora --backend sqlite --rag search "concise answers"
python -m memora --root .memora --backend sqlite --rag verify
python -m memora --root .memora --backend sqlite --rag rebuild-index
```

RAG v1 supports the dependency-free local `hash` embedding provider by default, the optional local `bge` provider for BGE-M3 embeddings, the local `sqlite` vector store, the optional `qdrant` vector store, and `none` or `deterministic` rerankers. Future embedding provider names such as `openai`, `cohere`, and `voyage`, and future vector stores such as `chroma`, remain reserved but report `reserved but not implemented` if selected.

To use a locally downloaded BGE-M3 model or Qdrant, install the optional extras you need. Memora does not download models and does not start or manage a Qdrant server.

```bash
pip install -e ".[bge]"
pip install -e ".[qdrant]"
pip install -e ".[bge,qdrant]"
```

A local `.env` file can hold embedding and vector-index settings. Copy `.env.example` to `.env`, then adjust the local model path, vector store, and Qdrant settings for your machine. Provider-specific vector store settings are parsed into `MemoryConfig.vector_store_options`; they are not top-level `MemoryConfig` fields. Dense-only BGE with SQLite looks like:

```env
MEMORA_BACKEND=sqlite
MEMORA_RAG=true
MEMORA_EMBEDDING_PROVIDER=bge
MEMORA_EMBEDDING_MODEL=bge-m3
MEMORA_EMBEDDING_MODEL_PATH=C:\Download\bge-m3
MEMORA_EMBEDDING_DIMENSION=1024
MEMORA_EMBEDDING_BATCH_SIZE=8
MEMORA_EMBEDDING_FP16=true
MEMORA_EMBEDDING_SPARSE=false
MEMORA_VECTOR_STORE=sqlite
MEMORA_RETRIEVAL_MODE=dense
MEMORA_KEYWORD_RECALL=auto
HF_OFFLINE=1
```

Dense-only Qdrant uses Qdrant as the derived vector index while MemoryStore remains authoritative:

```env
MEMORA_BACKEND=sqlite
MEMORA_RAG=true
MEMORA_VECTOR_STORE=qdrant
MEMORA_VECTOR_STORE_URL=http://localhost:6333
MEMORA_VECTOR_STORE_COLLECTION=memora_memories
MEMORA_EMBEDDING_PROVIDER=bge
MEMORA_EMBEDDING_MODEL_PATH=C:\Download\bge-m3
MEMORA_EMBEDDING_DIMENSION=1024
MEMORA_EMBEDDING_SPARSE=false
MEMORA_RETRIEVAL_MODE=dense
MEMORA_KEYWORD_RECALL=auto
HF_OFFLINE=1
```

Hybrid Qdrant retrieval requests BGE-M3 sparse lexical weights and uses Qdrant-side dense+sparse fusion:

```env
MEMORA_BACKEND=sqlite
MEMORA_RAG=true
MEMORA_VECTOR_STORE=qdrant
MEMORA_VECTOR_STORE_URL=http://localhost:6333
MEMORA_VECTOR_STORE_COLLECTION=memora_memories
MEMORA_EMBEDDING_PROVIDER=bge
MEMORA_EMBEDDING_MODEL_PATH=C:\Download\bge-m3
MEMORA_EMBEDDING_DIMENSION=1024
MEMORA_EMBEDDING_SPARSE=true
MEMORA_RETRIEVAL_MODE=hybrid
MEMORA_KEYWORD_RECALL=auto
MEMORA_HYBRID_PREFETCH_LIMIT=100
HF_OFFLINE=1
```

Then run commands with that env file:

```bash
python -m memora --env-file .env init
python -m memora --env-file .env rebuild-index
python -m memora --env-file .env search "结构化单据识别规则"
```

SQLite ignores sparse vectors and uses dense cosine search. Qdrant hybrid uses sparse vectors only when both `MEMORA_EMBEDDING_SPARSE=true` and `MEMORA_RETRIEVAL_MODE=hybrid` are set. `MEMORA_KEYWORD_RECALL` controls keyword candidates in RAG: `auto` uses SQLite FTS candidates when available and falls back to deterministic in-memory keyword scan; `fts` uses FTS only; `scan` uses deterministic scan only; `off` disables keyword candidates for pure vector retrieval. After changing `embedding_provider`, `embedding_model`, `embedding_model_path`, `embedding_dimension`, `vector_store`, or sparse/hybrid mode, run `rebuild-index` before relying on RAG search quality.

`verify` prints vector diagnostics when RAG is enabled:

```text
vector_ok=True missing=0 orphans=0 mismatches=0 sync_errors=0
```

`rebuild-index` rebuilds both the normal memory index and the RAG vector index when `--rag` is enabled. JSON `import` also syncs imported active memories into the vector index.

The selected local memory backend remains the source of truth. RAG is a retrieval index, not a separate authoritative memory store.

## Data Portability

Memora can export, import, verify, rebuild, and back up memory data:

```bash
python -m memora --root .memora export memories.json
python -m memora --root .memora import memories.json
python -m memora --root .memora verify
python -m memora --root .memora rebuild-index
python -m memora --root .memora backup backup.json
```

These commands cover memories only, not session history.

To move memories between backends, use the JSON export/import format:

```bash
python -m memora --root .memora --backend file export memories.json
python -m memora --root .memora --backend sqlite import memories.json
```

## Python Usage

```python
from memora.manager import MemoryManager


manager = MemoryManager()
manager.init_storage()
manager.save_memory(
    memory_type="preference",
    name="response-style",
    description="Response style preference.",
    content="Prefer concise answers.",
)
results = manager.retrieve_memory("concise answers")
print(manager.format_memories_for_prompt(results=results))
```

## Runtime Integration

External agent runtimes can use `MemoryRuntime` as a thin wrapper around the manager API.

```python
from memora.runtime import MemoryRuntime


runtime = MemoryRuntime()
runtime.init_storage()

context = runtime.build_context("response preferences and current project")
runtime.remember_message("session_1", "user", "What should we do next?")
runtime.remember_message("session_1", "assistant", "Continue the runtime integration.")
runtime.remember_summary("session_1", "The user accepted the runtime integration direction.")
```

Recommended agent integration pattern:

1. Build task context before the external agent call with `runtime.build_task_context(...)`.
2. Memora prepends pinned `preference` and `project` memories, then retrieves on-demand memories such as `tool` or `knowledge` by query and type.
3. Store user and assistant messages with `runtime.remember_message(...)`.
4. Let the external runtime, not Memora, extract candidate memories from conversations, traces, or task summaries.
5. Pass candidates to `runtime.remember_extracted(...)` so Memora can create, update, reject, or require confirmation.
6. If `result.action == "requires_confirmation"`, ask the user before calling `runtime.confirm_memory_candidate(result.candidate)`.

```python
context = runtime.build_task_context(
    "pytest failure after refactor",
    memory_types=["tool", "knowledge"],
)
```

Use `runtime.build_context(...)` only when you want query-only retrieval without pinned context.

## Agent Memory Write Pipeline

Memora does not call an LLM by default. Agent runtimes can either pass explicit candidate memories to Memora, or inject an `LLMMemoryExtractor`-compatible client that returns JSON-only extraction output. Extraction produces an auditable `ExtractionArtifact`; writing still goes through validation, policy evaluation, updating, rejection, or confirmation handling.

```python
from memora.runtime import MemoryRuntime


runtime = MemoryRuntime()
runtime.init_storage()

result = runtime.remember_extracted(
    memory_type="preference",
    name="response-style",
    description="Response style preference.",
    content="Prefer concise answers.",
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

## Write-Time Relation Flow

Memora separates extraction, relation recall, relation judgment, and storage:

1. The external agent or extractor produces a structured `MemoryCandidate`.
2. Memora rejects unsafe, transient, or noisy candidates before relation handling.
3. If semantic write relations or LLM relation judging is enabled, Memora uses the configured embedding provider to find one likely existing target in the same type and scope.
4. If an LLM relation judge is injected, Memora asks it to classify only that embedding hit as `none`, `duplicate`, `merge`, `conflict`, or `supersede`.
5. Policy decides whether to create, update, supersede, reject, or require confirmation.
6. The selected local backend writes the `MemoryItem`.
7. When RAG is enabled, the vector index is synced from the saved local item.

Duplicates and merges update the target memory. High-confidence conflicts and `supersede` decisions archive the old memory, create a new memory with `supersedes=[old.id]`, remove the old memory from the vector index, and sync the new memory.

`LLMMemoryRelationJudge` is provider-neutral. Any external client can be used if it implements `complete(messages) -> str` and returns JSON matching the relation decision schema:

```python
from memora.config import MemoryConfig
from memora.relations import LLMMemoryRelationJudge
from memora.runtime import MemoryRuntime


class MyRelationClient:
    def complete(self, messages):
        # Call your provider here and return JSON text only.
        return (
            '{"kind":"merge","confidence":0.9,"reason":"Refines existing memory.",'
            '"merged":{"name":"response-style","description":"Response style.",'
            '"content":"Prefer concise answers with short summaries.","tags":["style"]}}'
        )


runtime = MemoryRuntime(
    config=MemoryConfig(
        memory_backend="sqlite",
        rag_enabled=True,
        llm_relation_judge_enabled=True,
    ),
    relation_judge=LLMMemoryRelationJudge(MyRelationClient()),
)
```

OpenAI can be used with the same adapter shape. See `examples/openai_memory_clients.py` for a Responses API adapter that requests structured JSON output for both extraction and relation judging:

```python
from openai import OpenAI

from examples.openai_memory_clients import OpenAIRelationClient
from memora.relations import LLMMemoryRelationJudge


client = OpenAI()
judge = LLMMemoryRelationJudge(OpenAIRelationClient(client, "gpt-5.6"))
```

Production OpenAI integrations should prefer structured outputs, keep the relation prompt deterministic, and set conservative conflict replacement thresholds until the deployment has reviewed enough write logs. If OpenAI is unavailable or returns invalid JSON, Memora falls back to the deterministic embedding relation behavior.

## Write Result Diagnostics

`MemoryWriteResult` includes stable diagnostic fields for agent runtimes:

- `action`
- `reason`
- `target_memory_id`
- `memory`
- `candidate`
- `relation_kind`
- `relation_confidence`
- `relation_reason`
- `relation_judge_status`
- `relation_judge_error`
- `rag_sync_errors`

`relation_judge_status` can be:

- `accepted`: an injected relation judge returned a valid decision.
- `missing`: LLM relation judging was enabled and a relation target existed, but no judge was injected.
- `invalid`: the injected judge returned invalid relation JSON or an invalid decision.
- `failed`: the injected judge raised a non-validation exception.
- `null`: no LLM relation judge was called.

`rag_sync_errors` reports errors from the current write only. Memory writes degrade gracefully when RAG sync fails, and `verify` continues to expose accumulated RAG sync errors until `rebuild-index` repairs and clears them.

CLI `remember --json` prints the same machine-readable write result fields. CLI can simulate candidate writes and confirmation flow, but it does not instantiate LLM extractors, OpenAI clients, or LLM relation judges. Use the Python runtime API for hosted LLM integration.

For CLI debugging:

```bash
python -m memora --root .memora remember --type preference --name response-style --description "Response style preference." --content "Prefer concise answers." --session session_1 --json > candidate.json
python -m memora --root .memora confirm --candidate candidate.json --json
```

On Windows PowerShell, prefer explicit UTF-8 output for candidate files:

```powershell
python -m memora --root .memora remember --type preference --name response-style --description "Response style preference." --content "Prefer concise answers." --session session_1 --json | Set-Content -Encoding utf8 candidate.json
python -m memora --root .memora confirm --candidate candidate.json --json
```

If a pending candidate is later confirmed, Memora re-checks the current store state before writing. Stale confirmations return a new `requires_confirmation` result instead of overwriting newer memory.

## Configuration Behavior

Memora's policy-related configuration fields are active in the manager/runtime write paths:

- `max_memory_content_chars` controls the noisy-output content length limit.
- omitted write weights use type-specific defaults such as `default_preference_weight`, `default_project_weight`, `default_episodic_weight`, `default_reflective_weight`, `default_tool_weight`, `default_knowledge_weight`, and `default_general_weight`.
- explicit write weights are preserved.
- `allow_auto_save_user_preferences` and `allow_auto_save_project_facts` control whether automatic `runtime_extraction`, `session_extraction`, and `conversation` candidates can be written without confirmation.
- disabled auto-save returns `requires_confirmation`, not `rejected`.
- `semantic_write_relations_enabled` enables embedding-backed write-time duplicate, merge, and conflict relation detection. It is disabled by default.
- `semantic_relation_threshold`, `semantic_merge_threshold`, and `semantic_conflict_threshold` tune the similarity gates used before semantic write relations can affect policy decisions.
- `llm_relation_judge_enabled` lets an injected `MemoryRelationJudge` refine an embedding hit into `none`, `duplicate`, `merge`, `conflict`, or `supersede`.
- `llm_relation_confidence_threshold`, `llm_merge_confidence_threshold`, and `llm_conflict_auto_replace_threshold` control whether LLM relation decisions are accepted, merged, or allowed to auto-replace.
- `require_confirmation_for_conflicts` controls whether detected semantic conflicts require confirmation when high-confidence replacement is not allowed.
- `allow_high_confidence_conflict_replace` and `high_confidence_conflict_threshold` allow high-confidence semantic conflicts to supersede the target memory automatically.

Manual writes remain allowed unless rejected by safety, transient-state, noisy-output, or semantic conflict policy.

LLM relation judging is an integration hook, not a bundled hosted client. Construct an `LLMMemoryRelationJudge` with a compatible client and pass it to `MemoryRuntime(..., relation_judge=judge)` or `MemoryManager(..., relation_judge=judge)` while enabling `llm_relation_judge_enabled`.

## Examples

Run the fake agent runtime example:

```bash
python examples/simple_agent_runtime.py
```

Run the offline LLM relation demo:

```bash
python examples/llm_relation_runtime.py
```

Run the OpenAI relation judge demo:

```bash
pip install openai
set OPENAI_API_KEY=your-key
python examples/openai_llm_relation_runtime.py
```

Run the full OpenAI memory-turn demo:

```bash
pip install openai
set OPENAI_API_KEY=your-key
python examples/openai_full_memory_turn_runtime.py
```

Run the full OpenAI-backed memory-system demo:

```bash
pip install openai
set OPENAI_API_KEY=your-key
python examples/openai_memory_system_runtime.py
```

The memory-system demo shows typed context injection before a task, task-end LLM extraction, relation judging, local backend writes, RAG sync, and `supersedes` audit output. OpenAI examples use the Responses API through a small adapter in `examples/openai_memory_clients.py`. Override the default model with `OPENAI_MODEL` if needed.

## MVP Boundaries

This version includes a deterministic local RAG v1 path, an LLM extraction contract, embedding-backed write-time relation detection, hash embeddings, optional local BGE-M3 dense or dense+sparse embeddings, SQLite and optional Qdrant vector indexes, hybrid vector + keyword retrieval, write-result diagnostics, and RAG verify/rebuild diagnostics.

It does not include bundled external LLM clients, hosted embedding APIs, a bundled/managed Qdrant server, model rerankers, web UI, or hosted multi-tenant service.
