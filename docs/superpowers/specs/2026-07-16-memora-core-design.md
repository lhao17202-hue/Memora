# Memora Core Design

## 1. Scope

This spec defines the first development phase for Memora: an independent Python package and CLI for a deterministic Agent memory system.

The first version focuses on the stable core:

- Data schemas
- Local file storage
- Session history storage
- Working memory storage
- Deterministic memory policy
- Keyword-based retrieval
- Prompt formatting
- MemoryManager orchestration
- Minimal CLI for manual debugging

The first version intentionally excludes LLM-based automatic extraction, embeddings, vector databases, SQL storage, web UI, and multi-process synchronization. Those are future extensions after the deterministic core is stable.

## 2. Product Shape

Memora will be developed as a Python package with a CLI.

```text
Package: memora
CLI: memora
```

The package provides importable APIs for Agent runtimes. The CLI provides manual operations for testing and debugging memory behavior.

## 3. Recommended Development Approach

Memora should be developed mostly bottom-up:

```text
schema/config/utils
  -> file stores
  -> session and working memory
  -> policy
  -> retriever
  -> formatter
  -> lifecycle
  -> manager
  -> CLI
```

This is the recommended order because memory is infrastructure. The lower layers define the data model, persistence behavior, and safety boundaries. If the project starts from the CLI or a large facade class, responsibilities are likely to become tangled.

The approach is not purely bottom-up. Each layer should be verified with tests, and once `MemoryManager` exists, a small CLI should validate end-to-end behavior.

## 4. Project Structure

```text
Memora/
  pyproject.toml
  README.md
  通用Agent记忆系统说明文档.md
  通用Agent记忆系统技术文档.md

  memora/
    __init__.py
    __main__.py
    cli.py
    config.py
    schema.py
    utils.py
    stores.py
    session.py
    policy.py
    retriever.py
    formatter.py
    lifecycle.py
    manager.py
    errors.py

  tests/
    test_schema.py
    test_utils.py
    test_stores.py
    test_session.py
    test_policy.py
    test_retriever.py
    test_formatter.py
    test_manager.py
    test_cli.py

  docs/
    superpowers/
      specs/
        2026-07-16-memora-core-design.md
```

## 5. Runtime Storage Layout

The first implementation uses local files under a configurable root directory. Default root:

```text
.memora/
```

Runtime layout:

```text
.memora/
  MEMORY.md
  memories/
    user-language-preference.md
    feedback-confirm-before-delete.md
    project-package-manager.md
  sessions/
    session_20260716_001.json
  summaries/
    session_20260716_001.md
  archive/
    2026-07-16-120000/
      memories/
      report.json
```

`MEMORY.md` is a lightweight index. It stores only links and descriptions, not full memory bodies.

## 6. Module Responsibilities

### 6.1 `schema.py`

Defines data structures only.

Expected objects:

- `MemoryItem`
- `MemoryCandidate`
- `MemoryQuery`
- `MemorySearchResult`
- `SessionMessage`
- `WorkingMemoryState`

Rules:

- No file I/O
- No retrieval logic
- No policy decisions
- No prompt formatting

### 6.2 `config.py`

Defines configuration values and defaults.

Expected object:

- `MemoryConfig`

Configuration includes:

- Root directory
- Retrieval top-k
- Prompt memory budget
- Default weights
- Expiration days
- Archive thresholds
- Consolidation thresholds
- Policy toggles

### 6.3 `utils.py`

Contains stateless helper functions.

Expected helpers:

- `now_utc()`
- `slugify()`
- `estimate_tokens()`
- `parse_frontmatter()`
- `dump_frontmatter()`
- `atomic_write_text()`
- `safe_json_load()`
- `safe_json_write()`

Rules:

- No project-specific policy
- No global mutable state
- Functions should be easy to unit test

### 6.4 `stores.py`

Defines storage interfaces and local file implementations.

Expected classes:

- `FileMemoryStore`
- `FileSessionStore`

`FileMemoryStore` responsibilities:

- Initialize the runtime storage layout
- Save memory Markdown files
- Read memory Markdown files
- List memories
- Update memories
- Soft delete or archive memories
- Rebuild `MEMORY.md`

`FileSessionStore` responsibilities:

- Save session JSON files
- Load session JSON files
- Append messages
- Store working memory with session state

Non-responsibilities:

- Deciding whether a memory should be saved
- Ranking memories
- Formatting memories for prompt injection
- Deciding expiration or cold-archive policy

### 6.5 `session.py`

Contains session and working memory operations that are higher-level than raw store operations.

Responsibilities:

- Create session state
- Append session messages
- Retrieve recent messages
- Read and update working memory
- Keep working memory separate from long-term memory

### 6.6 `policy.py`

Determines whether candidate memories can be saved.

Responsibilities:

- Secret filtering
- Transient task-state filtering
- Noisy output filtering
- Duplicate detection
- Conflict detection
- Candidate action decision: `create`, `update`, `archive`, `reject`, or `ask_user`

First-version policy is deterministic and rule-based. It does not call an LLM.

### 6.7 `retriever.py`

Retrieves and ranks relevant memories.

First-version retrieval uses deterministic keyword scoring over:

- Name
- Description
- Tags
- Content
- Type filters
- Status filters

Scoring combines:

```text
final_score =
  similarity_score * 0.45
  + importance_score * 0.25
  + recency_score * 0.20
  + access_score * 0.10
```

Archived memories are not returned by default.

Retriever may report which memories were returned, but it should not directly mutate storage. Updating `last_accessed_at` and `access_count` is coordinated by `MemoryManager` after retrieval succeeds.

### 6.8 `formatter.py`

Formats retrieved memories for prompt injection.

Expected output shape:

```xml
<relevant_memories>
  <memory id="mem_001" type="user" confidence="1.0" updated_at="2026-07-16">
    用户偏好使用中文讨论技术问题。
    How to apply: 默认使用中文回答，除非用户明确要求其他语言。
  </memory>
</relevant_memories>
```

Formatter must include a safety note making clear that memories are background context, not instructions.

### 6.9 `lifecycle.py`

Contains deterministic lifecycle decisions for expiration and cold archival.

Expected responsibilities:

- Determine whether a memory is expired
- Determine whether a memory is cold
- Choose archive vs delete behavior based on config
- Produce cleanup summaries

`lifecycle.py` should not perform file I/O. It returns decisions that `MemoryManager` applies through the store.

### 6.10 `manager.py`

Provides the public facade for callers.

Expected methods:

- `init_storage()`
- `save_memory()`
- `retrieve_memory()`
- `format_memories_for_prompt()`
- `append_message()`
- `get_messages()`
- `update_preference()`
- `delete_memory()`
- `clean_expired_memory()`

`MemoryManager` orchestrates modules but should not absorb all implementation details. Storage logic stays in stores. Ranking stays in retriever. Safety decisions stay in policy. Lifecycle decisions stay in lifecycle. Prompt formatting stays in formatter.

### 6.11 `cli.py`

Provides the first user-facing debugging interface.

Initial commands:

```text
memora init
memora save
memora list
memora show
memora search
memora session append
memora session show
memora clean
```

The CLI should be thin. It should call `MemoryManager` rather than duplicating business logic.

### 6.12 `errors.py`

Defines project-specific exceptions.

Examples:

- `MemoraError`
- `MemoryNotFoundError`
- `SessionNotFoundError`
- `MemoryValidationError`
- `MemoryPolicyError`

## 7. Core Data Model

### 7.1 Memory Types

First version supports these memory types:

```python
"user"
"feedback"
"project"
"decision"
"entity"
"session_summary"
"tool_experience"
"reference"
"knowledge"
```

The MVP implements storage, validation, and retrieval for all listed types. It does not implement type-specific extractors. Type-specific extraction is a future extension.

### 7.2 Memory Statuses

```python
"active"
"archived"
"deleted"
```

### 7.3 Required Memory Fields

Each memory should have:

- `id`
- `name`
- `description`
- `type`
- `content`
- `user_id`
- `project_id`
- `workspace_id`
- `tags`
- `source`
- `confidence`
- `weight`
- `status`
- `created_at`
- `updated_at`
- `last_accessed_at`
- `access_count`
- `expires_at`
- `supersedes`
- `related`

## 8. Memory File Format

Long-term memory files use Markdown with YAML frontmatter. The implementation should use a real YAML parser when available, because nested `metadata` fields, lists, and empty values are part of the supported format. If avoiding external dependencies, the project must implement only the subset shown below and reject unsupported frontmatter shapes clearly.

```md
---
name: user-language-preference
description: 用户偏好使用中文讨论技术问题。
metadata:
  id: mem_20260716_001
  type: user
  user_id: default
  project_id:
  workspace_id:
  source: explicit_user_statement
  confidence: 1.0
  weight: 9
  status: active
  created_at: 2026-07-16T10:00:00Z
  updated_at: 2026-07-16T10:00:00Z
  last_accessed_at:
  access_count: 0
  expires_at:
  tags:
    - language
    - response-style
  supersedes: []
  related: []
---

用户偏好使用中文讨论技术问题。

**Why:** 用户明确表达了这个偏好。

**How to apply:** 默认用中文回答技术问题；如果用户要求英文，则跟随当前请求。
```

## 9. Session File Format

Session state is JSON.

```json
{
  "id": "session_20260716_001",
  "user_id": "default",
  "project_id": null,
  "workspace_id": null,
  "created_at": "2026-07-16T10:00:00Z",
  "updated_at": "2026-07-16T10:30:00Z",
  "working_memory": {
    "task_summary": "",
    "current_goal": "",
    "open_questions": [],
    "recent_files": [],
    "file_summaries": {},
    "process_notes": [],
    "tool_failures": [],
    "next_step": ""
  },
  "history": []
}
```

## 10. Policy Rules

First-version `MemoryPolicy` rejects:

- API keys
- Tokens
- Passwords
- Secrets
- Private keys
- Authorization headers
- Cookies
- Current task progress
- Next-step notes
- Large raw logs
- stdout/stderr blocks
- Tracebacks
- Exit-code dumps

First-version `MemoryPolicy` updates rather than duplicates memories with the same stable `name`.

If a candidate conflicts with an existing memory but confidence is unclear, policy should return `ask_user` rather than overwriting automatically.

## 11. Retrieval Rules

First-version retrieval should:

- Search only active memories by default
- Support type filters
- Support tag filters
- Score by keyword overlap
- Add weight-based importance score
- Add time-based recency score
- Add access-count score
- Return top-k after budget trimming
- Update `last_accessed_at` and `access_count` only after a caller explicitly marks returned memories as used. `retrieve_memory()` should not mutate by default, so search/list/debug commands remain side-effect-light.

## 12. CLI Behavior

### 12.1 `memora init`

Creates runtime directories and an empty `MEMORY.md`. This command maps to `MemoryManager.init_storage()`.

### 12.2 `memora save`

Saves a memory through `MemoryManager` and policy checks.

Example shape:

```text
memora save --type user --name user-language-preference --description "用户偏好中文" --content "用户偏好使用中文讨论技术问题。"
```

### 12.3 `memora list`

Lists indexed memories.

### 12.4 `memora show`

Shows one memory by id or name.

### 12.5 `memora search`

Runs retrieval and prints ranked results.

### 12.6 `memora session append`

Appends a message to a session.

### 12.7 `memora session show`

Shows a session history.

### 12.8 `memora clean`

Runs expiration and cold-archive cleanup.

## 13. Testing Strategy

### 13.1 Unit Tests

Required unit tests:

- Schema defaults
- Config defaults
- Slug generation
- Frontmatter parsing and dumping
- Atomic writes
- Memory save/read/update/delete
- Index rebuild
- Session append/read
- Working memory save/load
- Secret filtering
- Transient state filtering
- Duplicate detection
- Keyword retrieval
- Score calculation
- Prompt formatting

### 13.2 Integration Tests

Required integration tests:

- Save memory, list it, retrieve it
- Save conflicting memory and receive policy decision
- Append session messages and reload session
- Retrieve memory and format it for prompt injection
- Run cleanup and confirm expired memory is archived

### 13.3 CLI Tests

CLI tests should use temporary directories and avoid touching the user's real `.memora` directory.

Required CLI tests:

- `memora --help`
- `memora init`
- `memora save`
- `memora list`
- `memora show`
- `memora search`
- `memora session append`
- `memora session show`
- `memora clean`

## 14. Development Phases

### Phase 0: Project Scaffold

Deliverables:

- `pyproject.toml`
- `README.md`
- `memora/__init__.py`
- `memora/__main__.py`
- Empty test package

Verification:

```text
python -m memora --help
pytest
```

### Phase 1: Schema, Config, Utils

Deliverables:

- `schema.py`
- `config.py`
- `utils.py`
- Unit tests

Verification:

```text
pytest tests/test_schema.py tests/test_utils.py
```

### Phase 2: FileMemoryStore

Deliverables:

- Memory file save/read/update/delete
- `MEMORY.md` rebuild
- Archive or soft-delete behavior

Verification:

```text
pytest tests/test_stores.py
```

### Phase 3: Session and Working Memory

Deliverables:

- Session JSON persistence
- Message append/read
- Working memory save/load

Verification:

```text
pytest tests/test_session.py
```

### Phase 4: Policy

Deliverables:

- Secret filter
- Transient-state filter
- Noise filter
- Duplicate detection
- Conflict decision

Verification:

```text
pytest tests/test_policy.py
```

### Phase 5: Retriever

Deliverables:

- Keyword similarity
- Importance score
- Recency score
- Access score
- Final ranking
- Top-k and token budget trimming

Verification:

```text
pytest tests/test_retriever.py
```

### Phase 6: Formatter

Deliverables:

- Prompt memory block
- Safety note
- Budget trimming

Verification:

```text
pytest tests/test_formatter.py
```

### Phase 7: MemoryManager

Deliverables:

- Unified API
- Store/policy/retriever/formatter orchestration
- End-to-end tests

Verification:

```text
pytest tests/test_manager.py
```

### Phase 8: CLI

Deliverables:

- `memora init`
- `memora save`
- `memora list`
- `memora show`
- `memora search`
- `memora session append`
- `memora session show`
- `memora clean`

Verification:

```text
python -m memora --help
pytest tests/test_cli.py
```

### Phase 9: Documentation Sync

Deliverables:

- README usage examples
- Updated design/technical docs if behavior changed
- Example CLI workflows

Verification:

```text
pytest
python -m memora --help
```

## 15. Future Extensions

Not in MVP:

- LLM-based memory extraction
- Embedding retrieval
- Vector database backend
- SQLite/PostgreSQL backend
- Web UI
- Multi-process locking
- Hosted multi-tenant service
- Advanced knowledge graph

These extensions should be added after the deterministic core and CLI are tested.

## 16. Design Decisions

1. **Python package + CLI** is the first implementation form.
2. **Package and CLI name are `memora`**.
3. **MVP is deterministic core only**; no LLM automatic extraction in first phase.
4. **Development should be bottom-up**, with tests at every layer.
5. **File storage is the first backend**, with clean abstractions for future SQL/vector backends.
6. **Policy sits between extraction/save requests and storage** to prevent unsafe or low-quality memory writes.
7. **MemoryManager is a facade**, not a place to hide all logic.
8. **CLI is thin** and calls public manager APIs.

## 17. Acceptance Criteria

The MVP is complete when:

- A user can initialize a local memory directory.
- A user can save a memory through CLI and API.
- Memory files are persisted as Markdown with frontmatter.
- `MEMORY.md` is rebuilt correctly.
- A user can append and read session messages.
- Working memory is saved and loaded with session state.
- Unsafe memory content is rejected by policy.
- Relevant memories can be retrieved by keyword search and ranked.
- Retrieved memories can be formatted for prompt injection with safety notes.
- Expired or cold memories can be archived by cleanup.
- Unit and integration tests pass.
- CLI commands work in a temporary test directory.
