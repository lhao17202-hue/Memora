# Memora MVP Hardening Design

**Goal:** Strengthen the existing deterministic Memora MVP so invalid data is rejected early, project-specific errors are used consistently, local stores handle boundary cases predictably, and the CLI reports failures clearly.

**Scope:** This is a hardening pass only. It does not add JSON CLI output, update/delete/archive/restore commands, LLM-based extraction, SQLite/vector backends, or Agent runtime integration.

## Current State

Memora already has a working local core with dataclass schemas, Markdown/YAML memory files, JSON sessions, deterministic policy, keyword retrieval, prompt formatting, lifecycle cleanup, manager facade, CLI, and tests. The current MVP passes its test suite, but several boundaries are still loose:

- Runtime schema values are mostly trusted after dataclass construction.
- Some manager failures use `ValueError` instead of Memora-specific exceptions.
- Store parsing assumes valid datetime/type/status metadata.
- Session file paths are derived directly from session IDs.
- CLI errors are not normalized through project exceptions.

## Design

### 1. Schema validation

Add lightweight runtime validation in `memora/schema.py` without introducing Pydantic or another validation dependency.

Add constants:

- `VALID_MEMORY_TYPES`
- `VALID_MEMORY_STATUSES`
- `VALID_CANDIDATE_ACTIONS`
- `VALID_SESSION_ROLES`

Add validation functions:

- `validate_memory_type(value: str) -> None`
- `validate_memory_status(value: str) -> None`
- `validate_candidate_action(value: str) -> None`
- `validate_memory_item(item: MemoryItem) -> None`
- `validate_memory_candidate(candidate: MemoryCandidate) -> None`
- `validate_memory_query(query: MemoryQuery) -> None`
- `validate_session_message(message: SessionMessage) -> None`

Validation failures raise `MemoryValidationError`.

Rules:

- `MemoryItem.id`, `name`, `description`, and `content` must be non-empty strings.
- `MemoryItem.type` must be one of the supported memory types.
- `MemoryItem.status` must be `active`, `archived`, or `deleted`.
- `MemoryCandidate.action` must be a supported candidate action.
- `weight` must be from `1` to `10`.
- `confidence` must be from `0.0` to `1.0`.
- `access_count` must be `>= 0`.
- `MemoryQuery.query` must be a string.
- `MemoryQuery.top_k` must be `> 0`.
- `MemoryQuery.max_tokens` must be `> 0`.
- `SessionMessage.role` must be one of `user`, `assistant`, `system`, or `tool`.
- `SessionMessage.content` must be a string.

### 2. Typed errors

Use existing project exceptions from `memora/errors.py` consistently:

- `MemoryValidationError` for invalid schema/store/session input.
- `MemoryPolicyError` for deterministic policy rejection or confirmation-required saves.
- `MemoryNotFoundError` only when callers request a missing memory and the API contract requires failure. Existing optional lookup APIs may continue returning `None` for compatibility.

In `memora/manager.py`, replace direct `ValueError` policy failures with `MemoryPolicyError`. Validate inputs before constructing persisted memory items.

### 3. Store robustness

Strengthen `memora/stores.py` while keeping the file format stable.

Memory store:

- Validate `MemoryItem` objects before writing.
- Validate parsed memory objects after reading.
- Invalid datetime strings in frontmatter raise `MemoryValidationError` with a clear message.
- Invalid frontmatter values raise `MemoryValidationError` rather than producing partially trusted objects.

Session store:

- Add session ID validation before computing file paths.
- Reject empty IDs and IDs containing `/`, `\\`, or `..`.
- Validate `SessionMessage` before appending.

`delete_memory(identifier, soft_delete=True)` remains backward-compatible: soft delete archives the item by setting status to `archived`. A later CLI layer should expose this as `archive` to avoid semantic confusion.

### 4. CLI reliability

Update `memora/cli.py` so command execution is wrapped by project-level error handling:

- Catch `MemoraError`.
- Print `error: <message>` to stderr.
- Return exit code `1`.
- Keep argparse behavior unchanged for argument parsing errors.

This round does not add `--json`; it only makes current text behavior reliable.

## Tests

Add or update tests in:

- `tests/test_schema.py`
  - invalid memory type
  - invalid status
  - invalid weight/confidence
  - invalid query limits
  - invalid session role

- `tests/test_manager.py`
  - policy rejection raises `MemoryPolicyError`
  - invalid memory type raises `MemoryValidationError`

- `tests/test_stores.py`
  - invalid session ID is rejected
  - invalid datetime frontmatter is rejected

- `tests/test_cli.py`
  - saving secret-like content exits non-zero
  - stderr contains a clear `error:` message

Final verification command:

```bash
pytest -v
```

## Non-goals

This design intentionally excludes:

- JSON CLI output
- `update`, `delete`, `archive`, or `restore` commands
- LLM extraction
- SQLite storage
- vector retrieval
- Agent runtime integration
- automatic repair of corrupted files

## Self-review

- Placeholder scan: no placeholders or TBDs remain.
- Internal consistency: validation, manager errors, store robustness, and CLI reliability all reinforce the existing deterministic MVP boundaries.
- Scope check: this is one focused hardening pass, not a new subsystem.
- Ambiguity check: soft delete behavior remains compatible and is explicitly documented as archival semantics.
