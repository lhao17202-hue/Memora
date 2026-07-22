# Memora Policy/Config Hardening v1 Design

## Purpose

Memora already has a functional local memory core, SQLite/FTS backend, agent memory write pipeline, and deterministic local RAG v1. The next productization gap is configuration consistency: README currently admits that several policy-related configuration fields exist but do not fully affect behavior. This design closes that gap so Memora behaves like a trustworthy library rather than a collection of partly wired features.

This pass focuses on deterministic policy/config behavior only. It does not add new retrieval providers, JSON CLI output, CI, publishing metadata, or real external embedding/vector integrations.

## Goals

1. Make `MemoryConfig.max_memory_content_chars` control the policy noisy-output length limit.
2. Make default weight config fields control automatically assigned weights when callers do not pass an explicit weight.
3. Make auto-save config fields control whether externally extracted user/project memories can be written without confirmation.
4. Make conflict-confirmation config control whether policy conflicts require confirmation.
5. Keep manual writes ergonomic and backward-compatible where possible.
6. Update README so it no longer says these fields are placeholders.
7. Add regression tests proving these settings affect manager/runtime write behavior.

## Non-goals

This pass will not implement:

- CLI `--json` output.
- machine-readable error schemas.
- new LLM or agent examples.
- OpenAI/Cohere/Voyage embeddings.
- Qdrant/Chroma/PGVector/vector-store adapters.
- model rerankers.
- CI/lint/coverage/release automation.
- complex semantic conflict resolution.
- UI or hosted service features.

## Current behavior

Relevant current files:

- `memora/config.py` already defines:
  - `max_memory_content_chars`
  - `default_user_weight`
  - `default_feedback_weight`
  - `default_project_weight`
  - `default_summary_weight`
  - `default_tool_experience_weight`
  - `allow_auto_save_user_preferences`
  - `allow_auto_save_project_facts`
  - `require_confirmation_for_conflicts`
- `memora/policy.py` constructs `MemoryPolicy()` without config and hardcodes noisy output as `len(value) > 4000`.
- `memora/manager.py` constructs `MemoryPolicy()` without passing `MemoryConfig`.
- `save_memory()`, `remember_extracted()`, and candidate paths use a default weight of `5`, so configured default weights do not reliably affect writes.
- conflicts always return `ask_user`.
- auto-save switches are not applied.

## Design

### 1. Config-aware policy

`MemoryPolicy` will accept config:

```python
class MemoryPolicy:
    def __init__(self, config: MemoryConfig | None = None):
        self.config = config or MemoryConfig()
```

`MemoryManager.__init__` will construct it with:

```python
self.policy = MemoryPolicy(self.config)
```

This keeps direct `MemoryPolicy()` tests working while allowing manager/runtime behavior to honor config.

### 2. Content length limit

`MemoryPolicy.is_noisy_output()` will replace the hardcoded `4000` with:

```python
len(value) > self.config.max_memory_content_chars
```

Behavior:

- if content exceeds the configured maximum, policy returns `reject` with reason `noisy_output`.
- if content is under the configured maximum and does not match other noisy patterns, normal policy evaluation continues.

### 3. Default weight assignment

Default weight assignment belongs in `MemoryManager`, before validation and policy evaluation, because validation requires a concrete integer weight.

Add a helper:

```python
def _default_weight_for_type(self, memory_type: str) -> int:
    if memory_type == "user":
        return self.config.default_user_weight
    if memory_type == "feedback":
        return self.config.default_feedback_weight
    if memory_type == "session_summary":
        return self.config.default_summary_weight
    if memory_type == "tool_experience":
        return self.config.default_tool_experience_weight
    return self.config.default_project_weight
```

Add another helper:

```python
def _resolve_candidate_defaults(self, candidate: MemoryCandidate) -> MemoryCandidate:
    if candidate.weight is None:
        candidate.weight = self._default_weight_for_type(candidate.type)
    return candidate
```

To distinguish omitted weight from an explicit `5`, update `MemoryCandidate.weight` from `int = 5` to `int | None = None`, then ensure manager code resolves it before `validate_memory_candidate()`.

Public methods that currently default `weight=5` should become `weight: int | None = None` where they represent newly extracted/written memories:

- `MemoryManager.save_memory()`
- `MemoryRuntime.remember_extracted()` if it has a weight parameter
- `MemoryRuntime.remember_summary()` if it has a weight parameter or creates a session-summary memory internally
- CLI parsing may continue to default `--weight 5` only for explicit CLI ergonomics, but for config-driven behavior it is better to default CLI `--weight` to `None` and let manager apply configured defaults unless user passes `--weight`.

Rules:

- omitted weight → type-specific configured default.
- explicit weight → preserved exactly.
- all resolved weights still pass existing validation `1..10`.

### 4. Auto-save switches

Auto-save switches apply only to externally extracted/automatic candidate writes, not manual saves.

Automatic sources are:

- `conversation`
- `runtime_extraction`
- `session_extraction`

Manual source is:

- `manual`

Policy behavior:

- If candidate type is `user`, source is automatic, and `allow_auto_save_user_preferences=False`, return `ask_user` with reason `auto_save_user_preferences_disabled`.
- If candidate type is `project`, source is automatic, and `allow_auto_save_project_facts=False`, return `ask_user` with reason `auto_save_project_facts_disabled`.
- Do not reject these candidates. They are potentially valid; they simply require user confirmation.
- Manual `save_memory(... source="manual")` remains allowed unless rejected by safety/noise/conflict policy.

This matches Memora's role: external LLMs extract candidate memories, while Memora deterministically decides whether they may be written automatically.

### 5. Conflict confirmation setting

Existing `find_conflict()` is intentionally simple and currently detects examples like Chinese-vs-English language preference conflicts.

Behavior with `require_confirmation_for_conflicts=True`:

- unchanged: conflict returns `ask_user`, reason `conflict_requires_confirmation`, and target memory id.

Behavior with `require_confirmation_for_conflicts=False`:

- do not ask for confirmation solely because of `find_conflict()`.
- continue normal create path if no duplicate was found.
- do not overwrite the conflicting memory automatically.

Rationale: current conflict detection is too heuristic to safely overwrite existing memory. Creating a separate memory is safer than destructive replacement. Later versions can add richer merge/update policies.

### 6. Ordering of policy checks

Recommended evaluation order:

1. slugify candidate name.
2. reject secrets.
3. reject transient task state.
4. reject noisy output, using configured length.
5. apply auto-save confirmation gates.
6. duplicate detection and update.
7. conflict detection and confirmation if enabled.
8. create accepted candidate.

Auto-save gates run before duplicate/update so disabled auto-save cannot silently update existing memories.

### 7. README update

Replace the current configuration caveat with a concrete configuration section:

- content length limit is active.
- default weights are active when weight is omitted.
- auto-save user/project settings are active for runtime/session/conversation extraction sources.
- conflict confirmation is active.
- manual writes remain allowed unless rejected by safety/noise rules.

Keep remaining limitations honest:

- CLI remains primarily a debugging tool.
- RAG v1 remains deterministic local.
- external providers remain reserved but not implemented.

## Testing plan

Add or update tests in:

- `tests/test_policy.py`
- `tests/test_manager.py`
- `tests/test_runtime.py`
- `tests/test_cli.py` if CLI weight default changes need coverage.

Required coverage:

1. `max_memory_content_chars` rejects content above configured length.
2. content below configured length is not rejected as noisy output solely by length.
3. omitted user memory weight uses `default_user_weight`.
4. omitted feedback memory weight uses `default_feedback_weight`.
5. omitted project-like memory weight uses `default_project_weight`.
6. omitted session summary weight uses `default_summary_weight`.
7. omitted tool experience weight uses `default_tool_experience_weight`.
8. explicit weight is preserved and not overwritten by defaults.
9. `allow_auto_save_user_preferences=False` makes automatic user candidate return `requires_confirmation` with reason `auto_save_user_preferences_disabled`.
10. `allow_auto_save_project_facts=False` makes automatic project candidate return `requires_confirmation` with reason `auto_save_project_facts_disabled`.
11. manual `save_memory(... source="manual")` still writes user/project memories when auto-save is disabled.
12. `require_confirmation_for_conflicts=True` keeps current `requires_confirmation` behavior.
13. `require_confirmation_for_conflicts=False` allows conflict candidate to be created rather than requiring confirmation.
14. runtime `remember_extracted()` follows the same manager policy behavior.
15. existing non-RAG and RAG tests still pass.

## Verification

Focused tests during implementation:

```bash
pytest tests/test_policy.py tests/test_manager.py tests/test_runtime.py tests/test_cli.py -q
```

Full regression:

```bash
pytest -q --basetemp "${TEMP:-/tmp}/memora-policy-config-hardening-tests" -p no:cacheprovider
```

## Risks and mitigations

### Risk: breaking existing callers that rely on `MemoryCandidate.weight` always being an int

Mitigation: resolve defaults inside `MemoryManager` before validation and storage. Existing callers that explicitly pass an int continue to work.

### Risk: CLI behavior changes because `--weight` no longer defaults to 5

Mitigation: document that omitted CLI weight uses type-specific configured defaults. Existing tests should assert only meaningful behavior, not incidental default `5` unless user explicitly passes `--weight 5`.

### Risk: auto-save disabled feels like rejection

Mitigation: return `requires_confirmation`, not `rejected`, with clear reasons. This preserves the candidate and allows agent runtimes to ask the user.

### Risk: conflict confirmation disabled creates contradictory memories

Mitigation: this is explicitly the user's chosen configuration. The safer implementation creates a new memory rather than overwriting an existing conflicting memory.

## Implementation order

1. Make `MemoryPolicy` config-aware and wire `MemoryManager` to pass config.
2. Replace hardcoded content length with `config.max_memory_content_chars`.
3. Change candidate/new-memory weight defaults to resolve from config when omitted.
4. Add auto-save gates to policy evaluation.
5. Add conflict-confirmation setting behavior.
6. Update README caveat section.
7. Add focused tests.
8. Run full regression.
