# Memora Policy/Config Hardening v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Memora's policy-related configuration fields actively control write behavior so README no longer documents partially wired policy knobs.

**Architecture:** Keep policy decisions deterministic and local. `MemoryPolicy` becomes config-aware for safety/auto-save/conflict decisions, while `MemoryManager` resolves omitted write defaults before validation/storage. Runtime and CLI continue delegating to manager so behavior stays centralized.

**Tech Stack:** Python dataclasses, existing Memora manager/policy/runtime/CLI modules, pytest regression tests, no new dependencies.

## Global Constraints

- Do not implement CLI `--json` output.
- Do not implement machine-readable error schemas.
- Do not add new LLM or agent examples.
- Do not implement OpenAI/Cohere/Voyage embeddings.
- Do not implement Qdrant/Chroma/PGVector/vector-store adapters.
- Do not implement model rerankers.
- Do not add CI/lint/coverage/release automation.
- Do not implement complex semantic conflict resolution.
- Do not add UI or hosted service features.
- Auto-save disabled must return `requires_confirmation`, not `rejected`.
- Manual writes must remain allowed unless rejected by existing safety/noise/conflict policy.
- Omitted weight must use type-specific `MemoryConfig` defaults; explicit weight must be preserved.
- Existing non-RAG and RAG behavior must keep passing regression tests.

---

## File Structure

- `memora/policy.py`
  - Owns deterministic safety, noisy-output, auto-save, duplicate, and conflict policy decisions.
  - Will accept `MemoryConfig` and use config values instead of hardcoded policy settings.

- `memora/schema.py`
  - Owns dataclasses and validation.
  - Will allow `MemoryCandidate.weight` to be omitted (`None`) before manager resolves defaults.

- `memora/manager.py`
  - Owns public memory write orchestration.
  - Will inject config into policy, resolve omitted candidate/default weights before validation, and preserve existing write/RAG sync paths.

- `memora/runtime.py`
  - Thin top-level agent API.
  - Will allow `remember_extracted()` to omit weight so config defaults apply.

- `memora/cli.py`
  - Debug CLI.
  - Will let omitted `remember --weight` flow through as `None`, so config defaults apply unless user explicitly passes `--weight`.

- `README.md`
  - Will replace the current configuration caveat with an accurate configuration behavior section.

- `tests/test_policy.py`
  - Policy-level tests for config-aware content length, auto-save gates, and conflict confirmation.

- `tests/test_manager.py`
  - Manager-level tests for resolved default weights, explicit weight preservation, manual write behavior, and conflict creation behavior.

- `tests/test_runtime.py`
  - Runtime-level test proving `remember_extracted()` follows auto-save confirmation behavior.

- `tests/test_cli.py`
  - CLI-level test proving omitted `remember --weight` uses configured/manager default behavior indirectly without breaking existing CLI flows.

---

### Task 1: Make MemoryPolicy config-aware

**Files:**
- Modify: `memora/policy.py`
- Modify: `memora/manager.py`
- Test: `tests/test_policy.py`

**Interfaces:**
- Consumes: `MemoryConfig` from `memora.config`.
- Produces: `MemoryPolicy(config: MemoryConfig | None = None)` and `MemoryManager.policy = MemoryPolicy(self.config)`.

- [ ] **Step 1: Add failing policy tests for configured noisy-output length**

Append these tests to `tests/test_policy.py`:

```python
from memora.config import MemoryConfig


def test_noisy_output_uses_configured_content_length_limit():
    policy = MemoryPolicy(MemoryConfig(max_memory_content_chars=10))

    result = policy.evaluate(candidate("x" * 11), [])

    assert result.action == "reject"
    assert result.reason == "noisy_output"


def test_content_under_configured_length_is_not_noisy_by_length():
    policy = MemoryPolicy(MemoryConfig(max_memory_content_chars=20))

    result = policy.evaluate(candidate("durable"), [])

    assert result.action == "create"
    assert result.reason == "accepted"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_policy.py::test_noisy_output_uses_configured_content_length_limit tests/test_policy.py::test_content_under_configured_length_is_not_noisy_by_length -v
```

Expected: FAIL because `MemoryPolicy` does not accept a config argument yet.

- [ ] **Step 3: Update `memora/policy.py` imports and constructor**

In `memora/policy.py`, add `MemoryConfig` import and constructor:

```python
from .config import MemoryConfig
from .schema import MemoryCandidate, MemoryItem
```

Replace:

```python
class MemoryPolicy:
    def contains_secret(self, text: str) -> bool:
```

with:

```python
class MemoryPolicy:
    def __init__(self, config: MemoryConfig | None = None):
        self.config = config or MemoryConfig()

    def contains_secret(self, text: str) -> bool:
```

- [ ] **Step 4: Replace hardcoded noisy-output length**

In `MemoryPolicy.is_noisy_output()`, replace:

```python
return any(pattern.search(value) for pattern in NOISE_PATTERNS) or len(value) > 4000
```

with:

```python
return any(pattern.search(value) for pattern in NOISE_PATTERNS) or len(value) > self.config.max_memory_content_chars
```

- [ ] **Step 5: Wire manager to pass config**

In `memora/manager.py`, replace:

```python
self.policy = MemoryPolicy()
```

with:

```python
self.policy = MemoryPolicy(self.config)
```

- [ ] **Step 6: Run policy tests**

Run:

```bash
pytest tests/test_policy.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add memora/policy.py memora/manager.py tests/test_policy.py
git commit -m "feat: make memory policy config-aware" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Resolve omitted write weights from MemoryConfig

**Files:**
- Modify: `memora/schema.py`
- Modify: `memora/manager.py`
- Modify: `memora/runtime.py`
- Test: `tests/test_manager.py`
- Test: `tests/test_runtime.py`

**Interfaces:**
- Consumes: `MemoryConfig.default_*_weight` fields.
- Produces:
  - `MemoryCandidate.weight: int | None = None`
  - `MemoryManager._default_weight_for_type(memory_type: str) -> int`
  - `MemoryManager._resolve_candidate_defaults(candidate: MemoryCandidate) -> MemoryCandidate`
  - `MemoryManager.save_memory(..., weight: int | None = None)`
  - `MemoryRuntime.remember_extracted(..., weight: int | None = None)`

- [ ] **Step 1: Add failing manager tests for default weights**

Append these tests to `tests/test_manager.py`:

```python
def test_save_memory_uses_config_default_weight_when_omitted(tmp_path: Path):
    manager = MemoryManager(
        MemoryConfig(
            root_dir=tmp_path / ".memora",
            default_user_weight=10,
            default_feedback_weight=8,
            default_project_weight=6,
            default_summary_weight=4,
            default_tool_experience_weight=3,
        )
    )

    user = manager.save_memory("user", "prefers Chinese", "language", name="user-language")
    feedback = manager.save_memory("feedback", "likes concise answers", "style", name="feedback-style")
    project = manager.save_memory("project", "uses pytest", "tests", name="project-tests")
    summary = manager.save_memory("session_summary", "session summary", "summary", name="session-summary")
    tool = manager.save_memory("tool_experience", "pytest worked", "tool", name="tool-pytest")

    assert user.weight == 10
    assert feedback.weight == 8
    assert project.weight == 6
    assert summary.weight == 4
    assert tool.weight == 3


def test_explicit_weight_is_preserved_over_config_default(tmp_path: Path):
    manager = MemoryManager(MemoryConfig(root_dir=tmp_path / ".memora", default_user_weight=10))

    item = manager.save_memory("user", "prefers Chinese", "language", name="language", weight=5)

    assert item.weight == 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_manager.py::test_save_memory_uses_config_default_weight_when_omitted tests/test_manager.py::test_explicit_weight_is_preserved_over_config_default -v
```

Expected: first test FAILS because omitted `save_memory()` weight still becomes `5`.

- [ ] **Step 3: Allow omitted candidate weight in schema**

In `memora/schema.py`, replace:

```python
weight: int = 5
```

inside `MemoryCandidate` with:

```python
weight: int | None = None
```

Replace `validate_memory_candidate()` body tail:

```python
_validate_weight(candidate.weight)
_validate_confidence(candidate.confidence)
```

with:

```python
if candidate.weight is not None:
    _validate_weight(candidate.weight)
_validate_confidence(candidate.confidence)
```

- [ ] **Step 4: Add default weight helpers in manager**

In `memora/manager.py`, after `_write_result_from_decision()`, add:

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

    def _resolve_candidate_defaults(self, candidate: MemoryCandidate) -> MemoryCandidate:
        if candidate.weight is None:
            candidate.weight = self._default_weight_for_type(candidate.type)
        return candidate
```

- [ ] **Step 5: Resolve defaults before validation in manager candidate paths**

In `evaluate_memory_candidate()`, replace:

```python
validate_memory_candidate(candidate)
```

with:

```python
candidate = self._resolve_candidate_defaults(candidate)
validate_memory_candidate(candidate)
```

In `remember_candidate()`, replace:

```python
validate_memory_candidate(candidate)
```

with:

```python
candidate = self._resolve_candidate_defaults(candidate)
validate_memory_candidate(candidate)
```

In `save_memory()`, change signature:

```python
weight: int = 5,
```

to:

```python
weight: int | None = None,
```

Then after constructing `candidate = MemoryCandidate(...)`, replace:

```python
validate_memory_candidate(candidate)
```

with:

```python
candidate = self._resolve_candidate_defaults(candidate)
validate_memory_candidate(candidate)
```

- [ ] **Step 6: Assert resolved weight before creating MemoryItem**

At the top of `_new_memory_from_candidate()`, add:

```python
        if decision.weight is None:
            decision.weight = self._default_weight_for_type(decision.type)
```

This is a defensive guard so `MemoryItem.weight` always receives an int.

- [ ] **Step 7: Update runtime weight default**

In `memora/runtime.py`, change `remember_extracted()` signature:

```python
weight: int = 5,
```

to:

```python
weight: int | None = None,
```

Leave the `MemoryCandidate(... weight=weight ...)` call unchanged.

- [ ] **Step 8: Add runtime summary default weight test**

Append this test to `tests/test_runtime.py`:

```python
def test_remember_summary_uses_config_default_summary_weight(tmp_path: Path):
    runtime = MemoryRuntime(config=MemoryConfig(root_dir=tmp_path / ".memora", default_summary_weight=7))
    runtime.init_storage()

    item = runtime.remember_summary("session_1", "summary text")

    assert item.weight == 7
```

- [ ] **Step 9: Run focused tests**

Run:

```bash
pytest tests/test_manager.py::test_save_memory_uses_config_default_weight_when_omitted tests/test_manager.py::test_explicit_weight_is_preserved_over_config_default tests/test_runtime.py::test_remember_summary_uses_config_default_summary_weight -q
```

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add memora/schema.py memora/manager.py memora/runtime.py tests/test_manager.py tests/test_runtime.py
git commit -m "feat: apply configured default memory weights" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Add auto-save confirmation gates

**Files:**
- Modify: `memora/policy.py`
- Test: `tests/test_policy.py`
- Test: `tests/test_manager.py`
- Test: `tests/test_runtime.py`

**Interfaces:**
- Consumes: `MemoryConfig.allow_auto_save_user_preferences` and `MemoryConfig.allow_auto_save_project_facts`.
- Produces: automatic user/project candidate writes return `ask_user` / `requires_confirmation` when disabled.

- [ ] **Step 1: Add failing policy tests for auto-save gates**

Append to `tests/test_policy.py`:

```python
from memora.config import MemoryConfig


def test_auto_save_user_preferences_disabled_requires_confirmation():
    policy = MemoryPolicy(MemoryConfig(allow_auto_save_user_preferences=False))
    item = candidate("用户偏好中文回答。", name="language")
    item.source = "runtime_extraction"
    item.type = "user"

    result = policy.evaluate(item, [])

    assert result.action == "ask_user"
    assert result.reason == "auto_save_user_preferences_disabled"


def test_auto_save_project_facts_disabled_requires_confirmation():
    policy = MemoryPolicy(MemoryConfig(allow_auto_save_project_facts=False))
    item = candidate("Project uses pytest.", name="test-framework")
    item.source = "session_extraction"
    item.type = "project"

    result = policy.evaluate(item, [])

    assert result.action == "ask_user"
    assert result.reason == "auto_save_project_facts_disabled"
```

If `MemoryConfig` is already imported from Task 1, do not duplicate the import.

- [ ] **Step 2: Add manager/runtime behavior tests**

Append to `tests/test_manager.py`:

```python
def test_disabled_auto_save_user_returns_confirmation_without_writing(tmp_path: Path):
    manager = MemoryManager(MemoryConfig(root_dir=tmp_path / ".memora", allow_auto_save_user_preferences=False))
    manager.init_storage()
    candidate = MemoryCandidate(
        action="create",
        type="user",
        name="language",
        description="用户偏好中文。",
        content="用户偏好中文回答。",
        source="runtime_extraction",
    )

    result = manager.remember_candidate(candidate)

    assert result.action == "requires_confirmation"
    assert result.reason == "auto_save_user_preferences_disabled"
    assert result.memory is None
    assert manager.memory_store.list_memories() == []


def test_manual_save_ignores_auto_save_disabled(tmp_path: Path):
    manager = MemoryManager(
        MemoryConfig(
            root_dir=tmp_path / ".memora",
            allow_auto_save_user_preferences=False,
            allow_auto_save_project_facts=False,
        )
    )

    item = manager.save_memory("user", "用户偏好中文回答。", "用户偏好中文。", name="language", source="manual")

    assert item.name == "language"
    assert item.source == "manual"
```

Append to `tests/test_runtime.py`:

```python
def test_runtime_remember_extracted_respects_disabled_auto_save(tmp_path: Path):
    runtime = MemoryRuntime(config=MemoryConfig(root_dir=tmp_path / ".memora", allow_auto_save_user_preferences=False))
    runtime.init_storage()

    result = runtime.remember_extracted(
        memory_type="user",
        name="language",
        description="用户偏好中文。",
        content="用户偏好中文回答。",
    )

    assert result.action == "requires_confirmation"
    assert result.reason == "auto_save_user_preferences_disabled"
    assert result.memory is None
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
pytest tests/test_policy.py::test_auto_save_user_preferences_disabled_requires_confirmation tests/test_policy.py::test_auto_save_project_facts_disabled_requires_confirmation tests/test_manager.py::test_disabled_auto_save_user_returns_confirmation_without_writing tests/test_manager.py::test_manual_save_ignores_auto_save_disabled tests/test_runtime.py::test_runtime_remember_extracted_respects_disabled_auto_save -v
```

Expected: auto-save disabled tests FAIL because policy does not gate them yet.

- [ ] **Step 4: Implement automatic source helper in policy**

In `memora/policy.py`, add this constant near the other constants:

```python
AUTO_SAVE_SOURCES = {"conversation", "runtime_extraction", "session_extraction"}
```

Inside `MemoryPolicy`, add:

```python
    def requires_auto_save_confirmation(self, candidate: MemoryCandidate) -> str | None:
        if candidate.source not in AUTO_SAVE_SOURCES:
            return None
        if candidate.type == "user" and not self.config.allow_auto_save_user_preferences:
            return "auto_save_user_preferences_disabled"
        if candidate.type == "project" and not self.config.allow_auto_save_project_facts:
            return "auto_save_project_facts_disabled"
        return None
```

- [ ] **Step 5: Apply auto-save gate before duplicate detection**

In `MemoryPolicy.evaluate()`, after noisy-output rejection and before `duplicate = self.find_duplicate(...)`, add:

```python
        auto_save_reason = self.requires_auto_save_confirmation(candidate)
        if auto_save_reason:
            candidate.action = "ask_user"
            candidate.reason = auto_save_reason
            return candidate
```

- [ ] **Step 6: Run focused tests**

Run:

```bash
pytest tests/test_policy.py tests/test_manager.py tests/test_runtime.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add memora/policy.py tests/test_policy.py tests/test_manager.py tests/test_runtime.py
git commit -m "feat: gate automatic memory saves by config" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Honor conflict-confirmation configuration

**Files:**
- Modify: `memora/policy.py`
- Test: `tests/test_policy.py`
- Test: `tests/test_manager.py`

**Interfaces:**
- Consumes: `MemoryConfig.require_confirmation_for_conflicts`.
- Produces: conflicts require `ask_user` only when config is true; when false, non-duplicate conflicts proceed to create.

- [ ] **Step 1: Add failing policy test for disabled conflict confirmation**

Append to `tests/test_policy.py`:

```python
def test_conflict_confirmation_can_be_disabled():
    policy = MemoryPolicy(MemoryConfig(require_confirmation_for_conflicts=False))
    existing = [
        MemoryItem(
            id="mem_1",
            name="user-language-en",
            description="User prefers English.",
            type="user",
            content="用户偏好英文回答。",
        )
    ]

    result = policy.evaluate(candidate("用户偏好中文回答。", name="user-language-zh"), existing)

    assert result.action == "create"
    assert result.reason == "accepted"
    assert result.target_memory_id is None
```

- [ ] **Step 2: Add manager test for conflict create when confirmation disabled**

Append to `tests/test_manager.py`:

```python
def test_conflict_confirmation_disabled_creates_new_memory(tmp_path: Path):
    manager = MemoryManager(MemoryConfig(root_dir=tmp_path / ".memora", require_confirmation_for_conflicts=False))
    manager.init_storage()
    existing = manager.save_memory("user", "用户偏好英文回答。", "用户偏好英文。", name="language-en")
    candidate = MemoryCandidate(
        action="create",
        type="user",
        name="language-zh",
        description="用户偏好中文。",
        content="用户偏好中文回答。",
        source="runtime_extraction",
    )

    result = manager.remember_candidate(candidate)

    assert result.action == "created"
    assert result.memory is not None
    assert result.memory.id != existing.id
    assert result.reason == "accepted"
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
pytest tests/test_policy.py::test_conflict_confirmation_can_be_disabled tests/test_manager.py::test_conflict_confirmation_disabled_creates_new_memory -v
```

Expected: FAIL because conflicts always ask for confirmation.

- [ ] **Step 4: Update conflict branch in policy**

In `MemoryPolicy.evaluate()`, replace:

```python
        conflict = self.find_conflict(candidate, existing)
        if conflict:
            candidate.action = "ask_user"
            candidate.target_memory_id = conflict.id
            candidate.reason = "conflict_requires_confirmation"
            return candidate
```

with:

```python
        conflict = self.find_conflict(candidate, existing)
        if conflict and self.config.require_confirmation_for_conflicts:
            candidate.action = "ask_user"
            candidate.target_memory_id = conflict.id
            candidate.reason = "conflict_requires_confirmation"
            return candidate
```

Do not set `target_memory_id` when confirmation is disabled.

- [ ] **Step 5: Run focused tests**

Run:

```bash
pytest tests/test_policy.py tests/test_manager.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add memora/policy.py tests/test_policy.py tests/test_manager.py
git commit -m "feat: honor conflict confirmation config" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: Let CLI omitted remember weight use manager defaults

**Files:**
- Modify: `memora/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `MemoryCandidate.weight: int | None` from Task 2.
- Produces: `memora remember` with no `--weight` passes `None`; explicit `--weight` still passes an int.

- [ ] **Step 1: Add CLI regression test for omitted remember weight**

Append to `tests/test_cli.py`:

```python
def test_remember_command_omitted_weight_uses_type_default(tmp_path: Path):
    root = tmp_path / ".memora"

    result = run_cli(
        root,
        "remember",
        "--type",
        "user",
        "--name",
        "language",
        "--description",
        "用户偏好中文。",
        "--content",
        "用户偏好中文回答。",
    )
    shown = run_cli(root, "show", "language")

    assert result.returncode == 0
    assert "created" in result.stdout
    assert shown.returncode == 0
```

This test primarily protects CLI compatibility while allowing manager defaults to apply. The current `show` command does not print weight, so exact default weight is covered at manager/runtime level.

- [ ] **Step 2: Run CLI test before change**

Run:

```bash
pytest tests/test_cli.py::test_remember_command_omitted_weight_uses_type_default -v
```

Expected: PASS or FAIL is acceptable before implementation because this is a compatibility guard, not a precise weight inspection.

- [ ] **Step 3: Change CLI remember weight default**

In `memora/cli.py`, replace:

```python
remember_parser.add_argument("--weight", type=int, default=5)
```

with:

```python
remember_parser.add_argument("--weight", type=int)
```

This makes omitted `--weight` become `None`; explicit `--weight 5` still passes `5`.

- [ ] **Step 4: Run CLI tests**

Run:

```bash
pytest tests/test_cli.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add memora/cli.py tests/test_cli.py
git commit -m "feat: let CLI remember use configured weight defaults" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: Update README configuration caveat

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: behavior implemented in Tasks 1-5.
- Produces: accurate README section describing active policy/config fields.

- [ ] **Step 1: Replace README caveat section**

In `README.md`, replace:

```markdown
## Configuration caveats

Some configuration fields are placeholders for the next policy pass. Retrieval limits and lifecycle cleanup settings are active, but policy-related fields such as `max_memory_content_chars`, default memory weights, auto-save switches, and conflict-confirmation behavior are not fully wired through every write path yet.
```

with:

```markdown
## Configuration behavior

Memora's policy-related configuration fields are active in the manager/runtime write paths:

- `max_memory_content_chars` controls the noisy-output content length limit.
- omitted write weights use type-specific defaults such as `default_user_weight`, `default_feedback_weight`, `default_project_weight`, `default_summary_weight`, and `default_tool_experience_weight`.
- explicit write weights are preserved.
- `allow_auto_save_user_preferences` and `allow_auto_save_project_facts` control whether automatic `runtime_extraction`, `session_extraction`, and `conversation` candidates can be written without confirmation.
- disabled auto-save returns `requires_confirmation`, not `rejected`.
- `require_confirmation_for_conflicts` controls whether simple deterministic conflicts require confirmation.

Manual writes remain allowed unless rejected by safety, transient-state, noisy-output, or conflict policy.
```

- [ ] **Step 2: Review README for stale caveat wording**

Search the README text manually for the phrase `not fully wired` and remove it if still present.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document active policy configuration" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: Full regression verification

**Files:**
- Test-only task; no intended source modifications.

**Interfaces:**
- Consumes: all prior task changes.
- Produces: verified policy/config hardening branch.

- [ ] **Step 1: Run focused tests**

Run:

```bash
pytest tests/test_policy.py tests/test_manager.py tests/test_runtime.py tests/test_cli.py -q
```

Expected: PASS.

- [ ] **Step 2: Run RAG regression tests**

Run:

```bash
pytest tests/test_rag.py tests/test_vector_store.py tests/test_embeddings.py -q
```

Expected: PASS.

- [ ] **Step 3: Run full suite**

Run:

```bash
pytest -q --basetemp "${TEMP:-/tmp}/memora-policy-config-hardening-tests" -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 4: Check git status**

Run:

```bash
git status --short --branch
```

Expected: only intentional files changed/committed. If `.claude/` remains untracked, leave it uncommitted.

- [ ] **Step 5: Commit any missed test/doc fixes**

Only if Step 1-4 required fixes, commit them:

```bash
git add <changed-files>
git commit -m "test: verify policy config hardening" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

If no fixes were required, do not create an empty commit.

---

## Self-Review Notes

- Spec coverage: all design goals are covered by Tasks 1-6, with full verification in Task 7.
- No future provider, CLI JSON, agent example, CI, release, or complex conflict work is included.
- Type consistency: `MemoryCandidate.weight` becomes `int | None`, manager resolves it before validation/storage, and `MemoryItem.weight` remains an `int`.
- Auto-save disabled uses `ask_user` internally and `requires_confirmation` in `MemoryWriteResult`, as approved.
- Conflict confirmation disabled creates a new memory instead of overwriting the conflicting one.
