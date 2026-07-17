# Memora Retrieval Quality v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve deterministic retrieval ranking and result explanations with field-level matching while keeping Memora dependency-free and LLM-free.

**Architecture:** Update `MemoryRetriever.score()` to score `name`, `tags`, `description`, and `content` separately with fixed field weights. Keep the existing final score formula unchanged. Store the strongest matching field in `MemorySearchResult.reason` using the existing `reason` string field.

**Tech Stack:** Python standard library, existing Memora modules, pytest.

## Global Constraints

- Keep retrieval simple.
- Do not add embeddings.
- Do not add vector storage.
- Do not add Chinese segmentation dependencies.
- Do not add LLM calls.
- Focus on field weighting, clearer match reasons, and tests that lock in Chinese short-query behavior.
- Do not add external NLP libraries.
- Do not add `jieba` or other Chinese segmentation.
- Do not add automatic summarization.
- Do not change CLI output.
- Do not change Manager API.
- Do not change Runtime API.
- Do not change database/storage behavior.
- Do not add new dependencies.

---

## File Structure

- Modify `memora/retriever.py`: add field weights and field-level similarity inside deterministic scoring.
- Modify `tests/test_retriever.py`: add focused ranking, reason, and Chinese short-query tests.

---

### Task 1: Field-Weighted Retrieval Scoring

**Files:**
- Modify: `memora/retriever.py`
- Test: `tests/test_retriever.py`

**Interfaces:**
- Consumes:
  - Existing `MemoryRetriever.retrieve(memories: list[MemoryItem], query: MemoryQuery) -> list[MemorySearchResult]`
  - Existing `MemoryRetriever.score(memory: MemoryItem, query: MemoryQuery) -> MemorySearchResult | None`
  - Existing `_tokens(text: str) -> set[str]`
- Produces:
  - `FIELD_WEIGHTS = {"name": 1.00, "tags": 0.95, "description": 0.85, "content": 0.65}`
  - `MemorySearchResult.reason` values: `matched_name`, `matched_tags`, `matched_description`, `matched_content`
  - unchanged final score formula

- [ ] **Step 1: Add failing retriever tests**

Append these tests to `tests/test_retriever.py`:

```python

def test_name_match_ranks_above_content_only_match():
    name_match = item("pytest", "unrelated content")
    content_match = item("other", "pytest")

    results = MemoryRetriever().retrieve([content_match, name_match], MemoryQuery(query="pytest"))

    assert results[0].memory.name == "pytest"
    assert results[0].reason == "matched_name"
    assert results[0].similarity_score > results[1].similarity_score


def test_tag_match_ranks_above_content_only_match():
    tag_match = item("tagged", "unrelated content")
    tag_match.tags = ["pytest"]
    content_match = item("other", "pytest")

    results = MemoryRetriever().retrieve([content_match, tag_match], MemoryQuery(query="pytest"))

    assert results[0].memory.name == "tagged"
    assert results[0].reason == "matched_tags"
    assert results[0].similarity_score > results[1].similarity_score


def test_reason_identifies_strongest_matching_field():
    description_match = item("description", "pytest")
    description_match.content = "unrelated content"
    content_match = item("content", "pytest")
    content_match.description = "unrelated description"

    results = MemoryRetriever().retrieve([content_match, description_match], MemoryQuery(query="pytest"))

    assert results[0].memory.name == "description"
    assert results[0].reason == "matched_description"
    assert results[1].reason == "matched_content"


def test_chinese_short_query_matches_longer_chinese_memory():
    memory = item("language", "用户偏好中文回答。")
    retriever = MemoryRetriever()

    for query in ["中文", "偏好", "回答"]:
        results = retriever.retrieve([memory], MemoryQuery(query=query))
        assert len(results) == 1
        assert results[0].memory.name == "language"
```

- [ ] **Step 2: Run tests to verify field-weighting tests fail**

Run:

```bash
pytest tests/test_retriever.py -v
```

Expected: FAIL because current reason is `keyword_match` and current similarity does not distinguish field weights.

- [ ] **Step 3: Add field weights and field-level scoring**

Modify `memora/retriever.py`.

Add this constant after `HALF_LIFE_DAYS`:

```python
FIELD_WEIGHTS = {
    "name": 1.00,
    "tags": 0.95,
    "description": 0.85,
    "content": 0.65,
}
```

Replace this block in `MemoryRetriever.score()`:

```python
        haystack = " ".join([memory.name, memory.description, " ".join(memory.tags), memory.content])
        query_tokens = _tokens(query.query)
        memory_tokens = _tokens(haystack)
        if not query_tokens:
            similarity_score = 0.0
        else:
            similarity_score = len(query_tokens & memory_tokens) / len(query_tokens)
        if similarity_score <= 0:
            return None
```

with this exact block:

```python
        query_tokens = _tokens(query.query)
        field_texts = {
            "name": memory.name,
            "tags": " ".join(memory.tags),
            "description": memory.description,
            "content": memory.content,
        }
        similarity_score = 0.0
        reason = ""
        if query_tokens:
            for field_name, field_text in field_texts.items():
                field_tokens = _tokens(field_text)
                coverage = len(query_tokens & field_tokens) / len(query_tokens)
                weighted_score = coverage * FIELD_WEIGHTS[field_name]
                if weighted_score > similarity_score:
                    similarity_score = min(weighted_score, 1.0)
                    reason = f"matched_{field_name}"
        if similarity_score <= 0:
            return None
```

Then replace:

```python
            reason="keyword_match",
```

with:

```python
            reason=reason,
```

- [ ] **Step 4: Run retriever tests**

Run:

```bash
pytest tests/test_retriever.py -v
```

Expected: PASS with all retriever tests passing.

- [ ] **Step 5: Commit retrieval quality changes**

Run:

```bash
git add memora/retriever.py tests/test_retriever.py
git commit -m "feat: improve deterministic retrieval scoring" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Final Regression Verification

**Files:**
- No file changes expected.

**Interfaces:**
- Consumes: all existing tests.
- Produces: verified full-suite pass.

- [ ] **Step 1: Run full test suite**

Run:

```bash
pytest -v
```

Expected: PASS with all tests.

- [ ] **Step 2: Check git status**

Run:

```bash
git status --short
```

Expected: no modified tracked files from this task. Existing unrelated untracked files may remain.

---

## Self-Review

Spec coverage:
- Field weights: Task 1.
- Match reasons: Task 1.
- Chinese short-query tests: Task 1.
- No schema/API/CLI/runtime changes: Global Constraints and file structure.
- Full suite verification: Task 2.

Placeholder scan:
- No TBD, TODO, or incomplete implementation steps.

Type consistency:
- `MemorySearchResult.reason` already exists as a string field.
- `MemoryRetriever.score()` keeps the same public signature.
- `MemoryRetriever.retrieve()` keeps the same public signature.
