# Memora Retrieval Quality v2 Design

## Goal

Improve deterministic retrieval quality without changing Memora's architecture. Retrieval should rank field-specific matches more usefully and explain why each result matched, while staying simple, dependency-free, and LLM-free.

## User Direction

The user approved a minimal deterministic retrieval improvement:

- Keep retrieval simple.
- Do not add embeddings.
- Do not add vector storage.
- Do not add Chinese segmentation dependencies.
- Do not add LLM calls.
- Focus on field weighting, clearer match reasons, and tests that lock in Chinese short-query behavior.

## Current Behavior

`MemoryRetriever.score()` currently builds one combined haystack from:

- `memory.name`
- `memory.description`
- `memory.tags`
- `memory.content`

It then calculates similarity as:

```python
len(query_tokens & memory_tokens) / len(query_tokens)
```

The final score already combines similarity, importance, recency, and access count:

```python
final_score = (
    similarity_score * 0.45
    + importance_score * 0.25
    + recency_score * 0.20
    + access_score * 0.10
)
```

This final-score structure should remain unchanged.

## Proposed Design

### Field Weights

Replace the single combined-haystack similarity calculation with field-level similarity. Use these fixed weights:

```python
FIELD_WEIGHTS = {
    "name": 1.00,
    "tags": 0.95,
    "description": 0.85,
    "content": 0.65,
}
```

For each field:

1. Tokenize the field text with the existing `_tokens()` function.
2. Compute coverage: `len(query_tokens & field_tokens) / len(query_tokens)`.
3. Multiply coverage by the field weight.

The memory's `similarity_score` is the highest weighted field score, clamped to `1.0`.

This keeps the score in the existing 0-1 range while making matches in `name` and `tags` rank above content-only matches.

### Match Reason

Use the strongest matching field to set `MemorySearchResult.reason`:

- `matched_name`
- `matched_tags`
- `matched_description`
- `matched_content`

If no field has a positive score, return `None` as before.

If multiple fields tie, prefer the first field in this order:

1. `name`
2. `tags`
3. `description`
4. `content`

This tie-break order favors intentional metadata over body text.

### Chinese Short Query Behavior

Keep the existing tokenizer shape:

- English / number / underscore words
- Chinese chunks with length >= 2
- individual Chinese characters

Do not add a segmentation library. Instead, add tests that lock in the intended behavior:

- query `中文` matches memory text `用户偏好中文回答。`
- query `偏好` matches memory text `用户偏好中文回答。`
- query `回答` matches memory text `用户偏好中文回答。`

If current tokenization already supports this, implementation only needs field-level scoring. If a test exposes a gap, fix `_tokens()` with a tiny deterministic change only.

## Implementation Scope

Modify only:

- `memora/retriever.py`
- `tests/test_retriever.py`

No schema changes are required. `MemorySearchResult.reason` already exists as `str`.

## Non-Goals

This round does not add:

- embeddings
- vector databases
- external NLP libraries
- `jieba` or other Chinese segmentation
- LLM calls
- automatic summarization
- CLI output changes
- Manager API changes
- Runtime API changes
- database/storage changes
- new dependencies

## Tests

Add or update retriever tests for:

1. `test_name_match_ranks_above_content_only_match`
   - Query matches one memory by `name` and another by `content` only.
   - Name match ranks first.
   - Reason is `matched_name`.

2. `test_tag_match_ranks_above_content_only_match`
   - Query matches one memory by tag and another by content only.
   - Tag match ranks first.
   - Reason is `matched_tags`.

3. `test_reason_identifies_strongest_matching_field`
   - Query matches description/content in different memories.
   - Result reason identifies the strongest matching field.

4. `test_chinese_short_query_matches_longer_chinese_memory`
   - Queries `中文`, `偏好`, and `回答` all return the memory containing `用户偏好中文回答。`.

5. Existing tests must continue to pass:
   - archived filtering
   - include archived
   - type filtering
   - recency ranking
   - existing keyword ranking

## Success Criteria

- Field-level matches affect ranking deterministically.
- Result reasons identify the strongest matching field.
- Chinese short-query behavior is covered by tests.
- No new dependencies are added.
- Full test suite passes with `pytest -v`.
