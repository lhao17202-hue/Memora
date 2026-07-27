# Agent Integration Contract

This document defines the boundary between an external agent runtime and Memora.

## Responsibility Split

The external agent owns:

- Conversation orchestration and tool execution.
- Session traces, raw tool logs, and short-term task state.
- LLM-based memory extraction from conversations, traces, or task summaries.
- Deciding when to ask the user about a pending memory write.
- Providing hosted LLM, embedding, or vector database clients if the deployment needs them.

Memora owns:

- Validating structured memory candidates.
- Rejecting secrets, noisy output, and transient task state.
- Applying write policy for create/update/reject/confirmation decisions.
- Detecting embedding-backed write-time relations inside the same type and scope.
- Optionally delegating one embedding hit to an injected LLM relation judge.
- Persisting `MemoryItem` objects in the selected local backend.
- Syncing the RAG vector index from the local backend when RAG is enabled.
- Retrieving and formatting memory context for prompts.

## Candidate Contract

The agent should pass durable facts as `MemoryCandidate` data through `MemoryRuntime.remember_extracted(...)` or `MemoryManager.remember_candidate(...)`.

Required fields:

- `type`: one of `preference`, `project`, `episodic`, `reflective`, `tool`, `knowledge`, or `general`.
- `name`: stable short key for the memory.
- `description`: short human-readable summary.
- `content`: the durable memory body.

Recommended fields:

- `user_id`, `project_id`, and `workspace_id` for scope isolation.
- `tags` for filtering and explainability.
- `confidence` from the extractor.
- `session_id` when the candidate came from a specific session.

The agent should not pass:

- Raw secrets or credentials.
- Full chat transcripts as long-term memories.
- Tool logs that have not been summarized.
- Current task progress that will be obsolete after the turn.
- Speculative guesses with low confidence unless user confirmation is expected.

## Write Flow

1. The agent extracts a candidate from a conversation, task summary, or trace summary.
2. Memora fills defaults, validates the candidate, and applies safety policy.
3. If relation handling is enabled, Memora uses embeddings to find one possible target memory in the same type and scope.
4. If `llm_relation_judge_enabled` and a judge is injected, Memora asks the judge to classify that target as `none`, `duplicate`, `merge`, `conflict`, or `supersede`.
5. Policy chooses the write action.
6. Memora writes the selected local backend.
7. If RAG is enabled, Memora updates the vector index from the saved local item.

Duplicate and merge writes update the target memory. High-confidence conflicts and explicit `supersede` decisions archive the old memory, create a new memory with `supersedes=[old.id]`, delete the old vector entry, and sync the new vector entry.

## Write Result Diagnostics

Every write returns a `MemoryWriteResult`. Agent runtimes should log these fields:

- `action`
- `reason`
- `target_memory_id`
- `relation_kind`
- `relation_confidence`
- `relation_reason`
- `relation_judge_status`
- `relation_judge_error`
- `rag_sync_errors`

`relation_judge_status` values:

- `accepted`: the injected judge returned a valid decision.
- `missing`: LLM relation judging was enabled and a relation target existed, but no judge was injected.
- `invalid`: the injected judge returned invalid relation JSON or an invalid relation decision.
- `failed`: the injected judge raised a non-validation exception.
- `null`: no LLM relation judge was called.

`rag_sync_errors` contains errors from the current write only. A failed RAG sync does not cancel the local write. Use `verify` to inspect accumulated sync errors and `rebuild-index` to repair the vector index.

## Relation Judge Contract

An LLM relation client must implement:

```python
def complete(messages) -> str:
    ...
```

It must return JSON only:

```json
{
  "kind": "merge",
  "confidence": 0.9,
  "reason": "Candidate refines the existing response style.",
  "merged": {
    "name": "response-style",
    "description": "Response style preference.",
    "content": "Prefer concise answers with short summaries.",
    "tags": ["style", "summary"]
  }
}
```

Rules:

- Use `none` when the candidate should become a separate memory.
- Use `duplicate` when the candidate says the same durable fact.
- Use `merge` when the candidate refines or extends the target without contradiction.
- Use `conflict` when the candidate invalidates or contradicts the target.
- Use `supersede` when the candidate explicitly replaces the target and the old memory should be archived.
- `merge` decisions must include merged `description` and `content`.

If the judge fails or returns invalid JSON, Memora falls back to deterministic embedding relation behavior.

## OpenAI Adapter

OpenAI is supported through normal dependency injection. Memora does not import or configure the OpenAI SDK in the core package.

Example:

```python
from openai import OpenAI

from examples.openai_memory_clients import OpenAIExtractionClient, OpenAIRelationClient
from memora.extraction import LLMMemoryExtractor
from memora.relations import LLMMemoryRelationJudge
from memora.runtime import MemoryRuntime


client = OpenAI()
runtime = MemoryRuntime(
    extractor=LLMMemoryExtractor(OpenAIExtractionClient(client, "gpt-5.6")),
    relation_judge=LLMMemoryRelationJudge(OpenAIRelationClient(client, "gpt-5.6")),
)
```

Recommended OpenAI settings:

- Use the Responses API for new examples and integrations.
- Use structured JSON output for extraction and relation decisions.
- Keep the model configurable with `OPENAI_MODEL`.
- Treat `OPENAI_API_KEY` as an environment variable, not a Memora config field.
- Keep conflict auto-replace thresholds conservative until the deployment has enough audit data.

Runnable examples:

- `examples/openai_llm_relation_runtime.py`
- `examples/openai_full_memory_turn_runtime.py`

## CLI Debugging Boundary

The CLI is a local debugging surface. It can:

- Initialize stores.
- Save and search memories.
- Build typed agent context with `context`.
- Simulate an agent-extracted candidate with `remember`.
- Emit a machine-readable `MemoryWriteResult` with `remember --json`.
- Confirm a pending candidate with `confirm`.
- Verify and rebuild local indexes.

The CLI does not:

- Instantiate OpenAI clients.
- Read `OPENAI_API_KEY`.
- Run LLM extraction.
- Inject an LLM relation judge.
- Decide what should be remembered from a raw conversation.

Use the Python runtime API for hosted LLM integrations:

```python
runtime = MemoryRuntime(
    extractor=LLMMemoryExtractor(extraction_client),
    relation_judge=LLMMemoryRelationJudge(relation_client),
)
```

## Recommended Modes

Pure local:

- `memory_backend="file"` or `memory_backend="sqlite"`
- `rag_enabled=False`
- Good for simple local agents and debugging.

Local plus RAG:

- `memory_backend="sqlite"` or `file`
- `rag_enabled=True`
- Good when query-only lexical retrieval misses useful memories.

High-quality writes:

- `llm_relation_judge_enabled=True`
- Inject `LLMMemoryRelationJudge(provider_client)`.
- Keep extraction outside Memora.

Conservative conflict handling:

- `require_confirmation_for_conflicts=True`
- Raise `llm_conflict_auto_replace_threshold` or disable high-confidence replacement.

Aggressive conflict handling:

- `allow_high_confidence_conflict_replace=True`
- Set `llm_conflict_auto_replace_threshold` to the deployment's acceptable risk level.

## Storage Rule

Do not treat RAG as an independent memory database. The selected local memory backend is the source of truth. The vector store is rebuilt or synced from local `MemoryItem` records.
