# Vector Store Provider Boundary Design

Date: 2026-07-31

## Purpose

Memora should expose vector storage to upper layers through the existing `VectorStore` protocol, not through provider-specific fields on `MemoryConfig`. The current Qdrant integration adds `qdrant_url`, `qdrant_host`, `qdrant_port`, `qdrant_api_key`, `qdrant_collection`, `qdrant_timeout`, `qdrant_prefer_grpc`, and `qdrant_recreate_collection` directly to `MemoryConfig`. That makes Memora's core configuration know Qdrant-specific connection details and would not scale to future providers.

The goal is to keep `MemoryConfig` focused on Memora-wide and retrieval-wide settings while moving concrete vector-store configuration behind provider-owned configuration and factories.

## Design Summary

Use dependency injection as the primary extension point, with a small provider factory/options path for CLI and `.env` support.

Upper layers should be able to provide a concrete vector store directly:

```python
manager = MemoryManager(config=config, vector_store=my_vector_store)
```

When no vector store is injected, Memora builds an internal provider from generic config fields:

```python
MemoryConfig(
    vector_store="sqlite" | "qdrant",
    vector_store_options={...},
)
```

Provider-specific options are contained in `vector_store_options` or provider-owned dataclasses, not as top-level `MemoryConfig` fields.

## Configuration Boundary

`MemoryConfig` keeps generic RAG fields:

- `vector_store`
- `vector_path`
- `retrieval_mode`
- `hybrid_prefetch_limit`
- `vector_candidate_limit`
- `keyword_candidate_limit`
- `min_semantic_score`

`MemoryConfig` removes Qdrant-specific fields:

- `qdrant_url`
- `qdrant_host`
- `qdrant_port`
- `qdrant_api_key`
- `qdrant_collection`
- `qdrant_timeout`
- `qdrant_prefer_grpc`
- `qdrant_recreate_collection`

`MemoryConfig` adds:

```python
vector_store_options: dict[str, object] = field(default_factory=dict)
```

This field is passed to vector-store provider factories. Core retrieval logic must not read provider-specific keys directly.

## Qdrant Provider Configuration

Qdrant owns its concrete configuration. Add a provider-specific dataclass near the Qdrant implementation:

```python
@dataclass
class QdrantVectorStoreConfig:
    url: str | None = None
    host: str = "localhost"
    port: int = 6333
    api_key: str | None = None
    collection: str = "memora_memories"
    timeout: float = 5.0
    prefer_grpc: bool = False
    recreate_collection: bool = False
    dimension: int = 384
    retrieval_mode: str = "dense"
    hybrid_prefetch_limit: int = 100
```

`QdrantVectorStore` should accept this provider config instead of full `MemoryConfig`.

SQLite can either keep accepting `MemoryConfig` for now or get a small `SQLiteVectorStoreConfig`; the minimum required design change is to remove Qdrant-specific fields from core config.

## Factory and Injection Flow

`MemoryManager` gains an optional `vector_store` constructor argument:

```python
class MemoryManager:
    def __init__(..., vector_store: VectorStore | None = None):
        ...
```

If RAG is enabled:

1. Build or receive the embedding provider.
2. Use the injected vector store when provided.
3. Otherwise call `build_vector_store(config)`.
4. Pass the resulting `VectorStore` to `RagIndex` and `RagRetriever`.

`build_vector_store(config)` remains a convenience path for built-in providers. For Qdrant, it converts `config.vector_store_options` plus generic retrieval fields into `QdrantVectorStoreConfig`.

## `.env` and CLI Support

`.env` support remains, but provider-specific settings map into `vector_store_options`, not top-level config fields.

Recommended env shape:

```env
MEMORA_VECTOR_STORE=qdrant
MEMORA_VECTOR_STORE_URL=http://localhost:6333
MEMORA_VECTOR_STORE_COLLECTION=memora_memories
MEMORA_VECTOR_STORE_TIMEOUT=5
MEMORA_VECTOR_STORE_PREFER_GRPC=false
MEMORA_VECTOR_STORE_RECREATE_COLLECTION=false
```

CLI flags may remain user-facing for convenience, but internally they should populate `vector_store_options` rather than `qdrant_*` fields. This allows the CLI to support common built-in providers without changing the core config boundary.

## Error Handling

- Unknown `vector_store` names continue to raise `MemoryValidationError`.
- Reserved providers continue to report `reserved but not implemented`.
- Invalid provider option values from `.env` should raise clear `MemoryValidationError` messages naming the env key.
- Qdrant missing dependency should continue to report a clear install hint.
- Hybrid retrieval validation remains generic: hybrid requires sparse embeddings and a vector store that supports the requested sparse/hybrid behavior.

## Testing

Update tests to cover:

1. `MemoryConfig` no longer exposes Qdrant-specific top-level fields.
2. `MemoryManager` can accept an injected `VectorStore`.
3. `.env` Qdrant settings populate `vector_store_options`.
4. CLI Qdrant flags populate provider options and preserve current behavior.
5. Qdrant provider builds from `QdrantVectorStoreConfig`.
6. Existing RAG, BGE, SQLite vector store, and Qdrant tests continue to pass.

## Scope

This change is a boundary cleanup, not a new vector store feature. It should not change retrieval scoring, embedding behavior, or Qdrant query semantics. It should preserve existing CLI behavior where practical while improving the internal API shape.
