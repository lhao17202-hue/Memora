# Vector Store Provider Boundary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor vector-store configuration so Memora core exposes vector stores through the `VectorStore` protocol and removes Qdrant-specific fields from top-level `MemoryConfig` while preserving CLI and `.env` Qdrant support.

**Architecture:** `MemoryConfig` keeps generic RAG settings and a provider-neutral `vector_store_options` dictionary. `MemoryManager` accepts an injected `VectorStore` and otherwise uses `build_vector_store(config)`. Qdrant owns its provider-specific dataclass and converts generic `vector_store_options` plus retrieval dimensions into client configuration.

**Tech Stack:** Python 3.11+, dataclasses, Protocol-based interfaces, pytest, optional `qdrant-client` dependency mocked in tests.

## Global Constraints

- `MemoryConfig` must not expose `qdrant_url`, `qdrant_host`, `qdrant_port`, `qdrant_api_key`, `qdrant_collection`, `qdrant_timeout`, `qdrant_prefer_grpc`, or `qdrant_recreate_collection` as top-level fields.
- `MemoryConfig` must keep generic RAG fields: `vector_store`, `vector_path`, `retrieval_mode`, `hybrid_prefetch_limit`, `vector_candidate_limit`, `keyword_candidate_limit`, and `min_semantic_score`.
- `MemoryConfig` must add `vector_store_options: dict[str, object] = field(default_factory=dict)`.
- Upper layers must be able to pass a concrete `VectorStore` into `MemoryManager`.
- `.env` and CLI Qdrant settings must populate `vector_store_options`, not top-level Qdrant fields.
- Existing retrieval scoring, embedding behavior, and Qdrant query semantics must not change.
- Missing `qdrant-client` must continue to produce a clear `MemoryValidationError` install hint.
- Hybrid retrieval must continue to require sparse embeddings and a compatible vector store.

---

## File Structure

- Modify `memora/config.py`: add `field`, add `vector_store_options`, remove Qdrant-specific fields.
- Modify `memora/env.py`: map `MEMORA_VECTOR_STORE_*` provider env vars into `vector_store_options`; keep clear typed errors.
- Modify `memora/cli.py`: keep existing Qdrant CLI flags but convert them into `vector_store_options` before `MemoryConfig(**kwargs)`.
- Modify `memora/vector_store.py`: add `QdrantVectorStoreConfig`, make `QdrantVectorStore` accept provider config, add option conversion helper.
- Modify `memora/rag.py`: build Qdrant from `vector_store_options`; validate hybrid compatibility without reading Qdrant-specific config fields.
- Modify `memora/manager.py`: accept optional injected `VectorStore` and use it for RAG.
- Modify tests in `tests/test_schema.py`, `tests/test_env.py`, `tests/test_qdrant_vector_store.py`, `tests/test_rag.py`, `tests/test_manager.py`, and `tests/test_cli.py`.
- Modify docs `README.md`, `docs/Memora RAG存储说明文档.md`, and `docs/Memora RAG存储技术文档.md` to document provider-neutral vector options.

---

### Task 1: Move Qdrant Options Out of `MemoryConfig`

**Files:**
- Modify: `memora/config.py:5-37`
- Test: `tests/test_schema.py:73-92`

**Interfaces:**
- Consumes: existing `MemoryConfig` dataclass.
- Produces: `MemoryConfig.vector_store_options: dict[str, object]`; no top-level `qdrant_*` attributes.

- [ ] **Step 1: Write the failing config boundary test**

Add these assertions to `tests/test_schema.py::test_memory_config_defaults`:

```python
def test_memory_config_defaults():
    config = MemoryConfig()

    assert config.root_dir == ".memora"
    assert config.memory_backend == "file"
    assert config.sqlite_path is None
    assert config.fts_enabled is True
    assert config.fts_candidate_limit == 100
    assert config.vector_store == "sqlite"
    assert config.vector_store_options == {}
    assert not hasattr(config, "qdrant_url")
    assert not hasattr(config, "qdrant_host")
    assert not hasattr(config, "qdrant_port")
    assert not hasattr(config, "qdrant_api_key")
    assert not hasattr(config, "qdrant_collection")
    assert not hasattr(config, "qdrant_timeout")
    assert not hasattr(config, "qdrant_prefer_grpc")
    assert not hasattr(config, "qdrant_recreate_collection")
    assert config.max_retrieved_memories == 8
    assert config.max_memory_prompt_tokens == 2000
    assert config.default_preference_weight == 9
    assert config.default_project_weight == 8
    assert config.default_episodic_weight == 5
    assert config.default_reflective_weight == 7
    assert config.default_tool_weight == 6
    assert config.default_knowledge_weight == 6
    assert config.default_general_weight == 4
    assert config.archive_cold_days == 180
    assert config.require_confirmation_for_conflicts is True
```

- [ ] **Step 2: Run the targeted test and verify it fails**

Run:

```bash
python -m pytest tests/test_schema.py::test_memory_config_defaults -q
```

Expected: FAIL because `vector_store_options` does not exist and Qdrant top-level fields still exist.

- [ ] **Step 3: Update `MemoryConfig`**

Change `memora/config.py` imports and fields:

```python
from dataclasses import dataclass, field
```

Replace the Qdrant-specific field block with `vector_store_options`:

```python
    vector_store: str = "sqlite"
    vector_path: str | Path | None = None
    vector_store_options: dict[str, object] = field(default_factory=dict)
    retrieval_mode: str = "dense"
    hybrid_prefetch_limit: int = 100
    vector_candidate_limit: int = 50
```

Remove these fields entirely:

```python
    qdrant_url: str | None = None
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: str | None = None
    qdrant_collection: str = "memora_memories"
    qdrant_timeout: float = 5.0
    qdrant_prefer_grpc: bool = False
    qdrant_recreate_collection: bool = False
```

- [ ] **Step 4: Run the targeted test and verify it passes**

Run:

```bash
python -m pytest tests/test_schema.py::test_memory_config_defaults -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add memora/config.py tests/test_schema.py
git commit -m "refactor: remove qdrant fields from memory config"
```

---

### Task 2: Parse Vector Store Provider Options From `.env`

**Files:**
- Modify: `memora/env.py:11-130`
- Test: `tests/test_env.py:33-80`

**Interfaces:**
- Consumes: `config_kwargs_from_env(env: Mapping[str, str]) -> dict[str, object]`.
- Produces: Qdrant/provider env vars inside `kwargs["vector_store_options"]` with keys `url`, `host`, `port`, `api_key`, `collection`, `timeout`, `prefer_grpc`, `recreate_collection`.

- [ ] **Step 1: Update env tests to fail on the new shape**

Replace `tests/test_env.py::test_config_kwargs_from_env_coerces_memora_values` with:

```python
def test_config_kwargs_from_env_coerces_memora_values():
    kwargs = config_kwargs_from_env(
        {
            "MEMORA_BACKEND": "sqlite",
            "MEMORA_SQLITE_PATH": ".memora/memora.sqlite3",
            "MEMORA_RAG": "yes",
            "MEMORA_EMBEDDING_PROVIDER": "bge",
            "MEMORA_EMBEDDING_DIMENSION": "1024",
            "MEMORA_EMBEDDING_BATCH_SIZE": "8",
            "MEMORA_EMBEDDING_FP16": "on",
            "MEMORA_EMBEDDING_SPARSE": "true",
            "MEMORA_RETRIEVAL_MODE": "hybrid",
            "MEMORA_HYBRID_PREFETCH_LIMIT": "100",
            "MEMORA_MIN_SEMANTIC_SCORE": "0.30",
            "MEMORA_VECTOR_STORE": "qdrant",
            "MEMORA_VECTOR_STORE_URL": "http://localhost:6333",
            "MEMORA_VECTOR_STORE_PORT": "6333",
            "MEMORA_VECTOR_STORE_COLLECTION": "memora_memories",
            "MEMORA_VECTOR_STORE_TIMEOUT": "5.5",
            "MEMORA_VECTOR_STORE_PREFER_GRPC": "false",
        }
    )

    assert kwargs == {
        "memory_backend": "sqlite",
        "sqlite_path": ".memora/memora.sqlite3",
        "rag_enabled": True,
        "embedding_provider": "bge",
        "embedding_dimension": 1024,
        "embedding_batch_size": 8,
        "embedding_fp16": True,
        "embedding_sparse": True,
        "retrieval_mode": "hybrid",
        "hybrid_prefetch_limit": 100,
        "min_semantic_score": 0.30,
        "vector_store": "qdrant",
        "vector_store_options": {
            "url": "http://localhost:6333",
            "port": 6333,
            "collection": "memora_memories",
            "timeout": 5.5,
            "prefer_grpc": False,
        },
    }
```

Replace invalid Qdrant env assertions with provider-neutral names:

```python
def test_config_kwargs_from_env_reports_invalid_values():
    with pytest.raises(MemoryValidationError, match="MEMORA_VECTOR_STORE_PORT"):
        config_kwargs_from_env({"MEMORA_VECTOR_STORE_PORT": "abc"})

    with pytest.raises(MemoryValidationError, match="MEMORA_EMBEDDING_SPARSE"):
        config_kwargs_from_env({"MEMORA_EMBEDDING_SPARSE": "maybe"})

    with pytest.raises(MemoryValidationError, match="MEMORA_VECTOR_STORE_PREFER_GRPC"):
        config_kwargs_from_env({"MEMORA_VECTOR_STORE_PREFER_GRPC": "maybe"})
```

- [ ] **Step 2: Run env tests and verify they fail**

Run:

```bash
python -m pytest tests/test_env.py -q
```

Expected: FAIL because `MEMORA_VECTOR_STORE_*` vars are ignored or mapped incorrectly.

- [ ] **Step 3: Implement provider option parsing**

In `memora/env.py`, remove Qdrant-specific config mappings from `_BOOL_FIELDS`, `_INT_FIELDS`, `_FLOAT_FIELDS`, and `_STRING_FIELDS`.

Add provider option maps:

```python
_VECTOR_STORE_BOOL_OPTIONS = {
    "MEMORA_VECTOR_STORE_PREFER_GRPC": "prefer_grpc",
    "MEMORA_VECTOR_STORE_RECREATE_COLLECTION": "recreate_collection",
}

_VECTOR_STORE_INT_OPTIONS = {
    "MEMORA_VECTOR_STORE_PORT": "port",
}

_VECTOR_STORE_FLOAT_OPTIONS = {
    "MEMORA_VECTOR_STORE_TIMEOUT": "timeout",
}

_VECTOR_STORE_STRING_OPTIONS = {
    "MEMORA_VECTOR_STORE_URL": "url",
    "MEMORA_VECTOR_STORE_HOST": "host",
    "MEMORA_VECTOR_STORE_API_KEY": "api_key",
    "MEMORA_VECTOR_STORE_COLLECTION": "collection",
}

_VECTOR_STORE_OPTIONS = (
    _VECTOR_STORE_BOOL_OPTIONS
    | _VECTOR_STORE_INT_OPTIONS
    | _VECTOR_STORE_FLOAT_OPTIONS
    | _VECTOR_STORE_STRING_OPTIONS
)
```

Update `_ENV_TO_CONFIG` and `merge_env` to include these option keys:

```python
_ENV_TO_CONFIG = _STRING_FIELDS | _BOOL_FIELDS | _INT_FIELDS | _FLOAT_FIELDS
_ENV_KEYS = set(_ENV_TO_CONFIG) | set(_VECTOR_STORE_OPTIONS)
```

```python
def merge_env(file_env: Mapping[str, str]) -> dict[str, str]:
    merged = dict(file_env)
    for key in _ENV_KEYS | _OS_ENV_KEYS:
        if key in os.environ:
            merged[key] = os.environ[key]
    return merged
```

Update `config_kwargs_from_env` to collect provider options:

```python
def config_kwargs_from_env(env: Mapping[str, str]) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    vector_store_options: dict[str, object] = {}
    for key, value in env.items():
        field = _ENV_TO_CONFIG.get(key)
        if field is not None:
            if key in _BOOL_FIELDS:
                kwargs[field] = _parse_bool(value, key)
            elif key in _INT_FIELDS:
                kwargs[field] = _parse_int(value, key)
            elif key in _FLOAT_FIELDS:
                kwargs[field] = _parse_float(value, key)
            else:
                kwargs[field] = value
            continue

        option = _VECTOR_STORE_OPTIONS.get(key)
        if option is None:
            continue
        if key in _VECTOR_STORE_BOOL_OPTIONS:
            vector_store_options[option] = _parse_bool(value, key)
        elif key in _VECTOR_STORE_INT_OPTIONS:
            vector_store_options[option] = _parse_int(value, key)
        elif key in _VECTOR_STORE_FLOAT_OPTIONS:
            vector_store_options[option] = _parse_float(value, key)
        else:
            vector_store_options[option] = value

    if vector_store_options:
        kwargs["vector_store_options"] = vector_store_options
    return kwargs
```

Add helpers to keep error wording consistent:

```python
def _parse_int(value: str, key: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise MemoryValidationError(f"invalid integer value for {key}: {value}") from exc


def _parse_float(value: str, key: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise MemoryValidationError(f"invalid float value for {key}: {value}") from exc
```

- [ ] **Step 4: Run env tests and verify they pass**

Run:

```bash
python -m pytest tests/test_env.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add memora/env.py tests/test_env.py
git commit -m "refactor: parse vector store options from env"
```

---

### Task 3: Add Qdrant Provider Config and Option Conversion

**Files:**
- Modify: `memora/vector_store.py:11-421`
- Test: `tests/test_qdrant_vector_store.py:1-227`

**Interfaces:**
- Consumes: `QdrantVectorStoreConfig` with provider-owned options.
- Produces: `QdrantVectorStore(config: QdrantVectorStoreConfig)` and `QdrantVectorStoreConfig.from_options(options: dict[str, object], *, dimension: int, retrieval_mode: str, hybrid_prefetch_limit: int) -> QdrantVectorStoreConfig`.

- [ ] **Step 1: Update Qdrant tests to use provider config**

Update import in `tests/test_qdrant_vector_store.py`:

```python
from memora.vector_store import QdrantVectorStore, QdrantVectorStoreConfig, qdrant_point_id
```

Replace every direct `QdrantVectorStore(MemoryConfig(...))` call with `QdrantVectorStore(QdrantVectorStoreConfig(...))`.

Specific replacements:

```python
QdrantVectorStore(QdrantVectorStoreConfig())
```

```python
QdrantVectorStore(QdrantVectorStoreConfig(dimension=4, url="http://localhost:6333"))
```

```python
QdrantVectorStore(QdrantVectorStoreConfig(dimension=4, retrieval_mode="hybrid"))
```

```python
QdrantVectorStore(QdrantVectorStoreConfig(retrieval_mode="hybrid", dimension=2))
```

```python
QdrantVectorStore(QdrantVectorStoreConfig(dimension=2))
```

```python
QdrantVectorStore(QdrantVectorStoreConfig(retrieval_mode="hybrid", dimension=2, hybrid_prefetch_limit=25))
```

Add a new test for option conversion:

```python
def test_qdrant_config_builds_from_vector_store_options():
    config = QdrantVectorStoreConfig.from_options(
        {
            "url": "http://localhost:6333",
            "api_key": "secret",
            "collection": "custom_memories",
            "timeout": 7.5,
            "prefer_grpc": True,
            "recreate_collection": True,
        },
        dimension=1024,
        retrieval_mode="hybrid",
        hybrid_prefetch_limit=25,
    )

    assert config.url == "http://localhost:6333"
    assert config.api_key == "secret"
    assert config.collection == "custom_memories"
    assert config.timeout == 7.5
    assert config.prefer_grpc is True
    assert config.recreate_collection is True
    assert config.dimension == 1024
    assert config.retrieval_mode == "hybrid"
    assert config.hybrid_prefetch_limit == 25
```

Add a test for unknown option names:

```python
def test_qdrant_config_rejects_unknown_options():
    with pytest.raises(MemoryValidationError, match="unknown vector_store_options"):
        QdrantVectorStoreConfig.from_options({"bad": "value"}, dimension=384, retrieval_mode="dense", hybrid_prefetch_limit=100)
```

- [ ] **Step 2: Run Qdrant tests and verify they fail**

Run:

```bash
python -m pytest tests/test_qdrant_vector_store.py -q
```

Expected: FAIL because `QdrantVectorStoreConfig` does not exist and `QdrantVectorStore` still accepts `MemoryConfig`.

- [ ] **Step 3: Implement `QdrantVectorStoreConfig`**

In `memora/vector_store.py`, change the dataclass import:

```python
from dataclasses import dataclass, fields
```

Add the config dataclass after constants:

```python
@dataclass(frozen=True)
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

    @classmethod
    def from_options(
        cls,
        options: dict[str, object] | None,
        *,
        dimension: int,
        retrieval_mode: str,
        hybrid_prefetch_limit: int,
    ) -> "QdrantVectorStoreConfig":
        values = dict(options or {})
        allowed = {field.name for field in fields(cls)} - {"dimension", "retrieval_mode", "hybrid_prefetch_limit"}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise MemoryValidationError(f"unknown vector_store_options for qdrant: {', '.join(unknown)}")
        values.update(
            {
                "dimension": dimension,
                "retrieval_mode": retrieval_mode,
                "hybrid_prefetch_limit": hybrid_prefetch_limit,
            }
        )
        return cls(**values)
```

- [ ] **Step 4: Update `QdrantVectorStore` to use provider config**

Change constructor signature and field usage:

```python
class QdrantVectorStore:
    name = "qdrant"

    def __init__(self, config: QdrantVectorStoreConfig):
        self.config = config
        try:
            from qdrant_client import QdrantClient, models
        except Exception as exc:  # noqa: BLE001 - optional dependency may fail during import
            raise MemoryValidationError("vector_store 'qdrant' requires optional dependency qdrant-client; install with: pip install -e \".[qdrant]\"") from exc
        self.models = models
        self.collection = config.collection
        client_kwargs: dict[str, Any] = {
            "api_key": config.api_key,
            "timeout": config.timeout,
            "prefer_grpc": config.prefer_grpc,
        }
        if config.url:
            client_kwargs["url"] = config.url
        else:
            client_kwargs["host"] = config.host
            client_kwargs["port"] = config.port
        self.client = QdrantClient(**{key: value for key, value in client_kwargs.items() if value is not None})
```

Replace remaining `self.config.embedding_dimension` usages with `self.config.dimension`.

Replace remaining `self.config.qdrant_recreate_collection` usages with `self.config.recreate_collection`.

Keep `self.config.retrieval_mode` and `self.config.hybrid_prefetch_limit` usage.

- [ ] **Step 5: Run Qdrant tests and verify they pass**

Run:

```bash
python -m pytest tests/test_qdrant_vector_store.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add memora/vector_store.py tests/test_qdrant_vector_store.py
git commit -m "refactor: isolate qdrant vector store config"
```

---

### Task 4: Build Qdrant From Provider Options in RAG Factory

**Files:**
- Modify: `memora/rag.py:8-82`
- Test: `tests/test_rag.py:1-66`

**Interfaces:**
- Consumes: `MemoryConfig.vector_store_options` and `QdrantVectorStoreConfig.from_options(...)`.
- Produces: `build_vector_store(config: MemoryConfig) -> VectorStore` that builds Qdrant without top-level Qdrant fields.

- [ ] **Step 1: Add factory test for option conversion**

In `tests/test_rag.py`, extend imports:

```python
from memora.vector_store import QdrantVectorStoreConfig, VectorSearchHit
```

Add this test after `test_rag_factories_support_only_v1_values`:

```python
def test_build_vector_store_passes_options_to_qdrant(fake_flag_embedding, monkeypatch):
    captured = {}

    class FakeQdrantStore:
        name = "qdrant"

        def __init__(self, config):
            captured["config"] = config

    monkeypatch.setattr("memora.rag.QdrantVectorStore", FakeQdrantStore)

    store = build_vector_store(
        MemoryConfig(
            rag_enabled=True,
            vector_store="qdrant",
            embedding_dimension=1024,
            retrieval_mode="hybrid",
            hybrid_prefetch_limit=25,
            vector_store_options={"url": "http://localhost:6333", "collection": "custom"},
        )
    )

    assert store.name == "qdrant"
    assert isinstance(captured["config"], QdrantVectorStoreConfig)
    assert captured["config"].url == "http://localhost:6333"
    assert captured["config"].collection == "custom"
    assert captured["config"].dimension == 1024
    assert captured["config"].retrieval_mode == "hybrid"
    assert captured["config"].hybrid_prefetch_limit == 25
```

- [ ] **Step 2: Run the new test and verify it fails**

Run:

```bash
python -m pytest tests/test_rag.py::test_build_vector_store_passes_options_to_qdrant -q
```

Expected: FAIL because `build_vector_store` still passes full `MemoryConfig`.

- [ ] **Step 3: Update RAG imports and Qdrant factory**

In `memora/rag.py`, change vector store import:

```python
from .vector_store import RESERVED_VECTOR_STORES, QdrantVectorStore, QdrantVectorStoreConfig, SQLiteVectorStore, VectorSearchHit, VectorStore
```

Update `build_vector_store` Qdrant branch:

```python
def build_vector_store(config: MemoryConfig) -> VectorStore:
    _validate_retrieval_config(config)
    if config.vector_store == "sqlite":
        return SQLiteVectorStore(config)
    if config.vector_store == "qdrant":
        qdrant_config = QdrantVectorStoreConfig.from_options(
            config.vector_store_options,
            dimension=config.embedding_dimension,
            retrieval_mode=config.retrieval_mode,
            hybrid_prefetch_limit=config.hybrid_prefetch_limit,
        )
        return QdrantVectorStore(qdrant_config)
    if config.vector_store in RESERVED_VECTOR_STORES:
        ReservedVectorStore(config.vector_store)
    raise MemoryValidationError(f"unsupported vector_store for RAG v1: {config.vector_store}")
```

Do not change retrieval scoring code in this task.

- [ ] **Step 4: Run RAG factory tests and verify they pass**

Run:

```bash
python -m pytest tests/test_rag.py::test_rag_factories_support_only_v1_values tests/test_rag.py::test_build_vector_store_passes_options_to_qdrant -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add memora/rag.py tests/test_rag.py
git commit -m "refactor: build qdrant vector store from provider options"
```

---

### Task 5: Add Vector Store Injection to `MemoryManager`

**Files:**
- Modify: `memora/manager.py:17-82`
- Test: `tests/test_manager.py`

**Interfaces:**
- Consumes: `VectorStore` protocol from `memora.vector_store`.
- Produces: `MemoryManager(..., vector_store: VectorStore | None = None)`.

- [ ] **Step 1: Add manager injection test**

Append this test to `tests/test_manager.py`:

```python
def test_rag_manager_uses_injected_vector_store(tmp_path: Path):
    from memora.embeddings import EmbeddingVector
    from memora.vector_store import VectorSearchHit

    class InMemoryVectorStore:
        name = "custom"

        def __init__(self):
            self.vectors = {}
            self.metadata = {}
            self.initialized = False

        def init_storage(self) -> None:
            self.initialized = True

        def upsert(self, memory_id, vector, metadata):
            dense = vector.dense if isinstance(vector, EmbeddingVector) else vector
            self.vectors[memory_id] = dense
            self.metadata[memory_id] = dict(metadata)

        def delete(self, memory_id: str) -> None:
            self.vectors.pop(memory_id, None)
            self.metadata.pop(memory_id, None)

        def search(self, vector, top_k: int, filters=None, mode: str = "dense"):
            return [VectorSearchHit(memory_id=memory_id, score=0.9, metadata=metadata) for memory_id, metadata in self.metadata.items()][:top_k]

        def get_metadata(self, memory_id: str):
            return self.metadata.get(memory_id)

        def verify(self, expected_memory_ids: set[str]):
            vector_ids = set(self.metadata)
            return {
                "vector_missing": sorted(expected_memory_ids - vector_ids),
                "vector_orphans": sorted(vector_ids - expected_memory_ids),
                "vector_errors": [],
            }

    vector_store = InMemoryVectorStore()
    manager = MemoryManager(MemoryConfig(root_dir=tmp_path / ".memora", rag_enabled=True), vector_store=vector_store)

    manager.init_storage()
    item = manager.save_memory("preference", "custom vector store marker", "custom", name="custom-vector")
    results = manager.retrieve_memory("custom vector store marker")

    assert vector_store.initialized is True
    assert item.id in vector_store.metadata
    assert results[0].memory.id == item.id
    assert manager.verify_memories()["vector_ok"] is True
```

- [ ] **Step 2: Run the injection test and verify it fails**

Run:

```bash
python -m pytest tests/test_manager.py::test_rag_manager_uses_injected_vector_store -q
```

Expected: FAIL because `MemoryManager.__init__` does not accept `vector_store`.

- [ ] **Step 3: Update manager constructor**

In `memora/manager.py`, import `VectorStore`:

```python
from .vector_store import VectorStore
```

Change constructor signature:

```python
    def __init__(
        self,
        config: MemoryConfig | None = None,
        memory_store: MemoryStore | None = None,
        session_store: SessionStore | None = None,
        relation_judge: MemoryRelationJudge | None = None,
        vector_store: VectorStore | None = None,
    ):
```

Change RAG vector store creation block:

```python
        if self.config.rag_enabled and embedder is not None:
            resolved_vector_store = vector_store or build_vector_store(self.config)
            reranker = build_reranker(self.config)
            self.rag_index = RagIndex(self.memory_store, embedder, resolved_vector_store, self.config)
            candidate_store = self.memory_store if isinstance(self.memory_store, MemoryCandidateStore) else None
            self.rag_retriever = RagRetriever(
                self.memory_store,
                candidate_store,
                embedder,
                resolved_vector_store,
                self.retriever,
                reranker,
                self.config,
            )
```

- [ ] **Step 4: Run manager injection test and verify it passes**

Run:

```bash
python -m pytest tests/test_manager.py::test_rag_manager_uses_injected_vector_store -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add memora/manager.py tests/test_manager.py
git commit -m "feat: allow vector store injection"
```

---

### Task 6: Route CLI Qdrant Flags Into Provider Options

**Files:**
- Modify: `memora/cli.py:22-221`
- Test: `tests/test_cli.py:337-412`

**Interfaces:**
- Consumes: existing CLI Qdrant args `--qdrant-url`, `--qdrant-host`, `--qdrant-port`, `--qdrant-api-key`, `--qdrant-collection`, `--qdrant-timeout`, `--qdrant-prefer-grpc`, `--qdrant-recreate-collection`.
- Produces: `_config_kwargs_from_args(args) -> dict` with Qdrant CLI values merged into `vector_store_options`.

- [ ] **Step 1: Add CLI config kwargs unit test**

In `tests/test_cli.py`, import parser helpers at the top:

```python
from memora.cli import _config_kwargs_from_args, build_parser
```

Add this test after `test_cli_loads_env_file_and_cli_overrides_it`:

```python
def test_cli_qdrant_flags_populate_vector_store_options(tmp_path: Path):
    parser = build_parser()
    args = parser.parse_args(
        [
            "--env-file",
            "",
            "--rag",
            "--vector-store",
            "qdrant",
            "--qdrant-url",
            "http://localhost:6333",
            "--qdrant-collection",
            "custom_memories",
            "--qdrant-timeout",
            "7.5",
            "--qdrant-prefer-grpc",
            "init",
        ]
    )

    kwargs = _config_kwargs_from_args(args)

    assert kwargs["rag_enabled"] is True
    assert kwargs["vector_store"] == "qdrant"
    assert kwargs["vector_store_options"] == {
        "url": "http://localhost:6333",
        "collection": "custom_memories",
        "timeout": 7.5,
        "prefer_grpc": True,
    }
```

- [ ] **Step 2: Run the CLI unit test and verify it fails**

Run:

```bash
python -m pytest tests/test_cli.py::test_cli_qdrant_flags_populate_vector_store_options -q
```

Expected: FAIL because `_config_kwargs_from_args` currently writes `qdrant_*` keys.

- [ ] **Step 3: Update CLI config mapping**

In `memora/cli.py`, remove Qdrant keys from `cli_values`:

```python
        "qdrant_url": args.qdrant_url,
        "qdrant_host": args.qdrant_host,
        "qdrant_port": args.qdrant_port,
        "qdrant_api_key": args.qdrant_api_key,
        "qdrant_collection": args.qdrant_collection,
        "qdrant_timeout": args.qdrant_timeout,
```

Add provider option handling after the generic `cli_values` loop:

```python
    vector_store_options = dict(kwargs.get("vector_store_options") or {})
    qdrant_cli_options = {
        "url": args.qdrant_url,
        "host": args.qdrant_host,
        "port": args.qdrant_port,
        "api_key": args.qdrant_api_key,
        "collection": args.qdrant_collection,
        "timeout": args.qdrant_timeout,
    }
    for key, value in qdrant_cli_options.items():
        if value is not None:
            vector_store_options[key] = value
    if args.qdrant_prefer_grpc:
        vector_store_options["prefer_grpc"] = True
    if args.qdrant_recreate_collection:
        vector_store_options["recreate_collection"] = True
    if vector_store_options:
        kwargs["vector_store_options"] = vector_store_options
```

Remove old direct bool handling:

```python
    if args.qdrant_prefer_grpc:
        kwargs["qdrant_prefer_grpc"] = True
    if args.qdrant_recreate_collection:
        kwargs["qdrant_recreate_collection"] = True
```

- [ ] **Step 4: Run CLI tests around env and Qdrant behavior**

Run:

```bash
python -m pytest tests/test_cli.py::test_cli_qdrant_flags_populate_vector_store_options tests/test_cli.py::test_qdrant_cli_missing_dependency_reports_clear_error tests/test_cli.py::test_hybrid_cli_without_sparse_reports_clear_error -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add memora/cli.py tests/test_cli.py
git commit -m "refactor: route qdrant cli flags through vector store options"
```

---

### Task 7: Update Docs for Provider-Neutral Vector Store Options

**Files:**
- Modify: `README.md`
- Modify: `docs/Memora RAG存储说明文档.md`
- Modify: `docs/Memora RAG存储技术文档.md`
- Test: no dedicated doc test; use full suite in final verification.

**Interfaces:**
- Consumes: implemented `.env` keys `MEMORA_VECTOR_STORE_*`.
- Produces: docs that no longer recommend `MEMORA_QDRANT_*` keys.

- [ ] **Step 1: Update README env examples**

In `README.md`, replace Qdrant-specific env names:

```env
MEMORA_QDRANT_URL=http://localhost:6333
MEMORA_QDRANT_COLLECTION=memora_memories
MEMORA_QDRANT_TIMEOUT=5
MEMORA_QDRANT_PREFER_GRPC=false
MEMORA_QDRANT_RECREATE_COLLECTION=false
```

with provider-neutral names:

```env
MEMORA_VECTOR_STORE_URL=http://localhost:6333
MEMORA_VECTOR_STORE_COLLECTION=memora_memories
MEMORA_VECTOR_STORE_TIMEOUT=5
MEMORA_VECTOR_STORE_PREFER_GRPC=false
MEMORA_VECTOR_STORE_RECREATE_COLLECTION=false
```

Add one sentence near the RAG config section:

```markdown
Provider-specific vector store settings are parsed into `MemoryConfig.vector_store_options`; they are not top-level `MemoryConfig` fields.
```

- [ ] **Step 2: Update Chinese RAG docs**

In both Chinese RAG docs, replace any `MEMORA_QDRANT_` examples with `MEMORA_VECTOR_STORE_` equivalents:

```text
MEMORA_VECTOR_STORE_URL
MEMORA_VECTOR_STORE_HOST
MEMORA_VECTOR_STORE_PORT
MEMORA_VECTOR_STORE_API_KEY
MEMORA_VECTOR_STORE_COLLECTION
MEMORA_VECTOR_STORE_TIMEOUT
MEMORA_VECTOR_STORE_PREFER_GRPC
MEMORA_VECTOR_STORE_RECREATE_COLLECTION
```

Add the Chinese note:

```markdown
具体向量库的连接参数会进入 `vector_store_options`，不会作为 `MemoryConfig` 顶层字段暴露给核心逻辑。
```

- [ ] **Step 3: Check docs for old env names**

Run:

```bash
python - <<'PY'
from pathlib import Path
for path in [Path('README.md'), Path('docs/Memora RAG存储说明文档.md'), Path('docs/Memora RAG存储技术文档.md')]:
    text = path.read_text(encoding='utf-8')
    old = [line for line in text.splitlines() if 'MEMORA_QDRANT_' in line]
    assert not old, f'{path} still has old Qdrant env names: {old}'
PY
```

Expected: exits 0.

- [ ] **Step 4: Commit**

```bash
git add README.md "docs/Memora RAG存储说明文档.md" "docs/Memora RAG存储技术文档.md"
git commit -m "docs: document vector store provider options"
```

---

### Task 8: Final Integration Verification and Cleanup

**Files:**
- Modify if needed: files touched by Tasks 1-7 only.
- Test: full test suite and repository-wide searches.

**Interfaces:**
- Consumes: all prior task outputs.
- Produces: clean working tree with all tests passing and no old Qdrant top-level config references.

- [ ] **Step 1: Search for removed `MemoryConfig` fields**

Run:

```bash
python - <<'PY'
from pathlib import Path
removed = [
    'qdrant_url',
    'qdrant_host',
    'qdrant_port',
    'qdrant_api_key',
    'qdrant_collection',
    'qdrant_timeout',
    'qdrant_prefer_grpc',
    'qdrant_recreate_collection',
]
allowed_files = {
    Path('memora/cli.py'),
}
violations = []
for path in list(Path('memora').glob('*.py')) + list(Path('tests').glob('*.py')):
    text = path.read_text(encoding='utf-8')
    for token in removed:
        if token in text and path not in allowed_files:
            violations.append(f'{path}: {token}')
assert not violations, '\n'.join(violations)
PY
```

Expected: exits 0. `memora/cli.py` may still contain arg names because it preserves user-facing CLI flags.

- [ ] **Step 2: Run focused RAG/vector/env/CLI tests**

Run:

```bash
python -m pytest tests/test_schema.py tests/test_env.py tests/test_qdrant_vector_store.py tests/test_rag.py tests/test_manager.py tests/test_cli.py tests/test_vector_store.py -q
```

Expected: all selected tests pass.

- [ ] **Step 3: Run full test suite**

Run:

```bash
python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 4: Check diff whitespace**

Run:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 5: Commit cleanup only if Step 1-4 required changes**

If Step 1-4 required code or doc changes, commit them:

```bash
git add memora tests README.md docs
git commit -m "test: verify vector store provider boundary"
```

If Step 1-4 required no changes, skip this commit.

- [ ] **Step 6: Report evidence**

Report the exact outputs from:

```bash
git log --oneline -5
python -m pytest -q
git status --short
```

Expected: recent task commits are present, full test suite passes, and working tree is clean.
