# Memora RAG 存储技术文档

## 1. 技术目标

在现有 Memora 代码结构上增加轻量 RAG 存储能力，同时保留以下约束：

- 默认行为不变，未开启 RAG 时继续使用当前文件/SQLite + FTS/keyword 检索。
- MemoryStore 继续保存完整 MemoryItem，是权威事实源。
- 向量必须通过 VectorStore 持久化，不写入 MemoryItem 主结构。
- EmbeddingProvider、VectorStore、Reranker 必须可插拔；默认 provider 保持 deterministic/local，BGE-M3 作为显式 opt-in 的本地 provider。
- 所有 provider 失败时系统可降级，不影响基本记忆读写。

当前 scope 精确限定为：

- `HashEmbeddingProvider`
- `BgeM3EmbeddingProvider`（optional `FlagEmbedding` extra，本地模型路径，dense 默认，可选 sparse lexical weights）
- `SQLiteVectorStore`（dense-only）
- `QdrantVectorStore`（optional `qdrant-client` extra，dense 或 dense+sparse hybrid）
- `NoOpReranker`
- `DeterministicReranker`
- `RagRetriever`

OpenAI、Cohere、Voyage、SentenceTransformer、E5、Ollama、sqlite-vec、Chroma、PGVector、FAISS、Milvus、Weaviate、CrossEncoder、LLM rerank 等仍作为 future adapter backlog 记录，后续实现时再加入 registry 和 optional dependencies。

---

## 2. 建议新增文件

第一版新增文件保持轻量：

```text
memora/
  embeddings.py
  vector_store.py
  reranker.py
  rag.py
```

职责说明：

| 文件 | v1 职责 |
|---|---|
| embeddings.py | EmbeddingProvider 抽象、HashEmbeddingProvider、BgeM3EmbeddingProvider、embedding text/hash helpers |
| vector_store.py | VectorStore 抽象、VectorSearchHit、SQLiteVectorStore、QdrantVectorStore |
| reranker.py | Reranker 抽象、NoOpReranker、DeterministicReranker |
| rag.py | RagRetriever / RagIndex，协调 MemoryStore、FTS、vector、reranker |

第一版不要把 future adapter 写成 `NotImplemented` 类，也不要把 future provider 放进 registry。这样可以避免配置了未实现 provider 后被误导。

当 provider 增多后，再演进为：

```text
memora/embeddings/
  __init__.py
  base.py
  hash.py
  openai.py
  sentence_transformer.py

memora/vector_stores/
  __init__.py
  base.py
  sqlite.py
  sqlite_vec.py
  qdrant.py
  chroma.py
  pgvector.py
  faiss.py
  milvus.py
  weaviate.py

memora/rerankers/
  __init__.py
  base.py
  deterministic.py
  cross_encoder.py
  cohere.py
  llm.py
```

`MemoryManager` 和 `RagRetriever` 只依赖抽象协议，不直接 import 具体厂商 SDK。

---

## 3. 配置设计

在 `MemoryConfig` 中新增：

```python
@dataclass
class MemoryConfig:
    rag_enabled: bool = False

    embedding_provider: str = "hash"
    embedding_model: str = "memora-hash-v1"
    embedding_dimension: int = 384

    vector_store: str = "sqlite"
    vector_path: str | Path | None = None
    vector_candidate_limit: int = 50

    keyword_candidate_limit: int = 50

    reranker: str = "deterministic"
    rerank_candidate_limit: int = 100
```

第一版有效配置值：

| 配置 | v1 有效值 | 说明 |
|---|---|---|
| embedding_provider | `hash`, `bge` | `hash` 为默认零依赖 provider；`bge` 使用本地 BGE-M3 dense embedding |
| vector_store | `sqlite`, `qdrant` | `sqlite` 为默认 dense-only 本地索引；`qdrant` 为 optional extra，支持 dense 或 hybrid |
| reranker | `none`, `deterministic` | 不接模型 reranker |

配置原则：

- `rag_enabled=False` 时不初始化 embedding/vector/reranker。
- v1 中非有效值应报清晰配置错误，不应静默 fallback 到别的 provider。
- provider 相关密钥从环境变量读取，不写入 MemoryConfig。
- BGE-M3 使用 `embedding_model_path` 指向本地模型目录；Memora 默认不下载模型。
- `.env` 可保存 `MEMORA_*` 配置和 `HF_OFFLINE=1`，CLI 显式参数优先级最高。
- `bge` optional extra 只在显式安装时引入 `FlagEmbedding`。

---

## 4. EmbeddingProvider 抽象

### 4.1 接口

```python
from typing import Protocol


class EmbeddingProvider(Protocol):
    name: str
    model: str
    dimension: int
    supports_sparse: bool

    def embed(self, texts: list[str]) -> list[EmbeddingVector]:
        ...
```

要求：

- 输入和输出必须保持顺序一致。
- 每个向量长度必须等于 `dimension`。
- provider 内部负责 batch 和重试。
- 上层不依赖具体 SDK。
- provider 不负责把向量写入数据库。

### 4.2 记忆向量化文本

统一构造 embedding text：

```python
def memory_embedding_text(item: MemoryItem) -> str:
    return (
        f"name: {item.name}\n"
        f"type: {item.type}\n"
        f"description: {item.description}\n"
        f"tags: {', '.join(item.tags)}\n"
        f"content: {item.content}"
    )
```

query 直接使用 `MemoryQuery.query`。

### 4.3 第一版实现

```python
class HashEmbeddingProvider:
    name = "hash"
    model = "memora-hash-v1"
    dimension = 384

    supports_sparse = False

    def embed(self, texts: list[str]) -> list[EmbeddingVector]:
        # 使用稳定 hash 生成确定性向量，仅用于测试和离线开发。
        ...
```

HashEmbeddingProvider 不追求真实语义效果，只保证：

- 零依赖。
- 可重复。
- 测试稳定。
- 不调用外部网络。
- 能验证 RAG 写入、同步、重建、校验、召回融合管线。

### 4.4 Future embedding adapters

下面内容只作为 future adapter backlog 保存，不进入 v1 registry，也不写 `NotImplemented` 占位类。

| Future adapter | 可能配置名 | 后续依赖/extra | 状态 |
|---|---|---|---|
| OpenAIEmbeddingProvider | openai | `openai` | future only |
| CohereEmbeddingProvider | cohere | `cohere` | future only |
| VoyageEmbeddingProvider | voyage | `voyageai` | future only |
| SentenceTransformerEmbeddingProvider | sentence-transformer | `sentence-transformers` | future only |
| BgeM3EmbeddingProvider | bge | `FlagEmbedding` optional extra + 本地模型路径 | current optional dense / dense+sparse |
| E5EmbeddingProvider | e5 | `sentence-transformers` 或专用本地模型依赖 | future only |
| OllamaEmbeddingProvider | ollama | HTTP client / ollama local service | future only |

future adapter 实现前：

- 不加入 `EMBEDDING_PROVIDERS`。
- 不加入 CLI provider choices。
- 不加入 optional dependencies。
- 不影响 v1 deterministic/local 行为。

---

## 5. VectorStore 抽象

VectorStore 是和 EmbeddingProvider 平级的独立生态接口。EmbeddingProvider 只负责把文本变成向量；VectorStore 负责把向量保存到具体向量数据库，并执行相似度检索。

不允许把 Qdrant、Chroma、PGVector 等实现写进 `SQLiteMemoryStore`。SQLiteMemoryStore 是 MemoryItem 的结构化事实源，VectorStore 是向量检索适配层，两者通过 `memory_id` 关联。

### 5.1 数据结构

```python
from dataclasses import dataclass
from typing import Any


@dataclass
class VectorSearchHit:
    memory_id: str
    score: float
    metadata: dict[str, Any]
```

### 5.2 接口

```python
from typing import Protocol, Any


class VectorStore(Protocol):
    name: str

    def init_storage(self) -> None:
        ...

    def upsert(
        self,
        memory_id: str,
        vector: list[float],
        metadata: dict[str, Any],
    ) -> None:
        ...

    def delete(self, memory_id: str) -> None:
        ...

    def search(
        self,
        vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchHit]:
        ...

    def rebuild(
        self,
        items: list[MemoryItem],
        embedder: EmbeddingProvider,
    ) -> None:
        ...

    def verify(self, expected_memory_ids: set[str]) -> dict[str, Any]:
        ...
```

### 5.3 Vector metadata

每条向量至少保存：

```python
metadata = {
    "memory_id": item.id,
    "user_id": item.user_id,
    "project_id": item.project_id,
    "workspace_id": item.workspace_id,
    "type": item.type,
    "status": item.status,
    "tags": item.tags,
    "provider": embedder.name,
    "model": embedder.model,
    "dimension": embedder.dimension,
    "text_hash": sha256(memory_embedding_text(item)),
    "updated_at": item.updated_at.isoformat() if item.updated_at else None,
}
```

metadata 用于快速过滤和索引一致性检查，但最终权限、scope、status 仍以 MemoryStore 为准。

### 5.4 第一版 SQLiteVectorStore

SQLiteVectorStore 是第一版唯一 VectorStore。

可以和 `memora.sqlite3` 共用一个 SQLite 文件，但逻辑表独立：

```sql
CREATE TABLE IF NOT EXISTS memory_vectors (
    memory_id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    text_hash TEXT NOT NULL,
    vector_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memory_vectors_provider_model
ON memory_vectors(provider, model);
```

第一版 search 可以全扫 cosine：

```python
def cosine_similarity(a: list[float], b: list[float]) -> float:
    ...
```

这对轻量个人记忆和 v1 管线验证足够；需要服务化向量索引或 dense+sparse hybrid 时，可选择 QdrantVectorStore。后续还可以替换为 sqlite-vec、FAISS 等。

### 5.5 Future VectorStore adapters

下面内容只作为 future adapter backlog 保存，不进入 v1 registry，也不写 `ReservedVectorStore` / `NotImplemented` 占位类。

| Future adapter | 可能配置名 | 后续依赖/extra | 状态 |
|---|---|---|---|
| SQLiteVecVectorStore | sqlite-vec | sqlite-vec extension | future only |
| QdrantVectorStore | qdrant | `qdrant-client` | current optional dense / hybrid |
| ChromaVectorStore | chroma | `chromadb` | future only |
| PGVectorStore | pgvector | `psycopg`, `pgvector` | future only |
| FAISSVectorStore | faiss | `faiss-cpu` | future only |
| MilvusVectorStore | milvus | `pymilvus` | future only |
| WeaviateVectorStore | weaviate | `weaviate-client` | future only |

future adapter 实现前：

- v1 registry 只包含 `sqlite`。
- CLI 只暴露 `--vector-store sqlite`。
- 具体 adapter 实现时再加入配置值、依赖、测试和错误信息。

---

## 6. Reranker 抽象

Reranker 使用向量召回结果，但不等同于“再按向量距离排一次”。

执行分两层：

1. VectorStore search 阶段已经根据 query vector 和 memory vector 产生 `semantic_score`。
2. Reranker 阶段把 `semantic_score`、`keyword_score`、importance、recency、access 等信号融合排序。

第一版 `DeterministicReranker` 是基于向量分数参与的规则重排。

### 6.1 接口

```python
class Reranker(Protocol):
    name: str

    def rank(
        self,
        query: MemoryQuery,
        candidates: list[MemorySearchResult],
    ) -> list[MemorySearchResult]:
        ...
```

### 6.2 第一版实现

```python
class NoOpReranker:
    name = "none"

    def rank(self, query, candidates):
        return candidates


class DeterministicReranker:
    name = "deterministic"

    def rank(self, query, candidates):
        return sorted(candidates, key=lambda result: result.final_score, reverse=True)
```

### 6.3 Future reranker adapters

下面内容只作为 future adapter backlog 保存。

| Future adapter | 可能配置名 | 后续用途 | 状态 |
|---|---|---|---|
| CrossEncoderReranker | cross-encoder | 本地模型相关性重排 | future only |
| CohereReranker | cohere | 云端 rerank | future only |
| LLMReranker | llm | LLM 判断候选相关性 | future only |

future reranker 实现前：

- v1 registry 只包含 `none` 和 `deterministic`。
- 不加入 optional dependencies。
- 不引入网络调用或非确定性模型排序。

---

## 7. Schema 扩展

### 7.1 MemorySearchResult

建议增加可选分数字段：

```python
@dataclass
class MemorySearchResult:
    memory: MemoryItem
    similarity_score: float
    importance_score: float
    recency_score: float
    access_score: float
    final_score: float
    reason: str = ""
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    rerank_score: float | None = None
```

兼容原则：

- 现有字段保留。
- 新字段必须有默认值，避免破坏现有测试和调用方。
- `similarity_score` 可以继续表示最终检索相似度。
- RAG 开启时，`similarity_score = max(semantic_score, keyword_score)`。

### 7.2 MemoryQuery

第一版可以不扩展 `MemoryQuery`。

如需精细控制，可以后续增加：

```python
retrieval_mode: Literal["auto", "keyword", "vector", "hybrid"] = "auto"
```

---

## 8. MemoryManager 接入点

当前接入点是：

```python
def _candidate_memories(self, query: MemoryQuery) -> list[MemoryItem]:
    ...
```

建议演进为：

```python
def retrieve_memory(...):
    memory_query = MemoryQuery(...)
    validate_memory_query(memory_query)
    if not self.config.rag_enabled:
        return self.retriever.retrieve(self._candidate_memories(memory_query), memory_query)
    return self.rag_retriever.retrieve(memory_query)
```

第一版原则：

- RAG 关闭时完全保留现有路径。
- RAG 开启时走 `RagRetriever`。
- 写入同步由 MemoryManager / RagIndex 协调，不把 vector 逻辑塞进 SQLiteMemoryStore。
- 所有 vector hit 必须回 MemoryStore 二次校验 scope/status/type/tags。

---

## 9. RagRetriever 设计

```python
class RagRetriever:
    def __init__(
        self,
        memory_store: MemoryStore,
        candidate_store: MemoryCandidateStore | None,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
        retriever: MemoryRetriever,
        reranker: Reranker,
        config: MemoryConfig,
    ):
        ...

    def retrieve(self, query: MemoryQuery) -> list[MemorySearchResult]:
        allowed = self._allowed_memories(query)
        vector_hits = self._vector_recall(query)
        keyword_items = self._keyword_recall(query)
        merged = self._merge(allowed, vector_hits, keyword_items)
        scored = self._score(query, merged)
        reranked = self.reranker.rank(query, scored)
        return reranked[: query.top_k]
```

### 9.1 MemoryStore filter

MemoryStore filter 产生 allowed ids：

```python
allowed = {
    item.id: item
    for item in self.memory_store.list_memories(include_archived=query.include_archived)
    if item.user_id == query.user_id
    and (query.project_id is None or item.project_id == query.project_id)
    and (query.workspace_id is None or item.workspace_id == query.workspace_id)
    and (not query.memory_types or item.type in query.memory_types)
    and (not query.tags or set(query.tags).intersection(item.tags))
    and (query.include_knowledge or item.type != "knowledge")
}
```

### 9.2 Vector recall

```python
query_vector = embedder.embed([query.query])[0]
vector_hits = vector_store.search(
    query_vector,
    top_k=config.vector_candidate_limit,
    filters={
        "user_id": query.user_id,
        "project_id": query.project_id,
        "workspace_id": query.workspace_id,
        "status": "active" if not query.include_archived else None,
        "types": query.memory_types,
        "tags": query.tags,
    },
)
```

返回后必须执行：

```python
vector_hits = [hit for hit in vector_hits if hit.memory_id in allowed]
```

### 9.3 Keyword recall

沿用 SQLite FTS：

```python
keyword_items = candidate_store.search_candidates(query)
```

如果 FTS 不可用，则使用现有全量 keyword scan。

### 9.4 Merge

```python
scores_by_id = {}

for hit in vector_hits:
    scores_by_id.setdefault(hit.memory_id, {})["semantic_score"] = hit.score

for item in keyword_items:
    scores_by_id.setdefault(item.id, {})["keyword_score"] = 1.0
```

只保留 `allowed` 中存在的 memory_id。

---

## 10. 打分设计

MemoryRetriever 或 RagRetriever 增加 hybrid scoring：

```python
final_score = (
    semantic_score * 0.45
    + keyword_score * 0.20
    + importance_score * 0.15
    + recency_score * 0.15
    + access_score * 0.05
)
```

兼容旧逻辑：

- RAG 关闭时使用原来的 keyword scoring。
- RAG 开启但向量不可用时 `semantic_score=0`。
- FTS 不可用时 `keyword_score=0` 或全量 keyword fallback。

重排来源说明：

| 分数 | 来源 | 是否来自向量 |
|---|---|---|
| semantic_score | VectorStore.search 返回的向量相似度 | 是 |
| keyword_score | SQLite FTS 或 keyword fallback | 否 |
| importance_score | MemoryItem.weight | 否 |
| recency_score | MemoryItem.updated_at 衰减 | 否 |
| access_score | MemoryItem.access_count | 否 |
| rerank_score | 可选模型 reranker | future only |

因此第一版排序已经会根据向量相似度重排，但不会只根据向量相似度重排。

---

## 11. 写入同步

### 11.1 save/update

```text
MemoryManager.save_memory / remember_candidate
  -> MemoryStore.save_memory / update_memory
  -> if rag_enabled: RagIndex.sync_memory(saved_item)
```

`sync_memory`：

```python
def sync_memory(item: MemoryItem) -> None:
    if not config.rag_enabled:
        return
    text = memory_embedding_text(item)
    vector = embedder.embed([text])[0]
    vector_store.upsert(
        memory_id=item.id,
        vector=vector,
        metadata=build_vector_metadata(item, embedder, text),
    )
```

### 11.2 archive/delete

第一版 archive/delete 都删除向量，简单、安全，不会召回 archived/deleted。

`include_archived` 仍可通过 keyword/full scan 召回；后续再支持 archived 向量 metadata 更新。

### 11.3 hard delete

必须删除向量：

```python
vector_store.delete(memory_id)
```

### 11.4 rebuild-index

必须同时重建：

```text
FTS
VectorStore
```

---

## 12. Verify 设计

`verify` 报告增加：

```python
{
    "checked": 10,
    "errors": [],
    "index_ok": True,
    "vector_ok": True,
    "vector_missing": [],
    "vector_orphans": [],
    "embedding_mismatches": [],
}
```

检查项：

- active memory 是否都有 vector。
- vector memory_id 是否在 MemoryStore 中存在。
- provider/model/dimension 是否与当前配置一致。
- text_hash 是否匹配当前 memory_embedding_text。

---

## 13. CLI 设计

第一版新增最小参数：

```text
--rag
--embedding-provider hash
--vector-store sqlite
--reranker none|deterministic
```

示例：

```bash
python -m memora --root .memora --backend sqlite --rag init
python -m memora --root .memora --backend sqlite --rag search "用户喜欢什么语言回答"
python -m memora --root .memora --backend sqlite --rag rebuild-index
```

第一版不要暴露未实现的 future provider choices。OpenAI、CrossEncoder 等 CLI choices 等具体 adapter 实现时再加入；Qdrant 已作为 optional vector store 暴露，缺少 `qdrant-client` 时必须给出清晰错误。

---

## 14. Optional dependencies

第一版不修改 `pyproject.toml` 添加 provider extras。

后续具体 adapter 实现时再添加：

```toml
[project.optional-dependencies]
openai = ["openai>=1.0"]
cohere = ["cohere>=5.0"]
sentence-transformers = ["sentence-transformers>=3.0"]
qdrant = ["qdrant-client>=1.9"]
chroma = ["chromadb>=0.5"]
pgvector = ["psycopg[binary]>=3.0", "pgvector>=0.3"]
faiss = ["faiss-cpu>=1.8"]
milvus = ["pymilvus>=2.4"]
weaviate = ["weaviate-client>=4.0"]
```

这些 extras 是 future work，不属于 RAG v1。

---

## 15. 测试计划

### 15.1 EmbeddingProvider

- hash provider 输出维度稳定。
- 同一文本输出相同向量。
- 批量输入顺序保持。
- 空字符串行为稳定。

### 15.2 VectorStore

- upsert 后可 search。
- update 后旧向量被替换。
- delete 后不再命中。
- filters 生效。
- verify 能发现 missing/orphan/mismatch。

### 15.3 RAG recall

- 只返回同 user_id 的记忆。
- 只返回同 project_id/workspace_id 的记忆。
- archived/deleted 默认不返回。
- vector + keyword 合并去重。
- keyword fallback 能命中专有名词。
- RAG 关闭时旧测试全部通过。

注意：HashEmbeddingProvider 不保证真实语义相近文本一定相互命中，因此默认测试不应把真实语义质量当作验收条件。BGE-M3 的真实模型测试应通过本地模型路径和显式 marker/env var opt-in 运行，避免默认测试依赖大模型文件。

### 15.4 Rebuild

- 删除 vector 表后 rebuild-index 可恢复。
- 修改 embedding model 后 verify 能提示 mismatch。

---

## 16. 实施顺序

推荐按以下顺序开发 RAG v1：

1. 增加配置项，默认关闭 RAG。
2. 增加 `embeddings.py`，实现 HashEmbeddingProvider。
3. 增加 `vector_store.py`，实现 SQLiteVectorStore。
4. 增加 `reranker.py`，实现 NoOpReranker 和 DeterministicReranker。
5. 增加 memory embedding text 和 metadata builder。
6. 在 MemoryManager / RagIndex 写入路径同步 VectorStore。
7. 实现 RagRetriever 混合召回。
8. 扩展 MemorySearchResult 分数字段，保持默认值兼容。
9. 扩展 verify/rebuild-index。
10. 增加 CLI 最小 RAG 参数。
11. RAG v1 稳定后，再创建 future adapter 计划。

当前阶段不做 OpenAIEmbeddingProvider optional extra，不做 E5/sqlite-vec 等 provider；BGE-M3 支持本地 dense embedding，并可在 Qdrant hybrid 模式下输出 sparse lexical weights。当前不实现 ColBERT 检索。

---

## 17. 关键边界

必须遵守：

- MemoryItem 完整数据只以 MemoryStore 为准。
- VectorStore 不直接返回记忆正文作为权威源，只返回 memory_id 和分数。
- 任何 VectorStore 命中都要回 MemoryStore 二次校验。
- provider 不可用时必须降级。
- 未实现 provider 不进入 v1 registry，不用占位类伪装支持。
- RAG 开启不应破坏文件后端和 SQLite 后端现有行为。
- RAG 关闭时现有检索路径和测试保持不变。

---

## 18. Future Work

### 18.1 Embedding adapters

- OpenAIEmbeddingProvider
- CohereEmbeddingProvider
- VoyageEmbeddingProvider
- SentenceTransformerEmbeddingProvider
- BGEEmbeddingProvider
- E5EmbeddingProvider
- OllamaEmbeddingProvider

### 18.2 VectorStore adapters

- SQLiteVecVectorStore
- QdrantVectorStore
- ChromaVectorStore
- PGVectorStore
- FAISSVectorStore
- MilvusVectorStore
- WeaviateVectorStore

### 18.3 Reranker adapters

- CrossEncoderReranker
- CohereReranker
- LLMReranker

### 18.4 Operations and migration

- vector migration/rebuild workflows
- adapter-specific verify diagnostics
- optional dependency extras
- provider config validation
- vector table schema migrations

### 18.5 Retrieval quality and performance

- entity/relationship-enhanced retrieval
- graph + vector + keyword hybrid retrieval
- sqlite-vec acceleration
- FAISS ANN
- embedding batch and cache
- incremental vector rebuild

---

## 19. 参考资料

- Mem0 How it works: https://docs.mem0.ai/core-concepts/how-it-works
- Mem0 open source configuration: https://docs.mem0.ai/open-source/configuration
- Mem0 migration notes: https://docs.mem0.ai/migration/platform-v2-to-v3
- OpenAI embeddings guide: https://platform.openai.com/docs/guides/embeddings


具体向量库的连接参数会进入 `vector_store_options`，不会作为 `MemoryConfig` 顶层字段暴露给核心逻辑。
