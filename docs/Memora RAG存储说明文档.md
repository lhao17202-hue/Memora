# Memora RAG 存储说明文档

## 1. 文档目的

本文档用于说明 Memora 下一阶段 RAG 存储能力的定位、模块边界、第一版范围和后续演进方向。

Memora 的目标不是完整复制 Mem0，也不是把所有聊天记录直接塞进向量库；它是在现有 deterministic local memory system 基础上，补齐轻量化 RAG 记忆召回能力：

- MemoryStore 继续作为完整 MemoryItem 的权威事实源。
- VectorStore 只保存 `memory_id`、向量和索引 metadata。
- EmbeddingProvider 负责把 query 和 memory text 转换为向量。
- Keyword/FTS 检索继续作为专有名词、中文短查询、数字、代码符号和精确词的兜底。
- Reranker 作为可插拔重排层，第一版只使用确定性规则。
- RAG 默认关闭；未开启时继续使用当前文件/SQLite + FTS/keyword 检索行为。

第一版 RAG 的目标是把本地向量索引生命周期和 hybrid retrieval 架构做闭环，而不是立刻追求云端 embedding、外部向量数据库或模型 rerank。未来 provider 生态会在本文档中保存为 future adapter backlog，供后续阶段使用。

---

## 2. 系统定位

RAG 存储不是“记忆写入决策层”。

Memora 中的长期记忆仍然是经过外部 agent runtime 或抽取逻辑沉淀后的 durable memory，例如：

- 用户长期偏好
- 用户对助手行为的反馈
- 项目事实
- 技术决策
- 工具使用经验
- 会话摘要
- 可复用知识
- 实体信息
- 外部参考资料

是否保存、是否拒绝、是否需要确认，仍然由现有 Policy 和写入流程决定。RAG 存储层只解决：这些已经写入的记忆如何被更好地召回。

当前阶段继续保持 local-first：

- 默认仍使用零依赖 `HashEmbeddingProvider`。
- 可选支持本地离线 BGE-M3 dense embedding provider。
- 可选支持 BGE-M3 dense+sparse 输出，并在 Qdrant 中做 hybrid 检索。
- 可选支持用户自管的 Qdrant 向量索引；Memora 不启动、不托管 Qdrant。
- 不新增网络型 embedding provider。
- 不新增模型 reranker。
- `FlagEmbedding` 仅作为 `bge` optional extra，不进入默认安装依赖。
- `qdrant-client` 仅作为 `qdrant` optional extra，不进入默认安装依赖。
- 不注册未实现的 provider，也不写 `NotImplemented` 占位类来假装支持。

---

## 3. 总体架构

```text
MemoryRuntime
  -> MemoryManager
    -> MemoryStore / SQLiteMemoryStore
       保存完整 MemoryItem、scope、status、tags、timestamps
    -> EmbeddingProvider
       将 query / memory text 转换为向量
    -> VectorStore
       保存 memory_id + vector + metadata，执行语义召回
    -> Keyword / FTS recall
       精确词、中文短查询、术语、数字兜底
    -> Reranker / MemoryRetriever
       融合 semantic、keyword、importance、recency、access 分数
    -> Prompt Formatter
       格式化最终 top_k 记忆
```

存储分工：

| 模块 | 存什么 | 负责什么 | 不负责什么 |
|---|---|---|---|
| MemoryStore / SQLiteMemoryStore | 完整 MemoryItem | 权威数据、结构化过滤、生命周期、导入导出 | 语义相似度计算 |
| VectorStore | memory_id、向量、简易 metadata | 向量持久化、语义相似召回 | 保存完整记忆正文作为权威源 |
| FTS/Keyword | name、description、tags、content 的文本索引 | 精确词召回、兜底召回 | 模糊语义理解 |
| Reranker | 不持久化核心数据 | 多路候选统一排序 | 决定记忆是否写入 |

硬边界：VectorStore 和 EmbeddingProvider 平级，都是可替换生态入口；但第一版只实现 deterministic/local provider。

第一阶段模块关系：

```text
embeddings.py
  -> HashEmbeddingProvider
  -> BgeM3EmbeddingProvider（可选 FlagEmbedding extra，本地模型路径；可选 sparse 输出）

vector_store.py
  -> SQLiteVectorStore（dense-only）
  -> QdrantVectorStore（可选 qdrant-client extra；dense 或 dense+sparse hybrid）

reranker.py
  -> NoOpReranker / DeterministicReranker

rag.py
  -> RagRetriever
```

后续 provider 变多后，再演进为 `embeddings/`、`vector_stores/`、`rerankers/` 子包。上层只通过抽象接口选择 provider，不直接依赖具体厂商 SDK。

---

## 4. 召回执行流程

Memora RAG 检索采用“结构化过滤 + 向量召回 + 关键词兜底 + 融合重排”的流程。

```text
用户 query
  -> 构造 MemoryQuery
  -> MemoryStore 结构化过滤
  -> query embedding
  -> VectorStore search
  -> FTS / keyword fallback
  -> 合并去重
  -> 二次结构化校验
  -> Reranker / MemoryRetriever
  -> 返回 top_k MemorySearchResult
```

### 4.1 MemoryStore 结构化硬过滤

必须优先保证边界正确：

- user_id
- project_id
- workspace_id
- status
- memory_types
- tags
- include_archived
- include_knowledge

这一步的职责是权限、租户、状态、类型和项目隔离。

即使 VectorStore 支持 metadata filter，最终也必须回到 MemoryStore 做二次校验，避免向量库 metadata 不一致导致串用户、召回已删除记忆或召回错误项目记忆。

### 4.2 向量语义召回

EmbeddingProvider 将 query 向量化，VectorStore 返回语义相似的 memory_id 和 semantic_score。

向量库只保存：

```text
memory_id
vector
metadata: user_id / project_id / workspace_id / status / type / tags
embedding provider / model / dimension / text_hash
```

完整内容仍然从 MemoryStore 读取。

### 4.3 关键词兜底召回

纯向量召回容易漏掉：

- 中文短词
- API 名称
- 文件路径
- 类名/函数名
- 数字、版本号、错误码
- 用户明确说过的原词

因此需要保留当前 SQLite FTS 和 MemoryRetriever 中的 keyword scoring。

### 4.4 合并去重

向量命中和关键词命中按 memory_id 合并。

同一条记忆可能同时有：

- semantic_score
- keyword_score
- importance_score
- recency_score
- access_score

最终排序时统一融合。

### 4.5 融合重排

第一版建议使用确定性融合，不立刻接模型 rerank：

```text
final_score =
  semantic_score * 0.45
  + keyword_score * 0.20
  + importance_score * 0.15
  + recency_score * 0.15
  + access_score * 0.05
```

这里的重排会使用向量结果，但不是只按向量分数排序。

第一版排序逻辑：

- VectorStore 返回的相似度进入 `semantic_score`。
- Keyword/FTS 命中进入 `keyword_score`。
- 记忆自身的 `weight`、`updated_at`、`access_count` 继续参与排序。
- `DeterministicReranker` 根据上述融合分数排序。

后续如果接入模型 Reranker，建议输入 query + memory text + 原始 semantic_score/keyword_score，由模型校准候选真实相关性。模型 rerank 不应只看向量距离，因为向量距离已经在 VectorStore 阶段用过一次。

---

## 5. 第一版必备可插拔模块

### 5.1 EmbeddingProvider

EmbeddingProvider 是向量化生态入口。

第一版只实现：

| Provider | 当前状态 | 用途 |
|---|---|---|
| HashEmbeddingProvider | 默认实现 | 零依赖、确定性测试、离线开发、RAG 管线闭环 |
| BgeM3EmbeddingProvider | 可选实现 | 本地 BGE-M3 dense 或 dense+sparse embedding，需安装 `.[bge]` 并提供本地模型路径 |

HashEmbeddingProvider 不追求真实语义质量。它的作用是：

- 保证 RAG 写入、索引、检索、rebuild、verify 流程可测试。
- 保持第一版无网络、无外部模型、无新依赖。
- 为后续真实 embedding provider 留出稳定接口。

因此，HashEmbeddingProvider 适合验证“RAG 架构和生命周期是否正确”，不代表最终语义召回质量。BGE-M3 provider 用于本地真实语义召回：默认只请求 dense vector；当 `MEMORA_EMBEDDING_SPARSE=true` 且 `MEMORA_RETRIEVAL_MODE=hybrid` 时，会请求 `lexical_weights` 并交给 Qdrant sparse vector 参与 dense+sparse 融合。ColBERT vectors 暂不进入索引。

BGE-M3 推荐通过 `.env` 配置：

```env
MEMORA_BACKEND=sqlite
MEMORA_RAG=true
MEMORA_EMBEDDING_PROVIDER=bge
MEMORA_EMBEDDING_MODEL=bge-m3
MEMORA_EMBEDDING_MODEL_PATH=C:\Download\bge-m3
MEMORA_EMBEDDING_DIMENSION=1024
MEMORA_EMBEDDING_BATCH_SIZE=8
MEMORA_EMBEDDING_FP16=true
MEMORA_EMBEDDING_SPARSE=false
MEMORA_VECTOR_STORE=sqlite
MEMORA_RETRIEVAL_MODE=dense
HF_OFFLINE=1
```

切换 provider、model、model path 或 dimension 后，向量索引属于派生缓存，必须运行 `rebuild-index` 重新生成。

后续适配器只记录为 future adapter，不进入当前 registry：

| Future adapter | 可能配置名 | 后续用途 |
|---|---|---|
| OpenAIEmbeddingProvider | openai | 通用云端 embedding |
| CohereEmbeddingProvider | cohere | 多语言和搜索场景 |
| VoyageEmbeddingProvider | voyage | RAG/检索优化 embedding |
| SentenceTransformerEmbeddingProvider | sentence-transformer | 本地开源模型生态 |
| BGEEmbeddingProvider | bge | 中文/多语言本地 embedding |
| E5EmbeddingProvider | e5 | 检索型本地 embedding |
| OllamaEmbeddingProvider | ollama | 本地服务化 embedding |

第一版不要添加这些 provider 的 `NotImplemented` 占位类，也不要把它们加入注册表。optional dependencies 在具体 adapter 真正实现时再加入。

### 5.2 VectorStore

VectorStore 是向量数据库生态入口。

第一版只实现：

| VectorStore | 第一版状态 | 用途 |
|---|---|---|
| SQLiteVectorStore | 实现 | 本地轻量持久化，可全扫 cosine，dense-only |
| QdrantVectorStore | 可选实现 | 用户自管 Qdrant 服务，支持 dense-only 或 dense+sparse hybrid 检索 |

说明：

- 向量必须存储到 VectorStore 中。
- SQLiteVectorStore 可以共用 SQLite 文件，但逻辑上仍然是向量数据库适配器。
- QdrantVectorStore 使用 `qdrant-client` optional extra；Memora 不负责启动或托管 Qdrant。
- Qdrant 中的 point payload 只保存检索 metadata，完整 MemoryItem 仍以 MemoryStore 为准。
- MemoryStore 不应直接承担向量相似度检索职责。
- VectorStore 不直接返回记忆正文，只返回 memory_id 和分数。
- 所有 VectorStore 命中都必须回 MemoryStore 做二次校验。

后续适配器只记录为 future adapter，不进入 v1 registry：

| Future adapter | 可能配置名 | 后续用途 |
|---|---|---|
| SQLiteVecVectorStore | sqlite-vec | SQLite 原生向量扩展，优化本地性能 |
| ChromaVectorStore | chroma | 原型开发和本地集合管理 |
| PGVectorStore | pgvector | PostgreSQL 统一业务数据和向量 |
| FAISSVectorStore | faiss | 本地高性能 ANN |
| MilvusVectorStore | milvus | 大规模分布式向量检索 |
| WeaviateVectorStore | weaviate | 具备 schema/filter 的向量服务 |

未来适配器实现前，不应静默退回 SQLiteVectorStore。等具体 adapter 实现时，再添加明确配置、依赖和错误行为。

### 5.3 Reranker

Reranker 是候选重排生态入口。

第一版只实现：

| Reranker | 第一版状态 | 用途 |
|---|---|---|
| NoOpReranker | 实现 | 不额外重排 |
| DeterministicReranker | 实现 | 使用规则融合分数排序 |

后续适配器只记录为 future adapter：

| Future adapter | 可能配置名 | 后续用途 |
|---|---|---|
| CrossEncoderReranker | cross-encoder | 本地 rerank 模型 |
| CohereReranker | cohere | 云端 rerank |
| LLMReranker | llm | 用 LLM 判断候选相关性 |

第一版不接模型 rerank，避免引入网络依赖、延迟和非确定性。

---

## 6. 写入与同步策略

记忆写入流程保持现状，只在 RAG 开启时追加向量索引同步：

```text
MemoryCandidate
  -> Policy evaluate
  -> MemoryStore save/update/delete
  -> FTS sync
  -> if rag_enabled: EmbeddingProvider embed(memory text)
  -> if rag_enabled: VectorStore upsert/delete
```

同步规则：

- create/update 后 upsert vector。
- archive/delete 后第一版建议删除向量，避免默认检索召回 archived/deleted memory。
- hard delete 必须删除向量。
- rebuild-index 必须重建 FTS 和 VectorStore。
- verify 必须检查 MemoryStore memory_id 与 VectorStore memory_id 是否一致。

同步协调应放在 MemoryManager / RagIndex 层，不要让 SQLiteMemoryStore 直接知道 RagIndex 或 VectorStore。

---

## 7. 降级策略

Memora 必须支持降级：

| 组件不可用 | 降级行为 |
|---|---|
| EmbeddingProvider 不可用 | 跳过向量召回，保留 keyword/FTS |
| VectorStore 不可用 | 跳过向量召回，保留 keyword/FTS |
| Reranker 不可用 | 使用确定性 MemoryRetriever 或 NoOpReranker |
| FTS 不可用 | 使用全量 keyword scan |
| RAG 关闭 | 回到当前 SQLite/文件检索行为 |

这保证 Memora 仍然是本地可用、可测试、可移植的记忆系统。

---

## 8. 第一版范围

第一版建议做：

- RAG 配置项，默认关闭。
- EmbeddingProvider 抽象。
- VectorStore 抽象。
- Reranker 抽象。
- HashEmbeddingProvider。
- SQLiteVectorStore。
- NoOpReranker。
- DeterministicReranker。
- RagRetriever。
- vector + keyword 混合候选召回。
- MemorySearchResult 增加兼容的 semantic_score、keyword_score、rerank_score。
- rebuild-index 同步重建 SQLite vector index。
- verify 检查 SQLite vector index 一致性。

第一版不做：

- OpenAI / Cohere / Voyage embedding provider。
- SentenceTransformer / BGE / E5 / Ollama 本地模型 provider。
- SQLite-vec / Qdrant / Chroma / PGVector / FAISS / Milvus / Weaviate vector store。
- 图谱记忆。
- 实体链接。
- 大规模 ANN 优化。
- CrossEncoder / Cohere / LLM rerank。
- 自动智能合并。
- optional dependency extras。
- 未实现 provider 的占位类或注册表项。

---

## 9. 后续演进 / Future Work

未来工作需要保存，但不进入第一版实现。

### Phase 2: 云端 embedding provider

- OpenAIEmbeddingProvider
- CohereEmbeddingProvider
- VoyageEmbeddingProvider
- adapter-specific verify diagnostics
- provider-specific optional dependencies

### Phase 3: 本地 embedding provider

- SentenceTransformerEmbeddingProvider
- BGEEmbeddingProvider
- E5EmbeddingProvider
- OllamaEmbeddingProvider
- 本地模型 batch、缓存、维度校验

### Phase 4: 向量数据库适配器

- SQLiteVecVectorStore
- QdrantVectorStore
- ChromaVectorStore
- PGVectorStore
- FAISSVectorStore
- MilvusVectorStore
- WeaviateVectorStore
- vector migration / rebuild workflows

### Phase 5: 高级 reranker

- CrossEncoderReranker
- CohereReranker
- LLMReranker
- query + memory text + semantic_score + keyword_score 的综合重排输入

### Phase 6: 实体/关系增强召回

- entity extraction
- entity linking
- relationship memory
- graph-enhanced retrieval
- hybrid graph + vector + keyword retrieval

### Phase 7: 性能优化

- sqlite-vec 加速
- FAISS ANN
- embedding batch
- vector cache
- incremental rebuild

---

## 10. 参考资料

- Mem0 How it works: https://docs.mem0.ai/core-concepts/how-it-works
- Mem0 open source configuration: https://docs.mem0.ai/open-source/configuration
- Mem0 migration notes: https://docs.mem0.ai/migration/platform-v2-to-v3
- OpenAI embeddings guide: https://platform.openai.com/docs/guides/embeddings
