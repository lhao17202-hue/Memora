# Memora 重构共识与执行计划

日期：2026-07-27

本文记录本轮讨论和 grilling 过程得出的结论，以及后续执行步骤。它不是当前代码现状说明，而是下一阶段重构的目标共识。

## 1. 背景问题

当前 Memora 已经具备本地记忆保存、检索、RAG 索引、CLI、确认流程和基础 policy，但在继续发展为真正的 agent 长期记忆系统时，暴露出几个核心问题：

1. 记忆类型混杂。
   旧的 `MemoryType` 同时混入了归属对象、来源、内容性质和存储用途，例如 `user`、`feedback`、`session_summary`、`tool_experience`、`reference`、`knowledge`。

2. RAG 的定位容易混淆。
   旧架构中容易把 vector store 理解成第三种存储后端，但实际它只是索引和检索增强能力。

3. Agent 不能只靠用户 query 检索长期记忆。
   用户偏好、项目约束这类长期上下文不应该完全依赖 query 命中。

4. 自动长期记忆抽取缺失。
   当前系统主要依赖外部 agent 直接提供 `MemoryCandidate`，Memora 自己不负责从 session/trace 中判断什么应该变成长记忆。

5. 冲突和合并能力不足。
   旧逻辑主要依赖同名更新和非常简单的 conflict 判断，无法可靠处理“用户偏好改变”“旧项目事实失效”“重复内容需要合并”等情况。

6. Markdown `MEMORY.md` 的概念不清。
   它当前只是人类可读 catalog，不是检索索引。真正检索走 `.md` 文件扫描、SQLite FTS 或 RAG/vector。

## 2. 已确认的架构原则

### 2.1 RAG 不是存储后端

最终共识：

```text
Markdown / SQLite = MemoryBackend
RAG / Vector = RetrievalCapability / Index
```

RAG/vector 不作为正式权威存储后端。它只用于：

- on-demand 语义检索
- conflict/merge 前的相似记忆召回
- knowledge、episodic、reflective、tool、general 等按需记忆的语义召回

MemoryItem 的正式保存仍然发生在 Markdown 或 SQLite 后端中。RAG/vector 是可重建索引。

### 2.2 长期记忆类型硬切为 7 类

旧类型不做兼容迁移，直接切换为：

```python
MemoryType = Literal[
    "preference",
    "project",
    "episodic",
    "reflective",
    "tool",
    "knowledge",
    "general",
]
```

各类型含义：

- `preference`：用户偏好、个人信息、输出习惯、自定义约束。
- `project`：项目需求、技术栈、架构设定、文件/目录/代码依赖。
- `episodic`：带时间线的交互事件和重要历史经历。
- `reflective`：任务完成后的复盘、失败总结、优化思路。
- `tool`：从工具调用 trace 中总结出的长期工具使用经验。
- `knowledge`：外部导入的离线技术资料、参考知识。
- `general`：LLM 判断重要但不属于以上类别的长期记忆。

旧类型处理策略：

```text
不做自动迁移。
旧 .md 或 SQLite 旧类型数据可能不再通过 validate。
文档需要明确这是 breaking change。
```

### 2.3 默认上下文注入策略

长期记忆分为两类检索/注入策略。

每轮默认注入：

```text
preference
project
```

按需检索：

```text
episodic
reflective
tool
knowledge
general
```

`preference` 和 `project` 不依赖 RAG，也不完全依赖 query。它们应通过 scope、status、weight、recency、access_count 和 token cap 取出。

其他类型根据 query、任务场景和可用检索能力按需召回。

### 2.4 自动长期记忆抽取固定使用 LLM

长期记忆自动抽取主路径固定为 LLM。

保留手动 `save` / `remember candidate` 能力，用于：

- 调试
- 测试
- 上层 agent 直接传入 candidate
- CLI 手动写入

但自动从 session/trace 中提取长期记忆时，使用 LLM。

### 2.5 LLM 抽取失败策略

自动流程中：

```text
LLM extraction failed
-> 记录 extraction_errors
-> 本轮不写长期记忆
-> 不阻断 agent 主流程
```

显式 CLI extract 中：

```text
抽取失败 -> 报错并返回非零 exit code
```

### 2.6 抽取时机是 session/task end

长期记忆抽取主流程发生在 session end 或 task end。

不做每轮消息后的实时长期记忆抽取。

原因：

- 每轮抽取容易把临时状态写成长记忆。
- session/task end 上下文更完整。
- 成本更可控。
- 更适合结合 agent trace 做总结。

### 2.7 ExtractionArtifact 与 MemoryItem 分离

LLM 原始抽取结果不能直接作为正式长期记忆入库。

流程应为：

```text
session/trace
-> LLM extractor
-> ExtractionArtifact
-> safety/relation/merge/write plan
-> MemoryItem save/update/archive
```

`ExtractionArtifact` 用于审计、调试、回放；`MemoryItem` 是正式长期记忆。

正式 `MemoryItem` 必须经过：

- safety policy
- relation detection
- conflict/merge decision
- write plan

否则不能进入长期记忆库。

### 2.8 冲突/合并主路径使用 LLM relation classifier

冲突与合并判断主路径：

```text
embedding related recall
-> LLM relation classifier
-> structured JSON decision
-> deterministic validation
-> execute write plan
```

fallback：

```text
LLM 失败时 deterministic safe fallback。
不确定就 requires_confirmation。
不自动 create 可能冲突的记忆。
```

### 2.9 语义冲突/合并必须依赖 embedding

跨名称的语义冲突和合并必须依赖 embedding related recall。

没有 embedding 时：

- 不做跨名称语义冲突/合并。
- 同名 duplicate 仍可走 deterministic 处理。
- 新 candidate 仍可 create。

### 2.10 Relation classifier 必须结构化 JSON 输出

LLM relation classifier 不接受自然语言动作文本。

输出必须是结构化 JSON，例如：

```json
{
  "action": "merge",
  "target_memory_id": "mem_123",
  "confidence": 0.88,
  "reason": "Candidate refines an existing response-language preference.",
  "merged_memory": {
    "name": "response-language",
    "description": "用户回答语言和风格偏好。",
    "content": "用户偏好使用中文回答技术问题，并希望回答简洁。",
    "tags": ["language", "response-style"]
  }
}
```

允许 action：

```text
add
noop
update
merge
conflict
supersede
```

系统必须验证：

- action 是否允许
- target_memory_id 是否存在
- target type/scope 是否匹配
- merged_memory 是否合法
- confidence 是否达标

### 2.11 高置信 conflict/supersede 可自动替换

LLM 高置信 conflict/supersede 可以自动替换，但必须满足硬护栏：

- confidence 达到阈值
- candidate 有明确替代表达
- memory type 允许自动替换

明确替代表达包括：

```text
以后
改成
不再
现在用
从现在开始
instead
switch to
no longer
from now on
```

自动替换时：

```text
archive old
create/update new
new.supersedes = [old.id]
old 从默认检索和 vector index 中移除
不 hard delete
```

### 2.12 Tool memory 不保存完整工具日志

完整工具日志属于 agent/trace 层，不属于长期 `tool` memory。

Memora 不负责采集完整工具日志。Agent 层可以把 trace/session 发给 LLM，LLM 抽取出长期有效工具经验，再交给 Memora。

长期 tool memory 示例：

```text
在 Windows PowerShell 中，不要使用 Bash heredoc；应使用 PowerShell here-string 后管道给 python。
```

而不是完整工具调用 stdout/stderr。

## 3. 新的目标流程

### 3.1 写入流程

```text
agent session / trace
-> LLM extractor
-> ExtractionArtifact
-> MemoryCandidate[]
-> safety policy
-> embedding related recall
-> LLM relation classifier
-> MemoryWritePlan
-> backend save/update/archive
-> if RAG enabled: update vector index
```

### 3.2 查询流程

```text
agent current task / user query
-> build_agent_context(query)
-> load pinned context: preference + project
-> retrieve on-demand context: episodic / reflective / tool / knowledge / general
-> merge + dedupe + score + token cap
-> format prompt context
```

### 3.3 RAG 使用场景

RAG 只在这些地方使用：

- on-demand memory retrieval
- knowledge/reference 类语义召回
- episodic/reflective/tool/general 的语义召回
- conflict/merge 前找相似旧记忆

RAG 不用于：

- 判断什么该记
- 替代正式存储后端
- 每轮 pinned preference/project 注入

## 4. 第一阶段执行计划

第一阶段目标：完成 domain model reset，不引入 LLM relation，不拆 backend。

### 4.1 修改 MemoryType

文件：

```text
memora/schema.py
tests/*
README.md
```

任务：

- 替换 `VALID_MEMORY_TYPES`
- 替换 `MemoryType` Literal
- 所有测试和示例改成新类型
- 不保留旧类型兼容

验收：

```bash
pytest
```

### 4.2 更新类型权重与生命周期

文件：

```text
memora/config.py
memora/retriever.py
memora/lifecycle.py
tests/test_retriever.py
tests/test_lifecycle.py
```

建议默认值：

```text
preference: weight 9, half-life 365
project: weight 8, half-life 180
episodic: weight 5, half-life 45
reflective: weight 7, half-life 180
tool: weight 6, half-life 120
knowledge: weight 6, half-life 365
general: weight 4, half-life 90
```

验收：

```bash
pytest tests/test_retriever.py tests/test_lifecycle.py
```

### 4.3 新增 context strategy

新增：

```text
memora/context.py
tests/test_context.py
```

核心常量：

```python
PINNED_CONTEXT_TYPES = ("preference", "project")
ON_DEMAND_CONTEXT_TYPES = ("episodic", "reflective", "tool", "knowledge", "general")
```

核心行为：

- pinned context 不靠 query。
- on-demand context 靠 query 检索。
- 两者合并去重。
- 保留 importance、recency、access 影响。
- 有 token cap。

### 4.4 新增 manager/runtime context API

文件：

```text
memora/manager.py
memora/runtime.py
tests/test_manager.py
tests/test_runtime.py
```

新增方法建议：

```python
get_pinned_context(...)
retrieve_on_demand_context(...)
build_agent_context(...)
```

保留旧 `retrieve_memory(query)`，避免第一阶段破坏太多现有入口。

### 4.5 CLI 增加 context 调试命令

文件：

```text
memora/cli.py
tests/test_cli.py
```

命令：

```bash
memora context "当前任务 query"
```

输出应展示 pinned + on-demand context。

### 4.6 文档更新

文件：

```text
README.md
docs/*
```

说明：

- 新 MemoryType
- pinned/on-demand 策略
- RAG 定位
- breaking change
- LLM extraction 是后续阶段

## 5. 第二阶段执行计划

目标：引入 LLM extraction contract。

新增：

```text
memora/extraction.py
tests/test_extraction.py
```

内容：

- `ExtractionArtifact`
- LLM extractor interface
- structured JSON schema
- validation
- extraction error recording
- CLI `extract --json`

不做：

- relation classifier
- merge/supersede execution
- trace recorder

## 6. 第三阶段执行计划

目标：引入 relation/conflict/merge pipeline。

新增：

```text
memora/relations.py
memora/write_plan.py
memora/merger.py
tests/test_relations.py
tests/test_write_plan.py
tests/test_merger.py
```

内容：

- embedding related recall
- LLM relation classifier JSON contract
- deterministic safe fallback
- `MemoryWritePlan`
- merge/update/noop/conflict/supersede 执行
- high-confidence auto supersede
- archive old + `supersedes`

## 7. 第四阶段执行计划

目标：后端与检索能力解耦。

内容：

- 明确 Markdown/SQLite 是 backend。
- RAG/vector 是 retrieval capability。
- Manager 不再把 RAG 当成独立存储。
- `MEMORY.md` 更名或文档定位为 catalog。
- 可选新增 Markdown `index.json` 作为机器索引。

## 8. 验证策略

每个阶段必须满足：

```bash
pytest
python -m build
```

关键 CLI 冒烟：

```bash
python -m memora --root .memora-test init
python -m memora --root .memora-test save --type preference --name response-language --description "用户偏好中文回答。" --content "用户偏好中文回答。"
python -m memora --root .memora-test save --type project --name tech-stack --description "项目使用 Python。" --content "项目使用 Python 编写。"
python -m memora --root .memora-test context "继续实现记忆系统"
```

预期：

- `preference` 和 `project` 稳定进入 context。
- on-demand 类型只有相关时进入。
- RAG 未开启时仍可工作。
- RAG 开启时只是增强召回，不改变正式存储语义。

## 9. Implementation Status: Typed Context Foundation

Implemented in the second refactor pass:

- `memora/taxonomy.py` is now the central source for memory types, default weights, half-life values, and pinned/on-demand context modes.
- `MemoryManager.retrieve_pinned_memories(...)` retrieves `preference` and `project` memories without relying on query text.
- `MemoryRuntime.retrieve_pinned_context(...)` and `MemoryRuntime.build_pinned_context(...)` expose pinned context to agent runtimes.
- `MemoryRuntime.retrieve_task_context(...)` and `MemoryRuntime.build_task_context(...)` combine pinned context with typed on-demand retrieval.
- Existing `retrieve_memory(...)`, `retrieve_context(...)`, and `build_context(...)` remain available as query-only compatibility APIs.
- RAG remains an optional retrieval enhancement and is not used for pinned context.

## 10. Implementation Status: LLM Extraction Contract

Implemented in the third refactor pass:

- `memora/extraction.py` defines `ExtractionArtifact`, `ExtractedMemory`, `LLMClient`, `MemoryExtractor`, `LLMMemoryExtractor`, JSON prompt construction, and JSON parsing.
- LLM extraction is optional and injected by the caller. If no extractor is configured, runtime extraction returns `memory_extractor_not_configured` and writes nothing.
- Extraction output is auditable: raw text, parsed memories, and parser errors are preserved in the artifact.
- Invalid JSON and invalid memory types do not write memories.
- Low-confidence extracted memories return `requires_confirmation` instead of being written automatically.
- `MemoryRuntime.extract_memories(...)`, `remember_extraction_artifact(...)`, and `extract_and_remember(...)` connect extraction to the existing deterministic write pipeline.
