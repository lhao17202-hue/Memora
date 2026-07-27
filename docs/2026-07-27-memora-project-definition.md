# Memora 项目定义

日期：2026-07-27

本文按新的重构方向重新定义 Memora。它有意无视旧 README 的部分表述，以当前讨论后的目标架构为准。

## 1. Memora 是什么

Memora 是一个面向 agent 的长期记忆系统。

它的核心职责不是替 agent 思考，也不是替 agent 执行任务，而是：

```text
保存长期记忆
按记忆类型组织长期记忆
在 agent 需要时取回正确记忆
处理记忆的冲突、合并、替换和生命周期
把记忆格式化成可注入 prompt 的上下文
```

Memora 的目标是让 agent 不只依赖当前对话窗口，而能持续记住：

- 用户是谁
- 用户偏好什么
- 当前项目有什么长期约束
- 过去发生过哪些重要事件
- 之前踩过什么坑
- 哪些工具经验长期有效
- 哪些外部知识可按需检索

## 2. Memora 不是什么

Memora 不是完整 agent runtime。

它不负责：

- 执行 agent 主任务
- 采集完整 trace
- 保存完整工具调用日志
- 直接调用工具
- 替代聊天历史系统
- 替代文件系统或项目索引
- 让 LLM 直接写数据库

Agent 层可以有 session、trace、tool logs、message history。Memora 接收这些信息的摘要或 LLM 抽取结果，然后决定哪些内容应该成为长期记忆。

## 3. 记忆系统的两层

Memora 只重点管理长期记忆，但整体 agent memory 可以理解为两层：

```text
Short-term memory / session memory
Long-term memory
```

短期记忆：

- 当前会话内容
- 当前任务状态
- 工具调用日志
- 访问过的文件
- trace event

这部分主要属于 agent runtime 层。Memora 第一阶段不做完整管理。

长期记忆：

- 从 session/task/trace 中抽取出来
- 经过 safety、relation、merge、write plan
- 保存成 `MemoryItem`
- 后续可被 agent 检索和注入 prompt

这是 Memora 的核心领域。

## 4. 长期记忆类型

Memora 的长期记忆类型为：

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

### 4.1 Preference Memory

用户偏好、个人信息、输出习惯、自定义约束。

示例：

```text
用户偏好中文回答。
用户喜欢先给结论再给原因。
用户不喜欢过度设计。
用户希望代码审查时客观直接。
```

检索策略：

```text
每轮默认注入。
不依赖 query。
按 scope、weight、recency、access_count 选择。
```

### 4.2 Project Memory

项目需求、技术栈、架构设定、文件/目录/代码依赖。

示例：

```text
项目使用 Python 编写。
项目默认使用 SQLite 作为正式本地后端。
RAG/vector 不是存储后端，而是检索能力。
preference 和 project 记忆每轮默认注入。
```

检索策略：

```text
每轮默认注入。
按 user/project/workspace scope 过滤。
不完全依赖 query。
```

### 4.3 Episodic Memory

带时间线的交互事件和重要历史经历。

示例：

```text
2026-07-27，用户和 agent 讨论并确认 RAG 不作为存储后端。
某次重构中发现 README 中文示例出现真实乱码。
```

检索策略：

```text
按需检索。
当 query 涉及“之前”“上次”“历史过程”“当时为何这么做”时召回。
```

### 4.4 Reflective Memory

任务完成后的复盘、失败总结、优化思路。

示例：

```text
重构前应先建立行为基线，否则容易把功能变化伪装成重构。
policy 规则不能只靠关键词，否则会误杀 token_budget 和 max_tokens 这类非凭据配置。
```

检索策略：

```text
按需检索。
调试、修复、复盘、类似错误场景下召回。
```

### 4.5 Tool Memory

从工具调用和 trace 中总结出的长期工具经验。

它不是完整工具日志。

示例：

```text
在 Windows PowerShell 中，不要使用 Bash heredoc；应使用 here-string 后管道给 python。
PowerShell 下递归删除临时目录会触发安全策略时，应避免在冒烟脚本里做删除。
```

检索策略：

```text
按需检索。
当 agent 准备使用相关工具、shell、MCP、Skill 时召回。
```

### 4.6 Knowledge Memory

外部导入的离线技术资料、参考知识、稳定文档摘要。

示例：

```text
某库官方文档中的 API 使用限制。
某框架迁移指南中的关键兼容说明。
```

检索策略：

```text
按需检索。
最适合 RAG / semantic recall。
```

### 4.7 General Memory

LLM 判断重要但不属于以上类型的长期记忆。

示例：

```text
某个长期协作习惯。
某个暂时难以归类但未来可能有用的稳定事实。
```

检索策略：

```text
低优先级。
按需召回或少量兜底。
```

## 5. MemoryItem

`MemoryItem` 是 Memora 的正式长期记忆单位。

它包含：

- id
- name
- description
- type
- content
- scope
- tags
- source
- confidence
- weight
- status
- created_at / updated_at
- access statistics
- supersedes / related

`MemoryItem` 只有经过正式写入流程后才进入长期记忆库。

LLM 原始输出不是 `MemoryItem`。它应先成为 `ExtractionArtifact` 或 `MemoryCandidate`。

## 6. MemoryCandidate

`MemoryCandidate` 是候选记忆。

来源包括：

- LLM 从 session/task/trace 中抽取
- 上层 agent 手动传入
- CLI 调试输入

候选记忆必须经过：

```text
safety policy
related recall
relation classification
write plan
```

才能变成正式 `MemoryItem`。

## 7. ExtractionArtifact

`ExtractionArtifact` 是 LLM 抽取过程的审计记录。

它保存：

- 使用的 prompt
- 输入 session/trace 摘要
- LLM 原始 JSON 输出
- 解析后的 candidates
- errors
- created_at

它不是长期记忆，不参与默认检索。

它的作用是：

- 调试
- 审计
- 回放
- 解释为什么产生某条 candidate

## 8. 存储后端

Memora 的正式存储后端是：

```text
Markdown
SQLite
```

### 8.1 Markdown backend

适合：

- 小规模记忆
- 人类可读
- Git diff
- 本地开发
- 透明调试

保存形式：

```text
.memora/memories/<user>/<project>/<workspace>/<name>.md
```

`MEMORY.md` 是人类可读 catalog，不是检索索引。

### 8.2 SQLite backend

适合：

- 默认正式本地后端
- 中小规模长期使用
- 需要 FTS
- 需要事务和稳定过滤

保存形式：

```text
memories table
memory_fts table
```

`memories` 是正式数据表，`memory_fts` 是关键词候选召回索引。

## 9. RAG / Vector 的定位

RAG/vector 不是存储后端。

它是可选检索能力。

用途：

- on-demand semantic retrieval
- knowledge memory 检索
- episodic/reflective/tool/general 的语义召回
- conflict/merge 前找 related memories

不用 RAG 的系统仍然可以工作。

启用 RAG 后：

```text
MemoryItem 保存到 Markdown 或 SQLite
-> vector index 同步 embedding
-> 查询时使用 hybrid retrieval
```

RAG 索引可以重建，不是唯一数据来源。

## 10. Agent 上下文构建

Memora 不应该只通过一个 query 搜所有记忆。

新的上下文构建流程：

```text
build_agent_context(query)
-> load pinned context
-> retrieve on-demand context
-> merge / dedupe / score
-> format for prompt
```

Pinned context：

```text
preference
project
```

On-demand context：

```text
episodic
reflective
tool
knowledge
general
```

这样 agent 每轮都能稳定获得用户偏好和项目约束，同时避免把所有历史事件、工具经验、知识库内容塞进 prompt。

## 11. LLM 抽取

自动长期记忆抽取固定使用 LLM。

抽取时机：

```text
session end / task end
```

不是每轮消息后实时抽取。

LLM extractor 输入：

- session messages
- optional trace summary
- optional tool summary

LLM extractor 输出必须是结构化 JSON。

示例：

```json
{
  "memories": [
    {
      "type": "preference",
      "name": "response-language",
      "description": "用户偏好中文回答。",
      "content": "用户偏好使用中文回答技术问题。",
      "tags": ["language", "response-style"],
      "confidence": 0.92,
      "evidence": ["还是中文聊吧，看着轻松点"]
    }
  ]
}
```

系统严格 validate。

字段非法、JSON 解析失败、type 不合法，都应记录 extraction error。

## 12. 冲突、合并与替换

Memora 需要解决三类问题：

1. 重复记忆
2. 旧事实失效
3. 新候选补充旧记忆

流程：

```text
candidate
-> embedding related recall
-> LLM relation classifier
-> MemoryWritePlan
-> execute
```

Relation action：

```text
add
noop
update
merge
conflict
supersede
```

LLM relation classifier 必须输出结构化 JSON。

高置信 conflict/supersede 可以自动替换，但必须满足护栏：

- confidence 达标
- 语句有明确替代表达
- type 允许自动替换
- target scope/type 合法

自动替换时：

```text
archive old
write new or update target
new.supersedes = [old.id]
old 从默认检索和 vector index 中移除
```

不 hard delete。

## 13. Safety policy

即使使用 LLM，Memora 仍然必须保留 deterministic safety policy。

LLM 不允许绕过：

- secret 检测
- noisy output 检测
- transient task state 检测
- scope 校验
- target id 校验
- stale confirmation 校验

LLM 负责建议，Memora 负责执行。

## 14. 推荐默认模式

开发透明模式：

```text
backend = markdown
retrieval = lexical
```

正式本地模式：

```text
backend = sqlite
retrieval = fts + lexical
```

语义增强模式：

```text
backend = sqlite
retrieval = fts + lexical + rag
embedding = enabled
```

高质量 conflict/merge 模式：

```text
embedding related recall = required
llm relation classifier = enabled
```

## 15. Memora 的最终一句话定义

Memora 是一个 LLM-native、type-aware、local-first 的 agent 长期记忆系统。

它用 LLM 从 session/task/trace 中抽取长期记忆，用 Markdown 或 SQLite 保存正式 `MemoryItem`，用类型策略决定哪些记忆每轮注入、哪些按需召回，并用 embedding + LLM relation classifier 处理重复、冲突、合并和替换。

## Implementation Note: Runtime Context APIs

Current runtime-facing context APIs:

- `MemoryRuntime.build_pinned_context(...)`: formats stable pinned memories (`preference`, `project`) without requiring query text.
- `MemoryRuntime.build_task_context(query, memory_types=[...])`: formats pinned context first, then appends typed on-demand retrieval results.
- `MemoryRuntime.build_context(query)`: remains the compatibility path for query-only retrieval.

RAG/vector search can enhance typed on-demand retrieval when enabled. Pinned context is loaded from the selected backend by scope and policy, not by semantic vector search.
