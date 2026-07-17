# Memora

Memora is a deterministic local memory system for agent runtimes.

It provides:

- Markdown memory files with YAML frontmatter
- JSON session history
- Working memory state
- Deterministic safety policy
- Keyword retrieval and scoring
- Prompt formatting
- Lifecycle cleanup
- A thin CLI for debugging

## Install for development

```bash
pip install -e .[dev]
```

## Run tests

```bash
pytest
```

## CLI quickstart

```bash
python -m memora --root .memora init
python -m memora --root .memora save --type user --name language --description "用户偏好中文。" --content "用户偏好使用中文回答。"
python -m memora --root .memora list
python -m memora --root .memora search "中文回答"
python -m memora --root .memora show language
python -m memora --root .memora update language --tag language --weight 8
python -m memora --root .memora archive language
python -m memora --root .memora list --archived
python -m memora --root .memora restore language
python -m memora --root .memora search "中文回答" --type user --tag language --top-k 5
python -m memora --root .memora delete language
python -m memora --root .memora list --all
python -m memora --root .memora session append session_1 --role user --content "hello"
python -m memora --root .memora session show session_1
python -m memora --root .memora export memories.json
python -m memora --root .memora import memories.json
python -m memora --root .memora verify
python -m memora --root .memora rebuild-index
python -m memora --root .memora backup backup.json
python -m memora --root .memora clean
```

## CLI error behavior

Validation and policy failures are reported to stderr and return a non-zero exit code:

```bash
python -m memora --root .memora save --type user --name secret --description "secret" --content "api_key = sk-abcdef123456"
# stderr: error: memory rejected: contains_secret
```

## Data portability and integrity

Memora can export, import, verify, rebuild, and back up memory files:

```bash
python -m memora --root .memora export memories.json
python -m memora --root .memora import memories.json
python -m memora --root .memora verify
python -m memora --root .memora rebuild-index
python -m memora --root .memora backup backup.json
```

These commands cover memories only, not session history.

## Python usage

```python
from memora.manager import MemoryManager

manager = MemoryManager()
manager.init_storage()
manager.save_memory(
    memory_type="user",
    name="language",
    description="用户偏好中文。",
    content="用户偏好使用中文回答。",
)
results = manager.retrieve_memory("中文回答")
print(manager.format_memories_for_prompt(results=results))
```

## Runtime integration

External agent runtimes can use `MemoryRuntime` as a thin wrapper around the manager API:

```python
from memora.runtime import MemoryRuntime

runtime = MemoryRuntime()
runtime.init_storage()

context = runtime.build_context("用户偏好和当前项目")
runtime.remember_message("session_1", "user", "下一步做什么")
runtime.remember_message("session_1", "assistant", "建议做 runtime integration。")
runtime.remember_summary("session_1", "用户认可最简单的 runtime integration。")
```

## Agent runtime demo

Run the fake agent runtime example:

```bash
python examples/simple_agent_runtime.py
```

The demo uses `MemoryRuntime` with a local fake assistant response. It does not call an LLM.

## MVP boundaries

This version does not include LLM-based extraction, embeddings, vector databases, SQL backends, web UI, or hosted multi-tenant service.
