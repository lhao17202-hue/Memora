"""Interactive OpenAI-compatible Memora memory demo.

This demo keeps credentials out of source code. Configure the LLM provider and
Memora backend with environment variables or a local .env file, then choose one
of three operations:

1. extract: pass conversation messages plus optional working memory, then run
   the full extract -> relation judge -> policy -> write pipeline.
2. query: retrieve memory context for a question.
3. list: list the current memories in the configured store.

Run examples:
    python examples/openai_memory_demo.py extract --messages messages.json --working-memory working_memory.json
    python examples/openai_memory_demo.py query "我应该怎么回答用户？"
    python examples/openai_memory_demo.py list
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from openai_memory_clients import OpenAIExtractionClient, OpenAIRelationClient

from memora.config import MemoryConfig
from memora.env import apply_env_to_os, config_kwargs_from_env, load_env_file, merge_env
from memora.extraction import LLMMemoryExtractor
from memora.relations import LLMMemoryRelationJudge
from memora.runtime import MemoryRuntime
from memora.schema import WorkingMemoryState


DEFAULT_MODEL = "gpt-5.5"
DEFAULT_ROOT = ".memora-openai-demo"
DEFAULT_QDRANT_COLLECTION = "memora_openai_demo"
DEFAULT_MEMORY_TYPES = ["episodic", "reflective", "tool", "knowledge", "general"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a real LLM-backed Memora memory demo.")
    parser.add_argument("--env-file", default=".env", help="Path to local Memora environment config.")
    parser.add_argument("--user-id", default="demo-user", help="Memory user scope.")
    parser.add_argument("--project-id", default="memora-openai-demo", help="Memory project scope.")
    parser.add_argument("--workspace-id", default=None, help="Optional memory workspace scope.")
    parser.add_argument("--session-id", default="openai_memory_demo_session", help="Session ID used for extracted memories.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract", help="Extract and save memories from messages and working memory.")
    extract_parser.add_argument("--messages", required=True, help="JSON string or JSON file path containing a list of messages.")
    extract_parser.add_argument("--working-memory", help="JSON string or JSON file path containing a working memory object.")

    query_parser = subparsers.add_parser("query", help="Retrieve memories for a question.")
    query_parser.add_argument("question", help="Question/query to retrieve memory context for.")
    query_parser.add_argument("--type", action="append", dest="memory_types", help="On-demand memory type to retrieve; repeatable.")
    query_parser.add_argument("--tag", action="append", dest="tags", help="Exact-match tag filter; repeatable.")
    query_parser.add_argument("--top-k", type=int, default=8, help="Maximum on-demand memories to retrieve.")
    query_parser.add_argument("--no-pinned", action="store_true", help="Skip pinned preference/project context.")

    list_parser = subparsers.add_parser("list", help="List memories in the configured store.")
    list_parser.add_argument("--all", action="store_true", help="Include archived/deleted memories.")

    return parser


def make_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY before running this demo.")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Install the OpenAI Python SDK first: pip install openai") from exc

    kwargs: dict[str, str] = {"api_key": api_key}
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def make_runtime(env_file: str) -> MemoryRuntime:
    file_env = load_env_file(env_file) if env_file else {}
    env = merge_env(file_env)
    apply_env_to_os(env)
    kwargs = config_kwargs_from_env(env)
    kwargs.setdefault("root_dir", env.get("MEMORA_ROOT", DEFAULT_ROOT))
    kwargs.setdefault("memory_backend", "sqlite")
    kwargs.setdefault("rag_enabled", True)
    kwargs.setdefault("embedding_provider", "bge")
    kwargs.setdefault("embedding_model", "bge-m3")
    kwargs.setdefault("embedding_dimension", 1024)
    kwargs.setdefault("embedding_batch_size", 8)
    kwargs.setdefault("embedding_fp16", True)
    kwargs.setdefault("embedding_sparse", True)
    kwargs.setdefault("vector_store", "qdrant")
    kwargs.setdefault("retrieval_mode", "hybrid")
    kwargs.setdefault("keyword_recall", "auto")
    kwargs.setdefault("semantic_write_relations_enabled", True)
    kwargs.setdefault("llm_relation_judge_enabled", True)
    vector_options = dict(kwargs.get("vector_store_options") or {})
    vector_options.setdefault("url", env.get("MEMORA_VECTOR_STORE_URL", "http://localhost:6333"))
    vector_options.setdefault("collection", env.get("MEMORA_VECTOR_STORE_COLLECTION", DEFAULT_QDRANT_COLLECTION))
    kwargs["vector_store_options"] = vector_options

    client = make_openai_client()
    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    return MemoryRuntime(
        config=MemoryConfig(**kwargs),
        extractor=LLMMemoryExtractor(OpenAIExtractionClient(client, model)),
        relation_judge=LLMMemoryRelationJudge(OpenAIRelationClient(client, model)),
    )


def _load_json_value(value: str, field_name: str) -> Any:
    path = Path(value)
    if path.exists():
        raw_text = path.read_text(encoding="utf-8")
    else:
        raw_text = value
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{field_name} must be valid JSON or a path to a JSON file: {exc.msg}") from exc


def load_messages(value: str) -> list[dict[str, str]]:
    payload = _load_json_value(value, "--messages")
    if not isinstance(payload, list):
        raise SystemExit("--messages must decode to a JSON list.")
    messages = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise SystemExit(f"--messages[{index}] must be an object.")
        role = str(item.get("role") or "user")
        content = str(item.get("content") or "")
        if not content.strip():
            raise SystemExit(f"--messages[{index}].content must be non-empty.")
        messages.append({"role": role, "content": content})
    return messages


def load_working_memory(value: str | None) -> WorkingMemoryState | None:
    if value is None:
        return None
    payload = _load_json_value(value, "--working-memory")
    if not isinstance(payload, dict):
        raise SystemExit("--working-memory must decode to a JSON object.")
    allowed = {"task", "tool_notes", "recent_files", "file_summaries", "notes", "trace"}
    filtered = {key: payload[key] for key in allowed if key in payload}
    return WorkingMemoryState(**filtered)


def run_extract(runtime: MemoryRuntime, args: argparse.Namespace) -> int:
    messages = load_messages(args.messages)
    working_memory = load_working_memory(args.working_memory)

    runtime.init_storage()
    for message in messages:
        runtime.remember_message(args.session_id, message["role"], message["content"], user_id=args.user_id)

    artifact, results = runtime.extract_and_remember(
        messages,
        user_id=args.user_id,
        project_id=args.project_id,
        workspace_id=args.workspace_id,
        session_id=args.session_id,
        working_memory=working_memory,
    )

    print("=== Extraction Artifact ===")
    print(f"ok={artifact.ok} should_remember={artifact.should_remember} errors={artifact.errors}")
    for memory in artifact.memories:
        print(
            "candidate "
            f"type={memory.type} "
            f"name={memory.name} "
            f"confidence={memory.confidence} "
            f"requires_confirmation={memory.requires_confirmation} "
            f"tags={memory.tags}"
        )

    print("=== Write Results ===")
    for result in results:
        memory_name = result.memory.name if result.memory is not None else "-"
        memory_id = result.memory.id if result.memory is not None else "-"
        print(
            "write "
            f"action={result.action} "
            f"reason={result.reason} "
            f"memory={memory_name} "
            f"id={memory_id} "
            f"relation={result.relation_kind}:{result.relation_confidence} "
            f"judge={result.relation_judge_status}"
        )
    return 0


def run_query(runtime: MemoryRuntime, args: argparse.Namespace) -> int:
    runtime.init_storage()
    results = runtime.retrieve_task_context(
        args.question,
        user_id=args.user_id,
        project_id=args.project_id,
        workspace_id=args.workspace_id,
        memory_types=args.memory_types or DEFAULT_MEMORY_TYPES,
        tags=args.tags,
        top_k=args.top_k,
        include_pinned=not args.no_pinned,
    )
    print("=== Retrieved Memories ===")
    if not results:
        print("(no memory)")
        return 0
    for index, result in enumerate(results, start=1):
        memory = result.memory
        print(f"[{index}] {memory.id} {memory.type} {memory.name} score={result.final_score:.4f} reason={result.reason}")
        print(f"description: {memory.description}")
        print(f"tags: {memory.tags}")
        print(f"content: {memory.content}")
    return 0


def run_list(runtime: MemoryRuntime, args: argparse.Namespace) -> int:
    runtime.init_storage()
    memories = runtime.manager.list_memories(include_archived=args.all)
    print("=== Memories ===")
    if not memories:
        print("(no memory)")
        return 0
    for memory in memories:
        print(f"{memory.id} {memory.status} {memory.type} {memory.name}")
        print(f"description: {memory.description}")
        print(f"tags: {memory.tags}")
        print(f"content: {memory.content}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    runtime = make_runtime(args.env_file)
    if args.command == "extract":
        return run_extract(runtime, args)
    if args.command == "query":
        return run_query(runtime, args)
    if args.command == "list":
        return run_list(runtime, args)
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
