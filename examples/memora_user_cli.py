r"""User-facing Memora CLI demo for real memory workflow testing.

This script is intentionally a demo wrapper around Memora's runtime APIs. It
uses an OpenAI-compatible LLM for extraction/relation judging, BGE for local
embeddings, Qdrant for the vector index, and SQLite as the source-of-truth
memory backend.

Secrets are read from environment variables only. Do not put API keys in this
file or in committed dataset files.

CMD examples:
    python examples\memora_user_cli.py /cli extract --messages examples\demo_dataset\messages\sample_turn.json --working-memory examples\demo_dataset\working_memory\sample_turn.json
    python examples\memora_user_cli.py /cli query "用户偏好什么回答风格？"
    python examples\memora_user_cli.py /cli list
    python examples\memora_user_cli.py /cli clear --yes
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from openai_memory_clients import OpenAIExtractionClient, OpenAIRelationClient

from memora.config import MemoryConfig
from memora.env import apply_env_to_os, config_kwargs_from_env, load_env_file, merge_env
from memora.extraction import LLMMemoryExtractor
from memora.relations import LLMMemoryRelationJudge
from memora.runtime import MemoryRuntime
from memora.schema import WorkingMemoryState


DEFAULT_ENV_FILE = ".memora-user-cli.env"
DEFAULT_MODEL = "gpt-5.5"
DEFAULT_ROOT = ".memora-user-cli"
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_QDRANT_COLLECTION = "memora_user_cli"
DEFAULT_BGE_MODEL_PATH = r"C:\Download\bge-m3"
DEFAULT_USER_ID = "demo-user"
DEFAULT_PROJECT_ID = "memora-user-cli-demo"
DEFAULT_SESSION_ID = "memora_user_cli_session"
DEFAULT_ON_DEMAND_TYPES = ["episodic", "reflective", "tool", "knowledge", "general"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Memora user CLI demo. First argument must be /cli.")
    parser.add_argument("entrypoint", help="Demo entrypoint marker: /cli.")
    parser.add_argument("--env-file", default=DEFAULT_ENV_FILE, help="Optional git-ignored env file path.")
    parser.add_argument("--user-id", default=DEFAULT_USER_ID, help="Memory user scope.")
    parser.add_argument("--project-id", default=DEFAULT_PROJECT_ID, help="Memory project scope.")
    parser.add_argument("--workspace-id", default=None, help="Optional memory workspace scope.")
    parser.add_argument("--session-id", default=DEFAULT_SESSION_ID, help="Session ID used when extracting memories.")

    commands = parser.add_subparsers(dest="command", required=True)

    extract_parser = commands.add_parser("extract", help="Extract and save memories from messages + working memory.")
    extract_parser.add_argument("--messages", required=True, help="JSON file path or JSON string containing message list.")
    extract_parser.add_argument("--working-memory", required=True, help="JSON file path or JSON string containing working memory object.")

    query_parser = commands.add_parser("query", help="Retrieve the most relevant memory snippets for a question.")
    query_parser.add_argument("question", help="Question used to retrieve memories.")
    query_parser.add_argument("--type", action="append", dest="memory_types", help="On-demand memory type filter; repeatable.")
    query_parser.add_argument("--tag", action="append", dest="tags", help="Exact tag filter; repeatable.")
    query_parser.add_argument("--top-k", type=int, default=8, help="Maximum on-demand memory count.")
    query_parser.add_argument("--no-pinned", action="store_true", help="Skip pinned preference/project memories.")

    commands.add_parser("list", help="List all current memories.")

    clear_parser = commands.add_parser("clear", help="Hard-delete all current memories in this demo scope.")
    clear_parser.add_argument("--yes", action="store_true", help="Required confirmation for hard deletion.")

    return parser


def _load_demo_env(env_file: str) -> dict[str, str]:
    file_env = load_env_file(env_file) if env_file else {}
    env = merge_env(file_env)
    apply_env_to_os(env)
    for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"):
        if key in env and key not in os.environ:
            os.environ[key] = env[key]
    return env


def make_openai_client(env: Mapping[str, str] | None = None):
    source = env or os.environ
    api_key = source.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY before running extract with this demo.")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Install the OpenAI Python SDK first: pip install openai") from exc

    kwargs: dict[str, str] = {"api_key": api_key}
    base_url = source.get("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _base_config_kwargs(env_file: str, *, with_llm_relation_judge: bool) -> dict[str, object]:
    env = _load_demo_env(env_file)
    kwargs = config_kwargs_from_env(env)

    kwargs.setdefault("root_dir", env.get("MEMORA_ROOT", DEFAULT_ROOT))
    kwargs.setdefault("memory_backend", "sqlite")
    kwargs.setdefault("rag_enabled", True)
    kwargs.setdefault("embedding_provider", "bge")
    kwargs.setdefault("embedding_model", "bge-m3")
    kwargs.setdefault("embedding_model_path", env.get("MEMORA_EMBEDDING_MODEL_PATH", DEFAULT_BGE_MODEL_PATH))
    kwargs.setdefault("embedding_dimension", 1024)
    kwargs.setdefault("embedding_batch_size", 8)
    kwargs.setdefault("embedding_fp16", True)
    kwargs.setdefault("embedding_sparse", True)
    kwargs.setdefault("retrieval_mode", "hybrid")
    kwargs.setdefault("keyword_recall", "auto")
    kwargs.setdefault("vector_store", "qdrant")
    kwargs.setdefault("semantic_write_relations_enabled", True)
    kwargs["llm_relation_judge_enabled"] = with_llm_relation_judge
    kwargs["allow_auto_save_project_facts"] = True

    vector_options = dict(kwargs.get("vector_store_options") or {})
    vector_options.setdefault("url", env.get("MEMORA_VECTOR_STORE_URL", DEFAULT_QDRANT_URL))
    vector_options.setdefault("collection", env.get("MEMORA_VECTOR_STORE_COLLECTION", DEFAULT_QDRANT_COLLECTION))
    vector_options.setdefault("timeout", float(env.get("MEMORA_VECTOR_STORE_TIMEOUT", "30")))
    kwargs["vector_store_options"] = vector_options
    return kwargs


def make_runtime(env_file: str, *, with_llm: bool) -> MemoryRuntime:
    env = _load_demo_env(env_file)
    client = make_openai_client(env) if with_llm else None
    model = env.get("OPENAI_MODEL", DEFAULT_MODEL)
    config = MemoryConfig(**_base_config_kwargs(env_file, with_llm_relation_judge=with_llm))
    return MemoryRuntime(
        config=config,
        extractor=LLMMemoryExtractor(OpenAIExtractionClient(client, model)) if client is not None else None,
        relation_judge=LLMMemoryRelationJudge(OpenAIRelationClient(client, model)) if client is not None else None,
    )


def _load_json_value(value: str, field_name: str) -> Any:
    path = Path(value)
    raw_text = path.read_text(encoding="utf-8") if path.exists() else value
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


def load_working_memory(value: str) -> WorkingMemoryState:
    payload = _load_json_value(value, "--working-memory")
    if not isinstance(payload, dict):
        raise SystemExit("--working-memory must decode to a JSON object.")
    allowed = {"task", "tool_notes", "recent_files", "file_summaries", "notes", "trace"}
    filtered = {key: payload[key] for key in allowed if key in payload}
    return WorkingMemoryState(**filtered)


def run_extract(args: argparse.Namespace) -> int:
    runtime = make_runtime(args.env_file, with_llm=True)
    runtime.init_storage()
    messages = load_messages(args.messages)
    working_memory = load_working_memory(args.working_memory)

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

    print("=== Extracted Candidates ===")
    print(f"ok={artifact.ok} should_remember={artifact.should_remember} errors={artifact.errors}")
    for memory in artifact.memories:
        print(f"- {memory.type} {memory.name} confidence={memory.confidence} tags={memory.tags}")
        print(f"  {memory.content}")

    print("=== Write Results ===")
    for result in results:
        memory_id = result.memory.id if result.memory is not None else "-"
        memory_name = result.memory.name if result.memory is not None else "-"
        print(f"- action={result.action} reason={result.reason} id={memory_id} name={memory_name}")
    return 0


def _format_optional_score(value: float | None) -> str:
    return "None" if value is None else f"{value:.4f}"


def _format_datetime(value: Any) -> str:
    return "None" if value is None else value.isoformat()


def run_query(args: argparse.Namespace) -> int:
    runtime = make_runtime(args.env_file, with_llm=False)
    runtime.init_storage()
    results = runtime.retrieve_task_context(
        args.question,
        user_id=args.user_id,
        project_id=args.project_id,
        workspace_id=args.workspace_id,
        memory_types=args.memory_types or DEFAULT_ON_DEMAND_TYPES,
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
        print(
            "scores: "
            f"final={result.final_score:.4f} "
            f"similarity={result.similarity_score:.4f} "
            f"semantic={result.semantic_score:.4f} "
            f"keyword={result.keyword_score:.4f} "
            f"importance={result.importance_score:.4f} "
            f"recency={result.recency_score:.4f} "
            f"access={result.access_score:.4f} "
            f"rerank={_format_optional_score(result.rerank_score)}"
        )
        print(
            "metadata: "
            f"user_id={memory.user_id} "
            f"project_id={memory.project_id} "
            f"workspace_id={memory.workspace_id} "
            f"source={memory.source} "
            f"confidence={memory.confidence} "
            f"weight={memory.weight} "
            f"status={memory.status} "
            f"access_count={memory.access_count}"
        )
        print(
            "timestamps: "
            f"created_at={_format_datetime(memory.created_at)} "
            f"updated_at={_format_datetime(memory.updated_at)} "
            f"last_accessed_at={_format_datetime(memory.last_accessed_at)}"
        )
        print(f"relations: supersedes={memory.supersedes} related={memory.related}")
        print(f"description: {memory.description}")
        print(f"tags: {memory.tags}")
        print(f"content: {memory.content}")
    return 0


def run_list(args: argparse.Namespace) -> int:
    runtime = make_runtime(args.env_file, with_llm=False)
    runtime.init_storage()
    memories = runtime.manager.list_memories(include_archived=True)
    print("=== All Memories ===")
    if not memories:
        print("(no memory)")
        return 0
    for memory in memories:
        print(f"{memory.id} {memory.status} {memory.type} {memory.name}")
        print(f"description: {memory.description}")
        print(f"tags: {memory.tags}")
        print(f"content: {memory.content}")
    return 0


def run_clear(args: argparse.Namespace) -> int:
    if not args.yes:
        raise SystemExit("clear hard-deletes memories in this demo scope; rerun with --yes to confirm.")
    runtime = make_runtime(args.env_file, with_llm=False)
    runtime.init_storage()
    memories = runtime.manager.list_memories(include_archived=True)
    for memory in memories:
        runtime.manager.delete_memory(memory.id, hard=True)
    print(f"cleared {len(memories)} memories")
    return 0


def _is_cli_entrypoint(value: str) -> bool:
    normalized = value.replace("\\", "/").lower()
    return value == "/cli" or normalized.endswith("/cli")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not _is_cli_entrypoint(args.entrypoint):
        parser.error("first argument must be /cli")
    if args.command == "extract":
        return run_extract(args)
    if args.command == "query":
        return run_query(args)
    if args.command == "list":
        return run_list(args)
    if args.command == "clear":
        return run_clear(args)
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
