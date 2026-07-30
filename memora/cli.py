"""Command-line interface for Memora."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime

from .config import MemoryConfig
from .embeddings import EMBEDDING_PROVIDER_CHOICES
from .env import apply_env_to_os, config_kwargs_from_env, load_env_file, merge_env
from .errors import MemoraError, MemoryValidationError
from .manager import MemoryManager
from .reranker import RERANKER_CHOICES
from .runtime import MemoryRuntime
from .schema import MemoryCandidate, SessionMessage
from .vector_store import VECTOR_STORE_CHOICES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memora",
        description="Memora deterministic local memory system.",
    )
    parser.add_argument("--root", default=None, help="Memora runtime root directory.")
    parser.add_argument("--env-file", default=".env", help="Path to a Memora .env file for local config.")
    parser.add_argument("--backend", choices=("file", "sqlite"), default=None, help="Memory storage backend.")
    parser.add_argument("--sqlite-path", help="SQLite database path when --backend sqlite is used.")
    parser.add_argument("--no-fts", action="store_true", default=None, help="Disable SQLite FTS candidate recall.")
    parser.add_argument("--rag", action="store_true", default=None, help="Enable deterministic local RAG retrieval and vector indexing.")
    parser.add_argument("--embedding-provider", default=None, choices=EMBEDDING_PROVIDER_CHOICES, help="Embedding provider for RAG.")
    parser.add_argument("--embedding-model", default=None, help="Embedding model label for RAG metadata.")
    parser.add_argument("--embedding-dimension", type=int, default=None, help="Embedding vector dimension.")
    parser.add_argument("--embedding-model-path", default=None, help="Local embedding model path for providers such as bge.")
    parser.add_argument("--embedding-batch-size", type=int, default=None, help="Embedding batch size for local providers.")
    parser.add_argument("--embedding-fp16", action="store_true", default=None, help="Use fp16 for local embedding providers when supported.")
    parser.add_argument("--embedding-sparse", action="store_true", default=None, help="Request sparse embeddings from providers that support them.")
    parser.add_argument("--retrieval-mode", choices=("dense", "hybrid"), default=None, help="Vector retrieval mode for RAG.")
    parser.add_argument("--hybrid-prefetch-limit", type=int, default=None, help="Per-channel prefetch limit for hybrid vector retrieval.")
    parser.add_argument("--qdrant-url", default=None, help="Qdrant URL, for example http://localhost:6333.")
    parser.add_argument("--qdrant-host", default=None, help="Qdrant host when --qdrant-url is not set.")
    parser.add_argument("--qdrant-port", type=int, default=None, help="Qdrant port when --qdrant-url is not set.")
    parser.add_argument("--qdrant-api-key", default=None, help="Qdrant API key, if required.")
    parser.add_argument("--qdrant-collection", default=None, help="Qdrant collection name for Memora vectors.")
    parser.add_argument("--qdrant-timeout", type=float, default=None, help="Qdrant client timeout in seconds.")
    parser.add_argument("--qdrant-prefer-grpc", action="store_true", default=None, help="Prefer gRPC for Qdrant client calls.")
    parser.add_argument("--qdrant-recreate-collection", action="store_true", default=None, help="Recreate the Qdrant collection during init; deletes indexed vector data.")
    parser.add_argument("--vector-store", default=None, choices=VECTOR_STORE_CHOICES, help="Vector store for RAG.")
    parser.add_argument("--reranker", default=None, choices=RERANKER_CHOICES, help="Reranker for RAG.")
    parser.add_argument("--semantic-write-relations", action="store_true", help="Use embeddings to detect write-time duplicate, merge, and conflict relations.")
    parser.add_argument("--semantic-relation-threshold", type=float, default=0.78, help="Minimum similarity for write-time semantic relation detection.")
    parser.add_argument("--semantic-merge-threshold", type=float, default=0.82, help="Minimum similarity for write-time semantic merge/update decisions.")
    parser.add_argument("--semantic-conflict-threshold", type=float, default=0.90, help="Minimum similarity before conflict evidence can affect write decisions.")
    parser.add_argument("--no-high-confidence-conflict-replace", action="store_true", help="Disable automatic replacement for high-confidence semantic conflicts.")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("init", help="Initialize a Memora runtime directory.")

    save_parser = subparsers.add_parser("save", help="Save a memory.")
    save_parser.add_argument("--type", required=True)
    save_parser.add_argument("--name", required=True)
    save_parser.add_argument("--description", required=True)
    save_parser.add_argument("--content", required=True)

    remember_parser = subparsers.add_parser("remember", help="Evaluate and write an agent-extracted candidate memory.")
    remember_parser.add_argument("--type", required=True)
    remember_parser.add_argument("--name", required=True)
    remember_parser.add_argument("--description", required=True)
    remember_parser.add_argument("--content", required=True)
    remember_parser.add_argument("--source")
    remember_parser.add_argument("--session", dest="session_id")
    remember_parser.add_argument("--tag", action="append", dest="tags")
    remember_parser.add_argument("--weight", type=int)
    remember_parser.add_argument("--confidence", type=float, default=1.0)
    remember_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")

    confirm_parser = subparsers.add_parser("confirm", help="Confirm a pending candidate JSON file.")
    confirm_parser.add_argument("--candidate", required=True, help="Path to a pending candidate JSON file or remember --json result.")
    confirm_parser.add_argument("--action", choices=("create", "update", "supersede"), help="Override the candidate suggested action.")
    confirm_parser.add_argument("--target", dest="target_memory_id", help="Override the candidate target memory ID.")
    confirm_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")

    list_parser = subparsers.add_parser("list", help="List memories.")
    list_parser.add_argument("--archived", action="store_true", help="List archived memories only.")
    list_parser.add_argument("--all", action="store_true", help="List active, archived, and deleted memories.")

    show_parser = subparsers.add_parser("show", help="Show one memory.")
    show_parser.add_argument("identifier")

    update_parser = subparsers.add_parser("update", help="Update one memory.")
    update_parser.add_argument("identifier")
    update_parser.add_argument("--description")
    update_parser.add_argument("--content")
    update_parser.add_argument("--tag", action="append", dest="tags")
    update_parser.add_argument("--weight", type=int)
    update_parser.add_argument("--confidence", type=float)

    archive_parser = subparsers.add_parser("archive", help="Archive one memory.")
    archive_parser.add_argument("identifier")

    restore_parser = subparsers.add_parser("restore", help="Restore one archived or deleted memory.")
    restore_parser.add_argument("identifier")

    delete_parser = subparsers.add_parser("delete", help="Delete one memory.")
    delete_parser.add_argument("identifier")
    delete_parser.add_argument("--hard", action="store_true")

    export_parser = subparsers.add_parser("export", help="Export memories to JSON.")
    export_parser.add_argument("path")

    import_parser = subparsers.add_parser("import", help="Import memories from JSON.")
    import_parser.add_argument("path")

    subparsers.add_parser("verify", help="Verify memory store health.")
    subparsers.add_parser("rebuild-index", help="Rebuild the memory index.")

    backup_parser = subparsers.add_parser("backup", help="Back up memories to JSON.")
    backup_parser.add_argument("path")

    search_parser = subparsers.add_parser("search", help="Search memories.")
    search_parser.add_argument("query")
    search_parser.add_argument("--type", action="append", dest="memory_types")
    search_parser.add_argument("--tag", action="append", dest="tags")
    search_parser.add_argument("--top-k", type=int)
    search_parser.add_argument("--archived", action="store_true", help="Include archived memories.")

    context_parser = subparsers.add_parser("context", help="Build typed agent memory context.")
    context_parser.add_argument("query")
    context_parser.add_argument("--type", action="append", dest="memory_types", help="On-demand memory type to retrieve.")
    context_parser.add_argument("--tag", action="append", dest="tags")
    context_parser.add_argument("--top-k", type=int, help="Maximum on-demand memories.")
    context_parser.add_argument("--pinned-top-k", type=int, help="Maximum pinned memories.")
    context_parser.add_argument("--no-pinned", action="store_true", help="Skip pinned preference/project context.")
    context_parser.add_argument("--archived", action="store_true", help="Include archived memories.")
    context_parser.add_argument("--no-knowledge", action="store_true", help="Exclude knowledge memories.")
    context_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")

    subparsers.add_parser("clean", help="Archive expired or cold memories.")

    session_parser = subparsers.add_parser("session", help="Manage sessions.")
    session_subparsers = session_parser.add_subparsers(dest="session_command")

    append_parser = session_subparsers.add_parser("append", help="Append a message to a session.")
    append_parser.add_argument("session_id")
    append_parser.add_argument("--role", required=True)
    append_parser.add_argument("--content", required=True)

    show_session_parser = session_subparsers.add_parser("show", help="Show a session.")
    show_session_parser.add_argument("session_id")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config_kwargs = _config_kwargs_from_args(args)
        manager = MemoryManager(MemoryConfig(**config_kwargs))
        return _run_command(args, manager, parser)
    except MemoraError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _config_kwargs_from_args(args) -> dict:
    file_env = load_env_file(args.env_file) if args.env_file else {}
    env = merge_env(file_env)
    apply_env_to_os(env)
    kwargs = config_kwargs_from_env(env)
    cli_values = {
        "root_dir": args.root,
        "memory_backend": args.backend,
        "sqlite_path": args.sqlite_path,
        "embedding_provider": args.embedding_provider,
        "embedding_model": args.embedding_model,
        "embedding_dimension": args.embedding_dimension,
        "embedding_model_path": args.embedding_model_path,
        "embedding_batch_size": args.embedding_batch_size,
        "retrieval_mode": args.retrieval_mode,
        "hybrid_prefetch_limit": args.hybrid_prefetch_limit,
        "qdrant_url": args.qdrant_url,
        "qdrant_host": args.qdrant_host,
        "qdrant_port": args.qdrant_port,
        "qdrant_api_key": args.qdrant_api_key,
        "qdrant_collection": args.qdrant_collection,
        "qdrant_timeout": args.qdrant_timeout,
        "vector_store": args.vector_store,
        "reranker": args.reranker,
        "semantic_relation_threshold": args.semantic_relation_threshold,
        "semantic_merge_threshold": args.semantic_merge_threshold,
        "semantic_conflict_threshold": args.semantic_conflict_threshold,
    }
    for key, value in cli_values.items():
        if value is not None:
            kwargs[key] = value
    if args.no_fts is not None:
        kwargs["fts_enabled"] = not args.no_fts
    if args.rag:
        kwargs["rag_enabled"] = True
    if args.embedding_fp16:
        kwargs["embedding_fp16"] = True
    if args.embedding_sparse:
        kwargs["embedding_sparse"] = True
    if args.qdrant_prefer_grpc:
        kwargs["qdrant_prefer_grpc"] = True
    if args.qdrant_recreate_collection:
        kwargs["qdrant_recreate_collection"] = True
    if args.semantic_write_relations:
        kwargs["semantic_write_relations_enabled"] = True
    if args.no_high_confidence_conflict_replace:
        kwargs["allow_high_confidence_conflict_replace"] = False
    kwargs.setdefault("root_dir", ".memora")
    kwargs.setdefault("memory_backend", "file")
    kwargs.setdefault("embedding_provider", "hash")
    kwargs.setdefault("vector_store", "sqlite")
    kwargs.setdefault("retrieval_mode", "dense")
    kwargs.setdefault("reranker", "deterministic")
    return kwargs


def _json_default(value):
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_result_to_dict(result) -> dict:
    return {
        "action": result.action,
        "reason": result.reason,
        "target_memory_id": result.target_memory_id,
        "relation_kind": result.relation_kind,
        "relation_confidence": result.relation_confidence,
        "relation_reason": result.relation_reason,
        "relation_judge_status": result.relation_judge_status,
        "relation_judge_error": result.relation_judge_error,
        "rag_sync_errors": result.rag_sync_errors,
        "memory": asdict(result.memory) if result.memory is not None else None,
        "candidate": asdict(result.candidate) if result.candidate is not None else None,
    }


def _search_result_to_dict(result) -> dict:
    return {
        "memory": asdict(result.memory),
        "similarity_score": result.similarity_score,
        "importance_score": result.importance_score,
        "recency_score": result.recency_score,
        "access_score": result.access_score,
        "final_score": result.final_score,
        "reason": result.reason,
        "semantic_score": result.semantic_score,
        "keyword_score": result.keyword_score,
        "rerank_score": result.rerank_score,
    }


def _candidate_from_dict(data: dict) -> MemoryCandidate:
    candidate_data = data.get("candidate", data)
    if not isinstance(candidate_data, dict):
        raise ValueError("candidate JSON must contain an object")
    allowed_fields = MemoryCandidate.__dataclass_fields__.keys()
    values = {key: value for key, value in candidate_data.items() if key in allowed_fields}
    if isinstance(values.get("target_updated_at"), str):
        values["target_updated_at"] = datetime.fromisoformat(values["target_updated_at"])
    return MemoryCandidate(**values)


def _load_candidate_json(path: str) -> dict:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-16"):
        try:
            with open(path, encoding=encoding) as file:
                payload = json.load(file)
        except (UnicodeError, json.JSONDecodeError) as exc:
            last_error = exc
            continue
        if not isinstance(payload, dict):
            raise ValueError("candidate JSON must contain an object")
        return payload
    if last_error is not None:
        raise last_error
    raise ValueError("candidate JSON must contain an object")


def _print_write_result(result, json_output: bool = False) -> None:
    if json_output:
        print(json.dumps(_write_result_to_dict(result), ensure_ascii=False, default=_json_default))
        return
    if result.action in {"created", "updated"} and result.memory is not None:
        print(f"{result.action} {result.memory.id} {result.memory.name} {result.reason}")
        return
    if result.action == "requires_confirmation":
        print(f"requires_confirmation {result.target_memory_id} {result.reason}")
        return
    print(f"{result.action} {result.reason}")


def _run_command(args, manager: MemoryManager, parser: argparse.ArgumentParser) -> int:
    if args.command == "init":
        manager.init_storage()
        print(f"initialized {args.root}")
        return 0

    if args.command == "save":
        item = manager.save_memory(args.type, args.content, args.description, name=args.name)
        print(f"saved {item.id} {item.name}")
        return 0

    if args.command == "remember":
        tags = list(args.tags or [])
        source = args.source
        if args.session_id is not None:
            session_tag = f"session:{args.session_id}"
            if session_tag not in tags:
                tags.append(session_tag)
            if source is None:
                source = "session_extraction"
        if source is None:
            source = "runtime_extraction"
        candidate = MemoryCandidate(
            action="create",
            name=args.name,
            description=args.description,
            type=args.type,
            content=args.content,
            tags=tags,
            source=source,
            weight=args.weight,
            confidence=args.confidence,
        )
        result = manager.remember_candidate(candidate)
        _print_write_result(result, json_output=args.json)
        return 0

    if args.command == "confirm":
        try:
            payload = _load_candidate_json(args.candidate)
            candidate = _candidate_from_dict(payload)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise MemoryValidationError(f"invalid candidate JSON: {exc}") from exc
        result = manager.confirm_memory_candidate(
            candidate,
            action=args.action,
            target_memory_id=args.target_memory_id,
        )
        _print_write_result(result, json_output=args.json)
        return 0

    if args.command == "list":
        items = manager.list_memories(include_archived=args.archived or args.all)
        if args.archived and not args.all:
            items = [item for item in items if item.status == "archived"]
        if not args.all:
            items = [item for item in items if item.status == "active" or (args.archived and item.status == "archived")]
        for item in items:
            print(f"{item.id}\t{item.name}\t{item.type}\t{item.status}\t{item.description}")
        return 0

    if args.command == "show":
        item = manager.get_memory(args.identifier)
        if item is None:
            print("memory not found")
            return 1
        print(f"id: {item.id}")
        print(f"name: {item.name}")
        print(f"type: {item.type}")
        print(f"status: {item.status}")
        print(f"source: {item.source}")
        print(f"weight: {item.weight}")
        print(f"tags: {', '.join(item.tags)}")
        print(f"description: {item.description}")
        print(item.content)
        return 0

    if args.command == "update":
        item = manager.update_memory(
            args.identifier,
            description=args.description,
            content=args.content,
            tags=args.tags,
            weight=args.weight,
            confidence=args.confidence,
        )
        print(f"updated {item.id} {item.name}")
        return 0

    if args.command == "archive":
        item = manager.archive_memory(args.identifier)
        print(f"archived {item.id} {item.name}")
        return 0

    if args.command == "restore":
        item = manager.restore_memory(args.identifier)
        print(f"restored {item.id} {item.name}")
        return 0

    if args.command == "delete":
        manager.delete_memory(args.identifier, hard=args.hard)
        if args.hard:
            print(f"hard deleted {args.identifier}")
        else:
            print(f"deleted {args.identifier}")
        return 0

    if args.command == "export":
        report = manager.export_memories(args.path)
        print(f"exported {report['exported']} memories to {args.path}")
        return 0

    if args.command == "import":
        report = manager.import_memories(args.path)
        print(f"imported {report['imported']} skipped {report['skipped']} errors {len(report['errors'])}")
        return 0

    if args.command == "verify":
        report = manager.verify_memories()
        print(f"verified {report['checked']} memories index_ok={report['index_ok']} errors={len(report['errors'])}")
        if "vector_ok" in report:
            print(
                f"vector_ok={report['vector_ok']} "
                f"missing={len(report['vector_missing'])} "
                f"orphans={len(report['vector_orphans'])} "
                f"mismatches={len(report['embedding_mismatches'])} "
                f"sync_errors={len(report.get('rag_sync_errors', []))}"
            )
        for error in report["errors"]:
            print(f"error: {error}")
        return 0

    if args.command == "rebuild-index":
        manager.rebuild_index()
        print("rebuilt index")
        return 0

    if args.command == "backup":
        report = manager.backup(args.path)
        print(f"backed up {report['exported']} memories to {args.path}")
        return 0

    if args.command == "search":
        results = manager.retrieve_memory(
            args.query,
            memory_types=args.memory_types,
            tags=args.tags,
            top_k=args.top_k,
            include_archived=args.archived,
        )
        for result in results:
            print(f"{result.final_score:.3f}\t{result.memory.id}\t{result.memory.name}\t{result.memory.description}")
        return 0

    if args.command == "context":
        runtime = MemoryRuntime(manager=manager)
        results = runtime.retrieve_task_context(
            args.query,
            memory_types=args.memory_types,
            tags=args.tags,
            top_k=args.top_k,
            pinned_top_k=args.pinned_top_k,
            include_pinned=not args.no_pinned,
            include_archived=args.archived,
            include_knowledge=not args.no_knowledge,
        )
        if args.json:
            print(json.dumps([_search_result_to_dict(result) for result in results], ensure_ascii=False, default=_json_default))
            return 0
        print(manager.format_memories_for_prompt(results=results))
        return 0

    if args.command == "clean":
        print(manager.clean_expired_memory())
        return 0

    if args.command == "session" and args.session_command == "append":
        manager.append_message("default", args.session_id, SessionMessage(role=args.role, content=args.content))
        print(f"appended {args.role} message to {args.session_id}")
        return 0

    if args.command == "session" and args.session_command == "show":
        for message in manager.get_messages("default", args.session_id):
            print(f"{message.role}: {message.content}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
