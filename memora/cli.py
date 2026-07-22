"""Command-line interface for Memora."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime

from .config import MemoryConfig
from .errors import MemoraError
from .manager import MemoryManager
from .embeddings import EMBEDDING_PROVIDER_CHOICES
from .reranker import RERANKER_CHOICES
from .schema import MemoryCandidate, SessionMessage
from .vector_store import VECTOR_STORE_CHOICES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memora",
        description="Memora deterministic local memory system.",
    )
    parser.add_argument("--root", default=".memora", help="Memora runtime root directory.")
    parser.add_argument("--backend", choices=("file", "sqlite"), default="file", help="Memory storage backend.")
    parser.add_argument("--sqlite-path", help="SQLite database path when --backend sqlite is used.")
    parser.add_argument("--no-fts", action="store_true", help="Disable SQLite FTS candidate recall.")
    parser.add_argument("--rag", action="store_true", help="Enable deterministic local RAG retrieval and vector indexing.")
    parser.add_argument("--embedding-provider", default="hash", choices=EMBEDDING_PROVIDER_CHOICES, help="Embedding provider for RAG.")
    parser.add_argument("--vector-store", default="sqlite", choices=VECTOR_STORE_CHOICES, help="Vector store for RAG.")
    parser.add_argument("--reranker", default="deterministic", choices=RERANKER_CHOICES, help="Reranker for RAG.")
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
        manager = MemoryManager(
            MemoryConfig(
                root_dir=args.root,
                memory_backend=args.backend,
                sqlite_path=args.sqlite_path,
                fts_enabled=not args.no_fts,
                rag_enabled=args.rag,
                embedding_provider=args.embedding_provider,
                vector_store=args.vector_store,
                reranker=args.reranker,
            )
        )
        return _run_command(args, manager, parser)
    except MemoraError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _json_default(value):
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_result_to_dict(result) -> dict:
    return {
        "action": result.action,
        "reason": result.reason,
        "target_memory_id": result.target_memory_id,
        "memory": asdict(result.memory) if result.memory is not None else None,
        "candidate": asdict(result.candidate) if result.candidate is not None else None,
    }


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
