"""Command-line interface for Memora."""

from __future__ import annotations

import argparse
import sys

from .config import MemoryConfig
from .errors import MemoraError
from .manager import MemoryManager
from .schema import SessionMessage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memora",
        description="Memora deterministic local memory system.",
    )
    parser.add_argument("--root", default=".memora", help="Memora runtime root directory.")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("init", help="Initialize a Memora runtime directory.")

    save_parser = subparsers.add_parser("save", help="Save a memory.")
    save_parser.add_argument("--type", required=True)
    save_parser.add_argument("--name", required=True)
    save_parser.add_argument("--description", required=True)
    save_parser.add_argument("--content", required=True)

    subparsers.add_parser("list", help="List memories.")

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

    search_parser = subparsers.add_parser("search", help="Search memories.")
    search_parser.add_argument("query")

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
    manager = MemoryManager(MemoryConfig(root_dir=args.root))

    try:
        return _run_command(args, manager, parser)
    except MemoraError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _run_command(args, manager: MemoryManager, parser: argparse.ArgumentParser) -> int:
    if args.command == "init":
        manager.init_storage()
        print(f"initialized {args.root}")
        return 0

    if args.command == "save":
        item = manager.save_memory(args.type, args.content, args.description, name=args.name)
        print(f"saved {item.id} {item.name}")
        return 0

    if args.command == "list":
        for item in manager.memory_store.list_memories():
            print(f"{item.id}\t{item.name}\t{item.type}\t{item.description}")
        return 0

    if args.command == "show":
        item = manager.memory_store.get_memory(args.identifier)
        if item is None:
            print("memory not found")
            return 1
        print(f"id: {item.id}")
        print(f"name: {item.name}")
        print(f"type: {item.type}")
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

    if args.command == "search":
        results = manager.retrieve_memory(args.query)
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
