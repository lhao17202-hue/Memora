"""Command-line interface for Memora."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memora",
        description="Memora deterministic local memory system.",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("init", help="Initialize a Memora runtime directory.")
    subparsers.add_parser("save", help="Save a memory.")
    subparsers.add_parser("list", help="List memories.")
    subparsers.add_parser("show", help="Show one memory.")
    subparsers.add_parser("search", help="Search memories.")
    subparsers.add_parser("clean", help="Archive expired or cold memories.")

    session_parser = subparsers.add_parser("session", help="Manage sessions.")
    session_subparsers = session_parser.add_subparsers(dest="session_command")
    session_subparsers.add_parser("append", help="Append a message to a session.")
    session_subparsers.add_parser("show", help="Show a session.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
