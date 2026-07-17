"""Utility helpers for Memora."""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .errors import MemoryValidationError


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def slugify(value: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[\\/\\:_]+", "-", text)
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-") or "memory"


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---\n"):
        return {}, text.strip()
    end = text.find("\n---", 4)
    if end == -1:
        raise MemoryValidationError("frontmatter missing closing marker")
    raw_meta = text[4:end]
    body = text[end + len("\n---") :].lstrip("\n")
    loaded = yaml.safe_load(raw_meta) or {}
    if not isinstance(loaded, dict):
        raise MemoryValidationError("frontmatter must be a mapping")
    return loaded, body.strip()


def dump_frontmatter(metadata: dict[str, Any], body: str) -> str:
    raw_meta = yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False).strip()
    return f"---\n{raw_meta}\n---\n\n{str(body).strip()}\n"


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        delete=False,
        dir=str(path.parent),
        newline="",
    ) as tmp:
        tmp.write(text)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def safe_json_load(path: Path, default: Any | None = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def safe_json_write(path: Path, data: Any) -> None:
    text = json.dumps(data, ensure_ascii=False, indent=2)
    atomic_write_text(Path(path), text)
