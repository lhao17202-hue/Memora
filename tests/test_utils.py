from pathlib import Path

import pytest

from memora.errors import MemoryValidationError
from memora.utils import (
    atomic_write_text,
    dump_frontmatter,
    estimate_tokens,
    parse_frontmatter,
    safe_json_load,
    safe_json_write,
    slugify,
)


def test_slugify_normalizes_text():
    assert slugify("User Language Preference") == "user-language-preference"
    assert slugify("用户 语言 偏好") == "用户-语言-偏好"
    assert slugify(" a/b:c ") == "a-b-c"


def test_estimate_tokens_is_at_least_one_for_text():
    assert estimate_tokens("hello") == 2
    assert estimate_tokens("") == 0


def test_frontmatter_round_trip_nested_metadata():
    text = dump_frontmatter(
        {
            "name": "user-language-preference",
            "description": "用户偏好中文。",
            "metadata": {"tags": ["language", "style"], "weight": 9},
        },
        "用户偏好使用中文。",
    )

    metadata, body = parse_frontmatter(text)

    assert metadata["name"] == "user-language-preference"
    assert metadata["metadata"]["tags"] == ["language", "style"]
    assert body == "用户偏好使用中文。"


def test_parse_frontmatter_rejects_missing_closing_marker():
    with pytest.raises(MemoryValidationError):
        parse_frontmatter("---\nname: bad\nbody")


def test_atomic_write_and_safe_json(tmp_path: Path):
    text_path = tmp_path / "nested" / "file.txt"
    atomic_write_text(text_path, "hello")
    assert text_path.read_text(encoding="utf-8") == "hello"

    json_path = tmp_path / "data" / "file.json"
    safe_json_write(json_path, {"ok": True})
    assert safe_json_load(json_path) == {"ok": True}
    assert safe_json_load(tmp_path / "missing.json", default={}) == {}
