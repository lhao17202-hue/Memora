from datetime import datetime, timedelta, timezone

from memora.retriever import MemoryRetriever
from memora.schema import MemoryItem, MemoryQuery


def item(name: str, content: str, weight: int = 5, status: str = "active") -> MemoryItem:
    return MemoryItem(
        id=name,
        name=name,
        description=content,
        type="user",
        content=content,
        weight=weight,
        status=status,
        updated_at=datetime.now(timezone.utc),
    )


def test_retrieve_ranks_keyword_match_first():
    memories = [
        item("python-style", "用户偏好中文 Python 代码风格。"),
        item("language", "用户偏好中文回答。"),
    ]

    results = MemoryRetriever().retrieve(memories, MemoryQuery(query="中文回答"))

    assert results[0].memory.name == "language"
    assert results[0].final_score > results[-1].final_score


def test_archived_memories_are_excluded_by_default():
    memories = [item("archived", "中文回答", status="archived")]

    results = MemoryRetriever().retrieve(memories, MemoryQuery(query="中文"))

    assert results == []


def test_include_archived_allows_archived_results():
    memories = [item("archived", "中文回答", status="archived")]

    results = MemoryRetriever().retrieve(memories, MemoryQuery(query="中文", include_archived=True))

    assert len(results) == 1


def test_type_filter_excludes_other_types():
    memory = item("project", "项目使用 pytest。")
    memory.type = "project"

    results = MemoryRetriever().retrieve([memory], MemoryQuery(query="pytest", memory_types=["user"]))

    assert results == []


def test_recency_score_decays_old_memory():
    fresh = item("fresh", "中文回答", weight=5)
    old = item("old", "中文回答", weight=5)
    old.updated_at = datetime.now(timezone.utc) - timedelta(days=365)

    results = MemoryRetriever().retrieve([old, fresh], MemoryQuery(query="中文回答"))

    assert results[0].memory.name == "fresh"
