from datetime import datetime, timedelta, timezone

from memora.config import MemoryConfig
from memora.lifecycle import LifecycleManager
from memora.schema import MemoryItem


def memory(**kwargs) -> MemoryItem:
    base = MemoryItem(id="mem_1", name="memory", description="desc", type="project", content="body")
    for key, value in kwargs.items():
        setattr(base, key, value)
    return base


def test_expired_memory_is_archived():
    manager = LifecycleManager(MemoryConfig())
    item = memory(expires_at=datetime.now(timezone.utc) - timedelta(days=1))

    assert manager.is_expired(item) is True
    assert manager.decide(item) == "archive"


def test_cold_low_weight_memory_is_archived():
    manager = LifecycleManager(MemoryConfig(archive_cold_days=30))
    item = memory(
        weight=3,
        updated_at=datetime.now(timezone.utc) - timedelta(days=40),
        last_accessed_at=None,
    )

    assert manager.is_cold(item) is True
    assert manager.decide(item) == "archive"


def test_high_weight_cold_memory_is_kept():
    manager = LifecycleManager(MemoryConfig(archive_cold_days=30))
    item = memory(
        weight=9,
        updated_at=datetime.now(timezone.utc) - timedelta(days=40),
        last_accessed_at=None,
    )

    assert manager.is_cold(item) is False
    assert manager.decide(item) == "keep"
