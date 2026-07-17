"""Deterministic lifecycle decisions for memories."""

from __future__ import annotations

from datetime import datetime, timezone

from .config import MemoryConfig
from .schema import MemoryItem


class LifecycleManager:
    def __init__(self, config: MemoryConfig):
        self.config = config

    def is_expired(self, memory: MemoryItem, now: datetime | None = None) -> bool:
        now = now or datetime.now(timezone.utc)
        if memory.expires_at is None:
            return False
        expires_at = memory.expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        return expires_at <= now

    def is_cold(self, memory: MemoryItem, now: datetime | None = None) -> bool:
        if memory.weight > 5:
            return False
        now = now or datetime.now(timezone.utc)
        anchor = memory.last_accessed_at or memory.updated_at or memory.created_at
        if anchor is None:
            return False
        if anchor.tzinfo is None:
            anchor = anchor.replace(tzinfo=timezone.utc)
        return (now - anchor).days >= self.config.archive_cold_days

    def decide(self, memory: MemoryItem, now: datetime | None = None) -> str:
        if memory.status != "active":
            return "keep"
        if self.is_expired(memory, now=now):
            return "archive"
        if self.is_cold(memory, now=now):
            return "archive"
        return "keep"
