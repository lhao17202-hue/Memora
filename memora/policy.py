"""Deterministic memory save policy."""

from __future__ import annotations

import re

from .config import MemoryConfig
from .schema import MemoryCandidate, MemoryItem
from .utils import slugify

SECRET_PATTERNS = [
    re.compile(r"(?i)api[_ -]?key"),
    re.compile(r"(?i)token"),
    re.compile(r"(?i)secret"),
    re.compile(r"(?i)password"),
    re.compile(r"(?i)private[_ -]?key"),
    re.compile(r"sk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"(?i)authorization:\s*bearer"),
    re.compile(r"(?i)cookie"),
]

TRANSIENT_PREFIXES = (
    "当前目标",
    "当前阶段",
    "下一步",
    "已完成",
    "当前阻塞",
    "临时计划",
    "current goal",
    "next step",
    "current blocker",
)

NOISE_PATTERNS = [
    re.compile(r"(?i)stdout"),
    re.compile(r"(?i)stderr"),
    re.compile(r"(?i)traceback"),
    re.compile(r"(?i)exit_code"),
]


AUTO_SAVE_SOURCES = {"conversation", "runtime_extraction", "session_extraction"}


class MemoryPolicy:
    def __init__(self, config: MemoryConfig | None = None):
        self.config = config or MemoryConfig()

    def contains_secret(self, text: str) -> bool:
        return any(pattern.search(text or "") for pattern in SECRET_PATTERNS)

    def is_transient_task_state(self, text: str) -> bool:
        lowered = (text or "").strip().lower()
        return any(lowered.startswith(prefix.lower()) for prefix in TRANSIENT_PREFIXES)

    def is_noisy_output(self, text: str) -> bool:
        value = text or ""
        return any(pattern.search(value) for pattern in NOISE_PATTERNS) or len(value) > self.config.max_memory_content_chars

    def requires_auto_save_confirmation(self, candidate: MemoryCandidate) -> str | None:
        if candidate.source not in AUTO_SAVE_SOURCES:
            return None
        if candidate.type == "user" and not self.config.allow_auto_save_user_preferences:
            return "auto_save_user_preferences_disabled"
        if candidate.type == "project" and not self.config.allow_auto_save_project_facts:
            return "auto_save_project_facts_disabled"
        return None

    def find_duplicate(self, candidate: MemoryCandidate, existing: list[MemoryItem]) -> MemoryItem | None:
        wanted = slugify(candidate.name)
        for item in existing:
            if item.status == "active" and item.name == wanted:
                return item
        return None

    def find_conflict(self, candidate: MemoryCandidate, existing: list[MemoryItem]) -> MemoryItem | None:
        content = candidate.content
        for item in existing:
            if item.status != "active" or item.type != candidate.type:
                continue
            if "中文" in content and "英文" in item.content:
                return item
            if "英文" in content and "中文" in item.content:
                return item
        return None

    def evaluate(self, candidate: MemoryCandidate, existing: list[MemoryItem]) -> MemoryCandidate:
        candidate.name = slugify(candidate.name)
        if self.contains_secret(candidate.content):
            candidate.action = "reject"
            candidate.reason = "contains_secret"
            return candidate
        if self.is_transient_task_state(candidate.content):
            candidate.action = "reject"
            candidate.reason = "transient_task_state"
            return candidate
        if self.is_noisy_output(candidate.content):
            candidate.action = "reject"
            candidate.reason = "noisy_output"
            return candidate
        auto_save_reason = self.requires_auto_save_confirmation(candidate)
        if auto_save_reason:
            candidate.action = "ask_user"
            candidate.reason = auto_save_reason
            return candidate
        duplicate = self.find_duplicate(candidate, existing)
        if duplicate:
            candidate.action = "update"
            candidate.target_memory_id = duplicate.id
            candidate.reason = "duplicate_or_same_key"
            return candidate
        conflict = self.find_conflict(candidate, existing)
        if conflict:
            candidate.action = "ask_user"
            candidate.target_memory_id = conflict.id
            candidate.reason = "conflict_requires_confirmation"
            return candidate
        candidate.action = "create"
        candidate.reason = "accepted"
        return candidate
