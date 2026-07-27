"""Deterministic memory save policy."""

from __future__ import annotations

import re

from .config import MemoryConfig
from .errors import MemoryValidationError
from .schema import MemoryCandidate, MemoryItem, MemoryRelation, MemoryRelationDecision
from .utils import slugify

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"(?i)authorization:\s*bearer\s+\S+"),
]

SECRET_ASSIGNMENT_PATTERN = re.compile(r"(?i)\b([A-Z0-9_ -]+)\s*[:=]\s*\S+")
SECRET_KEY_FRAGMENTS = ("api_key", "apikey", "token", "secret", "password", "private_key", "privatekey", "cookie")
NON_SECRET_ASSIGNMENT_KEYS = {"max_tokens"}
NON_SECRET_ASSIGNMENT_SUFFIXES = ("_budget", "_policy")

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
        self._validate_conflict_config()

    def contains_secret(self, text: str) -> bool:
        value = text or ""
        if any(pattern.search(value) for pattern in SECRET_PATTERNS):
            return True
        for match in SECRET_ASSIGNMENT_PATTERN.finditer(value):
            key = re.sub(r"[^a-z0-9]+", "_", match.group(1).casefold()).strip("_")
            if key in NON_SECRET_ASSIGNMENT_KEYS or key.endswith(NON_SECRET_ASSIGNMENT_SUFFIXES):
                continue
            compact_key = key.replace("_", "")
            if any(fragment in key or fragment in compact_key for fragment in SECRET_KEY_FRAGMENTS):
                return True
        return False

    def is_transient_task_state(self, text: str) -> bool:
        lowered = (text or "").strip().lower()
        return any(lowered.startswith(prefix.lower()) for prefix in TRANSIENT_PREFIXES)

    def is_noisy_output(self, text: str) -> bool:
        value = text or ""
        return any(pattern.search(value) for pattern in NOISE_PATTERNS) or len(value) > self.config.max_memory_content_chars

    def rejection_reason(self, candidate: MemoryCandidate) -> str | None:
        if self.contains_secret(candidate.content):
            return "contains_secret"
        if self.is_transient_task_state(candidate.content):
            return "transient_task_state"
        if self.is_noisy_output(candidate.content):
            return "noisy_output"
        return None

    def requires_auto_save_confirmation(self, candidate: MemoryCandidate) -> str | None:
        if candidate.source not in AUTO_SAVE_SOURCES:
            return None
        if candidate.type == "preference" and not self.config.allow_auto_save_user_preferences:
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

    def evaluate(
        self,
        candidate: MemoryCandidate,
        existing: list[MemoryItem],
        relation: MemoryRelation | None = None,
        relation_decision: MemoryRelationDecision | None = None,
    ) -> MemoryCandidate:
        """Return a deterministic policy decision for a candidate.

        This method normalizes and annotates the candidate, but it does not
        resolve manager-owned defaults such as omitted write weights. Use
        MemoryManager APIs for persistence-ready decisions and writes.
        """
        candidate.name = slugify(candidate.name)
        rejection_reason = self.rejection_reason(candidate)
        if rejection_reason is not None:
            candidate.action = "reject"
            candidate.reason = rejection_reason
            return candidate

        duplicate = self.find_duplicate(candidate, existing)
        relation = self._usable_relation(relation, existing)
        relation_kind = relation_decision.kind if relation_decision is not None else (relation.kind if relation is not None else "none")
        if relation and relation_kind in {"conflict", "supersede"}:
            candidate.target_memory_id = relation.target_memory_id
            candidate.target_updated_at = relation.target_updated_at
            candidate.suggested_action = "supersede"
        elif duplicate:
            candidate.target_memory_id = duplicate.id
            candidate.target_updated_at = duplicate.updated_at
            candidate.suggested_action = "update"
        elif relation and relation_kind in {"duplicate", "merge", "conflict"}:
            candidate.target_memory_id = relation.target_memory_id
            candidate.target_updated_at = relation.target_updated_at
            candidate.suggested_action = "update"
        else:
            candidate.target_memory_id = None
            candidate.target_updated_at = None
            candidate.suggested_action = "create"

        auto_save_reason = self.requires_auto_save_confirmation(candidate)
        if auto_save_reason:
            candidate.action = "ask_user"
            candidate.reason = auto_save_reason
            return candidate
        if duplicate and relation_kind not in {"conflict", "supersede"}:
            candidate.action = "update"
            candidate.reason = "duplicate_or_same_key"
            return candidate
        if relation and relation_kind == "duplicate":
            candidate.action = "update"
            candidate.reason = "llm_semantic_duplicate" if relation_decision is not None else "semantic_duplicate"
            return candidate
        if relation and relation_kind == "merge":
            candidate.action = "update"
            candidate.reason = "llm_semantic_merge" if relation_decision is not None else "semantic_merge"
            return candidate
        if relation and relation_kind in {"conflict", "supersede"}:
            confidence = relation_decision.confidence if relation_decision is not None else candidate.confidence
            threshold = (
                self.config.llm_conflict_auto_replace_threshold
                if relation_decision is not None
                else self.config.high_confidence_conflict_threshold
            )
            if (
                self.config.allow_high_confidence_conflict_replace
                and confidence >= threshold
            ):
                candidate.action = "supersede"
                candidate.reason = (
                    f"llm_semantic_{relation_kind}_high_confidence_replace"
                    if relation_decision is not None
                    else "semantic_conflict_high_confidence_replace"
                )
                return candidate
            if self.config.require_confirmation_for_conflicts:
                candidate.action = "ask_user"
                candidate.reason = (
                    f"llm_semantic_{relation_kind}_requires_confirmation"
                    if relation_decision is not None
                    else "semantic_conflict_requires_confirmation"
                )
                return candidate
            candidate.target_memory_id = None
            candidate.target_updated_at = None
            candidate.suggested_action = "create"
        candidate.action = "create"
        candidate.reason = "accepted"
        return candidate

    def _usable_relation(self, relation: MemoryRelation | None, existing: list[MemoryItem]) -> MemoryRelation | None:
        if relation is None or relation.kind == "none" or relation.target_memory_id is None:
            return None
        targets = {item.id: item for item in existing if item.status == "active"}
        target = targets.get(relation.target_memory_id)
        if target is None:
            return None
        if relation.target_updated_at is None:
            relation.target_updated_at = target.updated_at
        return relation

    def _validate_conflict_config(self) -> None:
        for field_name in (
            "high_confidence_conflict_threshold",
            "llm_relation_confidence_threshold",
            "llm_merge_confidence_threshold",
            "llm_conflict_auto_replace_threshold",
        ):
            threshold = getattr(self.config, field_name)
            if isinstance(threshold, bool) or not isinstance(threshold, int | float) or threshold < 0.0 or threshold > 1.0:
                raise MemoryValidationError(f"{field_name} must be from 0.0 to 1.0")
        if not isinstance(self.config.allow_high_confidence_conflict_replace, bool):
            raise MemoryValidationError("allow_high_confidence_conflict_replace must be a boolean")
        if not isinstance(self.config.llm_relation_judge_enabled, bool):
            raise MemoryValidationError("llm_relation_judge_enabled must be a boolean")
