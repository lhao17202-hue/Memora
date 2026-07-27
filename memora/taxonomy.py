"""Memory type taxonomy and ranking policy defaults."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MemoryType = Literal[
    "preference",
    "project",
    "episodic",
    "reflective",
    "tool",
    "knowledge",
    "general",
]

ContextMode = Literal["pinned", "on_demand"]


@dataclass(frozen=True)
class MemoryTypePolicy:
    default_weight: int
    half_life_days: int
    context_mode: ContextMode
    weight_config_field: str


MEMORY_TYPE_POLICIES: dict[MemoryType, MemoryTypePolicy] = {
    "preference": MemoryTypePolicy(
        default_weight=9,
        half_life_days=365,
        context_mode="pinned",
        weight_config_field="default_preference_weight",
    ),
    "project": MemoryTypePolicy(
        default_weight=8,
        half_life_days=180,
        context_mode="pinned",
        weight_config_field="default_project_weight",
    ),
    "episodic": MemoryTypePolicy(
        default_weight=5,
        half_life_days=45,
        context_mode="on_demand",
        weight_config_field="default_episodic_weight",
    ),
    "reflective": MemoryTypePolicy(
        default_weight=7,
        half_life_days=180,
        context_mode="on_demand",
        weight_config_field="default_reflective_weight",
    ),
    "tool": MemoryTypePolicy(
        default_weight=6,
        half_life_days=120,
        context_mode="on_demand",
        weight_config_field="default_tool_weight",
    ),
    "knowledge": MemoryTypePolicy(
        default_weight=6,
        half_life_days=365,
        context_mode="on_demand",
        weight_config_field="default_knowledge_weight",
    ),
    "general": MemoryTypePolicy(
        default_weight=4,
        half_life_days=90,
        context_mode="on_demand",
        weight_config_field="default_general_weight",
    ),
}

VALID_MEMORY_TYPES: tuple[MemoryType, ...] = tuple(MEMORY_TYPE_POLICIES)
PINNED_CONTEXT_TYPES: tuple[MemoryType, ...] = tuple(
    memory_type
    for memory_type, policy in MEMORY_TYPE_POLICIES.items()
    if policy.context_mode == "pinned"
)
ON_DEMAND_CONTEXT_TYPES: tuple[MemoryType, ...] = tuple(
    memory_type
    for memory_type, policy in MEMORY_TYPE_POLICIES.items()
    if policy.context_mode == "on_demand"
)


def policy_for_type(memory_type: str) -> MemoryTypePolicy:
    return MEMORY_TYPE_POLICIES.get(memory_type, MEMORY_TYPE_POLICIES["general"])


def half_life_days_for_type(memory_type: str) -> int:
    return policy_for_type(memory_type).half_life_days


def configured_default_weight(memory_type: str, config: object) -> int:
    policy = policy_for_type(memory_type)
    return int(getattr(config, policy.weight_config_field, policy.default_weight))
