from memora.taxonomy import (
    MEMORY_TYPE_POLICIES,
    ON_DEMAND_CONTEXT_TYPES,
    PINNED_CONTEXT_TYPES,
    VALID_MEMORY_TYPES,
    configured_default_weight,
    half_life_days_for_type,
)


class Config:
    default_preference_weight = 10


def test_memory_type_taxonomy_defines_valid_types_and_context_modes():
    assert VALID_MEMORY_TYPES == (
        "preference",
        "project",
        "episodic",
        "reflective",
        "tool",
        "knowledge",
        "general",
    )
    assert PINNED_CONTEXT_TYPES == ("preference", "project")
    assert ON_DEMAND_CONTEXT_TYPES == ("episodic", "reflective", "tool", "knowledge", "general")


def test_memory_type_taxonomy_defines_weight_and_half_life_defaults():
    assert MEMORY_TYPE_POLICIES["preference"].default_weight == 9
    assert MEMORY_TYPE_POLICIES["project"].default_weight == 8
    assert half_life_days_for_type("preference") == 365
    assert half_life_days_for_type("project") == 180
    assert half_life_days_for_type("knowledge") == 365
    assert half_life_days_for_type("unknown") == MEMORY_TYPE_POLICIES["general"].half_life_days


def test_configured_default_weight_allows_config_overrides():
    assert configured_default_weight("preference", Config()) == 10
    assert configured_default_weight("unknown", Config()) == MEMORY_TYPE_POLICIES["general"].default_weight
