from memora.config import MemoryConfig
from memora.policy import MemoryPolicy
from memora.schema import MemoryCandidate, MemoryItem


def candidate(content: str, name: str = "memory") -> MemoryCandidate:
    return MemoryCandidate(
        action="create",
        name=name,
        description="desc",
        type="user",
        content=content,
    )


def test_rejects_secret_shaped_content():
    result = MemoryPolicy().evaluate(candidate("api_key = sk-abcdef123456"), [])

    assert result.action == "reject"
    assert result.reason == "contains_secret"


def test_allows_non_secret_security_and_token_guidance():
    policy = MemoryPolicy()

    for content in (
        "Use environment variables for API keys.",
        "Keep responses concise to save tokens.",
        "Never store passwords in memory.",
    ):
        result = policy.evaluate(candidate(content), [])
        assert result.action == "create"
        assert result.reason == "accepted"


def test_rejects_transient_task_state():
    result = MemoryPolicy().evaluate(candidate("下一步：实现 CLI"), [])

    assert result.action == "reject"
    assert result.reason == "transient_task_state"


def test_rejects_noisy_output():
    result = MemoryPolicy().evaluate(candidate("stderr:\nTraceback most recent call last"), [])

    assert result.action == "reject"
    assert result.reason == "noisy_output"


def test_noisy_output_uses_configured_content_length_limit():
    policy = MemoryPolicy(MemoryConfig(max_memory_content_chars=10))

    result = policy.evaluate(candidate("x" * 11), [])

    assert result.action == "reject"
    assert result.reason == "noisy_output"


def test_content_under_configured_length_is_not_noisy_by_length():
    policy = MemoryPolicy(MemoryConfig(max_memory_content_chars=20))

    result = policy.evaluate(candidate("durable"), [])

    assert result.action == "create"
    assert result.reason == "accepted"
    assert result.suggested_action == "create"


def test_policy_evaluate_is_decision_only_and_does_not_resolve_weight():
    item = candidate("durable")

    result = MemoryPolicy().evaluate(item, [])

    assert result.action == "create"
    assert result.weight is None
    assert result.suggested_action == "create"


def test_auto_save_user_preferences_disabled_requires_confirmation():
    policy = MemoryPolicy(MemoryConfig(allow_auto_save_user_preferences=False))
    item = candidate("用户偏好中文回答。", name="language")
    item.source = "runtime_extraction"
    item.type = "user"

    result = policy.evaluate(item, [])

    assert result.action == "ask_user"
    assert result.reason == "auto_save_user_preferences_disabled"
    assert result.target_memory_id is None
    assert result.suggested_action == "create"


def test_auto_save_user_preferences_disabled_duplicate_includes_update_target():
    policy = MemoryPolicy(MemoryConfig(allow_auto_save_user_preferences=False))
    existing = [MemoryItem(id="mem_1", name="language", description="old", type="user", content="old")]
    item = candidate("用户偏好中文回答。", name="language")
    item.source = "runtime_extraction"
    item.type = "user"

    result = policy.evaluate(item, existing)

    assert result.action == "ask_user"
    assert result.reason == "auto_save_user_preferences_disabled"
    assert result.target_memory_id == "mem_1"
    assert result.suggested_action == "update"


def test_auto_save_project_facts_disabled_requires_confirmation():
    policy = MemoryPolicy(MemoryConfig(allow_auto_save_project_facts=False))
    item = candidate("Project uses pytest.", name="test-framework")
    item.source = "session_extraction"
    item.type = "project"

    result = policy.evaluate(item, [])

    assert result.action == "ask_user"
    assert result.reason == "auto_save_project_facts_disabled"
    assert result.suggested_action == "create"


def test_same_name_updates_existing_memory():
    existing = [MemoryItem(id="mem_1", name="user-language", description="old", type="user", content="old")]

    result = MemoryPolicy().evaluate(candidate("new", name="user-language"), existing)

    assert result.action == "update"
    assert result.target_memory_id == "mem_1"
    assert result.suggested_action == "update"
    assert result.reason == "duplicate_or_same_key"


def test_conflict_requires_confirmation_for_same_type_different_content():
    existing = [
        MemoryItem(
            id="mem_1",
            name="user-language-en",
            description="User prefers English.",
            type="user",
            content="用户偏好英文回答。",
        )
    ]

    result = MemoryPolicy().evaluate(candidate("用户偏好中文回答。", name="user-language-zh"), existing)

    assert result.action == "ask_user"
    assert result.target_memory_id == "mem_1"
    assert result.suggested_action == "update"
    assert result.reason == "conflict_requires_confirmation"


def test_conflict_confirmation_can_be_disabled():
    policy = MemoryPolicy(MemoryConfig(require_confirmation_for_conflicts=False))
    existing = [
        MemoryItem(
            id="mem_1",
            name="user-language-en",
            description="User prefers English.",
            type="user",
            content="用户偏好英文回答。",
        )
    ]

    result = policy.evaluate(candidate("用户偏好中文回答。", name="user-language-zh"), existing)

    assert result.action == "create"
    assert result.reason == "accepted"
    assert result.target_memory_id is None
