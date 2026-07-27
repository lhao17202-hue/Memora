from memora.config import MemoryConfig
from memora.errors import MemoryValidationError
from memora.policy import MemoryPolicy
from memora.schema import MemoryCandidate, MemoryItem, MemoryRelation, MemoryRelationDecision


def candidate(content: str, name: str = "memory") -> MemoryCandidate:
    return MemoryCandidate(
        action="create",
        name=name,
        description="desc",
        type="preference",
        content=content,
    )


def test_rejects_secret_shaped_content():
    result = MemoryPolicy().evaluate(candidate("api_key = sk-abcdef123456"), [])

    assert result.action == "reject"
    assert result.reason == "contains_secret"


def test_rejects_sensitive_environment_variable_assignments():
    policy = MemoryPolicy()

    for content in (
        "OPENAI_API_KEY=abc123",
        "GITHUB_TOKEN=ghp_abcdef123456",
        "DATABASE_PASSWORD=hunter2",
        "JWT_SECRET=abcdef123456",
        "SESSION_COOKIE=abc123",
    ):
        result = policy.evaluate(candidate(content), [])
        assert result.action == "reject"
        assert result.reason == "contains_secret"


def test_rejects_camel_case_sensitive_assignment_keys():
    policy = MemoryPolicy()

    for content in (
        "apiKey=abc123",
        "openaiApiKey=abc123",
        "privateKey=abc123",
    ):
        result = policy.evaluate(candidate(content), [])
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


def test_allows_non_credential_config_assignments():
    policy = MemoryPolicy()

    for content in (
        "token_budget=1000",
        "max_tokens=2048",
        "api_key_policy=use env vars",
        "password_policy=never store passwords",
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


def test_policy_rejects_invalid_high_confidence_conflict_config():
    for config in (
        MemoryConfig(high_confidence_conflict_threshold=True),
        MemoryConfig(high_confidence_conflict_threshold=1.1),
        MemoryConfig(allow_high_confidence_conflict_replace="yes"),
        MemoryConfig(llm_relation_judge_enabled="yes"),
        MemoryConfig(llm_conflict_auto_replace_threshold=1.1),
    ):
        try:
            MemoryPolicy(config)
        except MemoryValidationError as exc:
            assert (
                "confidence_conflict" in str(exc)
                or "allow_high_confidence_conflict_replace" in str(exc)
                or "llm_relation" in str(exc)
                or "llm_conflict" in str(exc)
            )
        else:
            raise AssertionError("expected MemoryValidationError")


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
    item.type = "preference"

    result = policy.evaluate(item, [])

    assert result.action == "ask_user"
    assert result.reason == "auto_save_user_preferences_disabled"
    assert result.target_memory_id is None
    assert result.suggested_action == "create"


def test_auto_save_user_preferences_disabled_duplicate_includes_update_target():
    policy = MemoryPolicy(MemoryConfig(allow_auto_save_user_preferences=False))
    existing = [MemoryItem(id="mem_1", name="language", description="old", type="preference", content="old")]
    item = candidate("用户偏好中文回答。", name="language")
    item.source = "runtime_extraction"
    item.type = "preference"

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
    existing = [MemoryItem(id="mem_1", name="user-language", description="old", type="preference", content="old")]

    result = MemoryPolicy().evaluate(candidate("new", name="user-language"), existing)

    assert result.action == "update"
    assert result.target_memory_id == "mem_1"
    assert result.suggested_action == "update"
    assert result.reason == "duplicate_or_same_key"


def test_different_name_language_preference_does_not_conflict_without_semantic_relation():
    existing = [
        MemoryItem(
            id="mem_1",
            name="user-language-en",
            description="User prefers English.",
            type="preference",
            content="用户偏好英文回答。",
        )
    ]

    result = MemoryPolicy().evaluate(candidate("用户偏好中文回答。", name="user-language-zh"), existing)

    assert result.action == "create"
    assert result.target_memory_id is None
    assert result.suggested_action == "create"
    assert result.reason == "accepted"


def test_semantic_conflict_requires_confirmation():
    policy = MemoryPolicy(MemoryConfig(allow_high_confidence_conflict_replace=False))
    existing = [
        MemoryItem(
            id="mem_1",
            name="user-language-en",
            description="User prefers English.",
            type="preference",
            content="用户偏好英文回答。",
        )
    ]

    result = policy.evaluate(
        candidate("用户偏好中文回答。", name="user-language-zh"),
        existing,
        relation=MemoryRelation(kind="conflict", target_memory_id="mem_1", similarity_score=0.95),
    )

    assert result.action == "ask_user"
    assert result.reason == "semantic_conflict_requires_confirmation"
    assert result.target_memory_id == "mem_1"
    assert result.suggested_action == "supersede"


def test_high_confidence_semantic_conflict_can_replace():
    existing = [MemoryItem(id="mem_1", name="language", description="old", type="preference", content="Prefer English.")]
    item = candidate("Prefer Chinese.", name="language-zh")
    item.confidence = 0.95

    result = MemoryPolicy().evaluate(
        item,
        existing,
        relation=MemoryRelation(kind="conflict", target_memory_id="mem_1", similarity_score=0.95),
    )

    assert result.action == "supersede"
    assert result.reason == "semantic_conflict_high_confidence_replace"
    assert result.target_memory_id == "mem_1"
    assert result.suggested_action == "supersede"


def test_semantic_conflict_confirmation_can_be_disabled():
    policy = MemoryPolicy(MemoryConfig(require_confirmation_for_conflicts=False, allow_high_confidence_conflict_replace=False))
    existing = [MemoryItem(id="mem_1", name="language", description="old", type="preference", content="Prefer English.")]

    result = policy.evaluate(
        candidate("Prefer Chinese.", name="language-zh"),
        existing,
        relation=MemoryRelation(kind="conflict", target_memory_id="mem_1", similarity_score=0.95),
    )

    assert result.action == "create"
    assert result.reason == "accepted"
    assert result.target_memory_id is None


def test_semantic_merge_updates_existing_memory():
    existing = [MemoryItem(id="mem_1", name="style", description="old", type="preference", content="Prefer concise responses.")]

    result = MemoryPolicy().evaluate(
        candidate("Prefer concise responses with short summaries.", name="response-style"),
        existing,
        relation=MemoryRelation(kind="merge", target_memory_id="mem_1", similarity_score=0.90),
    )

    assert result.action == "update"
    assert result.reason == "semantic_merge"
    assert result.target_memory_id == "mem_1"


def test_llm_relation_decision_none_overrides_embedding_relation():
    existing = [MemoryItem(id="mem_1", name="style", description="old", type="preference", content="Prefer concise responses.")]

    result = MemoryPolicy().evaluate(
        candidate("Prefer vim.", name="editor"),
        existing,
        relation=MemoryRelation(kind="merge", target_memory_id="mem_1", similarity_score=0.90),
        relation_decision=MemoryRelationDecision(kind="none", confidence=0.95, reason="Different facts."),
    )

    assert result.action == "create"
    assert result.reason == "accepted"
    assert result.target_memory_id is None


def test_llm_relation_decision_merge_updates_existing_memory():
    existing = [MemoryItem(id="mem_1", name="style", description="old", type="preference", content="Prefer concise responses.")]

    result = MemoryPolicy().evaluate(
        candidate("Prefer concise responses with short summaries.", name="response-style"),
        existing,
        relation=MemoryRelation(kind="merge", target_memory_id="mem_1", similarity_score=0.90),
        relation_decision=MemoryRelationDecision(kind="merge", confidence=0.85, reason="Refinement."),
    )

    assert result.action == "update"
    assert result.reason == "llm_semantic_merge"
    assert result.target_memory_id == "mem_1"


def test_llm_conflict_auto_replace_uses_decision_confidence():
    existing = [MemoryItem(id="mem_1", name="language", description="old", type="preference", content="Prefer English.")]
    item = candidate("Prefer Chinese.", name="language-zh")
    item.confidence = 0.10

    result = MemoryPolicy().evaluate(
        item,
        existing,
        relation=MemoryRelation(kind="merge", target_memory_id="mem_1", similarity_score=0.95),
        relation_decision=MemoryRelationDecision(kind="conflict", confidence=0.95, reason="Preference changed."),
    )

    assert result.action == "supersede"
    assert result.reason == "llm_semantic_conflict_high_confidence_replace"
