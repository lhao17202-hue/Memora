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


def test_rejects_transient_task_state():
    result = MemoryPolicy().evaluate(candidate("下一步：实现 CLI"), [])

    assert result.action == "reject"
    assert result.reason == "transient_task_state"


def test_rejects_noisy_output():
    result = MemoryPolicy().evaluate(candidate("stderr:\nTraceback most recent call last"), [])

    assert result.action == "reject"
    assert result.reason == "noisy_output"


def test_same_name_updates_existing_memory():
    existing = [MemoryItem(id="mem_1", name="user-language", description="old", type="user", content="old")]

    result = MemoryPolicy().evaluate(candidate("new", name="user-language"), existing)

    assert result.action == "update"
    assert result.target_memory_id == "mem_1"
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
    assert result.reason == "conflict_requires_confirmation"
