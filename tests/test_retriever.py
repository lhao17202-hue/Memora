from datetime import datetime, timedelta, timezone

from memora.retriever import HALF_LIFE_DAYS, MemoryRetriever
from memora.schema import MemoryItem, MemoryQuery


def item(name: str, content: str, weight: int = 5, status: str = "active") -> MemoryItem:
    return MemoryItem(
        id=name,
        name=name,
        description=content,
        type="preference",
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

    results = MemoryRetriever().retrieve([memories[0]], MemoryQuery(query="中文", include_archived=True))

    assert len(results) == 1


def test_type_filter_excludes_other_types():
    memory = item("project", "项目使用 pytest。")
    memory.type = "project"

    results = MemoryRetriever().retrieve([memory], MemoryQuery(query="pytest", memory_types=["preference"]))

    assert results == []


def test_memory_type_half_life_defaults_match_taxonomy():
    assert HALF_LIFE_DAYS == {
        "preference": 365,
        "project": 180,
        "episodic": 45,
        "reflective": 180,
        "tool": 120,
        "knowledge": 365,
        "general": 90,
    }


def test_recency_score_decays_old_memory():
    fresh = item("fresh", "中文回答", weight=5)
    old = item("old", "中文回答", weight=5)
    old.updated_at = datetime.now(timezone.utc) - timedelta(days=365)

    results = MemoryRetriever().retrieve([old, fresh], MemoryQuery(query="中文回答"))

    assert results[0].memory.name == "fresh"


def test_name_match_ranks_above_content_only_match():
    name_match = item("pytest", "unrelated content")
    content_match = item("other", "pytest")

    results = MemoryRetriever().retrieve([content_match, name_match], MemoryQuery(query="pytest"))

    assert results[0].memory.name == "pytest"
    assert results[0].reason == "exact_name"
    assert results[0].similarity_score > results[1].similarity_score


def test_tag_match_ranks_above_content_only_match():
    tag_match = item("tagged", "unrelated content")
    tag_match.tags = ["pytest"]
    content_match = item("other", "pytest")
    content_match.description = "unrelated description"

    results = MemoryRetriever().retrieve([content_match, tag_match], MemoryQuery(query="pytest"))

    assert results[0].memory.name == "tagged"
    assert results[0].reason == "tokens_tags"
    assert results[0].similarity_score > results[1].similarity_score


def test_reason_identifies_strongest_matching_field():
    description_match = item("description", "pytest")
    description_match.content = "unrelated content"
    content_match = item("content", "pytest")
    content_match.description = "unrelated description"

    results = MemoryRetriever().retrieve([content_match, description_match], MemoryQuery(query="pytest"))

    assert results[0].memory.name == "description"
    assert results[0].reason == "exact_description"
    assert results[1].reason == "phrase_content"


def test_chinese_short_query_matches_longer_chinese_memory():
    memory = item("language", "用户偏好中文回答。")
    retriever = MemoryRetriever()

    for query in ["中文", "偏好", "回答"]:
        results = retriever.retrieve([memory], MemoryQuery(query=query))
        assert len(results) == 1
        assert results[0].memory.name == "language"


def test_exact_name_phrase_beats_scattered_token_content_match():
    exact = item("pytest-fixture", "unrelated content")
    scattered = item("other", "pytest helpers create reusable fixture setup")

    results = MemoryRetriever().retrieve([scattered, exact], MemoryQuery(query="pytest fixture"))

    assert results[0].memory.id == "pytest-fixture"
    assert results[0].reason == "exact_name"
    assert results[0].similarity_score > results[1].similarity_score


def test_phrase_content_beats_partial_content_match():
    phrase = item("phrase", "incident response playbook")
    phrase.description = "unrelated description"
    partial = item("partial", "incident notes for unrelated operations")
    partial.description = "unrelated description"

    results = MemoryRetriever().retrieve([partial, phrase], MemoryQuery(query="incident response"))

    assert results[0].memory.id == "phrase"
    assert results[0].reason == "phrase_content"


def test_multi_field_match_beats_single_weak_content_match():
    multi = item("deploy", "api rollout checklist")
    multi.description = "deploy checklist"
    multi.tags = ["api"]
    content_only = item("content-only", "deploy api")
    content_only.description = "unrelated description"

    results = MemoryRetriever().retrieve([content_only, multi], MemoryQuery(query="deploy api"))

    assert results[0].memory.id == "deploy"
    assert results[0].similarity_score > results[1].similarity_score


def test_retrieve_deduplicates_by_memory_id_and_keeps_best_score():
    weak = item("duplicate", "pytest")
    weak.description = "unrelated"
    weak.content = "pytest"
    strong = item("duplicate", "pytest fixture")
    strong.description = "pytest fixture"

    results = MemoryRetriever().retrieve([weak, strong], MemoryQuery(query="pytest fixture"))
    ids = [result.memory.id for result in results]

    assert ids == ["duplicate"]
    assert results[0].memory.content == "pytest fixture"


def test_clear_reason_labels_cover_exact_description_tokens_tags_and_partial_content():
    exact_description = item("exact-description", "unrelated content")
    exact_description.description = "backup rotation"
    tag_match = item("tagged", "unrelated content")
    tag_match.description = "unrelated description"
    tag_match.tags = ["python", "cli"]
    partial_content = item("partial", "database notes")
    partial_content.description = "unrelated description"

    exact_results = MemoryRetriever().retrieve([exact_description], MemoryQuery(query="backup rotation"))
    tag_results = MemoryRetriever().retrieve([tag_match], MemoryQuery(query="python cli"))
    partial_results = MemoryRetriever().retrieve([partial_content], MemoryQuery(query="database archive"))

    assert exact_results[0].reason == "exact_description"
    assert tag_results[0].reason == "tokens_tags"
    assert partial_results[0].reason == "partial_content"


def test_ascii_short_query_does_not_substring_match_inside_longer_words():
    memories = [
        item("alpha", "unrelated content"),
        item("beta", "unrelated content"),
    ]

    results = MemoryRetriever().retrieve(memories, MemoryQuery(query="a"))

    assert results == []


def test_partial_name_and_description_matches_are_not_labeled_exact():
    name_partial = item("database", "unrelated content")
    name_partial.description = "unrelated description"
    description_partial = item("other", "unrelated content")
    description_partial.description = "database notes"

    name_results = MemoryRetriever().retrieve([name_partial], MemoryQuery(query="database archive"))
    description_results = MemoryRetriever().retrieve([description_partial], MemoryQuery(query="database archive"))

    assert name_results[0].reason == "partial_name"
    assert description_results[0].reason == "partial_description"


def test_query_terms_match_snake_case_name_phrase():
    memory = item("pytest_fixture", "unrelated content")
    memory.description = "unrelated description"

    results = MemoryRetriever().retrieve([memory], MemoryQuery(query="pytest fixture"))

    assert len(results) == 1
    assert results[0].memory.name == "pytest_fixture"
    assert results[0].reason == "exact_name"


def test_query_terms_match_snake_case_tag_phrase():
    memory = item("secrets", "unrelated content")
    memory.description = "unrelated description"
    memory.tags = ["api_key"]

    results = MemoryRetriever().retrieve([memory], MemoryQuery(query="api key"))

    assert len(results) == 1
    assert results[0].memory.name == "secrets"
    assert results[0].reason == "tokens_tags"


def test_ordered_non_adjacent_name_and_description_matches_are_partial():
    name_memory = item("api-production-key", "unrelated content")
    name_memory.description = "unrelated description"
    description_memory = item("other", "unrelated content")
    description_memory.description = "api production key"

    name_results = MemoryRetriever().retrieve([name_memory], MemoryQuery(query="api key"))
    description_results = MemoryRetriever().retrieve([description_memory], MemoryQuery(query="api key"))

    assert name_results[0].reason == "partial_name"
    assert description_results[0].reason == "partial_description"


def test_exact_name_phrase_ranks_above_ordered_non_adjacent_name():
    exact = item("api-key", "unrelated content")
    exact.description = "unrelated description"
    ordered = item("api-production-key", "unrelated content")
    ordered.description = "unrelated description"

    results = MemoryRetriever().retrieve([ordered, exact], MemoryQuery(query="api key"))

    assert [result.memory.name for result in results] == ["api-key", "api-production-key"]
    assert results[0].reason == "exact_name"
    assert results[1].reason == "partial_name"
    assert results[0].similarity_score > results[1].similarity_score
