from memora.config import MemoryConfig
from memora.errors import MemoryValidationError
from memora.relations import SemanticMemoryRelationResolver
from memora.schema import MemoryCandidate, MemoryItem


class KeywordEmbeddingProvider:
    name = "keyword"
    model = "keyword-test"
    dimension = 2

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "unrelated" in lowered:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([1.0, 0.0])
        return vectors


def candidate(content: str, name: str = "candidate") -> MemoryCandidate:
    return MemoryCandidate(
        action="create",
        name=name,
        description=content,
        type="preference",
        content=content,
    )


def item(content: str, name: str = "existing") -> MemoryItem:
    return MemoryItem(
        id=f"mem_{name}",
        name=name,
        description=content,
        type="preference",
        content=content,
    )


def resolver() -> SemanticMemoryRelationResolver:
    return SemanticMemoryRelationResolver(
        KeywordEmbeddingProvider(),
        MemoryConfig(
            semantic_relation_threshold=0.75,
            semantic_merge_threshold=0.80,
            semantic_conflict_threshold=0.90,
        ),
    )


def test_semantic_relation_resolver_returns_none_below_relation_threshold():
    relation = resolver().resolve(candidate("unrelated preference"), [item("concise responses")])

    assert relation.kind == "none"
    assert relation.target_memory_id is None
    assert relation.reason == "below_semantic_relation_threshold"


def test_semantic_relation_resolver_detects_duplicate_content():
    relation = resolver().resolve(candidate("Prefer concise responses."), [item("Prefer concise responses.")])

    assert relation.kind == "duplicate"
    assert relation.target_memory_id == "mem_existing"
    assert relation.reason == "semantic_duplicate"
    assert relation.similarity_score == 1.0


def test_semantic_relation_resolver_detects_merge_when_similar_and_compatible():
    relation = resolver().resolve(
        candidate("Prefer concise responses with short summaries."),
        [item("Prefer concise responses.")],
    )

    assert relation.kind == "merge"
    assert relation.target_memory_id == "mem_existing"
    assert relation.reason == "semantic_merge"


def test_semantic_relation_resolver_detects_conflict_only_after_embedding_match():
    relation = resolver().resolve(
        candidate("User prefers Chinese responses.", name="language-zh"),
        [item("User prefers English responses.", name="language-en")],
    )

    assert relation.kind == "conflict"
    assert relation.target_memory_id == "mem_language-en"
    assert relation.reason == "semantic_conflict"


def test_semantic_relation_resolver_ignores_other_types_and_scopes():
    project_item = item("Prefer concise responses.")
    project_item.type = "project"
    other_user = item("Prefer concise responses.", name="other-user")
    other_user.user_id = "alice"

    relation = resolver().resolve(candidate("Prefer concise responses with short summaries."), [project_item, other_user])

    assert relation.kind == "none"


def test_semantic_relation_resolver_rejects_invalid_thresholds():
    invalid_range = MemoryConfig(semantic_relation_threshold=-0.1)
    invalid_order = MemoryConfig(semantic_relation_threshold=0.90, semantic_merge_threshold=0.80)

    try:
        SemanticMemoryRelationResolver(KeywordEmbeddingProvider(), invalid_range)
    except MemoryValidationError as exc:
        assert "thresholds" in str(exc)
    else:
        raise AssertionError("expected MemoryValidationError")

    try:
        SemanticMemoryRelationResolver(KeywordEmbeddingProvider(), invalid_order)
    except MemoryValidationError as exc:
        assert "relation <= merge <= conflict" in str(exc)
    else:
        raise AssertionError("expected MemoryValidationError")
