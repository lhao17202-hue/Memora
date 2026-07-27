from memora.config import MemoryConfig
from memora.errors import MemoryValidationError
from memora.relations import LLMMemoryRelationJudge, SemanticMemoryRelationResolver, parse_relation_decision_json
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


class StaticRelationClient:
    def __init__(self, raw_text: str):
        self.raw_text = raw_text
        self.messages = None

    def complete(self, messages):
        self.messages = messages
        return self.raw_text


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


def test_parse_relation_decision_json_accepts_merge_decision():
    decision = parse_relation_decision_json(
        """
        {
          "kind": "merge",
          "confidence": 0.86,
          "reason": "Candidate refines the preference.",
          "merged": {
            "name": "response-style",
            "description": "Prefer concise responses.",
            "content": "Prefer concise responses with short summaries.",
            "tags": ["style", "summary"]
          }
        }
        """
    )

    assert decision.kind == "merge"
    assert decision.confidence == 0.86
    assert decision.merged_name == "response-style"
    assert decision.merged_content == "Prefer concise responses with short summaries."
    assert decision.merged_tags == ["style", "summary"]


def test_parse_relation_decision_json_rejects_invalid_payloads():
    for raw_text in (
        "not json",
        '{"kind":"merge","confidence":true,"reason":"bad","merged":{"description":"d","content":"c"}}',
        '{"kind":"merge","confidence":0.9,"reason":"bad"}',
        '{"kind":"merge","confidence":0.9,"reason":"bad","merged":{"description":"d","content":"c","tags":"style"}}',
        '{"kind":"unknown","confidence":0.9,"reason":"bad"}',
    ):
        try:
            parse_relation_decision_json(raw_text)
        except MemoryValidationError:
            pass
        else:
            raise AssertionError("expected MemoryValidationError")


def test_llm_memory_relation_judge_uses_json_prompt_and_parser():
    client = StaticRelationClient(
        '{"kind":"conflict","confidence":0.91,"reason":"Candidate changes the language preference."}'
    )
    existing = item("Prefer English responses.", name="language-en")
    relation = resolver().resolve(candidate("Prefer Chinese responses.", name="language-zh"), [existing])

    decision = LLMMemoryRelationJudge(client).judge(candidate("Prefer Chinese responses.", name="language-zh"), existing, relation)

    assert decision.kind == "conflict"
    assert decision.confidence == 0.91
    assert client.messages is not None
    assert client.messages[0]["role"] == "system"
    assert "candidate" in client.messages[1]["content"]
