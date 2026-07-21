from memora.config import MemoryConfig
from memora.embeddings import HashEmbeddingProvider, memory_embedding_text, sha256_text
from memora.schema import MemoryItem


def test_memory_config_rag_defaults_are_disabled():
    config = MemoryConfig()

    assert config.memory_backend == "file"
    assert config.rag_enabled is False
    assert config.embedding_provider == "hash"
    assert config.embedding_model == "memora-hash-v1"
    assert config.embedding_dimension == 384
    assert config.vector_store == "sqlite"
    assert config.reranker == "deterministic"


def test_hash_embedding_provider_is_deterministic_and_ordered():
    provider = HashEmbeddingProvider(dimension=16)

    vectors = provider.embed(["alpha 中文", "beta"])

    assert vectors[0] == provider.embed(["alpha 中文"])[0]
    assert vectors[1] == provider.embed(["beta"])[0]
    assert vectors[0] != vectors[1]
    assert len(vectors[0]) == 16
    assert len(vectors[1]) == 16


def test_hash_embedding_empty_string_is_stable_zero_vector():
    provider = HashEmbeddingProvider(dimension=8)

    vector = provider.embed([""])[0]

    assert vector == [0.0] * 8
    assert vector == provider.embed(["   "])[0]


def test_memory_embedding_text_contains_searchable_memory_fields():
    item = MemoryItem(
        id="mem_1",
        name="language",
        type="user",
        description="用户偏好中文。",
        content="用户偏好使用中文回答。",
        tags=["preference", "language"],
    )

    text = memory_embedding_text(item)

    assert "name: language" in text
    assert "type: user" in text
    assert "description: 用户偏好中文。" in text
    assert "tags: preference, language" in text
    assert "content: 用户偏好使用中文回答。" in text
    assert sha256_text(text) == sha256_text(text)
