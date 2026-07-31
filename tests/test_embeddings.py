import sys
import types

import pytest

from memora.config import MemoryConfig
from memora.embeddings import (
    BgeM3EmbeddingProvider,
    EMBEDDING_PROVIDER_CHOICES,
    EmbeddingVector,
    HashEmbeddingProvider,
    SparseVector,
    embedding_dense,
    memory_embedding_text,
    sha256_text,
)
from memora.errors import MemoryValidationError
from memora.schema import MemoryItem


class DenseResult:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


class FakeBgeModel:
    encode_calls = []

    def __init__(self, model_path, use_fp16=False):
        self.model_path = model_path
        self.use_fp16 = use_fp16

    def encode(self, **kwargs):
        self.__class__.encode_calls.append(kwargs)
        vectors = [[float(index + 1)] * 4 for index, _ in enumerate(kwargs["sentences"])]
        result = {"dense_vecs": DenseResult(vectors)}
        if kwargs["return_sparse"]:
            result["lexical_weights"] = [{"3": 0.5, "1": 1.25, "2": 0.0} for _ in kwargs["sentences"]]
        return result


@pytest.fixture
def fake_flag_embedding(monkeypatch):
    FakeBgeModel.encode_calls = []
    module = types.ModuleType("FlagEmbedding")
    module.BGEM3FlagModel = FakeBgeModel
    monkeypatch.setitem(sys.modules, "FlagEmbedding", module)
    return FakeBgeModel


def test_memory_config_rag_defaults_are_disabled():
    config = MemoryConfig()

    assert config.memory_backend == "file"
    assert config.rag_enabled is False
    assert config.embedding_provider == "hash"
    assert config.embedding_model == "memora-hash-v1"
    assert config.embedding_dimension == 384
    assert config.embedding_model_path is None
    assert config.embedding_batch_size == 8
    assert config.embedding_fp16 is False
    assert config.embedding_sparse is False
    assert config.vector_store == "sqlite"
    assert config.vector_store_options == {}
    assert config.retrieval_mode == "dense"
    assert config.reranker == "deterministic"


def test_embedding_provider_choices_include_bge():
    assert "hash" in EMBEDDING_PROVIDER_CHOICES
    assert "bge" in EMBEDDING_PROVIDER_CHOICES


def test_hash_embedding_provider_is_deterministic_and_ordered():
    provider = HashEmbeddingProvider(dimension=16)

    vectors = provider.embed(["alpha 中文", "beta"])

    assert vectors[0] == provider.embed(["alpha 中文"])[0]
    assert vectors[1] == provider.embed(["beta"])[0]
    assert vectors[0] != vectors[1]
    assert len(vectors[0].dense) == 16
    assert len(vectors[1].dense) == 16
    assert vectors[0].sparse is None
    assert provider.supports_sparse is False


def test_hash_embedding_empty_string_is_stable_zero_vector():
    provider = HashEmbeddingProvider(dimension=8)

    vector = provider.embed([""])[0]

    assert vector.dense == [0.0] * 8
    assert vector == provider.embed(["   "])[0]
    assert embedding_dense(vector) == [0.0] * 8


def test_bge_provider_requires_model_path(fake_flag_embedding):
    with pytest.raises(MemoryValidationError, match="embedding_model_path"):
        BgeM3EmbeddingProvider(model_path=None)


def test_bge_provider_reports_missing_optional_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "FlagEmbedding", None)

    with pytest.raises(MemoryValidationError, match="FlagEmbedding"):
        BgeM3EmbeddingProvider(model_path="C:/Download/bge-m3")


def test_bge_provider_encodes_dense_vectors_with_expected_options(fake_flag_embedding):
    provider = BgeM3EmbeddingProvider(
        model_path="C:/Download/bge-m3",
        model="bge-m3",
        dimension=4,
        batch_size=8,
        fp16=True,
    )

    vectors = provider.embed(["结构化单据识别规则", "发票OCR提取规范"])

    assert provider.name == "bge"
    assert provider.model == "bge-m3"
    assert provider.dimension == 4
    assert provider.supports_sparse is True
    assert provider._model.model_path == "C:/Download/bge-m3"
    assert provider._model.use_fp16 is True
    assert vectors == [EmbeddingVector(dense=[1.0, 1.0, 1.0, 1.0]), EmbeddingVector(dense=[2.0, 2.0, 2.0, 2.0])]
    assert fake_flag_embedding.encode_calls == [
        {
            "sentences": ["结构化单据识别规则", "发票OCR提取规范"],
            "return_dense": True,
            "return_sparse": False,
            "return_colbert_vecs": False,
            "batch_size": 8,
        }
    ]


def test_bge_provider_can_return_sparse_vectors(fake_flag_embedding):
    provider = BgeM3EmbeddingProvider(model_path="C:/Download/bge-m3", dimension=4, return_sparse=True)

    vectors = provider.embed(["结构化单据识别规则"])

    assert vectors == [EmbeddingVector(dense=[1.0, 1.0, 1.0, 1.0], sparse=SparseVector(indices=[1, 3], values=[1.25, 0.5]))]
    assert fake_flag_embedding.encode_calls[0]["return_sparse"] is True


def test_bge_provider_empty_input_does_not_call_model(fake_flag_embedding):
    provider = BgeM3EmbeddingProvider(model_path="C:/Download/bge-m3", dimension=4)

    assert provider.embed([]) == []
    assert fake_flag_embedding.encode_calls == []


def test_bge_provider_rejects_dimension_mismatch(fake_flag_embedding):
    provider = BgeM3EmbeddingProvider(model_path="C:/Download/bge-m3", dimension=8)

    with pytest.raises(MemoryValidationError, match="dimension"):
        provider.embed(["结构化单据识别规则"])


def test_bge_provider_requires_sparse_output_when_requested(fake_flag_embedding, monkeypatch):
    def encode_without_sparse(self, **kwargs):
        return {"dense_vecs": DenseResult([[1.0] * 4])}

    monkeypatch.setattr(FakeBgeModel, "encode", encode_without_sparse)
    provider = BgeM3EmbeddingProvider(model_path="C:/Download/bge-m3", dimension=4, return_sparse=True)

    with pytest.raises(MemoryValidationError, match="lexical_weights"):
        provider.embed(["结构化单据识别规则"])


def test_memory_embedding_text_contains_searchable_memory_fields():
    item = MemoryItem(
        id="mem_1",
        name="language",
        type="preference",
        description="用户偏好中文。",
        content="用户偏好使用中文回答。",
        tags=["preference", "language"],
    )

    text = memory_embedding_text(item)

    assert "name: language" in text
    assert "type: preference" in text
    assert "description: 用户偏好中文。" in text
    assert "tags: preference, language" in text
    assert "content: 用户偏好使用中文回答。" in text
    assert sha256_text(text) == sha256_text(text)
