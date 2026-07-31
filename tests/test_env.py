import os

import pytest

from memora.env import apply_env_to_os, config_kwargs_from_env, load_env_file, merge_env
from memora.errors import MemoryValidationError


def test_load_env_file_parses_simple_values(tmp_path):
    path = tmp_path / ".env"
    path.write_text(
        """
# comment
MEMORA_RAG=true
MEMORA_EMBEDDING_MODEL_PATH="C:\Download\bge-m3"
MEMORA_EMBEDDING_MODEL='bge-m3'

MEMORA_EMBEDDING_BATCH_SIZE=8
""".strip(),
        encoding="utf-8",
    )

    values = load_env_file(path)

    assert values == {
        "MEMORA_RAG": "true",
        "MEMORA_EMBEDDING_MODEL_PATH": "C:\Download\bge-m3",
        "MEMORA_EMBEDDING_MODEL": "bge-m3",
        "MEMORA_EMBEDDING_BATCH_SIZE": "8",
    }


def test_config_kwargs_from_env_coerces_memora_values():
    kwargs = config_kwargs_from_env(
        {
            "MEMORA_BACKEND": "sqlite",
            "MEMORA_SQLITE_PATH": ".memora/memora.sqlite3",
            "MEMORA_RAG": "yes",
            "MEMORA_EMBEDDING_PROVIDER": "bge",
            "MEMORA_EMBEDDING_DIMENSION": "1024",
            "MEMORA_EMBEDDING_BATCH_SIZE": "8",
            "MEMORA_EMBEDDING_FP16": "on",
            "MEMORA_EMBEDDING_SPARSE": "true",
            "MEMORA_RETRIEVAL_MODE": "hybrid",
            "MEMORA_KEYWORD_RECALL": "scan",
            "MEMORA_HYBRID_PREFETCH_LIMIT": "100",
            "MEMORA_MIN_SEMANTIC_SCORE": "0.30",
            "MEMORA_VECTOR_STORE": "qdrant",
            "MEMORA_VECTOR_STORE_URL": "http://localhost:6333",
            "MEMORA_VECTOR_STORE_PORT": "6333",
            "MEMORA_VECTOR_STORE_COLLECTION": "memora_memories",
            "MEMORA_VECTOR_STORE_TIMEOUT": "5.5",
            "MEMORA_VECTOR_STORE_PREFER_GRPC": "false",
        }
    )

    assert kwargs == {
        "memory_backend": "sqlite",
        "sqlite_path": ".memora/memora.sqlite3",
        "rag_enabled": True,
        "embedding_provider": "bge",
        "embedding_dimension": 1024,
        "embedding_batch_size": 8,
        "embedding_fp16": True,
        "embedding_sparse": True,
        "retrieval_mode": "hybrid",
        "keyword_recall": "scan",
        "hybrid_prefetch_limit": 100,
        "min_semantic_score": 0.30,
        "vector_store": "qdrant",
        "vector_store_options": {
            "url": "http://localhost:6333",
            "port": 6333,
            "collection": "memora_memories",
            "timeout": 5.5,
            "prefer_grpc": False,
        },
    }


def test_config_kwargs_from_env_reports_invalid_values():
    with pytest.raises(MemoryValidationError, match="MEMORA_VECTOR_STORE_PORT"):
        config_kwargs_from_env({"MEMORA_VECTOR_STORE_PORT": "abc"})

    with pytest.raises(MemoryValidationError, match="MEMORA_EMBEDDING_SPARSE"):
        config_kwargs_from_env({"MEMORA_EMBEDDING_SPARSE": "maybe"})

    with pytest.raises(MemoryValidationError, match="MEMORA_VECTOR_STORE_PREFER_GRPC"):
        config_kwargs_from_env({"MEMORA_VECTOR_STORE_PREFER_GRPC": "maybe"})


def test_merge_env_gives_process_env_precedence(monkeypatch):
    monkeypatch.setenv("MEMORA_EMBEDDING_PROVIDER", "hash")

    merged = merge_env({"MEMORA_EMBEDDING_PROVIDER": "bge", "MEMORA_RAG": "true"})

    assert merged["MEMORA_EMBEDDING_PROVIDER"] == "hash"
    assert merged["MEMORA_RAG"] == "true"


def test_merge_env_includes_vector_store_option_keys(monkeypatch):
    monkeypatch.setenv("MEMORA_VECTOR_STORE_COLLECTION", "from-process")

    merged = merge_env({"MEMORA_VECTOR_STORE_COLLECTION": "from-file"})

    assert merged["MEMORA_VECTOR_STORE_COLLECTION"] == "from-process"


def test_apply_env_to_os_sets_hf_offline_without_overriding(monkeypatch):
    monkeypatch.delenv("HF_OFFLINE", raising=False)
    apply_env_to_os({"HF_OFFLINE": "1"})
    assert os.environ["HF_OFFLINE"] == "1"

    monkeypatch.setenv("HF_OFFLINE", "0")
    apply_env_to_os({"HF_OFFLINE": "1"})
    assert os.environ["HF_OFFLINE"] == "0"
