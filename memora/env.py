"""Environment file helpers for Memora configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

from .errors import MemoryValidationError

_BOOL_FIELDS = {
    "MEMORA_RAG": "rag_enabled",
    "MEMORA_FTS_ENABLED": "fts_enabled",
    "MEMORA_EMBEDDING_FP16": "embedding_fp16",
    "MEMORA_EMBEDDING_SPARSE": "embedding_sparse",
    "MEMORA_SEMANTIC_WRITE_RELATIONS": "semantic_write_relations_enabled",
}

_INT_FIELDS = {
    "MEMORA_FTS_CANDIDATE_LIMIT": "fts_candidate_limit",
    "MEMORA_EMBEDDING_DIMENSION": "embedding_dimension",
    "MEMORA_EMBEDDING_BATCH_SIZE": "embedding_batch_size",
    "MEMORA_HYBRID_PREFETCH_LIMIT": "hybrid_prefetch_limit",
    "MEMORA_VECTOR_CANDIDATE_LIMIT": "vector_candidate_limit",
    "MEMORA_KEYWORD_CANDIDATE_LIMIT": "keyword_candidate_limit",
    "MEMORA_RERANK_CANDIDATE_LIMIT": "rerank_candidate_limit",
    "MEMORA_MAX_RETRIEVED_MEMORIES": "max_retrieved_memories",
    "MEMORA_MAX_MEMORY_PROMPT_TOKENS": "max_memory_prompt_tokens",
    "MEMORA_MAX_MEMORY_CONTENT_CHARS": "max_memory_content_chars",
}

_FLOAT_FIELDS = {
    "MEMORA_MIN_SEMANTIC_SCORE": "min_semantic_score",
    "MEMORA_SEMANTIC_RELATION_THRESHOLD": "semantic_relation_threshold",
    "MEMORA_SEMANTIC_MERGE_THRESHOLD": "semantic_merge_threshold",
    "MEMORA_SEMANTIC_CONFLICT_THRESHOLD": "semantic_conflict_threshold",
}

_STRING_FIELDS = {
    "MEMORA_ROOT": "root_dir",
    "MEMORA_BACKEND": "memory_backend",
    "MEMORA_SQLITE_PATH": "sqlite_path",
    "MEMORA_EMBEDDING_PROVIDER": "embedding_provider",
    "MEMORA_EMBEDDING_MODEL": "embedding_model",
    "MEMORA_EMBEDDING_MODEL_PATH": "embedding_model_path",
    "MEMORA_VECTOR_STORE": "vector_store",
    "MEMORA_VECTOR_PATH": "vector_path",
    "MEMORA_RETRIEVAL_MODE": "retrieval_mode",
    "MEMORA_RERANKER": "reranker",
}

_VECTOR_STORE_BOOL_OPTIONS = {
    "MEMORA_VECTOR_STORE_PREFER_GRPC": "prefer_grpc",
    "MEMORA_VECTOR_STORE_RECREATE_COLLECTION": "recreate_collection",
}

_VECTOR_STORE_INT_OPTIONS = {
    "MEMORA_VECTOR_STORE_PORT": "port",
}

_VECTOR_STORE_FLOAT_OPTIONS = {
    "MEMORA_VECTOR_STORE_TIMEOUT": "timeout",
}

_VECTOR_STORE_STRING_OPTIONS = {
    "MEMORA_VECTOR_STORE_URL": "url",
    "MEMORA_VECTOR_STORE_HOST": "host",
    "MEMORA_VECTOR_STORE_API_KEY": "api_key",
    "MEMORA_VECTOR_STORE_COLLECTION": "collection",
}

_VECTOR_STORE_OPTIONS = _VECTOR_STORE_BOOL_OPTIONS | _VECTOR_STORE_INT_OPTIONS | _VECTOR_STORE_FLOAT_OPTIONS | _VECTOR_STORE_STRING_OPTIONS
_ENV_TO_CONFIG = _STRING_FIELDS | _BOOL_FIELDS | _INT_FIELDS | _FLOAT_FIELDS
_ENV_KEYS = set(_ENV_TO_CONFIG) | set(_VECTOR_STORE_OPTIONS)

_OS_ENV_KEYS = {"HF_OFFLINE"}

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def load_env_file(path: str | Path = ".env") -> dict[str, str]:
    env_path = Path(path)
    if not env_path.exists():
        return {}
    values = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def merge_env(file_env: Mapping[str, str]) -> dict[str, str]:
    merged = dict(file_env)
    for key in _ENV_KEYS | _OS_ENV_KEYS:
        if key in os.environ:
            merged[key] = os.environ[key]
    return merged


def apply_env_to_os(env: Mapping[str, str]) -> None:
    for key in _OS_ENV_KEYS:
        if key in env and key not in os.environ:
            os.environ[key] = env[key]


def config_kwargs_from_env(env: Mapping[str, str]) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    vector_store_options: dict[str, object] = {}
    for key, value in env.items():
        field = _ENV_TO_CONFIG.get(key)
        if field is not None:
            if key in _BOOL_FIELDS:
                kwargs[field] = _parse_bool(value, key)
            elif key in _INT_FIELDS:
                kwargs[field] = _parse_int(value, key)
            elif key in _FLOAT_FIELDS:
                kwargs[field] = _parse_float(value, key)
            else:
                kwargs[field] = value
            continue

        option = _VECTOR_STORE_OPTIONS.get(key)
        if option is None:
            continue
        if key in _VECTOR_STORE_BOOL_OPTIONS:
            vector_store_options[option] = _parse_bool(value, key)
        elif key in _VECTOR_STORE_INT_OPTIONS:
            vector_store_options[option] = _parse_int(value, key)
        elif key in _VECTOR_STORE_FLOAT_OPTIONS:
            vector_store_options[option] = _parse_float(value, key)
        else:
            vector_store_options[option] = value
    if vector_store_options:
        kwargs["vector_store_options"] = vector_store_options
    return kwargs


def _parse_bool(value: str, key: str) -> bool:
    lowered = value.strip().lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    raise MemoryValidationError(f"invalid boolean value for {key}: {value}")


def _parse_int(value: str, key: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise MemoryValidationError(f"invalid integer value for {key}: {value}") from exc


def _parse_float(value: str, key: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise MemoryValidationError(f"invalid float value for {key}: {value}") from exc
