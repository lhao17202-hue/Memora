import os
import subprocess
import sys
import time
import uuid
from importlib import import_module
from pathlib import Path

import pytest

from memora.config import MemoryConfig
from memora.manager import MemoryManager
from memora.vector_store import QDRANT_DENSE_VECTOR_NAME, QDRANT_SPARSE_VECTOR_NAME, qdrant_point_id


RUN_TRUE_VALUES = {"1", "true", "yes", "on"}
DEFAULT_BGE_MODEL_PATH = Path(r"C:\Download\bge-m3")
DEFAULT_QDRANT_URL = "http://127.0.0.1:6333"


def _e2e_enabled() -> bool:
    return os.environ.get("RUN_MEMORA_E2E", "").strip().lower() in RUN_TRUE_VALUES


def _log(message: str) -> None:
    print(f"[memora-e2e] {message}", flush=True)


def _clip(value: str, *, limit: int = 4000) -> str:
    value = value.strip()
    if len(value) <= limit:
        return value
    return value[:limit] + f"\n... clipped {len(value) - limit} chars ..."


def _run_memora(root: Path, env_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.setdefault("HF_OFFLINE", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    command = [sys.executable, "-m", "memora", "--root", str(root), "--env-file", str(env_path), *args]
    _log("run: " + " ".join(str(part) for part in command))
    started = time.perf_counter()
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        env=env,
        timeout=int(os.environ.get("MEMORA_E2E_COMMAND_TIMEOUT", "240")),
    )
    elapsed = time.perf_counter() - started
    _log(f"done: exit={result.returncode} elapsed={elapsed:.2f}s args={' '.join(args)}")
    if result.stdout.strip():
        _log("stdout:\n" + _clip(result.stdout))
    if result.stderr.strip():
        _log("stderr:\n" + _clip(result.stderr))
    return result


def _assert_ok(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def _saved_memory_id(result: subprocess.CompletedProcess[str]) -> str:
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "saved":
            return parts[1]
    raise AssertionError(f"could not parse saved memory id from stdout:\n{result.stdout}")


def _timed(label: str, fn):
    _log(f"step: {label}")
    started = time.perf_counter()
    result = fn()
    _log(f"done: {label} elapsed={time.perf_counter() - started:.2f}s")
    return result


def _require_module(name: str) -> None:
    _log(f"preflight: import {name}")
    try:
        import_module(name)
    except Exception as exc:
        pytest.fail(f"RUN_MEMORA_E2E=1 but required module '{name}' cannot be imported: {exc!r}")
    _log(f"preflight ok: import {name}")


def _require_qdrant(qdrant_url: str) -> None:
    _log(f"preflight: connect qdrant {qdrant_url}")
    try:
        from qdrant_client import QdrantClient

        QdrantClient(url=qdrant_url, timeout=10).get_collections()
    except Exception as exc:
        pytest.fail(f"RUN_MEMORA_E2E=1 but Qdrant is not reachable at {qdrant_url}: {exc!r}")
    _log(f"preflight ok: qdrant reachable {qdrant_url}")


def _qdrant_client(qdrant_url: str):
    from qdrant_client import QdrantClient

    return QdrantClient(url=qdrant_url, timeout=10)


def _assert_qdrant_collection_ready(qdrant_url: str, collection: str) -> None:
    client = _qdrant_client(qdrant_url)
    assert client.collection_exists(collection_name=collection)
    info = client.get_collection(collection_name=collection)
    params = getattr(getattr(info, "config", None), "params", None)
    vectors = getattr(params, "vectors", None)
    assert isinstance(vectors, dict)
    assert QDRANT_DENSE_VECTOR_NAME in vectors
    dense = vectors[QDRANT_DENSE_VECTOR_NAME]
    assert int(getattr(dense, "size")) == 1024
    sparse_vectors = getattr(params, "sparse_vectors", None)
    assert isinstance(sparse_vectors, dict)
    assert QDRANT_SPARSE_VECTOR_NAME in sparse_vectors


def _assert_qdrant_payloads(qdrant_url: str, collection: str, expected_types: dict[str, str]) -> None:
    client = _qdrant_client(qdrant_url)
    points = client.retrieve(
        collection_name=collection,
        ids=[qdrant_point_id(memory_id) for memory_id in expected_types],
        with_payload=True,
        with_vectors=False,
    )
    payloads = {}
    for point in points:
        payload = getattr(point, "payload", None) or {}
        memory_id = payload.get("memory_id")
        if isinstance(memory_id, str):
            payloads[memory_id] = payload
    assert set(payloads) == set(expected_types)
    for memory_id, memory_type in expected_types.items():
        payload = payloads[memory_id]
        assert payload["memory_id"] == memory_id
        assert payload["type"] == memory_type
        assert payload["status"] == "active"
        assert payload["provider"] == "bge"
        assert payload["model"] == "bge-m3"
        assert payload["dimension"] == 1024
        assert payload["vector_store"] == "qdrant"
        assert payload["retrieval_mode"] == "hybrid"
        assert payload["sparse_enabled"] is True
        assert isinstance(payload["text_hash"], str)
        assert payload["text_hash"]


def _bge_qdrant_config(root: Path, *, model_path: Path, qdrant_url: str, collection: str) -> MemoryConfig:
    return MemoryConfig(
        root_dir=root,
        memory_backend="sqlite",
        rag_enabled=True,
        embedding_provider="bge",
        embedding_model="bge-m3",
        embedding_model_path=model_path,
        embedding_dimension=1024,
        embedding_batch_size=8,
        embedding_fp16=True,
        embedding_sparse=True,
        vector_store="qdrant",
        vector_store_options={
            "url": qdrant_url,
            "collection": collection,
            "timeout": 30.0,
        },
        retrieval_mode="hybrid",
        keyword_recall="auto",
        hybrid_prefetch_limit=30,
    )


def _write_env_file(env_path: Path, *, config: MemoryConfig, qdrant_url: str, collection: str) -> None:
    env_path.write_text(
        "\n".join(
            [
                f"MEMORA_BACKEND={config.memory_backend}",
                f"MEMORA_RAG={str(config.rag_enabled).lower()}",
                f"MEMORA_EMBEDDING_PROVIDER={config.embedding_provider}",
                f"MEMORA_EMBEDDING_MODEL={config.embedding_model}",
                f"MEMORA_EMBEDDING_MODEL_PATH={config.embedding_model_path}",
                f"MEMORA_EMBEDDING_DIMENSION={config.embedding_dimension}",
                f"MEMORA_EMBEDDING_BATCH_SIZE={config.embedding_batch_size}",
                f"MEMORA_EMBEDDING_FP16={str(config.embedding_fp16).lower()}",
                f"MEMORA_EMBEDDING_SPARSE={str(config.embedding_sparse).lower()}",
                f"MEMORA_VECTOR_STORE={config.vector_store}",
                f"MEMORA_VECTOR_STORE_URL={qdrant_url}",
                f"MEMORA_VECTOR_STORE_COLLECTION={collection}",
                f"MEMORA_VECTOR_STORE_TIMEOUT={config.vector_store_options['timeout']}",
                f"MEMORA_RETRIEVAL_MODE={config.retrieval_mode}",
                f"MEMORA_KEYWORD_RECALL={config.keyword_recall}",
                f"MEMORA_HYBRID_PREFETCH_LIMIT={config.hybrid_prefetch_limit}",
                "HF_OFFLINE=1",
            ]
        ),
        encoding="utf-8",
    )
    _log(f"wrote env file: {env_path}")


def _cleanup_qdrant_collection(qdrant_url: str, collection: str) -> None:
    _log(f"cleanup: collection={collection}")
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=qdrant_url, timeout=10)
        if client.collection_exists(collection_name=collection):
            client.delete_collection(collection_name=collection)
            _log(f"cleanup ok: deleted collection={collection}")
        else:
            _log(f"cleanup skipped: collection not found={collection}")
    except Exception:
        _log(f"cleanup skipped: failed to inspect/delete collection={collection}")


@pytest.mark.e2e
def test_real_bge_m3_qdrant_cli_flow(tmp_path: Path):
    if not _e2e_enabled():
        pytest.skip("set RUN_MEMORA_E2E=1 to run the real BGE-M3 + Qdrant E2E test")

    suite_started = time.perf_counter()
    _log("start real BGE-M3 + Qdrant CLI E2E")
    _require_module("FlagEmbedding")
    _require_module("qdrant_client")

    model_path = Path(os.environ.get("MEMORA_E2E_BGE_MODEL_PATH", str(DEFAULT_BGE_MODEL_PATH)))
    _log(f"preflight: model_path={model_path}")
    assert model_path.exists(), f"RUN_MEMORA_E2E=1 but BGE-M3 model path does not exist: {model_path}"
    _log("preflight ok: model path exists")

    qdrant_url = os.environ.get("MEMORA_E2E_QDRANT_URL", DEFAULT_QDRANT_URL)
    _require_qdrant(qdrant_url)
    configured_collection = os.environ.get("MEMORA_E2E_QDRANT_COLLECTION")
    collection = configured_collection or f"memora_e2e_{uuid.uuid4().hex}"
    root = tmp_path / ".memora"
    env_path = tmp_path / ".env"
    config = _bge_qdrant_config(root, model_path=model_path, qdrant_url=qdrant_url, collection=collection)
    _log(f"test root: {root}")
    _log(f"qdrant collection: {collection}")
    _write_env_file(env_path, config=config, qdrant_url=qdrant_url, collection=collection)

    try:
        _log("step: init storage and qdrant collection")
        initialized = _run_memora(root, env_path, "init")
        _assert_ok(initialized)

        _log("step: save preference memory")
        language = _run_memora(
            root,
            env_path,
            "save",
            "--type",
            "preference",
            "--name",
            "language",
            "--description",
            "Response language preference.",
            "--content",
            "The user prefers answers in Chinese.",
        )
        _assert_ok(language)
        language_id = _saved_memory_id(language)

        _log("step: save project memory")
        project = _run_memora(
            root,
            env_path,
            "save",
            "--type",
            "project",
            "--name",
            "test-stack",
            "--description",
            "Project test stack.",
            "--content",
            "This project uses pytest as its test framework.",
        )
        _assert_ok(project)
        project_id = _saved_memory_id(project)

        _timed(
            "assert qdrant collection and payloads after cli saves",
            lambda: (
                _assert_qdrant_collection_ready(qdrant_url, collection),
                _assert_qdrant_payloads(qdrant_url, collection, {language_id: "preference", project_id: "project"}),
            ),
        )

        _log("step: search preference memory")
        language_search = _run_memora(root, env_path, "search", "answer in Chinese")
        _assert_ok(language_search)
        assert "language" in language_search.stdout

        _log("step: search project memory")
        project_search = _run_memora(root, env_path, "search", "pytest test framework")
        _assert_ok(project_search)
        assert "test-stack" in project_search.stdout

        _log("step: verify memory and vector indexes")
        verified = _run_memora(root, env_path, "verify")
        _assert_ok(verified)
        assert "verified 2 memories" in verified.stdout
        assert "index_ok=True" in verified.stdout
        assert "vector_ok=True" in verified.stdout

        _log("step: rebuild indexes")
        rebuilt = _run_memora(root, env_path, "rebuild-index")
        _assert_ok(rebuilt)
        assert "rebuilt index" in rebuilt.stdout
        _timed(
            "assert qdrant payloads after cli rebuild",
            lambda: _assert_qdrant_payloads(qdrant_url, collection, {language_id: "preference", project_id: "project"}),
        )

        _log("step: search after rebuild")
        after_rebuild = _run_memora(root, env_path, "search", "Chinese response preference")
        _assert_ok(after_rebuild)
        assert "language" in after_rebuild.stdout
        _log(f"finished real E2E elapsed={time.perf_counter() - suite_started:.2f}s")
    finally:
        keep_collection = os.environ.get("MEMORA_E2E_KEEP_QDRANT_COLLECTION", "").strip().lower() in RUN_TRUE_VALUES
        if not keep_collection and configured_collection is None:
            _cleanup_qdrant_collection(qdrant_url, collection)


@pytest.mark.e2e
def test_real_bge_m3_qdrant_in_process_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    if not _e2e_enabled():
        pytest.skip("set RUN_MEMORA_E2E=1 to run the real in-process BGE-M3 + Qdrant E2E test")

    suite_started = time.perf_counter()
    _log("start real BGE-M3 + Qdrant in-process E2E")
    _require_module("FlagEmbedding")
    _require_module("qdrant_client")

    model_path = Path(os.environ.get("MEMORA_E2E_BGE_MODEL_PATH", str(DEFAULT_BGE_MODEL_PATH)))
    _log(f"preflight: model_path={model_path}")
    assert model_path.exists(), f"RUN_MEMORA_E2E=1 but BGE-M3 model path does not exist: {model_path}"
    _log("preflight ok: model path exists")

    qdrant_url = os.environ.get("MEMORA_E2E_QDRANT_URL", DEFAULT_QDRANT_URL)
    _require_qdrant(qdrant_url)
    configured_collection = os.environ.get("MEMORA_E2E_QDRANT_COLLECTION")
    collection = configured_collection or f"memora_e2e_inproc_{uuid.uuid4().hex}"
    root = tmp_path / ".memora-inprocess"
    config = _bge_qdrant_config(root, model_path=model_path, qdrant_url=qdrant_url, collection=collection)
    _log(f"test root: {root}")
    _log(f"qdrant collection: {collection}")

    monkeypatch.setenv("HF_OFFLINE", "1")
    try:
        manager = _timed("construct MemoryManager once", lambda: MemoryManager(config))
        _timed("init storage and qdrant collection", manager.init_storage)

        language = _timed(
            "save preference memory",
            lambda: manager.save_memory(
                "preference",
                "The user prefers answers in Chinese.",
                "Response language preference.",
                name="language",
            ),
        )
        project = _timed(
            "save project memory",
            lambda: manager.save_memory(
                "project",
                "This project uses pytest as its test framework.",
                "Project test stack.",
                name="test-stack",
            ),
        )

        _timed(
            "assert qdrant collection and payloads after in-process saves",
            lambda: (
                _assert_qdrant_collection_ready(qdrant_url, collection),
                _assert_qdrant_payloads(qdrant_url, collection, {language.id: "preference", project.id: "project"}),
            ),
        )

        language_results = _timed("search preference memory", lambda: manager.retrieve_memory("answer in Chinese"))
        assert language_results
        assert language_results[0].memory.id == language.id

        project_results = _timed("search project memory", lambda: manager.retrieve_memory("pytest test framework"))
        assert project_results
        assert project_results[0].memory.id == project.id

        verified = _timed("verify memory and vector indexes", manager.verify_memories)
        assert verified["checked"] == 2
        assert verified["index_ok"] is True
        assert verified["vector_ok"] is True
        assert verified["vector_missing"] == []
        assert verified["vector_orphans"] == []
        assert verified["embedding_mismatches"] == []

        _timed("rebuild indexes", manager.rebuild_index)
        _timed(
            "assert qdrant payloads after in-process rebuild",
            lambda: _assert_qdrant_payloads(qdrant_url, collection, {language.id: "preference", project.id: "project"}),
        )

        after_rebuild = _timed("search after rebuild", lambda: manager.retrieve_memory("Chinese response preference"))
        assert after_rebuild
        assert after_rebuild[0].memory.id == language.id
        _log(f"finished real in-process E2E elapsed={time.perf_counter() - suite_started:.2f}s")
    finally:
        keep_collection = os.environ.get("MEMORA_E2E_KEEP_QDRANT_COLLECTION", "").strip().lower() in RUN_TRUE_VALUES
        if not keep_collection and configured_collection is None:
            _cleanup_qdrant_collection(qdrant_url, collection)
