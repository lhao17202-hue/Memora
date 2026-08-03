import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest


RUN_TRUE_VALUES = {"1", "true", "yes", "on"}
DEFAULT_BGE_MODEL_PATH = Path(r"C:\Download\bge-m3")
DEFAULT_QDRANT_URL = "http://127.0.0.1:6333"


def _e2e_enabled() -> bool:
    return os.environ.get("RUN_MEMORA_E2E", "").strip().lower() in RUN_TRUE_VALUES


def _run_memora(root: Path, env_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.setdefault("HF_OFFLINE", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return subprocess.run(
        [sys.executable, "-m", "memora", "--root", str(root), "--env-file", str(env_path), *args],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        timeout=int(os.environ.get("MEMORA_E2E_COMMAND_TIMEOUT", "240")),
    )


def _assert_ok(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def _write_env_file(env_path: Path, *, model_path: Path, qdrant_url: str, collection: str) -> None:
    env_path.write_text(
        "\n".join(
            [
                "MEMORA_BACKEND=sqlite",
                "MEMORA_RAG=true",
                "MEMORA_EMBEDDING_PROVIDER=bge",
                "MEMORA_EMBEDDING_MODEL=bge-m3",
                f"MEMORA_EMBEDDING_MODEL_PATH={model_path}",
                "MEMORA_EMBEDDING_DIMENSION=1024",
                "MEMORA_EMBEDDING_BATCH_SIZE=8",
                "MEMORA_EMBEDDING_FP16=true",
                "MEMORA_EMBEDDING_SPARSE=true",
                "MEMORA_VECTOR_STORE=qdrant",
                f"MEMORA_VECTOR_STORE_URL={qdrant_url}",
                f"MEMORA_VECTOR_STORE_COLLECTION={collection}",
                "MEMORA_VECTOR_STORE_TIMEOUT=30",
                "MEMORA_RETRIEVAL_MODE=hybrid",
                "MEMORA_KEYWORD_RECALL=auto",
                "MEMORA_HYBRID_PREFETCH_LIMIT=30",
                "HF_OFFLINE=1",
            ]
        ),
        encoding="utf-8",
    )


def _cleanup_qdrant_collection(qdrant_url: str, collection: str) -> None:
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=qdrant_url, timeout=10)
        if client.collection_exists(collection_name=collection):
            client.delete_collection(collection_name=collection)
    except Exception:
        pass


@pytest.mark.e2e
def test_real_bge_m3_qdrant_cli_flow(tmp_path: Path):
    if not _e2e_enabled():
        pytest.skip("set RUN_MEMORA_E2E=1 to run the real BGE-M3 + Qdrant E2E test")

    pytest.importorskip("FlagEmbedding")
    pytest.importorskip("qdrant_client")

    model_path = Path(os.environ.get("MEMORA_E2E_BGE_MODEL_PATH", str(DEFAULT_BGE_MODEL_PATH)))
    if not model_path.exists():
        pytest.skip(f"BGE-M3 model path does not exist: {model_path}")

    qdrant_url = os.environ.get("MEMORA_E2E_QDRANT_URL", DEFAULT_QDRANT_URL)
    configured_collection = os.environ.get("MEMORA_E2E_QDRANT_COLLECTION")
    collection = configured_collection or f"memora_e2e_{uuid.uuid4().hex}"
    root = tmp_path / ".memora"
    env_path = tmp_path / ".env"
    _write_env_file(env_path, model_path=model_path, qdrant_url=qdrant_url, collection=collection)

    try:
        initialized = _run_memora(root, env_path, "init")
        _assert_ok(initialized)

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

        language_search = _run_memora(root, env_path, "search", "answer in Chinese")
        _assert_ok(language_search)
        assert "language" in language_search.stdout

        project_search = _run_memora(root, env_path, "search", "pytest test framework")
        _assert_ok(project_search)
        assert "test-stack" in project_search.stdout

        verified = _run_memora(root, env_path, "verify")
        _assert_ok(verified)
        assert "verified 2 memories" in verified.stdout
        assert "index_ok=True" in verified.stdout
        assert "vector_ok=True" in verified.stdout

        rebuilt = _run_memora(root, env_path, "rebuild-index")
        _assert_ok(rebuilt)
        assert "rebuilt index" in rebuilt.stdout

        after_rebuild = _run_memora(root, env_path, "search", "Chinese response preference")
        _assert_ok(after_rebuild)
        assert "language" in after_rebuild.stdout
    finally:
        keep_collection = os.environ.get("MEMORA_E2E_KEEP_QDRANT_COLLECTION", "").strip().lower() in RUN_TRUE_VALUES
        if not keep_collection and configured_collection is None:
            _cleanup_qdrant_collection(qdrant_url, collection)
