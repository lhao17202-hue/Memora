import subprocess
import sys


def test_openai_relation_example_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = subprocess.run(
        [sys.executable, "examples/openai_llm_relation_runtime.py"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Set OPENAI_API_KEY before running this example." in result.stderr


def test_openai_full_turn_example_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = subprocess.run(
        [sys.executable, "examples/openai_full_memory_turn_runtime.py"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Set OPENAI_API_KEY before running this example." in result.stderr


def test_llm_relation_runtime_example_runs_successfully():
    result = subprocess.run(
        [sys.executable, "examples/llm_relation_runtime.py"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "=== LLM Merge ===" in result.stdout
    assert "action=updated reason=llm_semantic_merge" in result.stdout
    assert "Prefer concise answers with short summaries." in result.stdout
    assert "=== LLM Conflict ===" in result.stdout
    assert "action=updated reason=llm_semantic_conflict_high_confidence_replace" in result.stdout
    assert "=== Fallback ===" in result.stdout
    assert "action=updated reason=semantic_merge" in result.stdout


def test_simple_agent_runtime_example_runs_successfully():
    result = subprocess.run(
        [sys.executable, "examples/simple_agent_runtime.py"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "=== Memory Context ===" in result.stdout
    assert "=== Assistant ===" in result.stdout
    assert "用户偏好使用中文回答。" in result.stdout
