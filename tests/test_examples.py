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


def test_openai_memory_system_example_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = subprocess.run(
        [sys.executable, "examples/openai_memory_system_runtime.py"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Set OPENAI_API_KEY before running this example." in result.stderr



def test_openai_memory_demo_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    result = subprocess.run(
        [sys.executable, "examples/openai_memory_demo.py", "list"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Set OPENAI_API_KEY before running this demo." in result.stderr



def test_openai_memory_demo_does_not_hardcode_api_key():
    from pathlib import Path

    source = Path("examples/openai_memory_demo.py").read_text(encoding="utf-8")

    assert "sk-" not in source
    assert "OPENAI_API_KEY" in source
    assert "OPENAI_BASE_URL" in source
    assert "OPENAI_MODEL" in source


def test_openai_relation_schema_supports_supersede():
    from pathlib import Path
    import importlib.util

    module_path = Path("examples/openai_memory_clients.py")
    spec = importlib.util.spec_from_file_location("openai_memory_clients", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert "supersede" in module.RELATION_DECISION_JSON_SCHEMA["properties"]["kind"]["enum"]



def test_openai_json_client_falls_back_to_chat_completions_for_compatible_apis():
    from examples.openai_memory_clients import OpenAIExtractionClient

    class FailingResponses:
        def create(self, **kwargs):
            raise RuntimeError("bad_response_body")

    class ChatCompletions:
        def __init__(self):
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            message = type("Message", (), {"content": '{"should_remember": false, "memories": []}'})()
            choice = type("Choice", (), {"message": message})()
            return type("ChatResponse", (), {"choices": [choice]})()

    class FakeClient:
        def __init__(self):
            self.responses = FailingResponses()
            self.chat_completions = ChatCompletions()
            self.chat = type("Chat", (), {"completions": self.chat_completions})()

    fake_client = FakeClient()
    result = OpenAIExtractionClient(fake_client, "gpt-5.5").complete([{"role": "user", "content": "nothing durable"}])

    assert result == '{"should_remember": false, "memories": []}'
    assert fake_client.chat_completions.kwargs["model"] == "gpt-5.5"
    assert fake_client.chat_completions.kwargs["response_format"] == {"type": "json_object"}


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
    assert "action=superseded reason=llm_semantic_conflict_high_confidence_replace" in result.stdout
    assert "supersedes=['mem_" in result.stdout
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
