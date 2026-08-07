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



def test_user_memory_cli_help_exposes_four_commands_without_api_key():
    result = subprocess.run(
        [sys.executable, "examples/memora_user_cli.py", "/cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "extract" in result.stdout
    assert "query" in result.stdout
    assert "list" in result.stdout
    assert "clear" in result.stdout



def test_user_memory_cli_accepts_git_bash_converted_cli_entrypoint():
    result = subprocess.run(
        [sys.executable, "examples/memora_user_cli.py", "C:/Program Files/Git/cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "extract" in result.stdout
    assert "query" in result.stdout
    assert "list" in result.stdout
    assert "clear" in result.stdout



def test_user_memory_cli_uses_safe_env_configuration():
    from pathlib import Path

    source = Path("examples/memora_user_cli.py").read_text(encoding="utf-8")

    assert "sk-" not in source
    assert "OPENAI_API_KEY" in source
    assert "OPENAI_BASE_URL" in source
    assert "OPENAI_MODEL" in source
    assert "MEMORA_VECTOR_STORE" in source
    assert "qdrant" in source
    assert "bge" in source
    assert "sqlite" in source



def test_user_memory_cli_dataset_files_exist():
    from pathlib import Path

    assert Path("examples/demo_dataset/messages/sample_turn.json").exists()
    assert Path("examples/demo_dataset/working_memory/sample_turn.json").exists()



def test_user_memory_cli_defaults_to_demo_specific_env_file():
    from examples import memora_user_cli

    parser = memora_user_cli.build_parser()
    args = parser.parse_args(["/cli", "list"])

    assert args.env_file == ".memora-user-cli.env"



def test_user_memory_cli_loads_openai_settings_from_env_file(monkeypatch, tmp_path):
    import sys
    import types

    from examples import memora_user_cli

    env_file = tmp_path / ".memora-user-cli.env"
    env_file.write_text(
        "OPENAI_API_KEY=file-key\n"
        "OPENAI_BASE_URL=https://example.test/v1\n"
        "OPENAI_MODEL=file-model\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))

    env = memora_user_cli._load_demo_env(str(env_file))
    client = memora_user_cli.make_openai_client(env)

    assert client is not None
    assert env["OPENAI_MODEL"] == "file-model"
    assert captured == {"api_key": "file-key", "base_url": "https://example.test/v1"}


def test_user_memory_cli_query_prints_score_breakdown_and_metadata(capsys, monkeypatch):
    from datetime import datetime
    from types import SimpleNamespace

    from examples import memora_user_cli
    from memora.schema import MemoryItem, MemorySearchResult

    memory = MemoryItem(
        id="mem_debug",
        name="sqlite-fixture-scope",
        description="SQLite in-memory fixtures should reuse a session-scoped connection.",
        type="tool",
        content="SQLite :memory: 单测数据丢失时，优先检查 fixture scope。",
        user_id="debug-user",
        project_id="debug-project",
        workspace_id="debug-workspace",
        tags=["testing", "database"],
        source="conversation",
        confidence=0.82,
        weight=7,
        status="active",
        created_at=datetime(2026, 8, 6, 10, 0, 0),
        updated_at=datetime(2026, 8, 6, 10, 5, 0),
        last_accessed_at=datetime(2026, 8, 6, 10, 6, 0),
        access_count=3,
        supersedes=["mem_old"],
        related=["mem_related"],
    )
    search_result = MemorySearchResult(
        memory=memory,
        similarity_score=0.91,
        importance_score=0.70,
        recency_score=0.60,
        access_score=0.50,
        final_score=0.8123,
        reason="matched_keyword",
        semantic_score=0.44,
        keyword_score=0.88,
        rerank_score=0.77,
    )

    fake_runtime = SimpleNamespace(
        init_storage=lambda: None,
        retrieve_task_context=lambda *args, **kwargs: [search_result],
    )
    monkeypatch.setattr(memora_user_cli, "make_runtime", lambda env_file, *, with_llm: fake_runtime)

    rc = memora_user_cli.run_query(
        SimpleNamespace(
            env_file=".env",
            question="SQLite fixture 为什么丢数据？",
            user_id="debug-user",
            project_id="debug-project",
            workspace_id="debug-workspace",
            memory_types=None,
            tags=None,
            top_k=8,
            no_pinned=False,
        )
    )

    output = capsys.readouterr().out
    assert rc == 0
    assert "scores: final=0.8123 similarity=0.9100 semantic=0.4400 keyword=0.8800 importance=0.7000 recency=0.6000 access=0.5000 rerank=0.7700" in output
    assert "metadata: user_id=debug-user project_id=debug-project workspace_id=debug-workspace source=conversation confidence=0.82 weight=7 status=active access_count=3" in output
    assert "timestamps: created_at=2026-08-06T10:00:00 updated_at=2026-08-06T10:05:00 last_accessed_at=2026-08-06T10:06:00" in output
    assert "relations: supersedes=['mem_old'] related=['mem_related']" in output



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
