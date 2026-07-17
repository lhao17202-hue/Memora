from pathlib import Path

from memora.config import MemoryConfig
from memora.schema import SessionMessage, WorkingMemoryState
from memora.session import SessionService
from memora.stores import FileSessionStore


def make_service(tmp_path: Path) -> SessionService:
    return SessionService(FileSessionStore(MemoryConfig(root_dir=str(tmp_path / ".memora"))))


def test_create_session_has_working_memory(tmp_path: Path):
    service = make_service(tmp_path)

    session = service.create_session(user_id="default", session_id="session_1")

    assert session["id"] == "session_1"
    assert session["working_memory"]["task_summary"] == ""
    assert session["history"] == []


def test_append_and_get_messages_with_limit(tmp_path: Path):
    service = make_service(tmp_path)
    service.create_session(session_id="session_1")
    service.append_message("default", "session_1", SessionMessage(role="user", content="one"))
    service.append_message("default", "session_1", SessionMessage(role="assistant", content="two"))

    messages = service.get_messages("default", "session_1", limit=1)

    assert len(messages) == 1
    assert messages[0].role == "assistant"
    assert messages[0].content == "two"


def test_update_and_get_working_memory(tmp_path: Path):
    service = make_service(tmp_path)
    service.create_session(session_id="session_1")
    state = WorkingMemoryState(task_summary="Design memory system", recent_files=["README.md"])

    service.update_working_memory("default", "session_1", state)
    loaded = service.get_working_memory("default", "session_1")

    assert loaded.task_summary == "Design memory system"
    assert loaded.recent_files == ["README.md"]
