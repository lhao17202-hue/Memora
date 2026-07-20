"""Session and working-memory operations."""

from __future__ import annotations

import uuid
from dataclasses import asdict

from .schema import SessionMessage, WorkingMemoryState
from .stores import SessionStore
from .utils import now_utc


class SessionService:
    def __init__(self, session_store: SessionStore):
        self.session_store = session_store

    def create_session(self, user_id: str = "default", session_id: str | None = None) -> dict:
        session_id = session_id or f"session_{now_utc().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        session = {
            "id": session_id,
            "user_id": user_id,
            "project_id": None,
            "workspace_id": None,
            "created_at": now_utc().isoformat(),
            "updated_at": now_utc().isoformat(),
            "working_memory": asdict(WorkingMemoryState()),
            "history": [],
        }
        self.session_store.save_session(session)
        return session

    def append_message(self, user_id: str, session_id: str, message: SessionMessage) -> None:
        if self.session_store.load_session(user_id, session_id) is None:
            self.create_session(user_id=user_id, session_id=session_id)
        self.session_store.append_message(user_id, session_id, message)

    def get_messages(self, user_id: str, session_id: str, limit: int | None = None) -> list[SessionMessage]:
        session = self.session_store.load_session(user_id, session_id)
        if session is None:
            return []
        raw_messages = session.get("history", [])
        if limit is not None:
            raw_messages = raw_messages[-limit:]
        return [SessionMessage(**message) for message in raw_messages]

    def get_working_memory(self, user_id: str, session_id: str) -> WorkingMemoryState:
        session = self.session_store.load_session(user_id, session_id)
        if session is None:
            return WorkingMemoryState()
        return WorkingMemoryState(**session.get("working_memory", {}))

    def update_working_memory(self, user_id: str, session_id: str, state: WorkingMemoryState) -> None:
        session = self.session_store.load_session(user_id, session_id)
        if session is None:
            session = self.create_session(user_id=user_id, session_id=session_id)
        session["working_memory"] = asdict(state)
        session["updated_at"] = now_utc().isoformat()
        self.session_store.save_session(session)
