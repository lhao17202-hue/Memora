import json
from pathlib import Path

from memora.config import MemoryConfig
from memora.manager import MemoryManager
from memora.portable import EXPORT_FORMAT


def manager_for(tmp_path: Path) -> MemoryManager:
    manager = MemoryManager(MemoryConfig(root_dir=tmp_path / ".memora"))
    manager.init_storage()
    return manager


def test_export_memories_writes_versioned_json_with_all_statuses(tmp_path: Path):
    manager = manager_for(tmp_path)
    active = manager.save_memory("user", "active content", "active desc", name="active")
    archived = manager.archive_memory(active.id)
    deleted = manager.save_memory("project", "deleted content", "deleted desc", name="deleted")
    manager.delete_memory(deleted.id)
    path = tmp_path / "memories.json"

    report = manager.export_memories(path)
    data = json.loads(path.read_text(encoding="utf-8"))

    assert report == {"exported": 2, "path": str(path)}
    assert data["format"] == EXPORT_FORMAT
    assert {item["status"] for item in data["memories"]} == {"archived", "deleted"}
    assert any(item["id"] == archived.id for item in data["memories"])


def test_import_memories_imports_new_and_skips_duplicates(tmp_path: Path):
    source = manager_for(tmp_path / "source")
    source.save_memory("user", "用户偏好中文回答。", "用户偏好中文。", name="language")
    export_path = tmp_path / "memories.json"
    source.export_memories(export_path)

    target = manager_for(tmp_path / "target")
    target.save_memory("user", "existing", "existing", name="language")

    report = target.import_memories(export_path)

    assert report["imported"] == 0
    assert report["skipped"] == 1
    assert report["errors"] == []


def test_import_memories_reports_item_errors_and_continues(tmp_path: Path):
    manager = manager_for(tmp_path)
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "format": EXPORT_FORMAT,
                "memories": [
                    {
                        "id": "mem_good",
                        "name": "good",
                        "description": "good desc",
                        "type": "user",
                        "content": "good content",
                        "user_id": "default",
                        "project_id": None,
                        "workspace_id": None,
                        "tags": [],
                        "source": "test",
                        "confidence": 1.0,
                        "weight": 5,
                        "status": "active",
                        "created_at": None,
                        "updated_at": None,
                        "last_accessed_at": None,
                        "access_count": 0,
                        "expires_at": None,
                        "supersedes": [],
                        "related": [],
                    },
                    {"id": "bad"},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = manager.import_memories(path)

    assert report["imported"] == 1
    assert report["skipped"] == 0
    assert len(report["errors"]) == 1
    assert manager.memory_store.get_memory("good") is not None


def test_verify_memories_reports_index_health_and_rebuild_repairs(tmp_path: Path):
    manager = manager_for(tmp_path)
    manager.save_memory("user", "content", "description", name="language")

    healthy = manager.verify_memories()
    assert healthy["checked"] == 1
    assert healthy["errors"] == []
    assert healthy["index_ok"] is True

    manager.memory_store.index_path.write_text("broken\n", encoding="utf-8")
    broken = manager.verify_memories()
    assert broken["checked"] == 1
    assert broken["index_ok"] is False

    manager.rebuild_index()
    repaired = manager.verify_memories()
    assert repaired["index_ok"] is True


def test_backup_writes_same_format_as_export(tmp_path: Path):
    manager = manager_for(tmp_path)
    manager.save_memory("user", "content", "description", name="language")
    path = tmp_path / "backup.json"

    report = manager.backup(path)
    data = json.loads(path.read_text(encoding="utf-8"))

    assert report == {"exported": 1, "path": str(path)}
    assert data["format"] == EXPORT_FORMAT
    assert len(data["memories"]) == 1
