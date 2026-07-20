import subprocess
import sys
from pathlib import Path


def run_cli(root: Path, *args: str):
    return subprocess.run(
        [sys.executable, "-m", "memora", "--root", str(root), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_python_module_help_exits_zero():
    result = subprocess.run(
        [sys.executable, "-m", "memora", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Memora" in result.stdout
    assert "init" in result.stdout


def test_init_save_list_show_search_clean(tmp_path: Path):
    root = tmp_path / ".memora"

    assert run_cli(root, "init").returncode == 0
    save = run_cli(
        root,
        "save",
        "--type",
        "user",
        "--name",
        "language",
        "--description",
        "用户偏好中文。",
        "--content",
        "用户偏好使用中文回答。",
    )
    assert save.returncode == 0
    assert "saved" in save.stdout

    listed = run_cli(root, "list")
    assert listed.returncode == 0
    assert "language" in listed.stdout

    shown = run_cli(root, "show", "language")
    assert shown.returncode == 0
    assert "用户偏好使用中文回答。" in shown.stdout

    search = run_cli(root, "search", "中文回答")
    assert search.returncode == 0
    assert "language" in search.stdout

    clean = run_cli(root, "clean")
    assert clean.returncode == 0
    assert "archived" in clean.stdout


def test_session_append_and_show(tmp_path: Path):
    root = tmp_path / ".memora"
    result = run_cli(root, "session", "append", "session_1", "--role", "user", "--content", "hello")
    assert result.returncode == 0

    shown = run_cli(root, "session", "show", "session_1")
    assert shown.returncode == 0
    assert "hello" in shown.stdout


def test_save_secret_reports_clear_error(tmp_path: Path):
    root = tmp_path / ".memora"

    result = run_cli(
        root,
        "save",
        "--type",
        "user",
        "--name",
        "secret",
        "--description",
        "secret",
        "--content",
        "api_key = sk-abcdef123456",
    )

    assert result.returncode == 1
    assert "error:" in result.stderr
    assert "contains_secret" in result.stderr


def save_language(root: Path):
    return run_cli(
        root,
        "save",
        "--type",
        "user",
        "--name",
        "language",
        "--description",
        "用户偏好中文。",
        "--content",
        "用户偏好使用中文回答。",
    )


def test_update_command_changes_memory(tmp_path: Path):
    root = tmp_path / ".memora"
    assert save_language(root).returncode == 0

    updated = run_cli(root, "update", "language", "--description", "updated desc", "--content", "updated content", "--tag", "language", "--weight", "8", "--confidence", "0.7")
    shown = run_cli(root, "show", "language")

    assert updated.returncode == 0
    assert "updated" in updated.stdout
    assert "updated desc" in shown.stdout
    assert "updated content" in shown.stdout


def test_archive_and_restore_commands(tmp_path: Path):
    root = tmp_path / ".memora"
    assert save_language(root).returncode == 0

    archived = run_cli(root, "archive", "language")
    listed = run_cli(root, "list")
    archived_list = run_cli(root, "list", "--archived")
    restored = run_cli(root, "restore", "language")
    listed_again = run_cli(root, "list")

    assert archived.returncode == 0
    assert "archived" in archived.stdout
    assert "language" not in listed.stdout
    assert "language" in archived_list.stdout
    assert restored.returncode == 0
    assert "restored" in restored.stdout
    assert "language" in listed_again.stdout


def test_delete_command_marks_deleted_and_hard_delete_removes(tmp_path: Path):
    root = tmp_path / ".memora"
    assert save_language(root).returncode == 0

    deleted = run_cli(root, "delete", "language")
    listed = run_cli(root, "list")
    all_list = run_cli(root, "list", "--all")
    hard_deleted = run_cli(root, "delete", "language", "--hard")
    all_after_hard_delete = run_cli(root, "list", "--all")

    assert deleted.returncode == 0
    assert "deleted" in deleted.stdout
    assert "language" not in listed.stdout
    assert "language" in all_list.stdout
    assert hard_deleted.returncode == 0
    assert "hard deleted" in hard_deleted.stdout
    assert "language" not in all_after_hard_delete.stdout


def test_search_filters_type_tag_and_top_k(tmp_path: Path):
    root = tmp_path / ".memora"
    assert run_cli(root, "save", "--type", "user", "--name", "language", "--description", "用户偏好中文。", "--content", "用户偏好中文回答。").returncode == 0
    assert run_cli(root, "save", "--type", "project", "--name", "project-language", "--description", "项目使用中文。", "--content", "项目中文文档。").returncode == 0
    assert run_cli(root, "update", "language", "--tag", "language").returncode == 0

    result = run_cli(root, "search", "中文", "--type", "user", "--tag", "language", "--top-k", "1")

    assert result.returncode == 0
    assert "language" in result.stdout
    assert "project-language" not in result.stdout


def test_export_import_verify_rebuild_and_backup_commands(tmp_path: Path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    export_path = tmp_path / "memories.json"
    backup_path = tmp_path / "backup.json"

    assert save_language(source).returncode == 0

    exported = run_cli(source, "export", str(export_path))
    assert exported.returncode == 0
    assert "exported 1 memories" in exported.stdout
    assert export_path.exists()

    imported = run_cli(target, "import", str(export_path))
    assert imported.returncode == 0
    assert "imported 1 skipped 0 errors 0" in imported.stdout

    duplicate = run_cli(target, "import", str(export_path))
    assert duplicate.returncode == 0
    assert "imported 0 skipped 1 errors 0" in duplicate.stdout

    verified = run_cli(target, "verify")
    assert verified.returncode == 0
    assert "verified 1 memories" in verified.stdout
    assert "index_ok=True" in verified.stdout
    assert "errors=0" in verified.stdout

    rebuilt = run_cli(target, "rebuild-index")
    assert rebuilt.returncode == 0
    assert "rebuilt index" in rebuilt.stdout

    backed_up = run_cli(target, "backup", str(backup_path))
    assert backed_up.returncode == 0
    assert "backed up 1 memories" in backed_up.stdout
    assert backup_path.exists()


def test_import_missing_file_reports_cli_error_without_traceback(tmp_path: Path):
    root = tmp_path / ".memora"
    missing = tmp_path / "missing.json"

    result = run_cli(root, "import", str(missing))

    assert result.returncode == 1
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr


def test_import_bad_format_reports_cli_error_without_traceback(tmp_path: Path):
    root = tmp_path / ".memora"
    import_path = tmp_path / "bad.json"
    import_path.write_text('{"format": "wrong", "memories": []}', encoding="utf-8")

    result = run_cli(root, "import", str(import_path))

    assert result.returncode == 1
    assert "error:" in result.stderr
    assert "unsupported import format" in result.stderr
    assert "Traceback" not in result.stderr


def test_import_non_list_memories_reports_cli_error_without_traceback(tmp_path: Path):
    root = tmp_path / ".memora"
    import_path = tmp_path / "bad.json"
    import_path.write_text('{"format": "memora.memories.v1", "memories": {}}', encoding="utf-8")

    result = run_cli(root, "import", str(import_path))

    assert result.returncode == 1
    assert "error:" in result.stderr
    assert "memories must be a list" in result.stderr
    assert "Traceback" not in result.stderr


def test_sqlite_backend_cli_memory_and_session_flow(tmp_path: Path):
    root = tmp_path / ".memora"

    initialized = run_cli(root, "--backend", "sqlite", "init")
    saved = run_cli(
        root,
        "--backend",
        "sqlite",
        "save",
        "--type",
        "user",
        "--name",
        "language",
        "--description",
        "用户偏好中文。",
        "--content",
        "用户偏好使用中文回答。",
    )
    listed = run_cli(root, "--backend", "sqlite", "list")
    shown = run_cli(root, "--backend", "sqlite", "show", "language")
    searched = run_cli(root, "--backend", "sqlite", "search", "中文回答")
    verified = run_cli(root, "--backend", "sqlite", "verify")
    appended = run_cli(root, "--backend", "sqlite", "session", "append", "session_1", "--role", "user", "--content", "hello")
    session = run_cli(root, "--backend", "sqlite", "session", "show", "session_1")

    assert initialized.returncode == 0
    assert saved.returncode == 0
    assert "saved" in saved.stdout
    assert (root / "memora.sqlite3").exists()
    assert listed.returncode == 0
    assert "language" in listed.stdout
    assert shown.returncode == 0
    assert "用户偏好使用中文回答。" in shown.stdout
    assert searched.returncode == 0
    assert "language" in searched.stdout
    assert verified.returncode == 0
    assert "verified 1 memories" in verified.stdout
    assert "index_ok=True" in verified.stdout
    assert appended.returncode == 0
    assert session.returncode == 0
    assert "hello" in session.stdout
    assert (root / "sessions" / "default" / "session_1.json").exists()


def test_remember_command_creates_and_updates_candidate_memory(tmp_path: Path):
    root = tmp_path / ".memora"

    created = run_cli(
        root,
        "remember",
        "--type",
        "user",
        "--name",
        "language",
        "--description",
        "用户偏好中文。",
        "--content",
        "用户偏好使用中文回答。",
        "--session",
        "session_1",
        "--tag",
        "preference",
    )
    updated = run_cli(
        root,
        "remember",
        "--type",
        "user",
        "--name",
        "language",
        "--description",
        "updated desc",
        "--content",
        "updated content",
    )
    shown = run_cli(root, "show", "language")

    assert created.returncode == 0
    assert "created" in created.stdout
    assert "accepted" in created.stdout
    assert updated.returncode == 0
    assert "updated" in updated.stdout
    assert "duplicate_or_same_key" in updated.stdout
    assert "updated content" in shown.stdout


def test_remember_command_rejects_secret_as_normal_policy_result(tmp_path: Path):
    root = tmp_path / ".memora"

    result = run_cli(
        root,
        "remember",
        "--type",
        "user",
        "--name",
        "secret",
        "--description",
        "secret",
        "--content",
        "api_key = sk-abcdef123456",
    )
    listed = run_cli(root, "list", "--all")

    assert result.returncode == 0
    assert "rejected" in result.stdout
    assert "contains_secret" in result.stdout
    assert "secret" not in listed.stdout


def test_remember_command_invalid_type_reports_cli_error(tmp_path: Path):
    root = tmp_path / ".memora"

    result = run_cli(
        root,
        "remember",
        "--type",
        "invalid",
        "--name",
        "bad",
        "--description",
        "bad",
        "--content",
        "bad",
    )

    assert result.returncode == 1
    assert "error:" in result.stderr
    assert "memory type" in result.stderr
