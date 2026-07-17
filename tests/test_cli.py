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
