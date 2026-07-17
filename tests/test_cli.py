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
