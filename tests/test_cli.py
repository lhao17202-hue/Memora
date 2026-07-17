import subprocess
import sys


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
