from pathlib import Path
import subprocess
import sys


def test_sensitivity_selftest(tmp_path):
    script = Path(__file__).resolve().parents[1] / "tools" / "compiler_core" / "sensitivity.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
    )
    diagnostic = (
        f"sensitivity.py self-test exited with {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert result.returncode == 0, diagnostic
    assert "[OK] self-test passed" in result.stdout, diagnostic
