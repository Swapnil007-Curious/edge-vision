"""Environment snapshot recorded alongside generated artifacts."""

import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

# psutil ships no type stubs, and types-psutil is not one of this project's dev dependencies.
import psutil  # type: ignore[import-untyped]

_WINDOWS_CPU_KEY = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"


def _git(*args: str) -> str | None:
    """Return stdout of a git command run beside this file, or None if it cannot run."""
    try:
        completed = subprocess.run(
            ("git", *args),
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _git_state() -> tuple[str | None, bool | None]:
    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return None, None
    status = _git("status", "--porcelain")
    return commit, None if status is None else bool(status)


def _cpu_model() -> str:
    if sys.platform == "win32":
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _WINDOWS_CPU_KEY) as key:
                value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
        except OSError:
            return platform.processor()
        return str(value).strip()

    try:
        with Path("/proc/cpuinfo").open(encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def collect_run_info() -> dict[str, Any]:
    commit, dirty = _git_state()
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_model": _cpu_model(),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "ram_total_bytes": psutil.virtual_memory().total,
        "packages": {name: version(name) for name in ("numpy", "pandas", "pyarrow")},
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
