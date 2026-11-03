"""Machine state recorded beside every benchmark: CPU features detected at runtime, clock speed,
power source and Windows power settings. Each probe returns None when it cannot answer."""

import subprocess
import sys
import time
from typing import Any

import onnx

# psutil ships no type stubs, and types-psutil is not one of this project's dev dependencies.
import psutil  # type: ignore[import-untyped]
import torch

from edgevision.runtime import available_providers, onnxruntime_version
from edgevision.utils.runinfo import collect_run_info

# Reported flag -> names py-cpuinfo may use for it.
CPU_FLAGS: dict[str, tuple[str, ...]] = {
    "avx2": ("avx2",),
    "avx512f": ("avx512f",),
    "avx512_vnni": ("avx512_vnni", "avx512vnni"),
}
_POWER_KEY = r"SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes"
# Windows 11 power mode overlays (Settings > System > Power); an all-zero GUID is Balanced.
POWER_MODES = {
    "961cc777-2547-4f9d-8174-7d86181b8a7a": "best_power_efficiency",
    "00000000-0000-0000-0000-000000000000": "balanced",
    "ded574b5-45a0-4f42-8737-46345c09c238": "best_performance",
}
BATTERY_WARNING = "running on battery power: latencies are not comparable to runs on AC power"


def cpu_flags() -> dict[str, bool | None]:
    """Instruction-set support from py-cpuinfo's CPUID query; all None if it is unavailable."""
    try:
        # py-cpuinfo ships no type information.
        import cpuinfo  # type: ignore[import-untyped]

        flags = set(cpuinfo.get_cpu_info().get("flags", []))
    except Exception:
        return dict.fromkeys(CPU_FLAGS)
    return {name: any(alias in flags for alias in aliases) for name, aliases in CPU_FLAGS.items()}


def cpu_frequency() -> dict[str, float] | None:
    try:
        frequency = psutil.cpu_freq()
    except Exception:
        return None
    if frequency is None:
        return None
    return {
        "current_mhz": float(frequency.current),
        "min_mhz": float(frequency.min),
        "max_mhz": float(frequency.max),
    }


def power_source() -> dict[str, Any]:
    """{"on_ac_power", "battery_percent"}, None where psutil cannot tell (e.g. no battery)."""
    try:
        battery = psutil.sensors_battery()
    except Exception:
        battery = None
    if battery is None:
        return {"on_ac_power": None, "battery_percent": None}
    plugged = battery.power_plugged
    return {
        "on_ac_power": None if plugged is None else bool(plugged),
        "battery_percent": float(battery.percent),
    }


def windows_power_scheme() -> str | None:
    """`powercfg /getactivescheme` output, on Windows only."""
    if sys.platform != "win32":
        return None
    try:
        completed = subprocess.run(
            ("powercfg", "/getactivescheme"),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = completed.stdout.strip()
    return output if completed.returncode == 0 and output else None


def windows_power_mode() -> dict[str, str | None] | None:
    """The Windows 11 power mode overlay for AC and battery, read from the registry."""
    if sys.platform != "win32":
        return None
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _POWER_KEY) as key:
            overlays = {
                source: str(winreg.QueryValueEx(key, f"ActiveOverlay{source}PowerScheme")[0])
                for source in ("Ac", "Dc")
            }
    except OSError:
        return None
    return {
        "ac_overlay": overlays["Ac"],
        "ac_mode": POWER_MODES.get(overlays["Ac"].lower()),
        "battery_overlay": overlays["Dc"],
        "battery_mode": POWER_MODES.get(overlays["Dc"].lower()),
    }


def collect_bench_env(protocol: dict[str, Any]) -> dict[str, Any]:
    """Everything known at the start of a benchmark; ``protocol`` records its settings."""
    power = power_source()
    warnings = [BATTERY_WARNING] if power["on_ac_power"] is False else []
    return {
        "run_info": collect_run_info(),
        "cpu_flags": cpu_flags(),
        "torch_cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "versions": {
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": onnxruntime_version(),
        },
        "onnxruntime_providers": available_providers(),
        "cpu_freq_start": cpu_frequency(),
        "cpu_freq_end": None,
        "power": power,
        "windows_power_scheme": windows_power_scheme(),
        "windows_power_mode": windows_power_mode(),
        "perf_counter_resolution_s": time.get_clock_info("perf_counter").resolution,
        "protocol": protocol,
        "warnings": warnings,
    }
