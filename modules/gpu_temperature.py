"""
GPU temperature protection with Kalman-filter-based predictive throttling.

Integrated as a first-party reForge module.  Temperature is read at each
sampler step (via store_latent) and compared against configurable thresholds.
A 1-D Kalman filter tracks both the current temperature and its rate of
change, enabling early pauses before the threshold is actually breached.

Thermal plateau detection handles the case where passive cooling is saturated
and the temperature refuses to drop — the user can choose to warn-and-continue
or abort the generation.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path
from typing import Callable

# ---------------------------------------------------------------------------
# Module-level mutable state
# ---------------------------------------------------------------------------

_temperature_func: Callable[[], float] | None = None
_guard: "TemperatureGuard | None" = None
_last_call_time: float = 0.0
_initialized: bool = False

# Windows-only OpenHardwareMonitor handles
_ohm_computer = None
_ohm_hardware = None
_ohm_sensor = None

# ---------------------------------------------------------------------------
# Sensor backends
# ---------------------------------------------------------------------------

def _read_nvidia_smi() -> float:
    from modules import shared
    try:
        idx = getattr(shared.opts, "gpu_temp_device_index", 0)
        lines = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"]
        ).decode().strip().splitlines()
        return float(lines[idx])
    except subprocess.CalledProcessError as e:
        print(f"\n[GPU Temp] nvidia-smi error: {e.output.decode('utf-8').strip()}")
    except Exception as e:
        print(f"\n[GPU Temp] nvidia-smi: {e}")
    return 0.0


_rocm_re = re.compile(r"Temperature \(Sensor edge\) \(C\): (\d+(?:\.\d+)?)")


def _read_rocm_smi() -> float:
    try:
        output = subprocess.check_output(["rocm-smi", "--showtemp"]).decode().strip()
        m = _rocm_re.search(output)
        if m:
            return float(m.group(1))
        print("[GPU Temp] rocm-smi: could not parse temperature")
    except subprocess.CalledProcessError as e:
        print(f"\n[GPU Temp] rocm-smi error: {e.output.decode('utf-8').strip()}")
    except Exception as e:
        print(f"\n[GPU Temp] rocm-smi: {e}")
    return 0.0


def _ohm_dll_path() -> Path:
    from modules.paths_internal import data_path
    return Path(data_path) / "tools" / "OpenHardwareMonitorLib.dll"


def _init_ohm() -> None:
    global _ohm_computer, _ohm_hardware, _ohm_sensor
    try:
        import launch
        if not launch.is_installed("pythonnet"):
            launch.run_pip("install pythonnet==3.0.2", "requirements for OpenHardwareMonitorLib")
        import clr  # type: ignore  # noqa: F401

        dll_path = _ohm_dll_path()
        if not dll_path.is_file():
            import urllib.request, zipfile
            dll_path.parent.mkdir(parents=True, exist_ok=True)
            url = "https://openhardwaremonitor.org/files/openhardwaremonitor-v0.9.6.zip"
            print("[GPU Temp] Downloading OpenHardwareMonitor…")
            zip_tmp, _ = urllib.request.urlretrieve(url)
            with zipfile.ZipFile(zip_tmp) as z:
                dll_path.write_bytes(z.read("OpenHardwareMonitor/OpenHardwareMonitorLib.dll"))

        if _ohm_computer is None:
            clr.AddReference(str(dll_path.with_suffix("")))
            from OpenHardwareMonitor.Hardware import Computer  # type: ignore  # noqa: F401
            _ohm_computer = Computer()
            _ohm_computer.CPUEnabled = False
            _ohm_computer.GPUEnabled = True
            _ohm_computer.Open()

        from modules import shared
        gpu_name = getattr(shared.opts, "gpu_temp_ohm_gpu_name", "")
        _ohm_hardware = None
        _ohm_sensor = None
        for hw in _ohm_computer.Hardware:
            if gpu_name in str(hw.Name):
                for sensor in hw.Sensors:
                    if "/temperature" in str(sensor.Identifier):
                        _ohm_hardware = hw
                        _ohm_sensor = sensor
                        return
        print(f"[GPU Temp] OpenHardwareMonitor: no temperature sensor found for '{gpu_name}'")
    except Exception as e:
        print(f"[GPU Temp] OpenHardwareMonitor init failed: {e}")


def _read_ohm() -> float:
    global _ohm_hardware, _ohm_sensor
    if _ohm_hardware is None or _ohm_sensor is None:
        return 0.0
    try:
        _ohm_hardware.Update()
        return float(_ohm_sensor.get_Value())
    except Exception as e:
        print(f"\n[GPU Temp] OpenHardwareMonitor read error: {e}")
    return 0.0


SENSOR_BACKENDS: dict[str, Callable[[], float]] = {
    "NVIDIA - nvidia-smi": _read_nvidia_smi,
    "AMD - ROCm-smi": _read_rocm_smi,
    "NVIDIA & AMD - OpenHardwareMonitor": _read_ohm,
}

# ---------------------------------------------------------------------------
# 1-D Kalman filter (pure Python — no numpy dependency)
# ---------------------------------------------------------------------------

class KalmanFilter1D:
    """
    Tracks GPU temperature and its rate of change.

    State vector: x = [temperature (°C), rate (°C/s)]
    Constant-velocity model: T(k+1) = T(k) + rate*dt, rate(k+1) = rate(k).
    """

    def __init__(self) -> None:
        self._x = [0.0, 0.0]
        # Initial covariance: large temperature uncertainty, small rate uncertainty.
        self._P = [[100.0, 0.0], [0.0, 1.0]]
        self.initialized = False

    def reset(self) -> None:
        self._x = [0.0, 0.0]
        self._P = [[100.0, 0.0], [0.0, 1.0]]
        self.initialized = False

    def update(self, measurement: float, dt: float) -> None:
        """Incorporate a new temperature reading taken `dt` seconds after the previous."""
        from modules import shared
        q_pos = float(getattr(shared.opts, "gpu_temp_kalman_q_pos", 0.1))
        q_vel = float(getattr(shared.opts, "gpu_temp_kalman_q_vel", 0.05))
        r     = float(getattr(shared.opts, "gpu_temp_kalman_r",     2.0))

        if not self.initialized:
            self._x = [measurement, 0.0]
            self.initialized = True
            return

        # --- Predict ---
        # F = [[1, dt], [0, 1]],  Q = diag(q_pos, q_vel)
        x0, x1 = self._x
        xp = [x0 + x1 * dt, x1]

        P = self._P
        # F @ P
        fp = [
            [P[0][0] + dt * P[1][0], P[0][1] + dt * P[1][1]],
            [P[1][0],                 P[1][1]],
        ]
        # (F @ P) @ F^T
        Pp = [
            [fp[0][0] + dt * fp[0][1], fp[0][1]],
            [fp[1][0] + dt * fp[1][1], fp[1][1]],
        ]
        Pp[0][0] += q_pos
        Pp[1][1] += q_vel

        # --- Update (H = [1, 0]) ---
        # S = Pp[0][0] + R
        s = Pp[0][0] + r
        k0, k1 = Pp[0][0] / s, Pp[1][0] / s
        innov = measurement - xp[0]
        self._x = [xp[0] + k0 * innov, xp[1] + k1 * innov]
        self._P = [
            [(1 - k0) * Pp[0][0],          (1 - k0) * Pp[0][1]],
            [Pp[1][0] - k1 * Pp[0][0],     Pp[1][1] - k1 * Pp[0][1]],
        ]

    def predict_at(self, horizon: float) -> float:
        """Predicted temperature `horizon` seconds from now."""
        return self._x[0] + self._x[1] * horizon

    @property
    def temperature(self) -> float:
        return self._x[0]

    @property
    def rate(self) -> float:
        """Rate of change in °C/s (positive = heating)."""
        return self._x[1]


# ---------------------------------------------------------------------------
# Temperature guard
# ---------------------------------------------------------------------------

class TemperatureGuard:
    """
    Main protection logic.  Called at each sampler step via store_latent.

    Three-stage response:
      1. Normal:    raw (or predicted) temp ≤ sleep_temp  → continue.
      2. Throttle:  raw or predicted temp > sleep_temp    → pause until
                    raw temp drops below wake_temp (or max_sleep_time elapses).
      3. Plateau:   GPU is not cooling after plateau_timeout seconds →
                    either warn-and-continue or abort, per user setting.
    """

    def __init__(self) -> None:
        self._kalman = KalmanFilter1D()
        self._last_measure_t: float | None = None
        self._plateau_start_t: float | None = None

    def reset(self) -> None:
        self._kalman.reset()
        self._last_measure_t = None
        self._plateau_start_t = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _measure(self) -> float:
        temp = _temperature_func()  # type: ignore[misc]
        now = time.time()
        dt = (now - self._last_measure_t) if self._last_measure_t is not None else 0.0
        self._last_measure_t = now
        if _is_kalman_enabled():
            self._kalman.update(temp, dt)
        return temp

    def _trigger_temp(self, raw: float) -> float:
        """Effective temperature used for threshold comparison.

        When Kalman prediction is on, takes the max of raw and the prediction
        at the configured horizon so we pause *before* the threshold is hit.
        """
        if not _is_kalman_enabled() or not self._kalman.initialized:
            return raw
        from modules import shared
        horizon = float(getattr(shared.opts, "gpu_temp_kalman_horizon", 10.0))
        return max(raw, self._kalman.predict_at(horizon))

    def _plateau_triggered(self, raw: float) -> bool:
        """True once the GPU has been thermally stuck above wake_temp long enough."""
        if not _is_kalman_enabled():
            return False
        from modules import shared
        timeout = float(getattr(shared.opts, "gpu_temp_plateau_timeout", 30.0))
        if timeout <= 0:
            return False
        rate = self._kalman.rate if self._kalman.initialized else 0.0
        wake = float(getattr(shared.opts, "gpu_temp_wake_temp", 75.0))
        # Plateau = still hot and rate is near zero or positive (not dropping)
        if raw > wake and rate > -0.1:
            if self._plateau_start_t is None:
                self._plateau_start_t = time.time()
            elif time.time() - self._plateau_start_t >= timeout:
                return True
        else:
            self._plateau_start_t = None
        return False

    def _log(self, raw: float, predicted: float | None = None) -> None:
        from modules import shared
        if not getattr(shared.opts, "gpu_temp_print", True):
            return
        rate = self._kalman.rate if self._kalman.initialized else 0.0
        if predicted is not None and _is_kalman_enabled():
            h = float(getattr(shared.opts, "gpu_temp_kalman_horizon", 10.0))
            print(f"[GPU Temp] {raw:.1f}°C  rate: {rate:+.2f}°C/s  predicted({h:.0f}s): {predicted:.1f}°C")
        else:
            print(f"[GPU Temp] {raw:.1f}°C  rate: {rate:+.2f}°C/s")

    # ------------------------------------------------------------------
    # Main check — called from store_latent
    # ------------------------------------------------------------------

    def check(self) -> None:
        from modules import shared
        global _last_call_time

        min_interval = float(getattr(shared.opts, "gpu_temp_min_interval", 5.0))
        now = time.time()
        if now - _last_call_time < min_interval:
            return

        raw = self._measure()
        trigger = self._trigger_temp(raw)
        sleep_temp = float(getattr(shared.opts, "gpu_temp_sleep_temp", 83.0))

        if trigger <= sleep_temp:
            _last_call_time = now
            return

        # ---- Pause loop ----
        predicted = (
            self._kalman.predict_at(float(getattr(shared.opts, "gpu_temp_kalman_horizon", 10.0)))
            if self._kalman.initialized else None
        )
        print("\n[GPU Temp] Threshold exceeded — pausing generation.")
        self._log(raw, predicted)

        sleep_step = float(getattr(shared.opts, "gpu_temp_sleep_step", 1.0))
        max_sleep = float(getattr(shared.opts, "gpu_temp_max_sleep", 60.0))
        wake_temp = float(getattr(shared.opts, "gpu_temp_wake_temp", 75.0))
        call_time = now

        time.sleep(sleep_step)
        raw = self._measure()

        while (
            raw > wake_temp
            and (not max_sleep or max_sleep > time.time() - call_time)
            and getattr(shared.opts, "gpu_temp_enable", True)
        ):
            self._log(raw)

            if shared.state.interrupted or shared.state.skipped:
                break

            if self._plateau_triggered(raw):
                action = getattr(shared.opts, "gpu_temp_plateau_action", "warn_and_continue")
                rate = self._kalman.rate if self._kalman.initialized else 0.0
                print(
                    f"\n[GPU Temp] Thermal plateau — GPU has reached maximum passive heat exchange.\n"
                    f"[GPU Temp] {raw:.1f}°C  rate: {rate:+.2f}°C/s  action: {action}"
                )
                if action == "abort_generation":
                    shared.state.interrupted = True
                break

            time.sleep(sleep_step)
            raw = self._measure()

        self._plateau_start_t = None
        _last_call_time = time.time()


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _is_kalman_enabled() -> bool:
    from modules import shared
    return bool(getattr(shared.opts, "gpu_temp_kalman_enable", True))


def _ensure_initialized() -> None:
    global _temperature_func, _guard, _initialized
    if _initialized:
        return
    from modules import shared
    src = getattr(shared.opts, "gpu_temp_sensor", "NVIDIA - nvidia-smi")
    _temperature_func = SENSOR_BACKENDS.get(src, _read_nvidia_smi)
    if src == "NVIDIA & AMD - OpenHardwareMonitor":
        if os.name != "nt":
            print("[GPU Temp] OpenHardwareMonitor is Windows-only — falling back to nvidia-smi")
            _temperature_func = _read_nvidia_smi
        else:
            _init_ohm()
    elif src == "AMD - ROCm-smi" and os.name == "nt":
        print("[GPU Temp] ROCm-smi is Linux-only — falling back to nvidia-smi")
        _temperature_func = _read_nvidia_smi
    _guard = TemperatureGuard()
    _initialized = True


def reinitialize() -> None:
    """Re-run sensor selection (called when the sensor setting changes)."""
    global _initialized
    _initialized = False
    _ensure_initialized()


def check() -> None:
    """Entry point called from sd_samplers_common.store_latent at each step."""
    from modules import shared
    if not getattr(shared.opts, "gpu_temp_enable", False):
        return
    _ensure_initialized()
    assert _guard is not None
    _guard.check()
