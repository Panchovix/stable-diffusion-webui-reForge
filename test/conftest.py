"""
test/conftest.py — session-level bootstrap for the reForge test suite.

Server lifecycle
----------------
When tests that hit the HTTP API are collected, ``webui_server`` launches
``webui.py`` as a subprocess (``--api --nowebui``) once for the whole
session, waits up to ``SERVER_READY_TIMEOUT`` seconds for it to accept
requests, and tears it down after all tests finish.

Opt-out
-------
Pass ``--no-server`` on the pytest command line (or set the environment
variable ``REFORGE_NO_SERVER=1``) to skip the subprocess launch entirely
— useful when a server is already running externally or when only running
pure-unit tests (e.g. ``test_diff_pipeline_pipeline.py``).

  pytest test/ --no-server                 # assume server already up
  pytest test/test_diff_pipeline_pipeline.py  # no server needed at all
"""

from __future__ import annotations

import base64
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Generator

import pytest
import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parent.parent
test_files_path = str(Path(__file__).parent / "test_files")
test_outputs_path = str(Path(__file__).parent / "test_outputs")

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

_DEFAULT_BASE_URL = "http://127.0.0.1:7860"
_SERVER_READY_TIMEOUT = 120   # seconds to wait for the server to accept HTTP
_SERVER_POLL_INTERVAL = 2     # seconds between readiness probes
_SERVER_HEALTH_PATH = "/sdapi/v1/options"  # endpoint used as readiness probe


# ---------------------------------------------------------------------------
# pytest hooks
# ---------------------------------------------------------------------------

def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--no-server",
        action="store_true",
        default=False,
        help="Skip launching the webui server subprocess (assume it is already running).",
    )


def pytest_configure(config: pytest.Config) -> None:
    # Prevent webui argument parser from choking on pytest's own CLI flags.
    os.environ.setdefault("IGNORE_CMD_ARGS_ERRORS", "1")


# ---------------------------------------------------------------------------
# Server bootstrap fixture
# ---------------------------------------------------------------------------

def _wait_for_server(base_url: str, timeout: float, interval: float) -> bool:
    """Poll the health endpoint until it returns 200 or timeout expires."""
    url = base_url.rstrip("/") + _SERVER_HEALTH_PATH
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(interval)
    return False


@pytest.fixture(scope="session")
def webui_server(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    """Launch the webui API server once for the entire test session.

    Only activated when a test requests ``base_url`` (or ``webui_server``
    directly), so pure unit tests are never affected.

    Skipped when:
      * ``--no-server`` CLI flag is passed, or
      * ``REFORGE_NO_SERVER=1`` is set in the environment.
    """
    skip = (
        request.config.getoption("--no-server", default=False)
        or os.environ.get("REFORGE_NO_SERVER", "0") == "1"
    )

    if skip:
        print("\n[conftest] --no-server: skipping server launch.")
        yield
        return

    base_url: str = _DEFAULT_BASE_URL
    try:
        base_url = request.config.getoption("--base-url") or _DEFAULT_BASE_URL
    except ValueError:
        pass  # pytest-base-url not installed; use default

    ckpt = _ROOT / "test" / "test_files" / "empty.pt"

    cmd = [
        sys.executable,
        str(_ROOT / "webui.py"),
        "--api",
        "--nowebui",
        "--ckpt", str(ckpt),
        "--skip-torch-cuda-test",
        "--disable-nan-check",
        "--no-half",
        "--port", str(_parse_port(base_url)),
    ]

    print(f"\n[conftest] Starting webui server: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        cwd=str(_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    ready = _wait_for_server(base_url, _SERVER_READY_TIMEOUT, _SERVER_POLL_INTERVAL)

    if not ready:
        # Dump whatever the server printed before giving up.
        proc.terminate()
        try:
            out, _ = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            out = ""
        pytest.exit(
            f"[conftest] webui server did not become ready within "
            f"{_SERVER_READY_TIMEOUT}s.\nServer output:\n{out}",
            returncode=3,
        )

    print(f"[conftest] webui server ready at {base_url}")

    yield  # --- tests run here ---

    print("\n[conftest] Stopping webui server …")
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    print("[conftest] webui server stopped.")


def _parse_port(base_url: str) -> int:
    """Extract the port number from a URL string, defaulting to 7860."""
    try:
        from urllib.parse import urlparse
        return urlparse(base_url).port or 7860
    except Exception:
        return 7860


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def base_url(webui_server: None, request: pytest.FixtureRequest) -> str:  # noqa: ARG001
    """Return the base URL of the running webui server.

    Depends on ``webui_server`` so the server is started exactly once, and
    only when a test actually needs the API.  Falls back to the ini option
    ``base_url`` (set in pyproject.toml) or the hardcoded default.
    """
    try:
        url = request.config.getini("base_url")
        if url:
            return url
    except ValueError:
        pass
    return _DEFAULT_BASE_URL


def file_to_base64(filename: str) -> str:
    with open(filename, "rb") as f:
        data = f.read()
    return "data:image/png;base64," + str(base64.b64encode(data), "utf-8")


@pytest.fixture(scope="session")
def img2img_basic_image_base64() -> str:
    return file_to_base64(os.path.join(test_files_path, "img2img_basic.png"))


@pytest.fixture(scope="session")
def mask_basic_image_base64() -> str:
    return file_to_base64(os.path.join(test_files_path, "mask_basic.png"))


@pytest.fixture(scope="session")
def initialize() -> None:
    # Kept for backward compatibility; real bootstrap is in webui_server.
    import webui  # noqa: F401
