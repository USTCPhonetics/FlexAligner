from __future__ import annotations

import importlib
import os
import socket
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

from tests._support import SRC_ROOT


@pytest.fixture(autouse=True)
def _deny_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stage-1 tests are model-free and may never need a network socket."""

    def blocked(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Stage-1 package/API tests attempted network access")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)


@pytest.fixture
def public_api() -> ModuleType:
    """Import through the public package, never through implementation modules."""

    return importlib.import_module("flexaligner")


@pytest.fixture
def run_cli(tmp_path: Path) -> Callable[..., subprocess.CompletedProcess[str]]:
    def run(*args: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            str(SRC_ROOT) if not existing else os.pathsep.join((str(SRC_ROOT), existing))
        )
        env.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        return subprocess.run(
            [sys.executable, "-m", "flexaligner", *args],
            cwd=tmp_path,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )

    return run
