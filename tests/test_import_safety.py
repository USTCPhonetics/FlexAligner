from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from tests._support import SRC_ROOT


def test_package_import_has_no_model_network_or_cwd_side_effects(tmp_path: Path) -> None:
    probe = r"""
import json
import pathlib
import socket
import sys

def blocked(*args, **kwargs):
    raise AssertionError("network access during package import")

socket.create_connection = blocked
socket.socket.connect = blocked
before = sorted(p.name for p in pathlib.Path.cwd().iterdir())
import flexaligner
after = sorted(p.name for p in pathlib.Path.cwd().iterdir())
print(json.dumps({
    "before": before,
    "after": after,
    "torch_loaded": "torch" in sys.modules,
    "transformers_loaded": "transformers" in sys.modules,
    "version": flexaligner.__version__,
}, sort_keys=True))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_ROOT)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["before"] == payload["after"] == []
    assert payload["torch_loaded"] is False
    assert payload["transformers_loaded"] is False
    assert isinstance(payload["version"], str) and payload["version"]
