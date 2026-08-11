"""Isolated loader for the immutable algorithm reference snapshot.

The reference imports Torch and Transformers at module scope even though its
pure helpers and records do not need them. This loader installs temporary stubs
for those two modules, imports the snapshot under a private name, and restores
the prior ``sys.modules`` state before returning.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import TypeVar

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_PATH = REPOSITORY_ROOT / "reference" / "align_single_cpu.py"
REFERENCE_SHA256 = "9ed4e21e615718ddfd10930359f55769fb27a0d284599cce45a3fc755e835de1"
REFERENCE_MODULE_NAME = "_flexaligner_algorithm_reference"
_MISSING = object()
_Function = TypeVar("_Function", bound=Callable[..., object])


class _UnavailableHeavyDependency:
    """Import-only sentinel; reference model methods are never called here."""


def _identity_inference_mode() -> Callable[[_Function], _Function]:
    """Stand in for ``torch.inference_mode()`` during module definition."""

    def decorate(function: _Function) -> _Function:
        return function

    return decorate


def _module_stubs() -> dict[str, ModuleType]:
    torch_stub = ModuleType("torch")
    torch_stub.__dict__["inference_mode"] = _identity_inference_mode
    transformers_stub = ModuleType("transformers")
    transformers_stub.__dict__["AutoModelForCTC"] = _UnavailableHeavyDependency
    transformers_stub.__dict__["AutoProcessor"] = _UnavailableHeavyDependency
    return {"torch": torch_stub, "transformers": transformers_stub}


@contextmanager
def _temporary_heavy_module_stubs() -> Iterator[None]:
    stubs = _module_stubs()
    previous: dict[str, ModuleType | object] = {
        name: sys.modules.get(name, _MISSING) for name in stubs
    }
    sys.modules.update(stubs)
    try:
        yield
    finally:
        for name, prior in previous.items():
            if prior is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior  # type: ignore[assignment]


def load_reference_module(path: Path = REFERENCE_PATH) -> ModuleType:
    """Load and return the hash-guarded snapshot without heavy dependencies."""

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != REFERENCE_SHA256:
        raise RuntimeError(
            "Reference snapshot hash mismatch: "
            f"expected={REFERENCE_SHA256}, actual={digest}, path={path}"
        )

    spec = importlib.util.spec_from_file_location(REFERENCE_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create import spec for reference snapshot: {path}")

    module = importlib.util.module_from_spec(spec)
    previous_reference = sys.modules.get(REFERENCE_MODULE_NAME, _MISSING)
    try:
        with _temporary_heavy_module_stubs():
            sys.modules[REFERENCE_MODULE_NAME] = module
            spec.loader.exec_module(module)
    finally:
        if previous_reference is _MISSING:
            sys.modules.pop(REFERENCE_MODULE_NAME, None)
        else:
            sys.modules[REFERENCE_MODULE_NAME] = previous_reference  # type: ignore[assignment]
    return module
