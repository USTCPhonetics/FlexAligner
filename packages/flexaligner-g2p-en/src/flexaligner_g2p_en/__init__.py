"""Pinned offline checkpoint resource for FlexAligner English G2P."""

from __future__ import annotations

from importlib.resources import files

__version__ = "0.3.0a1"
CHECKPOINT_SHA256 = "b8af35e4596d8dd5836dfd3fe9b2ba4f97b9c311efe8879544cbcfcbd566d8c6"


def checkpoint_bytes() -> bytes:
    """Read the packaged checkpoint without creating a cache or using the network."""

    return files(__package__).joinpath("checkpoint20.npz").read_bytes()


__all__ = ["CHECKPOINT_SHA256", "__version__", "checkpoint_bytes"]
