"""Guards and minimal behavioral facts for the immutable reference snapshot."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import sys

from tests.characterization.reference_loader import (
    REFERENCE_MODULE_NAME,
    REFERENCE_PATH,
    REFERENCE_SHA256,
    REPOSITORY_ROOT,
    load_reference_module,
)

EXPECTED_LINE_COUNT = 2548
EXPECTED_BYTE_COUNT = 96230
SOURCE_PATH = "/Users/yiyi0369/projects/flexaligner/align_single_cpu.py"


def test_reference_snapshot_hash_line_count_and_size() -> None:
    snapshot = REFERENCE_PATH.read_bytes()
    assert hashlib.sha256(snapshot).hexdigest() == REFERENCE_SHA256
    assert len(snapshot.splitlines()) == EXPECTED_LINE_COUNT
    assert len(snapshot) == EXPECTED_BYTE_COUNT


def test_reference_manifest_matches_snapshot_and_authority() -> None:
    manifest_path = REPOSITORY_ROOT / "reference" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == "1"
    assert manifest["snapshot"] == {
        "source_path": SOURCE_PATH,
        "repository_path": "reference/align_single_cpu.py",
        "copy_mode": "byte-for-byte",
        "line_count": EXPECTED_LINE_COUNT,
        "byte_count": EXPECTED_BYTE_COUNT,
        "sha256": REFERENCE_SHA256,
    }
    authority = manifest["authority"]
    assert "behavior oracle" in authority["algorithm"]
    assert "main@c5361efe4b5d8ad02574dae1bd7caa89ed3e4af0" in authority["remote"]
    assert "must not import" in authority["production_import_policy"]
    assert "Current files override" in authority["conflict_policy"]


def test_reference_loader_restores_heavy_and_private_modules() -> None:
    prior_torch = sys.modules.get("torch")
    prior_transformers = sys.modules.get("transformers")
    prior_reference = sys.modules.get(REFERENCE_MODULE_NAME)

    module = load_reference_module()

    assert module.__name__ == REFERENCE_MODULE_NAME
    assert sys.modules.get("torch") is prior_torch
    assert sys.modules.get("transformers") is prior_transformers
    assert sys.modules.get(REFERENCE_MODULE_NAME) is prior_reference


def test_production_package_does_not_import_reference_snapshot() -> None:
    source_root = REPOSITORY_ROOT / "src" / "flexaligner"
    forbidden_literals = {"align_single_cpu", "reference.align_single_cpu"}

    for source_path in source_root.rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(not alias.name.startswith("reference") for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert node.module is None or not node.module.startswith("reference")
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert not any(literal in node.value for literal in forbidden_literals)


def _toml_section(text: str, section_name: str) -> str:
    pattern = rf"(?ms)^\[{re.escape(section_name)}\]\s*(.*?)(?=^\[|\Z)"
    match = re.search(pattern, text)
    assert match is not None, f"Missing TOML section: {section_name}"
    return match.group(1)


def test_wheel_and_sdist_build_configuration_excludes_reference() -> None:
    pyproject = (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    wheel = _toml_section(pyproject, "tool.hatch.build.targets.wheel")
    sdist = _toml_section(pyproject, "tool.hatch.build.targets.sdist")

    assert 'packages = ["src/flexaligner"]' in wheel
    assert "reference" not in wheel
    assert '"/reference/**"' in sdist


def test_align_config_defaults_match_authoritative_snapshot() -> None:
    reference = load_reference_module()
    config = reference.AlignConfig()

    assert config.optional_sil is True
    assert config.sil_phone == "sil"
    assert config.sil_cost == -0.5
    assert config.sil_enter_cost == -0.5
    assert config.min_sil_dur_ms == 65.0
    assert config.optional_sph is True
    assert config.sph_phone == "sph"
    assert config.sph_cost == -2.0
    assert config.sph_enter_cost == -3.0
    assert config.sph_word_label == "[missing]"
    assert config.min_sph_dur_ms == 50.0
    assert config.beam == 400
    assert config.p_stay == 0.92
    assert config.boundary_lambda == 200.0
    assert config.boundary_context_s == 0.03
    assert config.frame_hop_s == 0.01
    assert config.word_sil_label == "sil"


def test_current_validator_accepts_leading_internal_and_tail_gaps() -> None:
    """Record the current-file fact that conflicts with an older session claim."""

    reference = load_reference_module()
    intervals = [
        reference.Interval(xmin=0.10, xmax=0.20, text="first"),
        reference.Interval(xmin=0.40, xmax=0.50, text="second"),
    ]
    textgrid = reference.TextGrid(
        xmin=0.0,
        xmax=1.0,
        tiers=[
            reference.IntervalTier(name="phones", xmin=0.0, xmax=1.0, intervals=list(intervals)),
            reference.IntervalTier(name="words", xmin=0.0, xmax=1.0, intervals=list(intervals)),
        ],
    )

    # No exception is the characterized reference behavior. This test does not
    # endorse the gap; a correction remains a separate algorithm decision.
    reference.validate_textgrid_structure(textgrid, context="gap-characterization")
