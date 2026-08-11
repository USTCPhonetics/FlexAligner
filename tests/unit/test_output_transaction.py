from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

import flexaligner.textgrid as textgrid_module
from flexaligner.errors import ArtifactExistsError, OutputError, OutputValidationError
from flexaligner.textgrid import (
    Interval,
    IntervalTier,
    TextGridDocument,
    parse_textgrid_long,
    write_validated_artifacts,
)


def _document(word: str = "hello") -> TextGridDocument:
    return TextGridDocument(
        0.0,
        1.0,
        (
            IntervalTier("phones", 0.0, 1.0, (Interval(0.0, 1.0, "HH"),)),
            IntervalTier("words", 0.0, 1.0, (Interval(0.0, 1.0, word),)),
        ),
    )


def _metadata(*, calibrated: bool = False) -> dict[str, object]:
    return {
        "schema_version": "1",
        "score_kind": "chunker_emission_geometric_mean",
        "calibrated": calibrated,
        "words": [{"word_index": 0, "word": "hello", "value": 0.8}],
    }


def _write(
    tmp_path: Path,
    *,
    metadata: dict[str, object] | None = None,
    output_path: Path | None = None,
    metadata_path: Path | None = None,
) -> tuple[Path, Path | None, str]:
    output = output_path or tmp_path / "nested" / "result.TextGrid"
    selected_metadata_path = metadata_path
    if metadata is not None and selected_metadata_path is None:
        selected_metadata_path = tmp_path / "nested" / "result.chunker.json"
    digest = write_validated_artifacts(
        textgrid=_document(),
        output_path=output,
        expected_words=("hello",),
        word_sil_label="sil",
        sph_word_label="[missing]",
        metadata_path=selected_metadata_path,
        metadata=metadata,
    )
    return output, selected_metadata_path, digest


def _assert_no_temps(*paths: Path) -> None:
    for path in paths:
        assert not path.with_name(path.name + ".tmp").exists()


def test_success_stages_validates_commits_and_returns_textgrid_sha256(tmp_path: Path) -> None:
    output, metadata_path, digest = _write(tmp_path, metadata=_metadata())

    assert output.is_file()
    assert metadata_path is not None and metadata_path.is_file()
    assert digest == hashlib.sha256(output.read_bytes()).hexdigest()
    assert parse_textgrid_long(output).tiers[1].intervals[0].text == "hello"
    parsed_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert parsed_metadata["calibrated"] is False
    _assert_no_temps(output, metadata_path)


def test_textgrid_only_transaction_is_supported(tmp_path: Path) -> None:
    output, metadata_path, digest = _write(tmp_path)

    assert metadata_path is None
    assert len(digest) == 64
    assert output.exists()
    _assert_no_temps(output)


def test_metadata_is_published_before_textgrid_success_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    publications: list[tuple[str, str]] = []
    original_link = os.link

    def recording_link(
        source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        target: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        follow_symlinks: bool = True,
    ) -> None:
        publications.append((Path(source).name, Path(target).name))
        original_link(source, target, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(textgrid_module.os, "link", recording_link)
    output, metadata_path, _digest = _write(tmp_path, metadata=_metadata())

    assert metadata_path is not None
    assert publications == [
        (metadata_path.name + ".tmp", metadata_path.name),
        (output.name + ".tmp", output.name),
    ]


@pytest.mark.parametrize("existing_role", ["output", "metadata", "output_temp", "metadata_temp"])
def test_existing_official_or_temporary_artifact_is_never_overwritten(
    tmp_path: Path, existing_role: str
) -> None:
    output = tmp_path / "result.TextGrid"
    metadata_path = tmp_path / "result.json"
    selected = {
        "output": output,
        "metadata": metadata_path,
        "output_temp": output.with_name(output.name + ".tmp"),
        "metadata_temp": metadata_path.with_name(metadata_path.name + ".tmp"),
    }[existing_role]
    selected.write_text("sentinel", encoding="utf-8")

    with pytest.raises(ArtifactExistsError):
        _write(
            tmp_path,
            metadata=_metadata(),
            output_path=output,
            metadata_path=metadata_path,
        )

    assert selected.read_text(encoding="utf-8") == "sentinel"
    for candidate in (output, metadata_path):
        if candidate != selected:
            assert not candidate.exists()


def test_official_and_temporary_paths_must_be_distinct(tmp_path: Path) -> None:
    output = tmp_path / "result.TextGrid"
    with pytest.raises(OutputError, match="must be distinct"):
        write_validated_artifacts(
            textgrid=_document(),
            output_path=output,
            expected_words=("hello",),
            word_sil_label="sil",
            sph_word_label="[missing]",
            metadata_path=output,
            metadata=_metadata(),
        )
    assert not output.exists()


def test_metadata_path_and_payload_are_an_exact_pair(tmp_path: Path) -> None:
    output = tmp_path / "result.TextGrid"
    with pytest.raises(OutputError, match="provided together"):
        write_validated_artifacts(
            textgrid=_document(),
            output_path=output,
            expected_words=("hello",),
            word_sil_label="sil",
            sph_word_label="[missing]",
            metadata_path=tmp_path / "metadata.json",
        )
    assert not output.exists()


def test_wrong_written_word_fails_readback_and_leaves_no_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "result.TextGrid"
    with pytest.raises(OutputValidationError) as caught:
        write_validated_artifacts(
            textgrid=_document("wrong"),
            output_path=output,
            expected_words=("hello",),
            word_sil_label="sil",
            sph_word_label="[missing]",
        )

    assert isinstance(caught.value.__cause__, ValueError)
    assert not output.exists()
    _assert_no_temps(output)


def test_injected_bad_textgrid_readback_is_typed_and_cleaned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "result.TextGrid"

    def fail_parse(_path: Path) -> TextGridDocument:
        raise ValueError("injected malformed TextGrid")

    monkeypatch.setattr(textgrid_module, "parse_textgrid_long", fail_parse)
    with pytest.raises(OutputValidationError) as caught:
        _write(tmp_path, output_path=output)

    assert isinstance(caught.value.__cause__, ValueError)
    assert not output.exists()
    _assert_no_temps(output)


@pytest.mark.parametrize(
    "metadata",
    [
        _metadata(calibrated=True),
        {"schema_version": "1", "calibrated": None},
        {"schema_version": "1", "calibrated": False, "value": float("nan")},
    ],
)
def test_metadata_must_be_strict_json_and_explicitly_uncalibrated(
    tmp_path: Path, metadata: dict[str, object]
) -> None:
    output = tmp_path / "result.TextGrid"
    metadata_path = tmp_path / "result.json"
    with pytest.raises(OutputValidationError):
        _write(
            tmp_path,
            metadata=metadata,
            output_path=output,
            metadata_path=metadata_path,
        )

    assert not output.exists()
    assert not metadata_path.exists()
    _assert_no_temps(output, metadata_path)


@pytest.mark.parametrize("failing_link_number", [1, 2])
def test_publish_failure_rolls_back_temps_and_any_partially_committed_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failing_link_number: int,
) -> None:
    output = tmp_path / "result.TextGrid"
    metadata_path = tmp_path / "result.json"
    original_link = os.link
    calls = 0

    def injected_link(
        source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        target: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        follow_symlinks: bool = True,
    ) -> None:
        nonlocal calls
        calls += 1
        if calls == failing_link_number:
            raise OSError(f"injected publish failure {calls}")
        original_link(source, target, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(textgrid_module.os, "link", injected_link)
    with pytest.raises(OutputError) as caught:
        _write(
            tmp_path,
            metadata=_metadata(),
            output_path=output,
            metadata_path=metadata_path,
        )

    assert isinstance(caught.value.__cause__, OSError)
    assert not output.exists()
    assert not metadata_path.exists()
    _assert_no_temps(output, metadata_path)


def test_official_race_is_no_clobber_and_external_sentinel_survives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "result.TextGrid"
    original_link = os.link

    def racing_link(
        source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        target: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        follow_symlinks: bool = True,
    ) -> None:
        Path(target).write_text("EXTERNAL-SENTINEL", encoding="utf-8")
        original_link(source, target, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(textgrid_module.os, "link", racing_link)
    with pytest.raises(ArtifactExistsError):
        _write(tmp_path, output_path=output)

    assert output.read_text(encoding="utf-8") == "EXTERNAL-SENTINEL"
    _assert_no_temps(output)


def test_temporary_tamper_after_validation_is_typed_and_rolled_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "result.TextGrid"
    original_link = os.link

    def tampering_link(
        source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        target: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        follow_symlinks: bool = True,
    ) -> None:
        Path(source).write_text("RACED-TEMP", encoding="utf-8")
        original_link(source, target, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(textgrid_module.os, "link", tampering_link)
    with pytest.raises(OutputValidationError, match="bytes differ"):
        _write(tmp_path, output_path=output)

    assert not output.exists()
    _assert_no_temps(output)


def test_external_replacement_during_post_commit_validation_is_not_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "result.TextGrid"
    original_read_bytes = Path.read_bytes

    def racing_read_bytes(path: Path) -> bytes:
        if path == output:
            path.unlink()
            path.write_text("EXTERNAL-SENTINEL", encoding="utf-8")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", racing_read_bytes)
    with pytest.raises(OutputValidationError):
        _write(tmp_path, output_path=output)

    assert output.read_text(encoding="utf-8") == "EXTERNAL-SENTINEL"
    _assert_no_temps(output)


@pytest.mark.parametrize("role", ["official", "temporary"])
def test_dangling_symlink_is_treated_as_existing_and_preserved(tmp_path: Path, role: str) -> None:
    output = tmp_path / "result.TextGrid"
    candidate = output if role == "official" else output.with_name(output.name + ".tmp")
    candidate.symlink_to(tmp_path / "missing-target")

    with pytest.raises(ArtifactExistsError):
        _write(tmp_path, output_path=output)

    assert candidate.is_symlink()
    assert os.path.lexists(candidate)


def test_symlink_parent_alias_collision_is_typed_and_cleans_owned_temps(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    output = real / "result.TextGrid"
    metadata_path = alias / "result.TextGrid.tmp"

    with pytest.raises(ArtifactExistsError):
        _write(
            tmp_path,
            metadata=_metadata(),
            output_path=output,
            metadata_path=metadata_path,
        )

    assert not output.exists()
    assert not metadata_path.exists()
    assert not (real / "result.TextGrid.tmp").exists()
    assert not (real / "result.TextGrid.tmp.tmp").exists()


def test_post_commit_read_failure_rolls_back_owned_official_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "result.TextGrid"
    metadata_path = tmp_path / "result.json"
    original_read_bytes = Path.read_bytes

    def injected_read_bytes(path: Path) -> bytes:
        if path == output:
            raise OSError("injected post-commit read failure")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", injected_read_bytes)
    with pytest.raises(OutputError) as caught:
        _write(
            tmp_path,
            metadata=_metadata(),
            output_path=output,
            metadata_path=metadata_path,
        )

    assert isinstance(caught.value.__cause__, OSError)
    assert not output.exists()
    assert not metadata_path.exists()
    _assert_no_temps(output, metadata_path)
