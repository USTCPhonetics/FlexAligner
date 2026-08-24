from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

import flexaligner.cli as cli
from flexaligner import Language, LocalModelBundle, ModelValidationError


class _InteractiveInput(io.StringIO):
    def isatty(self) -> bool:
        return True


def _bundle(tmp_path: Path) -> LocalModelBundle:
    return LocalModelBundle(
        chunker_dir=tmp_path / "snapshot/en/chunker",
        aligner_dir=tmp_path / "snapshot/en/aligner",
        manifest_path=tmp_path / "snapshot/model_manifest.json",
    )


def test_explicit_model_pair_bypasses_cache_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(**kwargs: object) -> LocalModelBundle | None:
        raise AssertionError(f"cache resolver called: {kwargs}")

    monkeypatch.setattr(cli, "find_cached_english_models", forbidden)
    args = cli.build_parser().parse_args(
        [
            "align",
            "--audio",
            "a.wav",
            "--text",
            "hello",
            "--lexicon",
            "a.dict",
            "--chunker-model",
            str(tmp_path / "chunker"),
            "--aligner-model",
            str(tmp_path / "aligner"),
            "--output",
            "a.TextGrid",
        ]
    )
    bundle = cli._resolve_cli_models(args)
    assert bundle.chunker_dir == tmp_path / "chunker"


def test_partial_explicit_model_pair_fails_before_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cli,
        "find_cached_english_models",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError(kwargs)),
    )
    status = cli.main(
        [
            "align",
            "--audio",
            "missing.wav",
            "--text",
            "hello",
            "--lexicon",
            "missing.dict",
            "--chunker-model",
            str(tmp_path / "chunker"),
            "--output",
            "missing.TextGrid",
        ]
    )
    payload = json.loads(capsys.readouterr().err)
    assert status != 0
    assert payload["code"] == "configuration_error"


def test_noninteractive_cache_miss_never_downloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "find_cached_english_models", lambda **kwargs: None)
    monkeypatch.setattr(
        cli,
        "download_english_models",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError(kwargs)),
    )
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO())
    status = cli.main(["models", "fetch", "--model-cache-dir", str(tmp_path)])
    streams = capsys.readouterr()
    assert streams.out == ""
    payload = json.loads(streams.err)
    assert status != 0
    assert payload["code"] == "model_cache_miss"
    assert payload["context"]["suggested_command"] == "flexaligner models fetch --yes"


def test_models_fetch_yes_defaults_to_mirror_and_prints_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    expected = _bundle(tmp_path)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(cli, "find_cached_english_models", lambda **kwargs: None)

    def download(**kwargs: object) -> LocalModelBundle:
        calls.append(dict(kwargs))
        return expected

    monkeypatch.setattr(cli, "download_english_models", download)
    status = cli.main(["models", "fetch", "--yes", "--model-cache-dir", str(tmp_path / "cache")])
    streams = capsys.readouterr()
    assert status == 0
    assert streams.err == ""
    assert calls == [{"cache_dir": tmp_path / "cache", "source": "mirror"}]
    payload = json.loads(streams.out)
    assert payload["revision"] == cli.DEFAULT_MODEL_REVISION
    assert payload["bundle_release"] == "v0.2.0a1"


def test_models_fetch_mandarin_uses_language_specific_resolver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected = LocalModelBundle(
        chunker_dir=tmp_path / "snapshot/zh/chunker",
        aligner_dir=tmp_path / "snapshot/zh/aligner",
        manifest_path=tmp_path / "snapshot/model_manifest.json",
    )
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(cli, "find_cached_models", lambda **kwargs: None)

    def download(**kwargs: object) -> LocalModelBundle:
        calls.append(dict(kwargs))
        return expected

    monkeypatch.setattr(cli, "download_models", download)
    status = cli.main(
        [
            "models",
            "fetch",
            "--language",
            "zh",
            "--yes",
            "--model-cache-dir",
            str(tmp_path / "cache"),
        ]
    )
    streams = capsys.readouterr()
    assert status == 0
    assert streams.err == ""
    assert calls == [
        {
            "language": Language.ZH,
            "cache_dir": tmp_path / "cache",
            "source": "mirror",
            "force_download": False,
        }
    ]
    payload = json.loads(streams.out)
    assert payload["language"] == "zh"


def test_invalid_cache_without_yes_fails_closed_and_never_downloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        cli,
        "find_cached_english_models",
        lambda **kwargs: (_ for _ in ()).throw(
            ModelValidationError("cached model hash mismatch", context={"path": "en/model"})
        ),
    )
    monkeypatch.setattr(
        cli,
        "download_english_models",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError(kwargs)),
    )

    status = cli.main(["models", "fetch", "--model-cache-dir", str(tmp_path)])

    streams = capsys.readouterr()
    assert status != 0
    assert streams.out == ""
    payload = json.loads(streams.err)
    assert payload["code"] == "model_validation_error"


def test_invalid_cache_with_yes_forces_redownload_and_revalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected = _bundle(tmp_path)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        cli,
        "find_cached_english_models",
        lambda **kwargs: (_ for _ in ()).throw(
            ModelValidationError("cached model file set is incomplete")
        ),
    )

    def repaired_download(**kwargs: object) -> LocalModelBundle:
        calls.append(dict(kwargs))
        return expected

    monkeypatch.setattr(cli, "download_english_models", repaired_download)

    status = cli.main(["models", "fetch", "--yes", "--model-cache-dir", str(tmp_path / "cache")])

    streams = capsys.readouterr()
    assert status == 0
    assert streams.err == ""
    assert calls == [
        {
            "cache_dir": tmp_path / "cache",
            "source": "mirror",
            "force_download": True,
        }
    ]
    payload = json.loads(streams.out)
    assert payload["chunker_model"] == str(expected.chunker_dir)


def test_interactive_prompt_accepts_custom_cache_and_official_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = _bundle(tmp_path)
    custom_cache = tmp_path / "chosen-cache"
    cache_calls: list[Path] = []
    download_calls: list[dict[str, object]] = []

    def find(*, cache_dir: Path) -> LocalModelBundle | None:
        cache_calls.append(cache_dir)
        return None

    def download(**kwargs: object) -> LocalModelBundle:
        download_calls.append(dict(kwargs))
        return expected

    monkeypatch.setattr(cli, "find_cached_english_models", find)
    monkeypatch.setattr(cli, "download_english_models", download)
    monkeypatch.setattr(
        cli.sys,
        "stdin",
        _InteractiveInput(f"y\n{custom_cache}\nofficial\n"),
    )
    result = cli._resolve_or_download_models(cache_dir=None, source=None, assume_yes=False)
    assert result == expected
    assert cache_calls[-1] == custom_cache
    assert download_calls == [{"cache_dir": custom_cache, "source": "official"}]


def test_cache_hit_never_prompts_or_downloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = _bundle(tmp_path)
    monkeypatch.setattr(cli, "find_cached_english_models", lambda **kwargs: expected)
    monkeypatch.setattr(
        cli,
        "_prompt_line",
        lambda prompt: (_ for _ in ()).throw(AssertionError(prompt)),
    )
    monkeypatch.setattr(
        cli,
        "download_english_models",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError(kwargs)),
    )
    assert (
        cli._resolve_or_download_models(cache_dir=tmp_path, source=None, assume_yes=False)
        == expected
    )
