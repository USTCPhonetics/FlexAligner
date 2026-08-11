from __future__ import annotations

from pathlib import Path

import pytest

from flexaligner import AlignmentOptions
from flexaligner.errors import FlexAlignerError
from flexaligner.pipeline import AlignmentPipeline
from tests.integration._support import FakeInferenceFactory, make_integration_fixture


def test_chunker_closes_before_aligner_loads_and_pipeline_close_is_idempotent(
    tmp_path: Path,
) -> None:
    fixture = make_integration_fixture(tmp_path, metadata=False)
    factory = FakeInferenceFactory()
    pipeline = AlignmentPipeline(inference_factory=factory)

    pipeline.align(
        request=fixture.request,
        models=fixture.models,
        lexicon_path=fixture.lexicon_path,
        options=AlignmentOptions(),
    )
    assert factory.trace[:6] == [
        "chunk.load",
        "chunk.infer",
        "chunk.close",
        "align.load",
        "align.infer",
        "align.close",
    ]
    assert factory.active is None
    pipeline.close()
    pipeline.close()
    assert factory.trace.count("factory.close") == 1
    assert factory.closed is True


@pytest.mark.parametrize("failure", ["chunk", "align"])
def test_inference_failure_still_closes_the_active_session(
    tmp_path: Path,
    failure: str,
) -> None:
    fixture = make_integration_fixture(tmp_path, metadata=False)
    factory = FakeInferenceFactory(fail_infer=failure)
    pipeline = AlignmentPipeline(inference_factory=factory)

    with pytest.raises(FlexAlignerError):
        pipeline.align(
            request=fixture.request,
            models=fixture.models,
            lexicon_path=fixture.lexicon_path,
            options=AlignmentOptions(),
        )

    assert factory.active is None
    assert f"{failure}.close" in factory.trace
    if failure == "chunk":
        assert "align.load" not in factory.trace
