from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from tide.config import load_config
from tide.pipeline import METRIC_COLUMNS, compute_metrics_file, compute_metrics_frame


class FakeBackend:
    def encode(self, texts: list[str]) -> NDArray[np.float64]:
        vectors = {
            "甲。": [1.0, 0.0],
            "乙。": [0.0, 1.0],
            "甲再说。": [1.0, 0.0],
        }
        return np.asarray([vectors[text] for text in texts], dtype=float)

    def score_surprisal(self, context: str, target: str) -> tuple[float, float]:
        value = float(len(target))
        return value, value + 1.0


class BatchFakeBackend(FakeBackend):
    def __init__(self) -> None:
        self.requests: list[tuple[str, str]] = []

    def score_surprisal(self, context: str, target: str) -> tuple[float, float]:
        raise AssertionError("The scalar scorer should not be called when batching is available")

    def score_surprisal_batch(
        self,
        requests: list[tuple[str, str]],
    ) -> list[tuple[float, float]]:
        self.requests = requests
        return [(float(len(target)), float(len(target) + 1)) for _context, target in requests]


def test_pipeline_computes_all_metrics_without_returning_text() -> None:
    frame = pd.DataFrame(
        {
            "dialogue_id": ["D1", "D1", "D1"],
            "turn_id": ["T1", "T2", "T3"],
            "turn": [1, 2, 3],
            "speaker": ["A", "B", "A"],
            "text": ["甲。", "乙。", "甲再说。"],
        }
    )
    output = compute_metrics_frame(frame, load_config(), FakeBackend())
    assert len(output) == 3
    assert set(METRIC_COLUMNS).issubset(output.columns)
    assert "text" not in output.columns
    assert output["n_words"].tolist() == [1, 1, 2]
    assert np.isnan(output.loc[0, "sem_dist_partner"])
    assert output.loc[1, "sem_dist_partner"] == pytest.approx(1.0)
    assert output.loc[2, "sem_dist_self"] == pytest.approx(0.0)
    assert output.loc[2, "self_novelty"] == pytest.approx(0.0)
    assert output.loc[2, "delta_surprisal"] == pytest.approx(len("甲再说。") - len("甲。"))
    assert output.loc[2, "peak_break"] == pytest.approx(len("甲再说。") - len("甲。"))
    assert output.loc[2, "surprisal_sent_max"] == len("甲再说。") + 1


def test_pipeline_accepts_minimal_three_column_input() -> None:
    frame = pd.DataFrame(
        {
            "turn": [1, 2],
            "speaker": ["A", "B"],
            "text": ["甲。", "乙。"],
        }
    )
    output = compute_metrics_frame(frame, load_config(), FakeBackend())
    assert output["dialogue_id"].tolist() == ["dialogue_001", "dialogue_001"]
    assert output["turn_id"].tolist() == ["dialogue_001-T001", "dialogue_001-T002"]


def test_pipeline_batches_surprisal_with_exact_prior_context() -> None:
    frame = pd.DataFrame(
        {
            "turn": [1, 2],
            "speaker": ["A", "B"],
            "text": ["甲。", "乙。"],
        }
    )
    backend = BatchFakeBackend()
    output = compute_metrics_frame(frame, load_config(), backend)
    assert backend.requests == [("S01:\n", "甲。"), ("S01:\n甲。\nS02:\n", "乙。")]
    assert output["surprisal_mean"].tolist() == [2.0, 2.0]


def test_pipeline_can_preserve_raw_speaker_labels_in_model_context() -> None:
    frame = pd.DataFrame(
        {
            "turn": [1, 2],
            "speaker": ["student", "tutor"],
            "text": ["甲。", "乙。"],
        }
    )
    backend = BatchFakeBackend()
    config = load_config()
    config.runtime.normalize_speakers = False
    compute_metrics_frame(frame, config, backend)
    assert backend.requests == [
        ("student:\n", "甲。"),
        ("student:\n甲。\ntutor:\n", "乙。"),
    ]


def test_neutral_serialization_is_invariant_to_source_speaker_names() -> None:
    first = pd.DataFrame(
        {
            "turn": [1, 2, 3],
            "speaker": ["student", "opponent", "student"],
            "text": ["甲。", "乙。", "甲再说。"],
        }
    )
    renamed = first.assign(speaker=["Alice", "Bob", "Alice"])
    first_backend = BatchFakeBackend()
    renamed_backend = BatchFakeBackend()
    compute_metrics_frame(first, load_config(), first_backend)
    compute_metrics_frame(renamed, load_config(), renamed_backend)
    assert first_backend.requests == renamed_backend.requests


def test_partner_distance_uses_most_recent_different_speaker() -> None:
    frame = pd.DataFrame(
        {
            "turn": [1, 2, 3],
            "speaker": ["A", "A", "B"],
            "text": ["甲。", "乙。", "甲再说。"],
        }
    )
    output = compute_metrics_frame(frame, load_config(), FakeBackend())
    assert np.isnan(output.loc[1, "sem_dist_partner"])
    assert output.loc[1, "sem_dist_self"] == pytest.approx(1.0)
    assert output.loc[2, "sem_dist_partner"] == pytest.approx(1.0)


def test_pipeline_rejects_invalid_schema() -> None:
    config = load_config()
    with pytest.raises(ValueError, match="missing required columns"):
        compute_metrics_frame(pd.DataFrame({"turn": [1]}), config, FakeBackend())
    duplicate = pd.DataFrame(
        {
            "dialogue_id": ["D1", "D1"],
            "turn": [1, 1],
            "speaker": ["A", "B"],
            "text": ["甲。", "乙。"],
        }
    )
    with pytest.raises(ValueError, match="unique"):
        compute_metrics_frame(duplicate, config, FakeBackend())
    blank = pd.DataFrame(
        {
            "turn": [1],
            "speaker": ["A"],
            "text": ["  \n\t"],
        }
    )
    with pytest.raises(ValueError, match="may not be empty"):
        compute_metrics_frame(blank, config, FakeBackend())


def test_compute_metrics_file_writes_text_free_csv(tmp_path: Path) -> None:
    source = tmp_path / "dialogue.csv"
    output = tmp_path / "metrics.csv"
    pd.DataFrame(
        {
            "turn": [1, 2],
            "speaker": ["A", "B"],
            "text": ["甲。", "乙。"],
        }
    ).to_csv(source, index=False)
    metrics = compute_metrics_file(source, output, backend=FakeBackend())
    assert output.is_file()
    assert len(metrics) == 2
    assert "text" not in pd.read_csv(output).columns
