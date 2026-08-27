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
    assert np.isnan(output.loc[0, "sem_dist_partner"])
    assert output.loc[1, "sem_dist_partner"] == pytest.approx(1.0)
    assert output.loc[2, "sem_dist_self"] == pytest.approx(0.0)
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
