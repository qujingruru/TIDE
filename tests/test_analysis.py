from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tide.analysis import (
    add_within_dialogue_zscores,
    hierarchical_delta_r2,
    partial_correlation,
)


def test_partial_correlation_recovers_constructed_signal() -> None:
    generator = np.random.default_rng(42)
    control = generator.normal(size=200)
    predictor = generator.normal(size=200)
    outcome = 2.5 * predictor + 3.0 * control + generator.normal(scale=0.1, size=200)
    frame = pd.DataFrame({"x": predictor, "y": outcome, "length": control})
    result = partial_correlation(frame, "x", "y", controls="length")
    assert result.r > 0.99
    assert result.p < 0.001
    assert result.n == 200


def test_hierarchical_delta_r2_uses_positive_f_test() -> None:
    generator = np.random.default_rng(7)
    length = generator.normal(size=240)
    metric = generator.normal(size=240)
    outcome = 0.2 * length + 1.8 * metric + generator.normal(scale=0.5, size=240)
    frame = pd.DataFrame({"outcome": outcome, "length": length, "metric": metric})
    result = hierarchical_delta_r2(frame, "outcome", ["length"], ["metric"])
    assert result.delta_r2 > 0.7
    assert result.f > 0
    assert result.p < 0.001
    assert result.df_num == 1
    assert result.df_den == 237


def test_within_dialogue_zscores_are_centered() -> None:
    frame = pd.DataFrame(
        {
            "dialogue_id": ["A", "A", "A", "B", "B", "B"],
            "metric": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        }
    )
    standardized = add_within_dialogue_zscores(frame, metrics=["metric"])
    means = standardized.groupby("dialogue_id")["metric_z"].mean()
    assert means.tolist() == pytest.approx([0.0, 0.0])
