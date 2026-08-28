from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tide.analysis import (
    add_within_dialogue_zscores,
    build_speaker_trajectories,
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


def test_speaker_relative_zscores_are_centered_within_dialogue_and_speaker() -> None:
    frame = pd.DataFrame(
        {
            "dialogue_id": ["A"] * 6 + ["B"] * 3,
            "speaker_id": ["S1"] * 3 + ["S2"] * 3 + ["S1"] * 3,
            "metric": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 5.0, 7.0, 9.0],
        }
    )
    standardized = add_within_dialogue_zscores(frame, metrics=["metric"])
    means = standardized.groupby(["dialogue_id", "speaker_id"])["metric_z"].mean()
    assert means.tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_trajectory_features_match_paper_definitions() -> None:
    values = np.arange(8, dtype=float)
    frame = pd.DataFrame(
        {
            "dialogue_id": ["D1"] * 8,
            "speaker_id": ["S1"] * 8,
            "turn": np.arange(1, 9),
            "n_chars": np.full(8, 20),
            "surprisal_mean": values,
            "surprisal_sent_max": values,
            "sem_dist_partner": values,
            "cr_turn_composite": np.full(8, 4.0),
            "CR03_avg": np.full(8, 1.5),
            "ct_composite": np.full(8, 3.0),
        }
    )
    result = build_speaker_trajectories(frame).iloc[0]
    standardized = (values - values.mean()) / values.std(ddof=0)
    assert result["trajectory_slope"] == pytest.approx(np.polyfit(np.arange(8), standardized, 1)[0])
    assert result["trajectory_variability"] == pytest.approx(1.0)
    assert result["trajectory_rise"] == pytest.approx(
        standardized[-2:].mean() - standardized[:2].mean()
    )
    assert result["trajectory_peak"] == pytest.approx(standardized.max())
