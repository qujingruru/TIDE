from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from statsmodels.tools.sm_exceptions import SingularMatrixWarning

from tide.validation import (
    _spline_control_design,
    benjamini_hochberg,
    cluster_bootstrap_icc_2_1,
    cluster_bootstrap_partial_correlation,
    clustered_group_interaction,
    clustered_incremental_test,
    clustered_linear_incremental_test,
    clustered_spline_association,
    cross_condition_transport_evaluation,
    extract_markdown_tables,
    held_out_dialogue_evaluation,
    paired_group_bootstrap_delta_r2,
    robust_validation_report,
    with_sequence_controls,
)


def _clustered_frame() -> pd.DataFrame:
    generator = np.random.default_rng(29)
    rows: list[dict[str, float | str]] = []
    for dialogue in range(30):
        dialogue_effect = generator.normal(scale=0.3)
        for _turn in range(12):
            length = generator.uniform(5, 100)
            metric = generator.normal()
            outcome = 0.6 * metric + np.log1p(length) + dialogue_effect
            outcome += generator.normal(scale=0.3)
            rows.append(
                {
                    "dialogue_id": f"D{dialogue:02d}",
                    "n_chars": length,
                    "metric": metric,
                    "noise": generator.normal(),
                    "outcome": outcome,
                }
            )
    return pd.DataFrame(rows)


def _full_validation_frame() -> pd.DataFrame:
    generator = np.random.default_rng(31)
    rows: list[dict[str, float | int | str]] = []
    for dialogue in range(30):
        dialogue_effect = generator.normal(scale=0.2)
        group = "AI" if dialogue % 2 else "HM"
        for turn in range(1, 25):
            length = generator.uniform(20, 120)
            entropy = generator.normal()
            surprisal = generator.normal()
            partner_distance = generator.uniform(0.05, 0.8)
            self_distance = generator.uniform(0.02, 0.7)
            fluency = 0.5 * entropy + 0.01 * length + dialogue_effect
            flexibility = 0.2 * entropy + 0.2 * self_distance + dialogue_effect
            originality = 0.4 * surprisal + 0.3 * partner_distance + dialogue_effect
            critical = 0.3 * surprisal - 0.2 * partner_distance + dialogue_effect
            noise = generator.normal(scale=0.2, size=4)
            fluency += noise[0]
            flexibility += noise[1]
            originality += noise[2]
            critical += noise[3]
            row: dict[str, float | int | str] = {
                "dialogue_id": f"D{dialogue:02d}",
                "turn_id": f"D{dialogue:02d}-T{turn:02d}",
                "turn": turn,
                "speaker_id": "S01" if turn % 2 else "S02",
                "group": group,
                "n_chars": length,
                "n_words": max(2.0, length / 2.0 + generator.normal()),
                "lexical_entropy": entropy,
                "mattr": generator.uniform(0.4, 1.0),
                "surprisal_mean": surprisal,
                "surprisal_sent_max": surprisal + abs(generator.normal(scale=0.2)),
                "sem_dist_partner": partner_distance,
                "sem_dist_self": self_distance,
                "self_novelty": self_distance + generator.normal(scale=0.03),
                "delta_surprisal": generator.normal(),
                "peak_break": generator.normal(),
                "CR01_avg": fluency,
                "CR02_avg": flexibility,
                "CR03_avg": originality,
                "cr_turn_composite": fluency + flexibility + originality,
                "ct_analysis": critical + generator.normal(scale=0.1),
                "ct_evaluation": critical + generator.normal(scale=0.1),
                "ct_reasoning": critical + generator.normal(scale=0.1),
                "ct_composite": critical,
            }
            for prefix, value in [
                ("CR01", fluency),
                ("CR02", flexibility),
                ("CR03", originality),
                ("CT01", critical),
                ("CT02", critical),
                ("CT03", critical),
            ]:
                row[f"{prefix}_R1"] = value + generator.normal(scale=0.08)
                row[f"{prefix}_R2"] = value + generator.normal(scale=0.08)
            rows.append(row)
    return pd.DataFrame(rows)


def test_clustered_spline_association_recovers_constructed_signal() -> None:
    result = clustered_spline_association(_clustered_frame(), "metric", "outcome")
    assert result.beta > 0.4
    assert result.partial_r > 0.7
    assert result.p < 0.001
    assert result.clusters == 30


def test_clustered_spline_association_accepts_design_controls() -> None:
    frame = _clustered_frame()
    frame["condition"] = frame["dialogue_id"].str.removeprefix("D").astype(int) % 2
    frame["confounded_metric"] = frame["condition"] + np.random.default_rng(7).normal(
        scale=0.05,
        size=len(frame),
    )
    frame["confounded_outcome"] = frame["condition"] + np.random.default_rng(8).normal(
        scale=0.05,
        size=len(frame),
    )
    unadjusted = clustered_spline_association(
        frame,
        "confounded_metric",
        "confounded_outcome",
    )
    adjusted = clustered_spline_association(
        frame,
        "confounded_metric",
        "confounded_outcome",
        controls=["condition"],
    )
    assert unadjusted.partial_r > 0.9
    assert abs(adjusted.partial_r) < 0.15


def test_length_and_progress_splines_form_a_well_conditioned_design() -> None:
    frame = _full_validation_frame()
    controlled = with_sequence_controls(frame)
    controlled["group_ai"] = controlled["group"].eq("AI").astype(int)
    design = _spline_control_design(
        controlled,
        "n_chars",
        4,
        ["n_words", "speaker_turn_progress", "first_speaker_turn", "group_ai"],
    )
    values = design.to_numpy(dtype=float)
    assert np.linalg.matrix_rank(values) == values.shape[1]
    assert np.linalg.cond(values) < 100


def test_clustered_models_drop_constant_complete_case_controls() -> None:
    frame = _clustered_frame()
    frame["first_turn"] = 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        association = clustered_spline_association(
            frame,
            "metric",
            "outcome",
            controls=["first_turn"],
        )
        increment = clustered_incremental_test(
            frame,
            "outcome",
            ["metric", "noise"],
            controls=["first_turn"],
        )
        linear_increment = clustered_linear_incremental_test(
            frame,
            "outcome",
            ["n_chars", "first_turn"],
            ["metric", "noise"],
        )
    assert association.partial_r > 0.7
    assert increment.delta_r2 > 0.2
    assert linear_increment.delta_r2 > 0.2
    assert not any(isinstance(item.message, SingularMatrixWarning) for item in caught)


def test_clustered_incremental_test_detects_predictor_family() -> None:
    result = clustered_incremental_test(
        _clustered_frame(),
        "outcome",
        ["metric", "noise"],
    )
    assert result.delta_r2 > 0.2
    assert result.p < 0.001
    assert result.df_num == 2

    linear = clustered_linear_incremental_test(
        _clustered_frame(),
        "outcome",
        ["n_chars"],
        ["metric", "noise"],
    )
    assert linear.delta_r2 > 0.2
    assert linear.p < 0.001


def test_clustered_group_interaction_detects_opposite_slopes() -> None:
    frame = _clustered_frame()
    frame["condition"] = frame["dialogue_id"].str.removeprefix("D").astype(int) % 2
    frame["moderated_outcome"] = np.where(
        frame["condition"].eq(0),
        frame["metric"],
        -frame["metric"],
    ) + np.random.default_rng(11).normal(scale=0.1, size=len(frame))
    result = clustered_group_interaction(
        frame,
        "metric",
        "moderated_outcome",
        "condition",
    )
    assert result.interaction_beta < -1
    assert result.p < 0.001


def test_cluster_bootstrap_returns_ordered_interval() -> None:
    estimate, lower, upper = cluster_bootstrap_partial_correlation(
        _clustered_frame(),
        "metric",
        "outcome",
        replicates=100,
    )
    assert lower < estimate < upper
    assert lower > 0


def test_cluster_bootstrap_icc_recovers_high_rater_agreement() -> None:
    generator = np.random.default_rng(17)
    target = generator.normal(size=360)
    frame = pd.DataFrame(
        {
            "dialogue_id": np.repeat([f"D{index:02d}" for index in range(30)], 12),
            "rater_1": target + generator.normal(scale=0.1, size=len(target)),
            "rater_2": target + generator.normal(scale=0.1, size=len(target)),
        }
    )
    result = cluster_bootstrap_icc_2_1(
        frame,
        ["rater_1", "rater_2"],
        replicates=100,
    )
    assert result.icc > 0.95
    assert result.lower < result.icc < result.upper
    assert result.n == 360
    assert result.clusters == 30


def test_benjamini_hochberg_matches_known_values() -> None:
    adjusted = benjamini_hochberg([0.01, 0.04, 0.03, np.nan])
    assert adjusted[:3] == pytest.approx([0.03, 0.04, 0.04])
    assert np.isnan(adjusted[3])


def test_sequence_controls_use_one_normalized_speaker_clock() -> None:
    frame = pd.DataFrame(
        {
            "dialogue_id": ["D1", "D1", "D1", "D1", "D2"],
            "speaker_id": ["S01", "S02", "S01", "S01", "S01"],
            "turn": [1, 2, 3, 4, 1],
        }
    )
    controlled = with_sequence_controls(frame)
    assert controlled["speaker_turn_progress"].tolist() == [0.0, 0.0, 0.5, 1.0, 0.0]
    assert controlled["speaker_turn_progress_squared"].tolist() == [
        0.0,
        0.0,
        0.25,
        1.0,
        0.0,
    ]
    assert controlled["first_speaker_turn"].tolist() == [1, 1, 0, 0, 1]
    assert "turn_squared" not in controlled


def test_extract_markdown_tables_uses_section_names_and_suffixes() -> None:
    report = """# Report

## Held-out evaluation

| Model | R2 |
|---|---:|
| Baseline | 0.10 |

| Outcome | Delta |
|---|---:|
| Fluency | 0.03 |
"""
    tables = extract_markdown_tables(report)
    assert list(tables) == ["held_out_evaluation", "held_out_evaluation_2"]
    assert tables["held_out_evaluation"].to_dict("records") == [{"Model": "Baseline", "R2": "0.10"}]


def test_held_out_dialogue_evaluation_excludes_test_dialogues() -> None:
    frame = _clustered_frame()
    useful = held_out_dialogue_evaluation(
        frame,
        "outcome",
        ["n_chars", "metric"],
        outer_splits=5,
        inner_splits=3,
    )
    noise = held_out_dialogue_evaluation(
        frame,
        "outcome",
        ["noise"],
        outer_splits=5,
        inner_splits=3,
    )
    assert useful.r2 > noise.r2
    assert useful.correlation > 0.7
    assert useful.spearman > 0.7
    assert useful.groups == 30


def test_repeated_stratified_group_evaluation_predicts_every_row() -> None:
    frame = _full_validation_frame()
    result = held_out_dialogue_evaluation(
        frame,
        "CR01_avg",
        ["n_chars", "lexical_entropy"],
        outer_splits=5,
        inner_splits=3,
        repeats=2,
        strata="group",
    )
    assert np.isfinite(result.predictions).all()
    assert result.n == len(frame)
    assert result.groups == 30


def test_cross_condition_transport_evaluation_uses_disjoint_dialogues() -> None:
    frame = _full_validation_frame()
    result = cross_condition_transport_evaluation(
        frame,
        "CR01_avg",
        ["n_chars", "lexical_entropy"],
        "HM",
        "AI",
        inner_splits=3,
    )
    assert result.correlation > 0.7
    assert result.groups == 15
    assert result.n == len(frame.loc[frame["group"].eq("AI")])


def test_paired_group_bootstrap_delta_r2_preserves_prediction_pairing() -> None:
    frame = _clustered_frame()
    target = frame["outcome"].to_numpy(dtype=float)
    baseline = np.repeat(target.mean(), len(target))
    improved = 0.75 * target + 0.25 * target.mean()
    result = paired_group_bootstrap_delta_r2(
        target,
        baseline,
        improved,
        frame["dialogue_id"].to_numpy(),
        replicates=100,
    )
    assert result.delta_r2 > 0.8
    assert result.lower > 0
    assert result.lower < result.upper


def test_paired_group_bootstrap_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="equal length"):
        paired_group_bootstrap_delta_r2(
            np.array([1.0, 2.0]),
            np.array([1.0]),
            np.array([1.0, 2.0]),
            np.array(["A", "B"]),
            replicates=100,
        )


def test_robust_validation_report_runs_all_validation_layers() -> None:
    frame = _full_validation_frame()
    frame["source_quality_flag"] = frame["dialogue_id"].eq("D00")
    report = robust_validation_report(
        frame,
        bootstrap_replicates=100,
        cv_repeats=1,
    )
    assert "Criterion distributions" in report
    assert "Human-rating criterion reliability" in report
    assert "Critical-thinking composite" in report
    assert "Interlocutor-type stability" in report
    assert "Held-out dialogue evaluation" in report
    assert "Speaker-trajectory localization" in report
    assert "Source-quality exclusion sensitivity" in report
    assert "normalized within-speaker dialogue progress" in report
