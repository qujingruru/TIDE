from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tide.pipeline import METRIC_COLUMNS
from tide.publication import (
    CREATIVE_RATER_COLUMNS,
    CRITICAL_RATER_COLUMNS,
    RATING_COLUMNS,
    SEMANTIC_SENSITIVITY_COLUMNS,
    build_public_table,
)


def _metrics() -> pd.DataFrame:
    rows = []
    for index, speaker in enumerate(["Alice", "Bob", "Alice", "Bob"], start=1):
        row: dict[str, float | int | str] = {
            "dialogue_id": "D1",
            "group": "peer",
            "turn_id": f"T{index}",
            "turn": index,
            "speaker": speaker,
            "n_chars": 10 + index,
            "n_words": 5 + index,
        }
        row.update({metric: float(index) for metric in METRIC_COLUMNS})
        if index < 3:
            row["self_novelty"] = np.nan
            row["delta_surprisal"] = np.nan
            row["peak_break"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _ratings() -> pd.DataFrame:
    rows = []
    for index, speaker in enumerate(["Alice", "Bob", "Alice", "Bob"], start=1):
        row: dict[str, float | int | str] = {
            "dlg": "D1",
            "group": "peer",
            "turn_id": f"T{index}",
            "turn": index,
            "speaker": speaker,
        }
        row.update({rating: 1.0 for rating in RATING_COLUMNS})
        row.update({rating: float(index % 3) for rating in CREATIVE_RATER_COLUMNS})
        row.update({rating: float((index + 1) % 4) for rating in CRITICAL_RATER_COLUMNS})
        rows.append(row)
    return pd.DataFrame(rows)


def test_public_table_removes_speaker_and_text_fields() -> None:
    public = build_public_table(_metrics(), _ratings())
    assert len(public) == 4
    assert "speaker" not in public.columns
    assert "text" not in public.columns
    assert public["speaker_id"].tolist() == ["S01", "S02", "S01", "S02"]
    assert "n_words" in public.columns
    assert set(CREATIVE_RATER_COLUMNS).issubset(public.columns)
    assert set(CRITICAL_RATER_COLUMNS).issubset(public.columns)


def test_public_table_requires_matching_metric_rows() -> None:
    metrics = _metrics().iloc[:-1]
    with pytest.raises(ValueError, match="no matching metric row"):
        build_public_table(metrics, _ratings())


def test_public_speaker_ids_follow_first_appearance_not_lexical_order() -> None:
    metrics = _metrics().assign(speaker=["Zed", "Amy", "Zed", "Amy"])
    ratings = _ratings().assign(speaker=["Zed", "Amy", "Zed", "Amy"])
    public = build_public_table(metrics, ratings)
    assert public["speaker_id"].tolist() == ["S01", "S02", "S01", "S02"]


def test_public_speaker_ids_include_unrated_earlier_speakers() -> None:
    metrics = _metrics().assign(speaker=["Agent", "Student", "Agent", "Student"])
    ratings = _ratings().iloc[[1, 3]].assign(speaker="Student")
    public = build_public_table(metrics, ratings)
    assert public["speaker_id"].tolist() == ["S02", "S02"]


def test_public_table_rejects_identifier_disagreement() -> None:
    ratings = _ratings().assign(dlg="wrong")
    with pytest.raises(ValueError, match="disagree on dialogue_id"):
        build_public_table(_metrics(), ratings)


def test_public_table_adds_text_free_dialogue_quality_flags() -> None:
    flags = pd.DataFrame(
        {
            "dialogue_id": ["D1"],
            "source_quality_issue": ["mixed_speaker_content"],
        }
    )
    public = build_public_table(_metrics(), _ratings(), quality_flags=flags)
    assert public["source_quality_flag"].tolist() == [True, True, True, True]
    assert public["source_quality_issue"].tolist() == ["mixed_speaker_content"] * 4


def test_public_table_rejects_duplicate_quality_flags() -> None:
    flags = pd.DataFrame(
        {
            "dialogue_id": ["D1", "D1"],
            "source_quality_issue": ["a", "b"],
        }
    )
    with pytest.raises(ValueError, match="duplicate dialogue_id"):
        build_public_table(_metrics(), _ratings(), quality_flags=flags)


def test_public_table_adds_text_free_semantic_sensitivity() -> None:
    sensitivity = _metrics().loc[:, ["turn_id"]].copy()
    for sensitivity_column, canonical_column in zip(
        SEMANTIC_SENSITIVITY_COLUMNS,
        ["sem_dist_partner", "sem_dist_self", "self_novelty"],
        strict=True,
    ):
        sensitivity[sensitivity_column] = _metrics()[canonical_column]
    public = build_public_table(
        _metrics(),
        _ratings(),
        semantic_sensitivity=sensitivity,
    )
    assert set(SEMANTIC_SENSITIVITY_COLUMNS).issubset(public.columns)
    assert "text" not in public.columns
