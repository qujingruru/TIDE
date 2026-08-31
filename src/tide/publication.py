"""Privacy-preserving preparation of paper reproduction tables."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from tide.pipeline import METRIC_COLUMNS

IDENTIFIER_COLUMNS = ["dialogue_id", "group", "turn_id", "turn", "speaker"]
SURFACE_COLUMNS = ["n_chars", "n_words"]
RATING_COLUMNS = [
    "CR01_avg",
    "CR02_avg",
    "CR03_avg",
    "cr_turn_composite",
    "ct_analysis",
    "ct_evaluation",
    "ct_reasoning",
    "ct_composite",
]
CREATIVE_RATER_COLUMNS = [
    "CR01_R1",
    "CR01_R2",
    "CR02_R1",
    "CR02_R2",
    "CR03_R1",
    "CR03_R2",
]
CRITICAL_RATER_COLUMNS = [
    "CT01_R1",
    "CT01_R2",
    "CT02_R1",
    "CT02_R2",
    "CT03_R1",
    "CT03_R2",
]
SEMANTIC_SENSITIVITY_COLUMNS = [
    "sem_dist_partner_truncate",
    "sem_dist_self_truncate",
    "self_novelty_truncate",
]


def _require_unique(frame: pd.DataFrame, name: str) -> None:
    if "turn_id" not in frame.columns:
        raise ValueError(f"{name} is missing turn_id")
    if bool(frame["turn_id"].duplicated().to_numpy().any()):
        raise ValueError(f"{name} contains duplicate turn_id values")


def _normalize_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    normalized = ratings.rename(columns={"dlg": "dialogue_id"}).copy()
    required = ["dialogue_id", "group", "turn_id", "turn", "speaker", *RATING_COLUMNS]
    missing = [column for column in required if column not in normalized.columns]
    if missing:
        raise ValueError("Ratings are missing columns: " + ", ".join(missing))
    rater_columns = [
        column
        for column in [*CREATIVE_RATER_COLUMNS, *CRITICAL_RATER_COLUMNS]
        if column in normalized.columns
    ]
    return cast(pd.DataFrame, normalized.loc[:, [*required, *rater_columns]].copy())


def build_public_table(
    metrics: pd.DataFrame,
    ratings: pd.DataFrame,
    *,
    quality_flags: pd.DataFrame | None = None,
    semantic_sensitivity: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge student ratings with text-free pipeline output and pseudonymize speakers."""

    _require_unique(metrics, "Metrics")
    _require_unique(ratings, "Ratings")
    required_metrics = [*IDENTIFIER_COLUMNS, *SURFACE_COLUMNS, *METRIC_COLUMNS]
    missing = [column for column in required_metrics if column not in metrics.columns]
    if missing:
        raise ValueError("Metrics are missing columns: " + ", ".join(missing))

    normalized_ratings = _normalize_ratings(ratings)
    rater_columns = [
        column
        for column in [*CREATIVE_RATER_COLUMNS, *CRITICAL_RATER_COLUMNS]
        if column in normalized_ratings.columns
    ]
    metric_table = cast(pd.DataFrame, metrics.loc[:, required_metrics].copy())
    identifier_audit = normalized_ratings.loc[:, IDENTIFIER_COLUMNS].merge(
        metric_table.loc[:, IDENTIFIER_COLUMNS],
        on="turn_id",
        how="left",
        suffixes=("_rating", "_metric"),
        validate="one_to_one",
    )
    if bool(identifier_audit["dialogue_id_metric"].isna().to_numpy().any()):
        raise ValueError("Some rated student turns have no matching metric row")
    for column in ["dialogue_id", "group", "turn", "speaker"]:
        rating_values = identifier_audit[f"{column}_rating"]
        metric_values = identifier_audit[f"{column}_metric"]
        if not rating_values.equals(metric_values):
            raise ValueError(f"Ratings and metrics disagree on {column}")

    speaker_map = cast(
        pd.DataFrame,
        metric_table.sort_values(["dialogue_id", "turn"], kind="mergesort")
        .loc[:, ["dialogue_id", "speaker"]]
        .drop_duplicates(),
    )
    speaker_map["speaker_id"] = (speaker_map.groupby("dialogue_id", sort=False).cumcount() + 1).map(
        lambda value: f"S{value:02d}"
    )
    merged = normalized_ratings.merge(
        metric_table.drop(columns=["dialogue_id", "group", "turn", "speaker"]),
        on="turn_id",
        how="left",
        validate="one_to_one",
    )
    if bool(merged[SURFACE_COLUMNS].isna().to_numpy().any()):
        raise ValueError("Some rated student turns have no matching metric row")
    merged = merged.merge(
        speaker_map,
        on=["dialogue_id", "speaker"],
        how="left",
        validate="many_to_one",
    )
    if bool(merged["speaker_id"].isna().to_numpy().any()):
        raise ValueError("Some rated speakers have no matching dialogue speaker")
    if quality_flags is None:
        merged["source_quality_flag"] = False
        merged["source_quality_issue"] = "none"
    else:
        required_flags = ["dialogue_id", "source_quality_issue"]
        missing_flags = [column for column in required_flags if column not in quality_flags.columns]
        if missing_flags:
            raise ValueError("Quality flags are missing columns: " + ", ".join(missing_flags))
        flag_table = cast(pd.DataFrame, quality_flags.loc[:, required_flags].copy())
        if bool(flag_table["dialogue_id"].duplicated().to_numpy().any()):
            raise ValueError("Quality flags contain duplicate dialogue_id values")
        unknown_dialogues = set(flag_table["dialogue_id"]) - set(metric_table["dialogue_id"])
        if unknown_dialogues:
            raise ValueError("Quality flags contain dialogue IDs absent from the metric table")
        if bool(flag_table["source_quality_issue"].isna().to_numpy().any()):
            raise ValueError("Quality flags may not contain missing issue labels")
        flag_table["source_quality_flag"] = True
        merged = merged.merge(
            flag_table,
            on="dialogue_id",
            how="left",
            validate="many_to_one",
        )
        merged["source_quality_flag"] = merged["source_quality_flag"].eq(True)
        merged["source_quality_issue"] = merged["source_quality_issue"].fillna("none")
    sensitivity_columns: list[str] = []
    if semantic_sensitivity is not None:
        _require_unique(semantic_sensitivity, "Semantic sensitivity")
        missing_sensitivity = [
            column
            for column in SEMANTIC_SENSITIVITY_COLUMNS
            if column not in semantic_sensitivity.columns
        ]
        if missing_sensitivity:
            raise ValueError(
                "Semantic sensitivity is missing columns: " + ", ".join(missing_sensitivity)
            )
        sensitivity_table = cast(
            pd.DataFrame,
            semantic_sensitivity.loc[:, ["turn_id", *SEMANTIC_SENSITIVITY_COLUMNS]].copy(),
        )
        merged = merged.merge(
            sensitivity_table,
            on="turn_id",
            how="left",
            validate="one_to_one",
        )
        for sensitivity_column, canonical_column in zip(
            SEMANTIC_SENSITIVITY_COLUMNS,
            ["sem_dist_partner", "sem_dist_self", "self_novelty"],
            strict=True,
        ):
            if not merged[sensitivity_column].isna().equals(merged[canonical_column].isna()):
                raise ValueError(
                    f"Semantic sensitivity has a different missing-value mask for "
                    f"{canonical_column}"
                )
        sensitivity_columns = SEMANTIC_SENSITIVITY_COLUMNS
    group_columns = ["dialogue_id", "speaker_id"]
    for metric in ["self_novelty", "delta_surprisal"]:
        mean = merged.groupby(group_columns)[metric].transform("mean")
        deviation = merged.groupby(group_columns)[metric].transform("std").replace(0, np.nan)
        merged[f"{metric}_z"] = (merged[metric] - mean) / deviation

    ordered = [
        "dialogue_id",
        "group",
        "source_quality_flag",
        "source_quality_issue",
        "turn_id",
        "turn",
        "speaker_id",
        *SURFACE_COLUMNS,
        *METRIC_COLUMNS[:6],
        *RATING_COLUMNS,
        *rater_columns,
        *METRIC_COLUMNS[6:],
        *sensitivity_columns,
        "self_novelty_z",
        "delta_surprisal_z",
    ]
    public = cast(pd.DataFrame, merged.loc[:, ordered].copy())
    forbidden = {"text", "utterance", "speaker", "name", "email"}
    if forbidden.intersection(public.columns):
        raise AssertionError("A direct text or identity field reached the public table")
    return public
