#!/usr/bin/env python3
"""Build the public text-free table from private project outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import pandas as pd

METRIC_COLUMNS = [
    "dlg",
    "group",
    "turn_id",
    "turn",
    "n_chars",
    "lexical_entropy",
    "mattr",
    "surprisal_mean",
    "surprisal_sent_max",
    "sem_dist_partner",
    "sem_dist_self",
]
RATING_COLUMNS = [
    "turn_id",
    "speaker",
    "CR01_avg",
    "CR02_avg",
    "CR03_avg",
    "cr_turn_composite",
    "ct_analysis",
    "ct_evaluation",
    "ct_reasoning",
    "ct_composite",
]
SELF_RELATIVE_COLUMNS = [
    "turn_id",
    "self_novelty",
    "delta_surprisal",
    "peak_break",
    "self_novelty_z",
    "delta_surprisal_z",
]


def _read_unique(path: Path, columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    if bool(frame["turn_id"].duplicated().to_numpy().any()):
        raise ValueError(f"{path} contains duplicate turn_id values")
    return cast(pd.DataFrame, frame.loc[:, columns].copy())


def build_public_table(
    metrics_path: Path,
    ratings_path: Path,
    self_relative_path: Path,
) -> pd.DataFrame:
    """Merge numerical fields and replace speaker labels with within-dialogue IDs."""

    metrics = _read_unique(metrics_path, METRIC_COLUMNS)
    ratings = _read_unique(ratings_path, RATING_COLUMNS)
    self_relative = _read_unique(self_relative_path, SELF_RELATIVE_COLUMNS)
    merged = metrics.merge(ratings, on="turn_id", how="left", validate="one_to_one")
    merged = merged.merge(self_relative, on="turn_id", how="left", validate="one_to_one")
    if bool(merged["speaker"].isna().to_numpy().any()):
        raise ValueError("Some metric rows have no matching speaker/rating row")

    merged["speaker_id"] = merged.groupby("dlg")["speaker"].transform(
        lambda values: pd.Series(
            pd.factorize(values, sort=True)[0] + 1,
            index=values.index,
        ).map(lambda value: f"S{value:02d}")
    )
    merged = merged.rename(columns={"dlg": "dialogue_id"}).drop(columns=["speaker"])
    ordered = [
        "dialogue_id",
        "group",
        "turn_id",
        "turn",
        "speaker_id",
        "n_chars",
        *METRIC_COLUMNS[5:],
        *RATING_COLUMNS[2:],
        *SELF_RELATIVE_COLUMNS[1:],
    ]
    public = cast(pd.DataFrame, merged.loc[:, ordered].copy())
    forbidden = {"text", "utterance", "speaker", "name", "email"}
    if forbidden.intersection(public.columns):
        raise AssertionError("A direct text or identity field reached the public table")
    return public


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--ratings", required=True, type=Path)
    parser.add_argument("--self-relative", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    public = build_public_table(args.metrics, args.ratings, args.self_relative)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    public.to_csv(args.output, index=False)
    print(f"Wrote {len(public)} text-free rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
