"""Statistical analyses reported with TIDE."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from numpy.typing import NDArray

METRICS = [
    "lexical_entropy",
    "mattr",
    "surprisal_mean",
    "surprisal_sent_max",
    "sem_dist_partner",
    "sem_dist_self",
]
RATINGS = {
    "CR01_avg": "Fluency",
    "CR02_avg": "Flexibility",
    "CR03_avg": "Originality",
    "ct_composite": "Critical thinking",
}


@dataclass(frozen=True)
class CorrelationResult:
    """A partial Pearson correlation and its effective sample size."""

    r: float
    p: float
    n: int


@dataclass(frozen=True)
class RegressionResult:
    """Incremental variance and nested-model F test."""

    delta_r2: float
    f: float
    df_num: int
    df_den: int
    p: float
    n: int


def _require_analysis_dependencies() -> tuple[Any, Any]:
    try:
        import statsmodels.api as sm
        from scipy import stats
    except ImportError as error:
        raise RuntimeError(
            "Analysis dependencies are not installed. Run "
            "`pip install 'tide-dialogue[analysis]'` or `uv sync --extra analysis`."
        ) from error
    return sm, stats


def partial_correlation(
    frame: pd.DataFrame,
    x: str,
    y: str,
    controls: str | list[str] = "n_chars",
    minimum_n: int = 30,
) -> CorrelationResult:
    """Correlate residuals of x and y after linear adjustment for controls."""

    sm, stats = _require_analysis_dependencies()
    control_columns = [controls] if isinstance(controls, str) else controls
    data = frame[[x, y, *control_columns]].dropna()
    if len(data) < minimum_n or data[x].std() == 0 or data[y].std() == 0:
        return CorrelationResult(float("nan"), float("nan"), len(data))
    design = sm.add_constant(data[control_columns], has_constant="add")
    residual_x = sm.OLS(data[x], design).fit().resid
    residual_y = sm.OLS(data[y], design).fit().resid
    result = stats.pearsonr(residual_x, residual_y)
    return CorrelationResult(float(result.statistic), float(result.pvalue), len(data))


def hierarchical_delta_r2(
    frame: pd.DataFrame,
    outcome: str,
    controls: list[str],
    added_predictors: list[str],
    minimum_n: int = 50,
) -> RegressionResult:
    """Compare nested OLS models and return the correctly signed F test."""

    sm, _stats = _require_analysis_dependencies()
    columns = [outcome, *controls, *added_predictors]
    data = frame[columns].dropna()
    if len(data) < minimum_n:
        return RegressionResult(
            float("nan"),
            float("nan"),
            len(added_predictors),
            0,
            float("nan"),
            len(data),
        )
    restricted = sm.OLS(
        data[outcome],
        sm.add_constant(data[controls], has_constant="add"),
    ).fit()
    full = sm.OLS(
        data[outcome],
        sm.add_constant(data[[*controls, *added_predictors]], has_constant="add"),
    ).fit()
    f_value, p_value, degrees = full.compare_f_test(restricted)
    return RegressionResult(
        delta_r2=float(full.rsquared - restricted.rsquared),
        f=float(f_value),
        df_num=int(degrees),
        df_den=int(full.df_resid),
        p=float(p_value),
        n=len(data),
    )


def add_within_dialogue_zscores(
    frame: pd.DataFrame,
    dialogue_column: str = "dialogue_id",
    speaker_column: str = "speaker_id",
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Add speaker-relative z scores within each dialogue."""

    selected = metrics or METRICS
    group_columns = [dialogue_column, speaker_column]
    missing = [column for column in group_columns if column not in frame.columns]
    if missing:
        raise ValueError(
            "Speaker-relative standardization is missing columns: " + ", ".join(missing)
        )
    output = frame.copy()
    for metric in selected:
        grouped = output.groupby(group_columns)[metric]
        mean = grouped.transform("mean")
        standard_deviation = grouped.transform("std").replace(0, np.nan)
        output[f"{metric}_z"] = (output[metric] - mean) / standard_deviation
    return output


def correlation_matrix(
    frame: pd.DataFrame,
    metric_columns: list[str],
    rating_columns: list[str],
) -> NDArray[np.float64]:
    """Return the partial-correlation matrix used by the diagnostic heatmap."""

    return np.asarray(
        [
            [partial_correlation(frame, metric, rating).r for rating in rating_columns]
            for metric in metric_columns
        ],
        dtype=float,
    )


def build_speaker_trajectories(frame: pd.DataFrame) -> pd.DataFrame:
    """Construct speaker-level novelty trajectory features for the paper analysis."""

    required = [
        "dialogue_id",
        "speaker_id",
        "turn",
        "n_chars",
        "surprisal_mean",
        "surprisal_sent_max",
        "sem_dist_partner",
        "cr_turn_composite",
        "CR03_avg",
        "ct_composite",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Trajectory input is missing columns: {', '.join(missing)}")

    standardized = frame.copy()
    for metric in ["surprisal_mean", "surprisal_sent_max", "sem_dist_partner"]:
        standard_deviation = standardized[metric].std(ddof=0)
        standardized[f"{metric}_z"] = (
            standardized[metric] - standardized[metric].mean()
        ) / standard_deviation
    standardized["novelty_composite"] = standardized[
        ["surprisal_mean_z", "surprisal_sent_max_z", "sem_dist_partner_z"]
    ].mean(axis=1)

    rows: list[dict[str, float | int | str]] = []
    grouped = standardized.groupby(["dialogue_id", "speaker_id"], sort=False)
    for keys, group in grouped:
        if not isinstance(keys, tuple) or len(keys) != 2:
            raise TypeError("Expected dialogue and speaker grouping keys")
        dialogue_id, speaker_id = keys
        group = group.sort_values("turn").dropna(subset=["novelty_composite"])
        if len(group) < 8:
            continue
        values = group["novelty_composite"].to_numpy(dtype=float)
        time = np.arange(len(values), dtype=float)
        third = max(1, len(values) // 3)
        rows.append(
            {
                "dialogue_id": str(dialogue_id),
                "speaker_id": str(speaker_id),
                "n_turns": len(group),
                "novelty_mean": float(values.mean()),
                "mean_turn_length": float(group["n_chars"].mean()),
                "trajectory_slope": float(np.polyfit(time, values, 1)[0]),
                "trajectory_variability": float(values.std(ddof=0)),
                "trajectory_rise": float(values[-third:].mean() - values[:third].mean()),
                "trajectory_peak": float(values.max()),
                "creative_thinking": float(group["cr_turn_composite"].mean()),
                "originality": float(group["CR03_avg"].mean()),
                "critical_thinking": float(group["ct_composite"].mean()),
            }
        )
    return pd.DataFrame(rows)


def build_metric_trajectories(frame: pd.DataFrame) -> pd.DataFrame:
    """Build raw-metric trajectory features without full-sample scaling.

    Keeping each metric on its native scale prevents held-out dialogues from
    influencing the weights of a composite before grouped cross-validation.
    Scaling for prediction is learned inside each training fold.
    """

    trajectory_metrics = ["surprisal_mean", "surprisal_sent_max", "sem_dist_partner"]
    required = [
        "dialogue_id",
        "speaker_id",
        "turn",
        "n_chars",
        *trajectory_metrics,
        "cr_turn_composite",
        "CR03_avg",
        "ct_composite",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Trajectory input is missing columns: {', '.join(missing)}")

    rows: list[dict[str, float | int | str]] = []
    grouped = frame.groupby(["dialogue_id", "speaker_id"], sort=False)
    for keys, group in grouped:
        if not isinstance(keys, tuple) or len(keys) != 2:
            raise TypeError("Expected dialogue and speaker grouping keys")
        dialogue_id, speaker_id = keys
        ordered = group.sort_values("turn")
        complete = ordered.dropna(subset=trajectory_metrics)
        if len(complete) < 8:
            continue
        row: dict[str, float | int | str] = {
            "dialogue_id": str(dialogue_id),
            "speaker_id": str(speaker_id),
            "n_turns": len(complete),
            "mean_turn_length": float(ordered["n_chars"].mean()),
            "creative_thinking": float(ordered["cr_turn_composite"].mean()),
            "originality": float(ordered["CR03_avg"].mean()),
            "critical_thinking": float(ordered["ct_composite"].mean()),
        }
        time = np.arange(len(complete), dtype=float)
        third = max(1, len(complete) // 3)
        for metric in trajectory_metrics:
            values = complete[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_slope"] = float(np.polyfit(time, values, 1)[0])
            row[f"{metric}_variability"] = float(values.std(ddof=0))
            row[f"{metric}_rise"] = float(values[-third:].mean() - values[:third].mean())
            row[f"{metric}_peak"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def _format_p(p_value: float) -> str:
    if np.isnan(p_value):
        return "NA"
    if p_value < 0.001:
        return "< .001"
    return f"= {p_value:.3f}".replace("0.", ".")


def validation_report(frame: pd.DataFrame) -> str:
    """Create an English Markdown report for the primary validation analyses."""

    scored = frame.dropna(subset=["CR01_avg"]).copy()
    standardized = add_within_dialogue_zscores(scored)
    lines = [
        "# TIDE validation report",
        "",
        (
            f"Sample: {standardized['dialogue_id'].nunique()} dialogues and "
            f"{len(standardized)} scored turns."
        ),
        "",
        "## Partial correlations controlling turn length",
        "",
        "| Metric | Fluency | Flexibility | Originality | Critical thinking |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in METRICS:
        values = [partial_correlation(standardized, metric, rating).r for rating in RATINGS]
        lines.append(f"| {metric} | " + " | ".join(f"{value:.3f}" for value in values) + " |")

    lines.extend(
        [
            "",
            "## Speaker-relative standardized metrics",
            "",
            "| Metric | Fluency | Flexibility | Originality | Critical thinking |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for metric in METRICS:
        metric_z = f"{metric}_z"
        values = [partial_correlation(standardized, metric_z, rating).r for rating in RATINGS]
        lines.append(f"| {metric_z} | " + " | ".join(f"{value:.3f}" for value in values) + " |")

    paper_table_metrics = [
        ("surprisal_mean_z", "Surprisal (speaker-relative z)"),
        ("surprisal_sent_max_z", "Sentence-max surprisal (z)"),
        ("sem_dist_partner_z", "Semantic distance to partner (z)"),
        ("self_novelty", "Self-novelty (vs. own history)"),
    ]
    lines.extend(
        [
            "",
            "## Paper Table 2: novelty metrics by rating dimension",
            "",
            "| Metric | Fluency | Flexibility | Originality | Critical thinking |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for metric, label in paper_table_metrics:
        values = [partial_correlation(standardized, metric, rating).r for rating in RATINGS]
        lines.append(f"| {label} | " + " | ".join(f"{value:.3f}" for value in values) + " |")

    lines.extend(
        [
            "",
            "## Incremental validity over turn length",
            "",
            "| Outcome | Delta R2 | F | df | p |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    predictors = [f"{metric}_z" for metric in METRICS]
    for rating, label in RATINGS.items():
        result = hierarchical_delta_r2(standardized, rating, ["n_chars"], predictors)
        lines.append(
            f"| {label} | {result.delta_r2:.3f} | {result.f:.2f} | "
            f"{result.df_num}, {result.df_den} | {_format_p(result.p)} |"
        )

    if {"speaker_id", "cr_turn_composite"}.issubset(standardized.columns):
        trajectories = build_speaker_trajectories(standardized)
        features = [
            "trajectory_slope",
            "trajectory_variability",
            "trajectory_rise",
            "trajectory_peak",
        ]
        lines.extend(
            [
                "",
                "## Speaker-level trajectory features",
                "",
                f"Trajectories with at least eight turns: {len(trajectories)}.",
                "",
                "| Outcome | Delta R2 | F | df | p |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for outcome, label in {
            "creative_thinking": "Creative thinking",
            "originality": "Originality",
            "critical_thinking": "Critical thinking",
        }.items():
            result = hierarchical_delta_r2(
                trajectories,
                outcome,
                ["novelty_mean", "mean_turn_length"],
                features,
            )
            lines.append(
                f"| {label} | {result.delta_r2:.3f} | {result.f:.2f} | "
                f"{result.df_num}, {result.df_den} | {_format_p(result.p)} |"
            )
    return "\n".join(lines) + "\n"


def write_validation_report(frame: pd.DataFrame, output_path: str | Path) -> Path:
    """Write the validation report to a Markdown file."""

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(validation_report(frame), encoding="utf-8")
    return output
