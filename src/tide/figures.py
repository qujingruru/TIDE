"""Publication-style figures generated from the de-identified numerical table."""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from tide.analysis import add_within_dialogue_zscores, correlation_matrix, partial_correlation

if TYPE_CHECKING:
    from matplotlib.figure import Figure

FIGURE_METRICS = [
    "lexical_entropy",
    "mattr",
    "surprisal_mean_z",
    "surprisal_sent_max_z",
    "sem_dist_partner_z",
]
FIGURE_METRIC_LABELS = [
    "Lexical entropy",
    "MATTR",
    "Surprisal ($z$)",
    "Sentence-max surprisal ($z$)",
    "Semantic dist. to partner ($z$)",
]
RATING_COLUMNS = ["CR01_avg", "CR02_avg", "CR03_avg", "ct_composite"]
RATING_LABELS = ["Fluency", "Flexibility", "Originality", "Critical\nthinking"]
A4_WIDTH_INCHES = 210 / 25.4


def _require_matplotlib() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Figure dependencies are not installed. Run "
            "`pip install 'tide-dialogue[analysis]'` or `uv sync --extra analysis`."
        ) from error
    return plt


@contextmanager
def publication_style() -> Iterator[Any]:
    """Use the manuscript's STIX serif family and embedded TrueType fonts."""

    plt = _require_matplotlib()
    settings = {
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 9.5,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "savefig.bbox": None,
        "savefig.pad_inches": 0.05,
    }
    with plt.rc_context(settings):
        yield plt


def _save_figure(figure: Figure, output_stem: Path) -> tuple[Path, Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")
    figure.savefig(pdf_path)
    figure.savefig(png_path, dpi=300)
    return pdf_path, png_path


def create_correlation_heatmap(
    frame: pd.DataFrame,
    output_stem: str | Path,
) -> tuple[Path, Path]:
    """Create an additional partial-correlation diagnostic heatmap."""

    scored = frame.dropna(subset=["CR01_avg"]).copy()
    standardized = add_within_dialogue_zscores(scored)
    matrix = correlation_matrix(standardized, FIGURE_METRICS, RATING_COLUMNS)
    output = Path(output_stem).expanduser().resolve()
    with publication_style() as plt:
        figure, axis = plt.subplots(figsize=(6.6, 3.9))
        image = axis.imshow(matrix, cmap="RdBu_r", vmin=-0.8, vmax=0.8, aspect="auto")
        axis.set_xticks(np.arange(len(RATING_LABELS)), labels=RATING_LABELS)
        axis.set_yticks(np.arange(len(FIGURE_METRIC_LABELS)), labels=FIGURE_METRIC_LABELS)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                color = "white" if abs(value) >= 0.45 else "black"
                axis.text(column, row, f"{value:.2f}", ha="center", va="center", color=color)
        colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        colorbar.set_label(r"Partial $r$ (turn length controlled)")
        axis.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
        figure.tight_layout()
        paths = _save_figure(figure, output)
        plt.close(figure)
    return paths


def _standard_error(values: pd.Series) -> float:
    return float(values.std(ddof=1) / np.sqrt(values.count()))


def summarize_distance_quintiles(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate both human-rating outcomes by partner-distance quintile."""

    columns = ["sem_dist_partner", "cr_turn_composite", "ct_composite"]
    data = frame.dropna(subset=["CR01_avg"])[columns].dropna().copy()
    data["distance_quintile"] = pd.qcut(
        data["sem_dist_partner"],
        5,
        labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
    )
    return cast(
        pd.DataFrame,
        data.groupby("distance_quintile", observed=True).agg(
            n=("sem_dist_partner", "size"),
            distance=("sem_dist_partner", "mean"),
            creative=("cr_turn_composite", "mean"),
            critical=("ct_composite", "mean"),
            creative_se=("cr_turn_composite", _standard_error),
            critical_se=("ct_composite", _standard_error),
        ),
    )


def grounded_novelty_partial_correlations(frame: pd.DataFrame) -> tuple[float, float]:
    """Return two turn-length-controlled correlations for a legacy diagnostic."""

    scored = frame.dropna(subset=["CR01_avg"])
    creative = partial_correlation(
        scored,
        "sem_dist_partner",
        "cr_turn_composite",
    ).r
    critical = partial_correlation(
        scored,
        "sem_dist_partner",
        "ct_composite",
    ).r
    return creative, critical


def _format_partial_correlation(value: float) -> str:
    formatted = f"{value:.2f}".replace("-0.", "-.").replace("0.", ".")
    return rf"Partial $r$ = ${formatted}$"


def _draw_grounded_novelty_panel(
    axis: Any,
    summary: pd.DataFrame,
    *,
    value: str,
    error: str,
    color: str,
    marker: str,
    title: str,
    ylabel: str,
    correlation: float,
    ylim: tuple[float, float],
) -> None:
    """Draw one outcome on its documented rating scale."""

    x_values = np.arange(1, 6)
    axis.errorbar(
        x_values,
        summary[value],
        yerr=summary[error],
        color=color,
        marker=marker,
        markersize=7,
        markerfacecolor="white",
        markeredgewidth=1.7,
        linewidth=2,
        capsize=3,
        capthick=1.1,
        zorder=3,
    )
    axis.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=10)
    axis.set_ylabel(ylabel)
    axis.set_ylim(*ylim)
    axis.set_xticks(
        x_values,
        [
            f"Q{index}\n{distance:.2f}"
            for index, distance in zip(x_values, summary["distance"], strict=True)
        ],
    )
    axis.grid(axis="y", color="#D6D9DC", linewidth=0.75, alpha=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#606970")
    axis.tick_params(colors="#3D464D")
    axis.text(
        0.98,
        0.94,
        _format_partial_correlation(correlation),
        ha="right",
        va="top",
        transform=axis.transAxes,
        fontsize=10.5,
        color=color,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 2.5, "alpha": 0.9},
    )


def create_grounded_novelty_plot(
    frame: pd.DataFrame,
    output_stem: str | Path,
) -> tuple[Path, Path]:
    """Create the legacy two-panel grounded-novelty diagnostic."""

    summary = summarize_distance_quintiles(frame)
    creative_correlation, critical_correlation = grounded_novelty_partial_correlations(frame)

    output = Path(output_stem).expanduser().resolve()
    with publication_style() as plt:
        figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.55), constrained_layout=True)
        _draw_grounded_novelty_panel(
            axes[0],
            summary,
            value="creative",
            error="creative_se",
            color="#32658f",
            marker="o",
            title="A  Creative thinking",
            ylabel="Mean rating (0-9)",
            correlation=creative_correlation,
            ylim=(5.5, 7.5),
        )
        _draw_grounded_novelty_panel(
            axes[1],
            summary,
            value="critical",
            error="critical_se",
            color="#A85A43",
            marker="s",
            title="B  Critical thinking",
            ylabel="Mean rating (1-6)",
            correlation=critical_correlation,
            ylim=(2.5, 4.05),
        )
        figure.supxlabel(
            "Semantic-distance quintile (mean distance shown below)",
            fontsize=10.5,
        )
        figure.text(
            0.5,
            -0.02,
            "closer to partner  <-                              ->  farther from partner",
            ha="center",
            fontsize=9.5,
            color="#59636D",
        )
        paths = _save_figure(figure, output)
        plt.close(figure)
    return paths


def _parse_interval(value: str) -> tuple[float, float]:
    numbers = re.findall(r"-?\d+(?:\.\d+)?", str(value))
    if len(numbers) != 2:
        raise ValueError(f"Could not parse confidence interval: {value}")
    return float(numbers[0]), float(numbers[1])


def create_held_out_increment_plot(
    table: pd.DataFrame,
    output_stem: str | Path,
) -> tuple[Path, Path]:
    """Plot paired held-out gains over the two principal rival baselines."""

    required = [
        "Outcome",
        "Delta R2 over structural length",
        "95% CI",
        "Delta R2 over established descriptors",
        "95% CI.1",
    ]
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError("Held-out increment table is missing: " + ", ".join(missing))
    output = Path(output_stem).expanduser().resolve()
    outcomes = table["Outcome"].astype(str).tolist()
    y_values = np.arange(len(outcomes))[::-1]
    panels = [
        (
            "Delta R2 over structural length",
            "95% CI",
            "A  Beyond structural length",
            "#163A5F",
        ),
        (
            "Delta R2 over established descriptors",
            "95% CI.1",
            "B  Beyond established descriptors",
            "#52697D",
        ),
    ]
    with publication_style() as plt:
        figure, axes = plt.subplots(
            1,
            2,
            figsize=(A4_WIDTH_INCHES, 3.55),
            sharey=True,
            constrained_layout=True,
        )
        for axis, (estimate_column, interval_column, title, color) in zip(
            axes,
            panels,
            strict=True,
        ):
            estimates = table[estimate_column].astype(float).to_numpy()
            intervals = [_parse_interval(value) for value in table[interval_column]]
            lower = np.array([interval[0] for interval in intervals])
            upper = np.array([interval[1] for interval in intervals])
            errors = np.vstack([estimates - lower, upper - estimates])
            axis.errorbar(
                estimates,
                y_values,
                xerr=errors,
                fmt="o",
                color=color,
                markerfacecolor="white",
                markeredgewidth=1.5,
                markersize=6.5,
                capsize=3,
                linewidth=1.4,
            )
            axis.axvline(0, color="#8A949C", linewidth=0.9, linestyle="--")
            axis.set_title(title, loc="left", fontsize=11, fontweight="bold")
            axis.set_xlabel(r"Paired held-out $\Delta R^2$")
            axis.grid(axis="x", color="#DDE2E6", linewidth=0.7)
            axis.spines[["top", "right", "left"]].set_visible(False)
            axis.tick_params(axis="y", length=0)
        axes[0].set_yticks(y_values, outcomes)
        paths = _save_figure(figure, output)
        plt.close(figure)
    return paths


def create_computational_diagnostic_plot(
    frame: pd.DataFrame,
    output_stem: str | Path,
) -> tuple[Path, Path]:
    """Visualize three controlled checks of the readout definitions."""

    indexed = frame.set_index("turn_id")
    required = [
        "REPEAT-T03",
        "PARAPHRASE-T03",
        "GROUNDED-T03",
        "DETACHED-T03",
        "LOW_ENTROPY-T01",
        "HIGH_ENTROPY-T01",
        "NO_CONTEXT-T01",
    ]
    missing = [turn_id for turn_id in required if turn_id not in indexed.index]
    if missing:
        raise ValueError("Diagnostic metrics are missing: " + ", ".join(missing))
    output = Path(output_stem).expanduser().resolve()
    navy = "#163A5F"
    light = "#A9B8C5"
    with publication_style() as plt:
        figure, axes = plt.subplots(
            1,
            3,
            figsize=(A4_WIDTH_INCHES, 3.50),
            constrained_layout=True,
        )
        labels = ["Repeat", "Paraphrase", "Grounded\nupdate", "Detached\nturn"]
        values = [
            float(indexed.loc[turn_id, "self_novelty"])
            for turn_id in ["REPEAT-T03", "PARAPHRASE-T03", "GROUNDED-T03", "DETACHED-T03"]
        ]
        axes[0].bar(np.arange(4), values, color=[light, navy, navy, navy], width=0.68)
        axes[0].set_xticks(np.arange(4), labels, rotation=20, ha="right")
        axes[0].set_ylabel("Self-novelty")
        axes[0].set_title("A  Reference behavior", loc="left", fontsize=11, fontweight="bold")

        lexical_values = [
            float(indexed.loc["LOW_ENTROPY-T01", "lexical_entropy"]),
            float(indexed.loc["HIGH_ENTROPY-T01", "lexical_entropy"]),
        ]
        axes[1].bar([0, 1], lexical_values, color=[light, navy], width=0.62)
        axes[1].set_xticks([0, 1], ["Repeated\nwords", "Diversified\nwords"])
        axes[1].set_ylabel("Lexical entropy (nats)")
        axes[1].set_title("B  Lexical redistribution", loc="left", fontsize=11, fontweight="bold")

        context_values = [
            float(indexed.loc["NO_CONTEXT-T01", "surprisal_mean"]),
            float(indexed.loc["GROUNDED-T03", "surprisal_mean"]),
        ]
        axes[2].bar([0, 1], context_values, color=[light, navy], width=0.62)
        axes[2].set_xticks([0, 1], ["No prior\ndialogue", "With prior\ndialogue"])
        axes[2].set_ylabel("Mean surprisal (nats)")
        axes[2].set_title("C  Context conditioning", loc="left", fontsize=11, fontweight="bold")
        for axis in axes:
            axis.grid(axis="y", color="#DDE2E6", linewidth=0.7)
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
        paths = _save_figure(figure, output)
        plt.close(figure)
    return paths


def reproduce_paper_figures(
    frame: pd.DataFrame,
    output_directory: str | Path,
) -> list[Path]:
    """Generate the two data-driven paper figures and an additional heatmap."""

    output = Path(output_directory).expanduser().resolve()
    heatmap = create_correlation_heatmap(frame, output / "correlation_heatmap")
    increment_table = pd.read_csv(
        output / "validation_tables" / "paired_held_out_improvement_of_the_full_tide_family.csv"
    )
    held_out_figure = create_held_out_increment_plot(
        increment_table,
        output / "fig3_heldout_increment",
    )
    diagnostic_path = Path("data/demo/diagnostic_metrics.csv")
    if not diagnostic_path.is_file():
        raise FileNotFoundError(
            "data/demo/diagnostic_metrics.csv is required to reproduce paper Figure 2"
        )
    diagnostic_figure = create_computational_diagnostic_plot(
        pd.read_csv(diagnostic_path),
        output / "fig2_computational_diagnostics",
    )
    return [*heatmap, *diagnostic_figure, *held_out_figure]
