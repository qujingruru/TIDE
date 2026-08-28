"""Publication-style figures generated from the de-identified numerical table."""

from __future__ import annotations

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
        "savefig.bbox": "tight",
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
    """Return the two turn-length-controlled correlations shown in Figure 2."""

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
    """Reproduce the manuscript's two-panel Figure 2."""

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


def reproduce_paper_figures(
    frame: pd.DataFrame,
    output_directory: str | Path,
) -> list[Path]:
    """Generate paper Figure 2 and an additional diagnostic heatmap."""

    output = Path(output_directory).expanduser().resolve()
    heatmap = create_correlation_heatmap(frame, output / "correlation_heatmap")
    figure_2 = create_grounded_novelty_plot(frame, output / "fig2_grounded_novelty")
    return [*heatmap, *figure_2]
