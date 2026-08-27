"""Publication-style figures generated from the de-identified numerical table."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from tide.analysis import add_within_dialogue_zscores, correlation_matrix

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
    """Reproduce Figure 1 with partial correlations controlling turn length."""

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


def create_grounded_novelty_plot(
    frame: pd.DataFrame,
    output_stem: str | Path,
) -> tuple[Path, Path]:
    """Reproduce Figure 2 across semantic-distance quintiles."""

    columns = ["sem_dist_partner", "cr_turn_composite", "ct_composite"]
    data = frame.dropna(subset=["CR01_avg"])[columns].dropna().copy()
    data["distance_quintile"] = pd.qcut(data["sem_dist_partner"], 5, duplicates="drop")
    grouped = data.groupby("distance_quintile", observed=True)
    summary = grouped.agg(
        distance=("sem_dist_partner", "mean"),
        creative=("cr_turn_composite", "mean"),
        critical=("ct_composite", "mean"),
        creative_se=("cr_turn_composite", _standard_error),
        critical_se=("ct_composite", _standard_error),
    )

    output = Path(output_stem).expanduser().resolve()
    with publication_style() as plt:
        figure, axis = plt.subplots(figsize=(6.6, 3.9))
        axis.errorbar(
            summary["distance"],
            summary["creative"],
            yerr=summary["creative_se"],
            color="#32658f",
            marker="o",
            markersize=5,
            linewidth=1.5,
            capsize=2.5,
            label="Creative thinking",
        )
        axis.errorbar(
            summary["distance"],
            summary["critical"],
            yerr=summary["critical_se"],
            color="#b64b38",
            marker="s",
            markersize=5,
            linewidth=1.5,
            capsize=2.5,
            label="Critical thinking",
        )
        axis.set_xlabel("Semantic distance to partner (quintile means)")
        axis.set_ylabel("Mean human rating")
        axis.legend(frameon=False, loc="upper right")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.margins(x=0.05)
        figure.tight_layout()
        paths = _save_figure(figure, output)
        plt.close(figure)
    return paths


def reproduce_paper_figures(
    frame: pd.DataFrame,
    output_directory: str | Path,
) -> list[Path]:
    """Generate data-driven Figures 1 and 2 in PDF and PNG formats."""

    output = Path(output_directory).expanduser().resolve()
    figure_1 = create_correlation_heatmap(frame, output / "fig1_heatmap")
    figure_2 = create_grounded_novelty_plot(frame, output / "fig2_grounded_novelty")
    return [*figure_1, *figure_2]
