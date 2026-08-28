from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tide.analysis import build_speaker_trajectories, validation_report, write_validation_report
from tide.figures import (
    grounded_novelty_partial_correlations,
    publication_style,
    reproduce_paper_figures,
)

PUBLIC_DATA = Path(__file__).parents[1] / "data" / "paper" / "deidentified_turn_metrics_ratings.csv"


def test_public_table_reproduces_primary_report_values(tmp_path: Path) -> None:
    frame = pd.read_csv(PUBLIC_DATA)
    report = validation_report(frame)
    assert "Sample: 103 dialogues and 1554 scored turns." in report
    assert "| lexical_entropy | 0.678 | 0.369 | 0.175 | 0.554 |" in report
    assert "| Surprisal (speaker-relative z) | -0.057 | -0.037 | 0.155 | 0.008 |" in report
    assert "| Self-novelty (vs. own history) | -0.204 | -0.223 | 0.143 | -0.202 |" in report
    assert "| Fluency | 0.081 | 28.44 | 6, 1404 | < .001 |" in report
    assert "Trajectories with at least eight turns: 135." in report
    assert "| Creative thinking | 0.059 | 4.53 | 4, 128 | = .002 |" in report
    assert "| Originality | 0.059 | 2.30 | 4, 128 | = .062 |" in report
    assert "| Critical thinking | 0.135 | 10.00 | 4, 128 | < .001 |" in report

    output = write_validation_report(frame, tmp_path / "report.md")
    assert output.read_text(encoding="utf-8") == report


def test_public_table_builds_135_speaker_trajectories() -> None:
    frame = pd.read_csv(PUBLIC_DATA).dropna(subset=["CR01_avg"])
    trajectories = build_speaker_trajectories(frame)
    assert len(trajectories) == 135
    assert {
        "trajectory_slope",
        "trajectory_variability",
        "trajectory_rise",
        "trajectory_peak",
    }.issubset(trajectories.columns)


def test_figure_2_correlations_are_computed_from_the_input() -> None:
    frame = pd.read_csv(PUBLIC_DATA)
    creative, critical = grounded_novelty_partial_correlations(frame)
    assert creative == pytest.approx(-0.22221828197968374)
    assert critical == pytest.approx(-0.4001385003814072)

    modified = frame.copy()
    modified["cr_turn_composite"] = modified["sem_dist_partner"]
    modified_creative, _ = grounded_novelty_partial_correlations(modified)
    assert modified_creative == pytest.approx(1.0)


def test_reproduction_writes_both_data_driven_figures(tmp_path: Path) -> None:
    frame = pd.read_csv(PUBLIC_DATA)
    paths = reproduce_paper_figures(frame, tmp_path)
    assert {path.name for path in paths} == {
        "correlation_heatmap.pdf",
        "correlation_heatmap.png",
        "fig2_grounded_novelty.pdf",
        "fig2_grounded_novelty.png",
    }
    assert all(path.stat().st_size > 1_000 for path in paths)


def test_publication_style_matches_manuscript_font() -> None:
    with publication_style() as plt:
        assert plt.rcParams["font.family"] == ["STIXGeneral"]
        assert plt.rcParams["mathtext.fontset"] == "stix"
        assert plt.rcParams["pdf.fonttype"] == 42
