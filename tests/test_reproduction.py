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
    assert "| lexical_entropy | 0.669 | 0.372 | 0.171 | 0.555 |" in report
    assert "| Surprisal (speaker-relative z) | -0.013 | -0.022 | 0.152 | 0.011 |" in report
    assert "| Self-novelty (vs. own history) | -0.166 | -0.197 | 0.142 | -0.172 |" in report
    assert "| Fluency | 0.074 | 25.55 | 6, 1404 | < .001 |" in report
    assert "Trajectories with at least eight turns: 135." in report
    assert "| Creative thinking | 0.063 | 4.75 | 4, 128 | = .001 |" in report
    assert "| Originality | 0.051 | 1.95 | 4, 128 | = .105 |" in report
    assert "| Critical thinking | 0.138 | 10.30 | 4, 128 | < .001 |" in report

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
    assert creative == pytest.approx(-0.20660195605652254)
    assert critical == pytest.approx(-0.3886885189517012)

    modified = frame.copy()
    modified["cr_turn_composite"] = modified["sem_dist_partner"]
    modified_creative, _ = grounded_novelty_partial_correlations(modified)
    assert modified_creative == pytest.approx(1.0)


def test_reproduction_writes_both_data_driven_figures(tmp_path: Path) -> None:
    frame = pd.read_csv(PUBLIC_DATA)
    tables = tmp_path / "validation_tables"
    tables.mkdir()
    pd.DataFrame(
        {
            "Outcome": ["Fluency", "Flexibility", "Originality", "Critical thinking"],
            "Delta R2 over structural length": [0.02, 0.01, 0.06, 0.03],
            "95% CI": ["[0.01, 0.03]", "[-0.01, 0.02]", "[0.03, 0.09]", "[0.01, 0.05]"],
            "Delta R2 over established descriptors": [0.01, 0.00, 0.03, 0.01],
            "95% CI.1": ["[0.00, 0.02]", "[-0.01, 0.01]", "[0.01, 0.05]", "[-0.01, 0.02]"],
        }
    ).to_csv(
        tables / "paired_held_out_improvement_of_the_full_tide_family.csv",
        index=False,
    )
    paths = reproduce_paper_figures(frame, tmp_path)
    assert {path.name for path in paths} == {
        "correlation_heatmap.pdf",
        "correlation_heatmap.png",
        "fig2_computational_diagnostics.pdf",
        "fig2_computational_diagnostics.png",
        "fig3_heldout_increment.pdf",
        "fig3_heldout_increment.png",
    }
    assert all(path.stat().st_size > 1_000 for path in paths)


def test_publication_style_matches_manuscript_font() -> None:
    with publication_style() as plt:
        assert plt.rcParams["font.family"] == ["STIXGeneral"]
        assert plt.rcParams["mathtext.fontset"] == "stix"
        assert plt.rcParams["pdf.fonttype"] == 42
