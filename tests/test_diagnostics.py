from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tide.diagnostics import diagnostic_report, write_diagnostic_report


def _diagnostics() -> pd.DataFrame:
    rows = []
    values = {
        "REPEAT-T03": (0.0, 0.2, 3.0, 1.0),
        "PARAPHRASE-T03": (0.1, 0.2, 3.2, 1.2),
        "GROUNDED-T03": (0.3, 0.3, 3.5, 1.5),
        "DETACHED-T03": (0.8, 0.9, 5.0, 1.8),
        "LOW_ENTROPY-T01": (0.0, 0.0, 4.0, 0.2),
        "HIGH_ENTROPY-T01": (0.0, 0.0, 4.0, 2.0),
        "NO_CONTEXT-T01": (0.0, 0.0, 4.5, 1.5),
    }
    for turn_id, (self_novelty, partner_distance, surprisal, entropy) in values.items():
        rows.append(
            {
                "turn_id": turn_id,
                "self_novelty": self_novelty,
                "sem_dist_partner": partner_distance,
                "sem_dist_self": self_novelty,
                "surprisal_mean": surprisal,
                "surprisal_sent_max": surprisal + 0.4,
                "delta_surprisal": surprisal - 3.0,
                "peak_break": surprisal - 3.2,
                "lexical_entropy": entropy,
                "mattr": 0.9 if "HIGH_ENTROPY" in turn_id else 0.4,
            }
        )
    return pd.DataFrame(rows)


def test_diagnostic_report_names_construct_boundaries() -> None:
    report = diagnostic_report(_diagnostics())
    assert "do not validate a psychological construct" in report
    assert "Exact repetition" in report
    assert "entropy is not an idea-quality score" in report
    assert "All nine readouts" in report
    assert "Change from prior self" in report


def test_diagnostic_report_rejects_nonzero_repetition_distance() -> None:
    frame = _diagnostics()
    frame.loc[frame["turn_id"] == "REPEAT-T03", "self_novelty"] = 0.1
    with pytest.raises(AssertionError, match="zero self-novelty"):
        diagnostic_report(frame)


def test_write_diagnostic_report_round_trips_text(tmp_path: Path) -> None:
    output = write_diagnostic_report(_diagnostics(), tmp_path / "diagnostics.md")
    assert output.read_text(encoding="utf-8") == diagnostic_report(_diagnostics())
