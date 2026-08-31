"""Constructed-input diagnostics for TIDE's measurement logic."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def diagnostic_report(frame: pd.DataFrame) -> str:
    """Return a Markdown report for the fixed diagnostic cases."""

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
        raise ValueError("Diagnostic metrics are missing turns: " + ", ".join(missing))

    repeat = indexed.loc["REPEAT-T03"]
    paraphrase = indexed.loc["PARAPHRASE-T03"]
    grounded = indexed.loc["GROUNDED-T03"]
    detached = indexed.loc["DETACHED-T03"]
    low_entropy = indexed.loc["LOW_ENTROPY-T01"]
    high_entropy = indexed.loc["HIGH_ENTROPY-T01"]
    no_context = indexed.loc["NO_CONTEXT-T01"]
    if abs(float(repeat["self_novelty"])) > 1e-6:
        raise AssertionError("Exact repetition should have zero self-novelty")
    if abs(float(repeat["sem_dist_self"])) > 1e-6:
        raise AssertionError("Exact repetition should have zero prior-self distance")
    if float(high_entropy["lexical_entropy"]) <= float(low_entropy["lexical_entropy"]):
        raise AssertionError("The diversified constructed turn should have higher entropy")
    if float(high_entropy["mattr"]) <= float(low_entropy["mattr"]):
        raise AssertionError("The diversified constructed turn should have higher MATTR")

    lines = [
        "# TIDE constructed-input diagnostic report",
        "",
        "These checks verify how the readouts respond to controlled text changes. They do "
        "not validate a psychological construct.",
        "All nine readouts are included across the two diagnostic tables below.",
        "",
        "## Same context, different third turn",
        "",
        "| Case | Self-novelty | Partner distance | Prior-self distance | "
        "Mean surprisal | Sentence maximum | Change from prior self | Prior-peak break |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row in [
        ("Exact repetition", repeat),
        ("Paraphrase", paraphrase),
        ("Grounded update", grounded),
        ("Topic-detached turn", detached),
    ]:
        lines.append(
            f"| {label} | {float(row['self_novelty']):.3f} | "
            f"{float(row['sem_dist_partner']):.3f} | "
            f"{float(row['sem_dist_self']):.3f} | "
            f"{float(row['surprisal_mean']):.3f} | "
            f"{float(row['surprisal_sent_max']):.3f} | "
            f"{float(row['delta_surprisal']):.3f} | "
            f"{float(row['peak_break']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Lexical redistribution",
            "",
            "| Case | Lexical entropy | MATTR |",
            "|---|---:|---:|",
            (
                f"| Repeated words | {float(low_entropy['lexical_entropy']):.3f} | "
                f"{float(low_entropy['mattr']):.3f} |"
            ),
            (
                f"| Diversified words | {float(high_entropy['lexical_entropy']):.3f} | "
                f"{float(high_entropy['mattr']):.3f} |"
            ),
            "",
            "## Diagnostic contrasts",
            "",
            (
                "Exact repetition produced self-novelty "
                f"{float(repeat['self_novelty']):.6f}; this checks the lower boundary."
            ),
            (
                "Holding the example length approximately fixed, lexical entropy rose from "
                f"{float(low_entropy['lexical_entropy']):.3f} in the repeated-word case to "
                f"{float(high_entropy['lexical_entropy']):.3f} in the diversified-word case. "
                "This verifies lexical sensitivity but also illustrates why entropy is not an "
                "idea-quality score."
            ),
            (
                "The identical grounded-update target had mean surprisal "
                f"{float(no_context['surprisal_mean']):.3f} without dialogue context and "
                f"{float(grounded['surprisal_mean']):.3f} after the fixed two-turn context. "
                "This verifies that surprisal depends on the declared reference context."
            ),
            "",
            "The paraphrase, grounded-update, and detached cases are model-instrument "
            "diagnostics. Their ordering is reported rather than treated as a formal guarantee.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_diagnostic_report(frame: pd.DataFrame, output_path: str | Path) -> Path:
    """Write a constructed-input diagnostic report and return its path."""

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(diagnostic_report(frame), encoding="utf-8")
    return path
