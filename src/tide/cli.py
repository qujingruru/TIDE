"""Command-line interface for TIDE."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from tide import __version__
from tide.config import load_config
from tide.pipeline import compute_metrics_file


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tide",
        description="Turn-level Information-theoretic Dialogue Evaluation",
    )
    parser.add_argument("--version", action="version", version=f"TIDE {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    compute = subparsers.add_parser("compute", help="Compute nine readouts from dialogue CSV data")
    compute.add_argument("input", type=Path, help="Input CSV with turn, speaker, and text")
    compute.add_argument("--output", "-o", type=Path, required=True, help="Output metric CSV")
    compute.add_argument("--config", "-c", type=Path, help="Pipeline YAML configuration")

    analyze = subparsers.add_parser(
        "analyze",
        help="Run the descriptive, non-clustered analysis report",
    )
    analyze.add_argument("input", type=Path, help="De-identified metric and rating CSV")
    analyze.add_argument("--output", "-o", type=Path, required=True, help="Markdown report")

    validate = subparsers.add_parser(
        "validate",
        help="Run cluster-aware, nonlinear, and held-out validation",
    )
    validate.add_argument("input", type=Path, help="De-identified metric and rating CSV")
    validate.add_argument("--output", "-o", type=Path, required=True, help="Markdown report")
    validate.add_argument(
        "--tables-dir",
        type=Path,
        help="Optional directory for machine-readable CSV copies of every report table",
    )
    validate.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=1_000,
        help="Dialogue-bootstrap replicates (default: 1000)",
    )
    validate.add_argument(
        "--cv-repeats",
        type=int,
        default=5,
        help="Repeated grouped cross-validation partitions (default: 5)",
    )

    diagnose = subparsers.add_parser(
        "diagnose",
        help="Report constructed-input behavior for all nine readouts",
    )
    diagnose.add_argument("input", type=Path, help="Metrics computed for diagnostic cases")
    diagnose.add_argument("--output", "-o", type=Path, required=True, help="Markdown report")

    reproduce = subparsers.add_parser(
        "reproduce",
        help="Reproduce the validation report, paper Figures 2--3, and a diagnostic heatmap",
    )
    reproduce.add_argument("input", type=Path, help="De-identified metric and rating CSV")
    reproduce.add_argument("--output-dir", "-o", type=Path, required=True)
    reproduce.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=1_000,
        help="Dialogue-bootstrap replicates for the validation report (default: 1000)",
    )
    reproduce.add_argument(
        "--cv-repeats",
        type=int,
        default=5,
        help="Repeated grouped cross-validation partitions (default: 5)",
    )

    show_config = subparsers.add_parser("show-config", help="Print the resolved configuration")
    show_config.add_argument("--config", "-c", type=Path)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the TIDE command-line interface."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    args = _parser().parse_args(arguments)
    if args.command == "compute":
        compute_metrics_file(args.input, args.output, args.config)
        return 0
    if args.command == "analyze":
        from tide.analysis import write_validation_report

        write_validation_report(pd.read_csv(args.input), args.output)
        return 0
    if args.command == "validate":
        from tide.validation import write_robust_validation_report

        write_robust_validation_report(
            pd.read_csv(args.input),
            args.output,
            bootstrap_replicates=args.bootstrap_replicates,
            cv_repeats=args.cv_repeats,
            tables_directory=args.tables_dir,
        )
        return 0
    if args.command == "diagnose":
        from tide.diagnostics import write_diagnostic_report

        write_diagnostic_report(pd.read_csv(args.input), args.output)
        return 0
    if args.command == "reproduce":
        from tide.figures import reproduce_paper_figures
        from tide.validation import write_robust_validation_report

        args.output_dir.mkdir(parents=True, exist_ok=True)
        frame = pd.read_csv(args.input)
        write_robust_validation_report(
            frame,
            args.output_dir / "brm_validation_report.md",
            bootstrap_replicates=args.bootstrap_replicates,
            cv_repeats=args.cv_repeats,
            tables_directory=args.output_dir / "validation_tables",
        )
        reproduce_paper_figures(frame, args.output_dir)
        return 0
    if args.command == "show-config":
        print(json.dumps(asdict(load_config(args.config)), indent=2, ensure_ascii=False))
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
