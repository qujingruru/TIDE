#!/usr/bin/env python3
"""Summarize constructed-input checks for the TIDE measurement logic."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tide.diagnostics import write_diagnostic_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", "-o", type=Path, required=True)
    args = parser.parse_args()
    output = write_diagnostic_report(pd.read_csv(args.input), args.output)
    print(output.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
