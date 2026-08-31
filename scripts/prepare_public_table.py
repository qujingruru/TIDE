#!/usr/bin/env python3
"""Build the public text-free table from corrected pipeline output."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tide.publication import build_public_table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--ratings", required=True, type=Path)
    parser.add_argument(
        "--quality-flags",
        type=Path,
        help="Optional text-free dialogue-level source-quality flag CSV",
    )
    parser.add_argument(
        "--semantic-sensitivity",
        type=Path,
        help="Optional text-free semantic readouts from an alternative embedding policy",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    public = build_public_table(
        pd.read_csv(args.metrics),
        pd.read_csv(args.ratings),
        quality_flags=pd.read_csv(args.quality_flags) if args.quality_flags else None,
        semantic_sensitivity=(
            pd.read_csv(args.semantic_sensitivity) if args.semantic_sensitivity else None
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    public.to_csv(args.output, index=False)
    print(f"Wrote {len(public)} text-free rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
