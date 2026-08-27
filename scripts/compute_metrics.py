#!/usr/bin/env python3
"""Compatibility entry point for the original compute_metrics.py workflow."""

from __future__ import annotations

import sys

from tide.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["compute", *sys.argv[1:]]))
