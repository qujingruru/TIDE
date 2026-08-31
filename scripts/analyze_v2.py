#!/usr/bin/env python3
"""Compatibility entry point for TIDE's cluster-aware validation workflow."""

from __future__ import annotations

import sys

from tide.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["validate", *sys.argv[1:]]))
