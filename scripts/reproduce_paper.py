#!/usr/bin/env python3
"""Generate the validation report and data-driven paper figures."""

from __future__ import annotations

import sys

from tide.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["reproduce", *sys.argv[1:]]))
