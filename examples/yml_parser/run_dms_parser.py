#!/usr/bin/env python3
"""Compatibility wrapper for the installed ``dms-parser run`` command."""

from __future__ import annotations

import sys

from dms_parser.cli import main as cli_main


def main(argv: list[str] | None = None) -> int:
    """Forward the legacy example arguments to ``dms-parser run``."""
    arguments = sys.argv[1:] if argv is None else argv
    return cli_main(["run", *arguments])


if __name__ == "__main__":
    sys.exit(main())
