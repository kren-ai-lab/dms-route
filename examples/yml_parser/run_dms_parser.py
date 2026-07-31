#!/usr/bin/env python3
"""Thin command-line example for the public dms_parser pipeline API."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dms_parser import (
    SourceConfigurationError,
    load_pipeline_config,
    run_pipeline,
)

logger = logging.getLogger("dms_parser.example_runner")


def main(argv: list[str] | None = None) -> int:
    """Run the YAML pipeline example and return its process exit code."""
    parser = argparse.ArgumentParser(
        description=(
            "Processes ProteinGym and/or MaveDB DMS datasets using dms_parser."
        )
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        type=Path,
        help="Configuration YAML file's location.",
    )
    parser.add_argument(
        "--only",
        choices=["all", "proteingym", "mavedb"],
        default="all",
        help="Restrict run to a single database (default: all).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Resolve metadata and WT sequences without downloading scores "
            "or writing processed output."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        config = load_pipeline_config(args.config)
    except SourceConfigurationError as exc:
        logger.error("Configuration error: %s", exc)
        return 1

    result = run_pipeline(
        config,
        only=args.only,
        dry_run=args.dry_run,
    )
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
