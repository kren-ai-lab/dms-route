"""Command-line interface for configuration-driven DMS pipelines."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dms_parser.config import load_pipeline_config
from dms_parser.exceptions import SourceConfigurationError
from dms_parser.pipeline import run_pipeline

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Build the parser for the installed ``dms-parser`` command."""
    parser = argparse.ArgumentParser(
        prog="dms-parser",
        description="Process DMS datasets using dms_parser.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Run a configuration-driven DMS pipeline.",
    )
    run_parser.add_argument(
        "--config",
        "-c",
        required=True,
        type=Path,
        help="Path to the pipeline configuration YAML file.",
    )
    run_parser.add_argument(
        "--only",
        choices=["all", "proteingym", "mavedb"],
        default="all",
        help="Restrict execution to one source (default: all).",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Resolve metadata and WT sequences without downloading scores "
            "or writing processed output."
        ),
    )
    run_parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set the process logging level (default: INFO).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface and return a process exit code."""
    args = _build_parser().parse_args(argv)

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
