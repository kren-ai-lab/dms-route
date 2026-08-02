"""Command-line interface for configuration-driven DMS pipelines."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import requests

from dms_parser.cache import FilesystemCache
from dms_parser.catalog import DatasetRecord, get_dataset_metadata, list_datasets
from dms_parser.config import load_pipeline_config, validate_source_dataset_id
from dms_parser.exceptions import (
    DMSParserError,
    InvalidPipelineOptionError,
    SourceConfigurationError,
)
from dms_parser.pipeline import (
    _validate_dataset_batch_request,
    download_and_standardize_dataset,
    download_and_standardize_datasets,
    run_pipeline,
)

logger = logging.getLogger(__name__)

_CATALOG_SOURCES = ("mavedb", "proteingym")
_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
_OUTPUT_FORMATS = ("text", "json")
_VARIANT_TYPES = ("substitutions", "indels")
_LIST_COLUMNS = (
    ("SOURCE", "source"),
    ("DATASET_ID", "dataset_id"),
    ("TITLE", "title"),
    ("TARGET_ID", "target_id"),
    ("VARIANT_TYPE", "variant_type"),
    ("N_VARIANTS", "n_variants"),
)
_METADATA_FIELDS = (
    "source",
    "dataset_id",
    "title",
    "target_id",
    "variant_type",
    "n_variants",
)


def _positive_integer(value: str) -> int:
    """Parse a positive integer for argparse."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _non_negative_integer(value: str) -> int:
    """Parse a non-negative integer for argparse."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _non_empty_dataset_id(value: str) -> str:
    """Reject empty dataset identifiers while preserving the supplied value."""
    if not value.strip():
        raise argparse.ArgumentTypeError("must be a non-empty dataset identifier")
    return value


def _non_empty_path(value: str) -> Path:
    """Reject empty paths while preserving normal pathlib parsing."""
    if not value.strip():
        raise argparse.ArgumentTypeError("must be a non-empty path")
    return Path(value)


def _batch_dataset_id(value: str) -> str:
    """Reject whitespace and control characters in a batch dataset ID."""
    _non_empty_dataset_id(value)
    if value != value.strip():
        raise argparse.ArgumentTypeError(
            "must not have leading or trailing whitespace"
        )
    if any(
        ord(character) < 32 or 127 <= ord(character) <= 159
        for character in value
    ):
        raise argparse.ArgumentTypeError("must not contain control characters")
    return value


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
        choices=_LOG_LEVELS,
        default="INFO",
        help="Set the process logging level (default: INFO).",
    )
    run_parser.set_defaults(handler=_run_command)

    list_parser = subparsers.add_parser(
        "list",
        help="List datasets from a source catalog.",
    )
    list_parser.add_argument("--source", choices=_CATALOG_SOURCES, required=True)
    list_parser.add_argument("--query", "-q")
    list_parser.add_argument("--limit", type=_positive_integer)
    list_parser.add_argument("--offset", type=_non_negative_integer, default=0)
    _add_catalog_options(list_parser)
    list_parser.set_defaults(handler=_list_command, command_parser=list_parser)

    metadata_parser = subparsers.add_parser(
        "metadata",
        help="Retrieve metadata for one dataset.",
    )
    metadata_parser.add_argument("--source", choices=_CATALOG_SOURCES, required=True)
    metadata_parser.add_argument(
        "--dataset-id",
        required=True,
        type=_non_empty_dataset_id,
    )
    _add_catalog_options(metadata_parser)
    metadata_parser.set_defaults(handler=_metadata_command, command_parser=metadata_parser)

    download_parser = subparsers.add_parser(
        "download",
        help="Download and standardize one substitutions dataset.",
    )
    download_parser.add_argument(
        "--source",
        choices=_CATALOG_SOURCES,
        required=True,
    )
    download_parser.add_argument(
        "--dataset-id",
        required=True,
        type=_non_empty_dataset_id,
    )
    download_parser.add_argument(
        "--output-dir",
        required=True,
        type=_non_empty_path,
    )
    download_parser.add_argument("--cache-dir", type=_non_empty_path)
    download_parser.add_argument("--refresh", action="store_true")
    download_parser.add_argument("--drop-failed", action="store_true")
    download_parser.add_argument("--add-wildtype-row", action="store_true")
    download_parser.add_argument("--overwrite", action="store_true")
    download_parser.add_argument(
        "--log-level",
        choices=_LOG_LEVELS,
        default="INFO",
        help="Set the process logging level (default: INFO).",
    )
    download_parser.set_defaults(
        handler=_download_command,
        command_parser=download_parser,
    )

    download_many_parser = subparsers.add_parser(
        "download-many",
        help="Download and standardize several substitutions datasets.",
    )
    download_many_parser.add_argument(
        "--source",
        choices=_CATALOG_SOURCES,
        required=True,
    )
    download_many_parser.add_argument(
        "--dataset-id",
        action="append",
        required=True,
        type=_batch_dataset_id,
    )
    download_many_parser.add_argument(
        "--output-dir",
        required=True,
        type=_non_empty_path,
    )
    download_many_parser.add_argument("--cache-dir", type=_non_empty_path)
    download_many_parser.add_argument("--refresh", action="store_true")
    download_many_parser.add_argument("--drop-failed", action="store_true")
    download_many_parser.add_argument("--add-wildtype-row", action="store_true")
    download_many_parser.add_argument("--overwrite", action="store_true")
    download_many_parser.add_argument(
        "--log-level",
        choices=_LOG_LEVELS,
        default="INFO",
        help="Set the process logging level (default: INFO).",
    )
    download_many_parser.set_defaults(
        handler=_download_many_command,
        command_parser=download_many_parser,
    )
    return parser


def _add_catalog_options(parser: argparse.ArgumentParser) -> None:
    """Add options shared by the catalog commands."""
    parser.add_argument("--variant-type", choices=_VARIANT_TYPES)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--format", choices=_OUTPUT_FORMATS, default="text")
    parser.add_argument("--output", "-o", type=Path)
    parser.add_argument(
        "--log-level",
        choices=_LOG_LEVELS,
        default="INFO",
        help="Set the process logging level (default: INFO).",
    )


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface and return a process exit code."""
    args = _build_parser().parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    return args.handler(args)


def _run_command(args: argparse.Namespace) -> int:
    """Execute the existing configuration-driven pipeline command."""
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


def _list_command(args: argparse.Namespace) -> int:
    """List catalog records and render them in the selected format."""
    try:
        cache = _catalog_cache(args)
        records = list_datasets(
            args.source,
            query=args.query,
            limit=args.limit,
            offset=args.offset,
            variant_type=args.variant_type,
            cache=cache,
            refresh=args.refresh,
        )
        rendered = (
            _render_list_text(records)
            if args.format == "text"
            else _render_json([_record_values(record) for record in records])
        )
        _publish_output(rendered, args.output)
    except (DMSParserError, requests.RequestException, OSError) as exc:
        logger.error("Catalog list failed: %s", exc)
        return 1
    return 0


def _metadata_command(args: argparse.Namespace) -> int:
    """Retrieve one catalog record and render it in the selected format."""
    try:
        cache = _catalog_cache(args)
        record = get_dataset_metadata(
            args.source,
            args.dataset_id,
            variant_type=args.variant_type,
            cache=cache,
            refresh=args.refresh,
        )
        rendered = (
            _render_metadata_text(record)
            if args.format == "text"
            else _render_json(_record_values(record))
        )
        _publish_output(rendered, args.output)
    except (DMSParserError, requests.RequestException, OSError) as exc:
        logger.error("Catalog metadata lookup failed: %s", exc)
        return 1
    return 0


def _download_command(args: argparse.Namespace) -> int:
    """Download and standardize one source dataset."""
    try:
        validate_source_dataset_id(args.source, args.dataset_id)
    except SourceConfigurationError as exc:
        args.command_parser.error(str(exc))

    cache_root = (
        args.cache_dir.expanduser()
        if args.cache_dir is not None
        else _default_cache_root()
    )
    try:
        result = download_and_standardize_dataset(
            args.source,
            args.dataset_id,
            output_dir=args.output_dir.expanduser(),
            cache=FilesystemCache(cache_root),
            refresh=args.refresh,
            drop_failed=args.drop_failed,
            add_wildtype_row=args.add_wildtype_row,
            overwrite=args.overwrite,
        )
    except (DMSParserError, requests.RequestException, OSError) as exc:
        logger.error("Dataset download failed: %s", exc)
        return 1

    logger.info(
        "Dataset download completed dataset=%s summary_csv=%s summary_json=%s",
        result.dataset_path,
        result.summary_csv_path,
        result.summary_json_path,
    )
    return 0


def _download_many_command(args: argparse.Namespace) -> int:
    """Download and standardize an ordered batch from one source."""
    output_dir = args.output_dir.expanduser()
    try:
        _validate_dataset_batch_request(
            args.source,
            args.dataset_id,
            output_dir=output_dir,
            refresh=args.refresh,
            drop_failed=args.drop_failed,
            add_wildtype_row=args.add_wildtype_row,
            overwrite=args.overwrite,
        )
    except (SourceConfigurationError, InvalidPipelineOptionError) as exc:
        args.command_parser.error(str(exc))
    except OSError as exc:
        logger.error("Dataset batch preflight failed: %s", exc)
        return 1

    cache_root = (
        args.cache_dir.expanduser()
        if args.cache_dir is not None
        else _default_cache_root()
    )
    try:
        result = download_and_standardize_datasets(
            args.source,
            args.dataset_id,
            output_dir=output_dir,
            cache=FilesystemCache(cache_root),
            refresh=args.refresh,
            drop_failed=args.drop_failed,
            add_wildtype_row=args.add_wildtype_row,
            overwrite=args.overwrite,
        )
    except (DMSParserError, requests.RequestException, OSError) as exc:
        logger.error("Dataset batch download failed: %s", exc)
        return 1

    logger.info(
        "Dataset batch completed successes=%d failures=%d summary_csv=%s "
        "summary_json=%s",
        result.success_count,
        result.failure_count,
        result.summary_csv_path,
        result.summary_json_path,
    )
    return result.exit_code


def _catalog_cache(args: argparse.Namespace) -> FilesystemCache | None:
    """Validate source-specific options and lazily create a ProteinGym cache."""
    if args.source == "mavedb":
        unsupported = [
            option
            for option, supplied in (
                ("--variant-type", args.variant_type is not None),
                ("--cache-dir", args.cache_dir is not None),
                ("--refresh", args.refresh),
            )
            if supplied
        ]
        if unsupported:
            args.command_parser.error(
                "source mavedb does not support " + ", ".join(unsupported)
            )
        return None

    cache_root = (
        args.cache_dir.expanduser()
        if args.cache_dir is not None
        else _default_cache_root()
    )
    return FilesystemCache(cache_root)


def _default_cache_root() -> Path:
    """Return the shared default cache root without creating it."""
    return Path.home() / ".cache" / "dms-parser"


def _record_values(record: DatasetRecord) -> dict[str, Any]:
    """Return all DatasetRecord fields in dataclass order."""
    return asdict(record)


def _render_json(value: object) -> str:
    """Render a JSON-safe value with the catalog output contract."""
    return json.dumps(
        value,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


def _display_value(value: object) -> str:
    """Render one human-readable value without mutating its source value."""
    if value is None:
        return "-"
    return re.sub(r"[\t\n\r\f\v]+", " ", str(value))


def _render_list_text(records: list[DatasetRecord]) -> str:
    """Render catalog records as a deterministic aligned table."""
    headers = [header for header, _ in _LIST_COLUMNS]
    rows = [
        [_display_value(getattr(record, field)) for _, field in _LIST_COLUMNS]
        for record in records
    ]
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        if rows
        else len(header)
        for index, header in enumerate(headers)
    ]

    def render_row(values: list[str]) -> str:
        return "  ".join(
            value.ljust(widths[index])
            for index, value in enumerate(values)
        ).rstrip()

    lines = [render_row(headers), render_row(["-" * width for width in widths])]
    lines.extend(render_row(row) for row in rows)
    if not rows:
        lines.append("No datasets found.")
    return "\n".join(lines) + "\n"


def _render_metadata_text(record: DatasetRecord) -> str:
    """Render one catalog record with its complete raw metadata."""
    lines = [
        f"{field}: {_display_value(getattr(record, field))}"
        for field in _METADATA_FIELDS
    ]
    raw_metadata = _render_json(record.raw_metadata).rstrip("\n")
    lines.append("raw_metadata:")
    lines.extend(f"  {line}" for line in raw_metadata.splitlines())
    return "\n".join(lines) + "\n"


def _publish_output(rendered: str, output_path: Path | None) -> None:
    """Write rendered output to stdout or publish it atomically to a file."""
    if output_path is None:
        sys.stdout.write(rendered)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(rendered)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                pass
    logger.info("Catalog output written path=%s", output_path)
