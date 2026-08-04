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
from dms_parser.downloads import (
    _validate_dataset_batch_request,
    download_and_standardize_dataset,
    download_and_standardize_datasets,
)
from dms_parser.exceptions import (
    DMSParserError,
    InvalidPipelineOptionError,
    MaveDBSnapshotTableError,
    SourceConfigurationError,
)
from dms_parser.pipeline import run_pipeline
from dms_parser.sources.mavedb_bulk_catalog import (
    MaveDBBulkCatalog,
    MaveDBDiscoveredExperiment,
    MaveDBDiscoveredScoreSet,
    MaveDBDiscoveryResult,
)
from dms_parser.sources.mavedb_snapshots import (
    MaveDBSnapshot,
    fetch_mavedb_snapshot,
)
from dms_parser.sources.mavedb_snapshot_tables import (
    MaveDBSnapshotTable,
    MaveDBSnapshotTableExtractionResult,
    _validate_dataset_ids as _validate_snapshot_table_dataset_ids,
    extract_mavedb_snapshot_tables,
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


def _positive_record_id(value: str) -> str:
    """Parse a positive Zenodo record ID without changing its representation."""
    if re.fullmatch(r"[1-9][0-9]*", value) is None:
        raise argparse.ArgumentTypeError("must be a positive Zenodo record ID")
    return value


def _snapshot_selector(value: str) -> str:
    """Parse ``latest`` or one positive concrete Zenodo record ID."""
    if value == "latest":
        return value
    return _positive_record_id(value)


def _non_empty_query(value: str) -> str:
    """Reject an empty search query while preserving its supplied casing."""
    if not value.strip():
        raise argparse.ArgumentTypeError("must be a non-empty query")
    return value


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


def _snapshot_table_dataset_id(value: str) -> str:
    """Parse one canonical MaveDB score-set URN for snapshot extraction."""
    _batch_dataset_id(value)
    try:
        validate_source_dataset_id("mavedb", value)
    except SourceConfigurationError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return value


def _add_snapshot_selector(parser: argparse.ArgumentParser) -> None:
    """Add the shared required snapshot selector arguments."""
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument(
        "--latest",
        action="store_true",
        help="Resolve the latest concrete MaveDB snapshot.",
    )
    selector.add_argument(
        "--record",
        type=_positive_record_id,
        help="Use one fixed concrete Zenodo record ID.",
    )


def _add_snapshot_options(parser: argparse.ArgumentParser) -> None:
    """Add cache and format options shared by snapshot commands."""
    parser.add_argument("--cache-dir", type=_non_empty_path)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--format",
        choices=_OUTPUT_FORMATS,
        default="text",
    )


def _add_log_level_option(parser: argparse.ArgumentParser) -> None:
    """Add the shared logging-level option."""
    parser.add_argument(
        "--log-level",
        choices=_LOG_LEVELS,
        default="INFO",
        help="Set the process logging level (default: INFO).",
    )


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

    snapshot_parser = subparsers.add_parser(
        "snapshot",
        help="Manage official MaveDB bulk snapshots.",
    )
    snapshot_subparsers = snapshot_parser.add_subparsers(
        dest="snapshot_command",
        required=True,
    )
    snapshot_fetch_parser = snapshot_subparsers.add_parser(
        "fetch",
        help="Resolve, verify, and cache a MaveDB bulk snapshot.",
    )
    _add_snapshot_selector(snapshot_fetch_parser)
    _add_snapshot_options(snapshot_fetch_parser)
    _add_log_level_option(snapshot_fetch_parser)
    snapshot_fetch_parser.set_defaults(handler=_snapshot_fetch_command)

    snapshot_extract_parser = snapshot_subparsers.add_parser(
        "extract",
        help="Extract selected raw score/count CSVs from a managed snapshot.",
        description=(
            "Extract selected raw score/count CSVs from a managed MaveDB "
            "snapshot. An uncached snapshot may download approximately 1.9 GB."
        ),
    )
    _add_snapshot_selector(snapshot_extract_parser)
    snapshot_extract_parser.add_argument(
        "--dataset-id",
        action="append",
        required=True,
        type=_snapshot_table_dataset_id,
        help="Canonical MaveDB score-set URN; repeat for multiple tables.",
    )
    snapshot_extract_parser.add_argument(
        "--include-superseded",
        action="store_true",
    )
    _add_snapshot_options(snapshot_extract_parser)
    snapshot_extract_parser.add_argument(
        "--output",
        "-o",
        type=_non_empty_path,
    )
    _add_log_level_option(snapshot_extract_parser)
    snapshot_extract_parser.set_defaults(
        handler=_snapshot_extract_command,
        command_parser=snapshot_extract_parser,
    )

    discover_parser = subparsers.add_parser(
        "discover",
        help="Search score sets in a local MaveDB bulk catalog.",
    )
    discovery_source = discover_parser.add_mutually_exclusive_group(
        required=True
    )
    discovery_source.add_argument(
        "--main-json",
        type=_non_empty_path,
        help="Use a local extracted MaveDB main.json.",
    )
    discovery_source.add_argument(
        "--snapshot",
        type=_snapshot_selector,
        help="Use 'latest' or one concrete managed snapshot record ID.",
    )
    discover_parser.add_argument(
        "--query",
        "-q",
        required=True,
        type=_non_empty_query,
    )
    discover_parser.add_argument("--include-superseded", action="store_true")
    discover_parser.add_argument("--cache-dir", type=_non_empty_path)
    discover_parser.add_argument("--refresh", action="store_true")
    discover_parser.add_argument(
        "--format",
        choices=_OUTPUT_FORMATS,
        default="text",
    )
    discover_parser.add_argument("--output", "-o", type=_non_empty_path)
    discover_parser.add_argument(
        "--log-level",
        choices=_LOG_LEVELS,
        default="INFO",
        help="Set the process logging level (default: INFO).",
    )
    discover_parser.set_defaults(
        handler=_discover_command,
        command_parser=discover_parser,
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


def _snapshot_fetch_command(args: argparse.Namespace) -> int:
    """Resolve and prepare one managed MaveDB bulk snapshot."""
    cache_root = (
        args.cache_dir.expanduser()
        if args.cache_dir is not None
        else _default_cache_root()
    )
    selector = "latest" if args.latest else args.record
    try:
        result = fetch_mavedb_snapshot(
            selector,
            cache=FilesystemCache(cache_root),
            refresh=args.refresh,
        )
        rendered = (
            _render_snapshot_text(result)
            if args.format == "text"
            else _render_json(_snapshot_values(result))
        )
        _publish_output(rendered, None)
    except (DMSParserError, requests.RequestException, OSError) as exc:
        logger.error("MaveDB snapshot fetch failed: %s", exc)
        return 1
    return 0


def _snapshot_extract_command(args: argparse.Namespace) -> int:
    """Resolve one managed snapshot and extract explicit raw score tables."""
    try:
        dataset_ids = _validate_snapshot_table_dataset_ids(args.dataset_id)
    except MaveDBSnapshotTableError as exc:
        args.command_parser.error(str(exc))

    cache_root = (
        args.cache_dir.expanduser()
        if args.cache_dir is not None
        else _default_cache_root()
    )
    cache = FilesystemCache(cache_root)
    selector = "latest" if args.latest else args.record
    try:
        snapshot = fetch_mavedb_snapshot(
            selector,
            cache=cache,
            refresh=args.refresh,
        )
        result = extract_mavedb_snapshot_tables(
            snapshot,
            dataset_ids,
            cache=cache,
            include_superseded=args.include_superseded,
            refresh=args.refresh,
        )
        rendered = (
            _render_snapshot_table_text(result)
            if args.format == "text"
            else _render_json(_snapshot_table_extraction_values(result))
        )
        _publish_output(rendered, args.output)
    except (DMSParserError, requests.RequestException, OSError) as exc:
        logger.error("MaveDB snapshot table extraction failed: %s", exc)
        return 1
    return 0


def _discover_command(args: argparse.Namespace) -> int:
    """Search one local or managed MaveDB bulk catalog."""
    if args.main_json is not None:
        unsupported = [
            option
            for option, supplied in (
                ("--cache-dir", args.cache_dir is not None),
                ("--refresh", args.refresh),
            )
            if supplied
        ]
        if unsupported:
            args.command_parser.error(
                "--main-json does not support " + ", ".join(unsupported)
            )

    try:
        if args.main_json is not None:
            catalog_path = args.main_json.expanduser()
            source = {"kind": "local"}
        else:
            cache_root = (
                args.cache_dir.expanduser()
                if args.cache_dir is not None
                else _default_cache_root()
            )
            snapshot = fetch_mavedb_snapshot(
                args.snapshot,
                cache=FilesystemCache(cache_root),
                refresh=args.refresh,
            )
            catalog_path = snapshot.main_json_path
            source = _discovery_snapshot_source(snapshot)

        catalog = MaveDBBulkCatalog.from_file(catalog_path)
        result = catalog.search_by_gene(
            args.query,
            include_superseded=args.include_superseded,
        )
        rendered = (
            _render_discovery_text(result, source)
            if args.format == "text"
            else _render_json(_discovery_values(result, source))
        )
        _publish_output(rendered, args.output)
    except (DMSParserError, requests.RequestException, OSError) as exc:
        logger.error("MaveDB bulk discovery failed: %s", exc)
        return 1
    return 0


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


def _snapshot_values(snapshot: MaveDBSnapshot) -> dict[str, Any]:
    """Return the deterministic JSON representation for one snapshot."""
    return {
        "record_id": snapshot.record.record_id,
        "doi": snapshot.record.doi,
        "concept_doi": snapshot.record.concept_doi,
        "publication_date": snapshot.record.publication_date,
        "archive_filename": snapshot.record.filename,
        "size": snapshot.record.size,
        "checksum": snapshot.record.checksum,
        "archive_path": str(snapshot.archive_path),
        "main_json_path": str(snapshot.main_json_path),
        "cache_hit": snapshot.cache_hit,
    }


def _snapshot_table_values(table: MaveDBSnapshotTable) -> dict[str, Any]:
    """Serialize one extracted raw-table bundle."""
    return {
        "dataset_id": table.dataset_id,
        "is_superseded": table.is_superseded,
        "scores_path": str(table.scores_path),
        "counts_path": (
            str(table.counts_path) if table.counts_path is not None else None
        ),
        "cache_hit": table.cache_hit,
    }


def _snapshot_table_extraction_values(
    result: MaveDBSnapshotTableExtractionResult,
) -> dict[str, Any]:
    """Return deterministic JSON values for snapshot-table extraction."""
    record = result.record
    return {
        "record_id": record.record_id,
        "doi": record.doi,
        "concept_doi": record.concept_doi,
        "publication_date": record.publication_date,
        "archive": {
            "filename": record.filename,
            "size": record.size,
            "checksum": record.checksum,
        },
        "dataset_count": result.dataset_count,
        "tables": [_snapshot_table_values(table) for table in result.tables],
    }


def _discovery_snapshot_source(snapshot: MaveDBSnapshot) -> dict[str, Any]:
    """Return deterministic provenance for one managed discovery source."""
    return {
        "kind": "snapshot",
        "record_id": snapshot.record.record_id,
        "doi": snapshot.record.doi,
        "concept_doi": snapshot.record.concept_doi,
        "publication_date": snapshot.record.publication_date,
        "filename": snapshot.record.filename,
        "size": snapshot.record.size,
        "checksum": snapshot.record.checksum,
    }


def _discovered_score_set_values(
    score_set: MaveDBDiscoveredScoreSet,
) -> dict[str, Any]:
    """Serialize one discovered score set with JSON lists."""
    return {
        "dataset_id": score_set.dataset_id,
        "title": score_set.title,
        "n_variants": score_set.n_variants,
        "targets": list(score_set.targets),
        "is_superseded": score_set.is_superseded,
    }


def _discovered_experiment_values(
    experiment: MaveDBDiscoveredExperiment,
) -> dict[str, Any]:
    """Serialize one discovered experiment group."""
    return {
        "experiment_set_id": experiment.experiment_set_id,
        "experiment_id": experiment.experiment_id,
        "title": experiment.title,
        "score_sets": [
            _discovered_score_set_values(score_set)
            for score_set in experiment.score_sets
        ],
    }


def _discovery_values(
    result: MaveDBDiscoveryResult,
    source: dict[str, Any],
) -> dict[str, Any]:
    """Return deterministic machine-readable discovery output."""
    return {
        "query": result.query,
        "snapshot_title": result.snapshot_title,
        "as_of": result.as_of,
        "experiment_count": result.experiment_count,
        "score_set_count": result.score_set_count,
        "source": source,
        "experiments": [
            _discovered_experiment_values(experiment)
            for experiment in result.experiments
        ],
    }


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


def _render_snapshot_text(snapshot: MaveDBSnapshot) -> str:
    """Render one managed snapshot as stable human-readable fields."""
    values = _snapshot_values(snapshot)
    return "\n".join(
        f"{field}: {_display_value(value)}"
        for field, value in values.items()
    ) + "\n"


def _render_snapshot_table_text(
    result: MaveDBSnapshotTableExtractionResult,
) -> str:
    """Render selected raw snapshot tables as stable human-readable fields."""
    lines = [
        f"record_id: {_display_value(result.record.record_id)}",
        f"doi: {_display_value(result.record.doi)}",
        f"archive_filename: {_display_value(result.record.filename)}",
        f"dataset_count: {result.dataset_count}",
    ]
    for table in result.tables:
        lines.extend(
            [
                f"dataset_id: {_display_value(table.dataset_id)}",
                "  status: "
                + ("superseded" if table.is_superseded else "current"),
                f"  scores_path: {_display_value(table.scores_path)}",
                "  counts_path: "
                + (
                    _display_value(table.counts_path)
                    if table.counts_path is not None
                    else "not available"
                ),
                "  cache: " + ("hit" if table.cache_hit else "extracted"),
            ]
        )
    return "\n".join(lines) + "\n"


def _render_discovery_text(
    result: MaveDBDiscoveryResult,
    source: dict[str, Any],
) -> str:
    """Render discovery results grouped by experiment."""
    source_label = "local" if source["kind"] == "local" else "managed snapshot"
    lines = [
        f"query: {_display_value(result.query)}",
        f"snapshot_title: {_display_value(result.snapshot_title)}",
        f"as_of: {_display_value(result.as_of)}",
        f"source: {source_label}",
    ]
    if source["kind"] == "snapshot":
        lines.append(f"record_id: {_display_value(source['record_id'])}")
    lines.extend(
        [
            f"experiment_count: {result.experiment_count}",
            f"score_set_count: {result.score_set_count}",
        ]
    )
    if not result.experiments:
        lines.append("No matching score sets found.")
        return "\n".join(lines) + "\n"

    for experiment in result.experiments:
        lines.extend(
            [
                f"experiment: {_display_value(experiment.experiment_id)}",
                "  experiment_set: "
                f"{_display_value(experiment.experiment_set_id)}",
                f"  title: {_display_value(experiment.title)}",
            ]
        )
        for score_set in experiment.score_sets:
            lines.extend(
                [
                    f"  score_set: {_display_value(score_set.dataset_id)}",
                    f"    title: {_display_value(score_set.title)}",
                    "    targets: "
                    + (
                        ", ".join(
                            _display_value(target)
                            for target in score_set.targets
                        )
                        if score_set.targets
                        else "-"
                    ),
                    f"    n_variants: {_display_value(score_set.n_variants)}",
                    "    status: "
                    + ("superseded" if score_set.is_superseded else "current"),
                ]
            )
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
