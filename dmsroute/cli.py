"""Command-line interface for configuration-driven DMS pipelines."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
import tempfile
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Annotated, Any

import requests
import typer

from dmsroute.acquisition.cache import FilesystemCache
from dmsroute.acquisition.cache_inventory import (
    CacheInventoryArtifact,
    CacheInventoryEntry,
    CacheInventoryResult,
    inventory_cache,
)
from dmsroute.catalog import DatasetRecord, get_dataset_metadata, list_datasets
from dmsroute.config import load_pipeline_config, validate_source_dataset_id
from dmsroute.downloads import (
    _standardization_options,
    _validate_batch_fallbacks,
    _validate_download_acquisition,
    _validate_download_variant_type,
    _validate_dataset_batch_request,
    download_and_standardize_dataset,
    download_and_standardize_datasets,
)
from dmsroute.core.exceptions import (
    CacheInventoryError,
    DMSParserError,
    InvalidPipelineOptionError,
    MaveDBSnapshotTableError,
    SourceConfigurationError,
)
from dmsroute.pipeline import run_pipeline
from dmsroute.sources.mavedb_bulk_catalog import (
    MaveDBBulkCatalog,
    MaveDBDiscoveredExperiment,
    MaveDBDiscoveredScoreSet,
    MaveDBDiscoveryResult,
)
from dmsroute.sources.mavedb_snapshots import (
    MaveDBSnapshot,
    fetch_mavedb_snapshot,
)
from dmsroute.sources.mavedb_snapshot_tables import (
    MaveDBSnapshotTable,
    MaveDBSnapshotTableExtractionResult,
    _validate_dataset_ids as _validate_snapshot_table_dataset_ids,
    extract_mavedb_snapshot_tables,
)

logger = logging.getLogger(__name__)

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

_HELP_CONTEXT = {"help_option_names": ["-h", "--help"]}
_COMPATIBILITY_OUTCOME: str | None = None


class _CatalogSource(str, Enum):
    """Supported source names exposed by catalog and download commands."""

    mavedb = "mavedb"
    proteingym = "proteingym"


class _LogLevel(str, Enum):
    """Supported process logging levels."""

    debug = "DEBUG"
    info = "INFO"
    warning = "WARNING"
    error = "ERROR"


class _OutputFormat(str, Enum):
    """Supported human- and machine-readable output formats."""

    text = "text"
    json = "json"


class _VariantType(str, Enum):
    """ProteinGym dataset family selections."""

    substitutions = "substitutions"
    indels = "indels"


class _PipelineSource(str, Enum):
    """Source selection accepted by the configuration pipeline."""

    all = "all"
    proteingym = "proteingym"
    mavedb = "mavedb"


class _Acquisition(str, Enum):
    """MaveDB standardized-download acquisition backends."""

    api = "api"
    snapshot = "snapshot"


class _RelativeMethod(str, Enum):
    """Supported WT-relative score transformations."""

    ratio = "ratio"
    log_ratio = "log_ratio"
    log2_ratio = "log2_ratio"
    difference = "difference"


class _BooleanValue(str, Enum):
    """Explicit boolean values for non-flag CLI options."""

    true = "true"
    false = "false"


def _positive_integer(value: str) -> int:
    """Parse a positive integer for a Typer option."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise typer.BadParameter("must be a positive integer") from exc
    if parsed <= 0:
        raise typer.BadParameter("must be a positive integer")
    return parsed


def _non_negative_integer(value: str) -> int:
    """Parse a non-negative integer for a Typer option."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise typer.BadParameter("must be a non-negative integer") from exc
    if parsed < 0:
        raise typer.BadParameter("must be a non-negative integer")
    return parsed


def _positive_record_id(value: str) -> str:
    """Parse a positive Zenodo record ID without changing its representation."""
    if re.fullmatch(r"[1-9][0-9]*", value) is None:
        raise typer.BadParameter("must be a positive Zenodo record ID")
    return value


def _snapshot_selector(value: str) -> str:
    """Parse ``latest`` or one positive concrete Zenodo record ID."""
    if value == "latest":
        return value
    return _positive_record_id(value)


def _non_empty_query(value: str) -> str:
    """Reject an empty search query while preserving its supplied casing."""
    if not value.strip():
        raise typer.BadParameter("must be a non-empty query")
    return value


def _non_empty_dataset_id(value: str) -> str:
    """Reject empty dataset identifiers while preserving the supplied value."""
    if not value.strip():
        raise typer.BadParameter("must be a non-empty dataset identifier")
    return value


def _non_empty_source(value: str) -> str:
    """Reject empty cache source filters while preserving their casing."""
    if not value.strip():
        raise typer.BadParameter("must be a non-empty source")
    return value


def _non_empty_path(value: str) -> Path:
    """Reject empty paths while preserving normal pathlib parsing."""
    if not value.strip():
        raise typer.BadParameter("must be a non-empty path")
    return Path(value)


def _batch_dataset_id(value: str) -> str:
    """Reject whitespace and control characters in a batch dataset ID."""
    _non_empty_dataset_id(value)
    if value != value.strip():
        raise typer.BadParameter(
            "must not have leading or trailing whitespace"
        )
    if any(
        ord(character) < 32 or 127 <= ord(character) <= 159
        for character in value
    ):
        raise typer.BadParameter("must not contain control characters")
    return value


def _read_dataset_id_file(path: Path) -> list[str]:
    """Read ordered dataset identifiers from a UTF-8 text file."""
    dataset_ids: list[str] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(),
        start=1,
    ):
        dataset_id = line.strip()
        if not dataset_id:
            continue
        try:
            _batch_dataset_id(dataset_id)
        except typer.BadParameter as exc:
            raise InvalidPipelineOptionError(
                f"Invalid dataset identifier in {path} at line "
                f"{line_number}: {exc.message}."
            ) from exc
        dataset_ids.append(dataset_id)
    return dataset_ids


def _batch_fallback_mapping(
    values: list[str] | None,
    dataset_ids: list[str],
    *,
    field: str,
) -> dict[str, str] | dict[str, float]:
    """Parse repeatable DATASET_ID=VALUE fallback options."""
    resolved: dict[str, Any] = {}
    requested = set(dataset_ids)
    for value in values or []:
        if "=" not in value:
            raise InvalidPipelineOptionError(
                f"{field} entries must use DATASET_ID=VALUE."
            )
        dataset_id, supplied = value.split("=", 1)
        if not dataset_id or not supplied:
            raise InvalidPipelineOptionError(
                f"{field} entries must use non-empty DATASET_ID=VALUE."
            )
        if dataset_id not in requested:
            raise InvalidPipelineOptionError(
                f"{field} contains an unrequested dataset: {dataset_id!r}."
            )
        if dataset_id in resolved:
            raise InvalidPipelineOptionError(
                f"Duplicate {field} mapping for {dataset_id!r}."
            )
        if field == "wt_score":
            try:
                parsed = float(supplied)
            except ValueError as exc:
                raise InvalidPipelineOptionError(
                    f"wt_score for {dataset_id!r} must be a finite number."
                ) from exc
            if not math.isfinite(parsed):
                raise InvalidPipelineOptionError(
                    f"wt_score for {dataset_id!r} must be a finite number."
                )
            resolved[dataset_id] = parsed
        else:
            resolved[dataset_id] = supplied
    return resolved


def _snapshot_table_dataset_id(value: str) -> str:
    """Parse one canonical MaveDB score-set URN for snapshot extraction."""
    _batch_dataset_id(value)
    try:
        validate_source_dataset_id("mavedb", value)
    except SourceConfigurationError as exc:
        raise typer.BadParameter(str(exc)) from exc
    return value


app = typer.Typer(
    name="dmsroute",
    help="Process DMS datasets using dmsroute.",
    add_completion=False,
    no_args_is_help=False,
    context_settings=_HELP_CONTEXT,
    pretty_exceptions_enable=False,
)
snapshot_app = typer.Typer(
    name="snapshot",
    help="Manage official MaveDB bulk snapshots.",
    add_completion=False,
    no_args_is_help=False,
    context_settings=_HELP_CONTEXT,
    pretty_exceptions_enable=False,
)


def _start_command(log_level: _LogLevel) -> None:
    """Mark command execution and configure the established logging format."""
    global _COMPATIBILITY_OUTCOME
    _COMPATIBILITY_OUTCOME = "command"
    logging.basicConfig(
        level=getattr(logging, log_level.value),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _usage_error(message: str) -> None:
    """Raise a Typer usage error while preserving ``main(argv)`` behavior."""
    global _COMPATIBILITY_OUTCOME
    _COMPATIBILITY_OUTCOME = "usage"
    raise typer.BadParameter(message)


def _finish(exit_code: int) -> None:
    """Exit from a command only when its established result is nonzero."""
    if exit_code:
        raise typer.Exit(exit_code)


@app.command(
    "run",
    help="Run a configuration-driven DMS pipeline.",
    context_settings=_HELP_CONTEXT,
)
def _run_command(
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            metavar="CONFIG",
            help=(
                "Path to pipeline YAML, including dataset-level WT fallbacks "
                "and opt-in score transforms."
            ),
        ),
    ],
    only: Annotated[
        _PipelineSource,
        typer.Option(
            "--only",
            help="Restrict execution to one source (default: all).",
        ),
    ] = _PipelineSource.all,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help=(
                "Resolve metadata and WT sequences without downloading scores "
                "or writing processed output."
            ),
        ),
    ] = False,
    log_level: Annotated[
        _LogLevel,
        typer.Option(
            "--log-level",
            help="Set the process logging level (default: INFO).",
        ),
    ] = _LogLevel.info,
) -> None:
    """Execute the existing configuration-driven pipeline command."""
    _start_command(log_level)
    try:
        config = load_pipeline_config(config_path)
    except SourceConfigurationError as exc:
        logger.error("Configuration error: %s", exc)
        raise typer.Exit(1) from None

    result = run_pipeline(config, only=only.value, dry_run=dry_run)
    _finish(result.exit_code)


@app.command(
    "cache",
    help="Inspect artifacts in the local DMSRoute cache without modifying them.",
    context_settings=_HELP_CONTEXT,
)
def _cache_command(
    cache_dir: Annotated[
        Path | None,
        typer.Option(
            "--cache-dir",
            parser=_non_empty_path,
            metavar="PATH",
        ),
    ] = None,
    source: Annotated[
        str | None,
        typer.Option(
            "--source",
            parser=_non_empty_source,
            metavar="TEXT",
        ),
    ] = None,
    dataset_id: Annotated[
        str | None,
        typer.Option(
            "--dataset-id",
            parser=_non_empty_dataset_id,
            metavar="TEXT",
        ),
    ] = None,
    output_format: Annotated[
        _OutputFormat,
        typer.Option("--format"),
    ] = _OutputFormat.text,
    log_level: Annotated[
        _LogLevel,
        typer.Option(
            "--log-level",
            help="Set the process logging level (default: INFO).",
        ),
    ] = _LogLevel.info,
) -> None:
    """Inspect managed cache artifacts without changing filesystem state."""
    _start_command(log_level)
    cache = FilesystemCache(_cache_root(cache_dir))
    try:
        result = inventory_cache(
            cache,
            source=source,
            dataset_id=dataset_id,
        )
    except CacheInventoryError as exc:
        logger.error("Cache inventory failed: %s", exc)
        raise typer.Exit(1) from None
    rendered = (
        _render_cache_inventory_text(cache.root, result)
        if output_format is _OutputFormat.text
        else _render_json(_cache_inventory_values(cache.root, result))
    )
    sys.stdout.write(rendered)


@app.command(
    "list",
    help=(
        "List datasets from a source catalog. MaveDB uses its active API, not "
        "a fixed snapshot."
    ),
    context_settings=_HELP_CONTEXT,
)
def _list_command(
    source: Annotated[_CatalogSource, typer.Option("--source")],
    query: Annotated[
        str | None,
        typer.Option(
            "--query",
            "-q",
            metavar="QUERY",
            help=(
                "For MaveDB, delegate a general text search to the active API; "
                "results follow API pagination and configured/default limits."
            ),
        ),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option("--limit", parser=_positive_integer, metavar="LIMIT"),
    ] = None,
    offset: Annotated[
        int,
        typer.Option("--offset", parser=_non_negative_integer, metavar="OFFSET"),
    ] = 0,
    variant_type: Annotated[
        _VariantType | None,
        typer.Option("--variant-type"),
    ] = None,
    cache_dir: Annotated[
        Path | None,
        typer.Option("--cache-dir", metavar="CACHE_DIR"),
    ] = None,
    refresh: Annotated[bool, typer.Option("--refresh")] = False,
    output_format: Annotated[
        _OutputFormat,
        typer.Option("--format"),
    ] = _OutputFormat.text,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", metavar="OUTPUT"),
    ] = None,
    log_level: Annotated[
        _LogLevel,
        typer.Option(
            "--log-level",
            help="Set the process logging level (default: INFO).",
        ),
    ] = _LogLevel.info,
) -> None:
    """List catalog records and render them in the selected format."""
    _start_command(log_level)
    source_name = source.value
    variant_name = variant_type.value if variant_type is not None else None
    try:
        cache = _catalog_cache(
            source_name,
            variant_type=variant_name,
            cache_dir=cache_dir,
            refresh=refresh,
        )
        records = list_datasets(
            source_name,
            query=query,
            limit=limit,
            offset=offset,
            variant_type=variant_name,
            cache=cache,
            refresh=refresh,
        )
        rendered = (
            _render_list_text(records)
            if output_format is _OutputFormat.text
            else _render_json([_record_values(record) for record in records])
        )
        _publish_output(rendered, output)
    except (DMSParserError, requests.RequestException, OSError) as exc:
        logger.error("Catalog list failed: %s", exc)
        raise typer.Exit(1) from None


@app.command(
    "metadata",
    help="Retrieve metadata for one dataset.",
    context_settings=_HELP_CONTEXT,
)
def _metadata_command(
    source: Annotated[_CatalogSource, typer.Option("--source")],
    dataset_id: Annotated[
        str,
        typer.Option(
            "--dataset-id",
            parser=_non_empty_dataset_id,
            metavar="DATASET_ID",
        ),
    ],
    variant_type: Annotated[
        _VariantType | None,
        typer.Option("--variant-type"),
    ] = None,
    cache_dir: Annotated[
        Path | None,
        typer.Option("--cache-dir", metavar="CACHE_DIR"),
    ] = None,
    refresh: Annotated[bool, typer.Option("--refresh")] = False,
    output_format: Annotated[
        _OutputFormat,
        typer.Option("--format"),
    ] = _OutputFormat.text,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", metavar="OUTPUT"),
    ] = None,
    log_level: Annotated[
        _LogLevel,
        typer.Option(
            "--log-level",
            help="Set the process logging level (default: INFO).",
        ),
    ] = _LogLevel.info,
) -> None:
    """Retrieve one catalog record and render it in the selected format."""
    _start_command(log_level)
    source_name = source.value
    variant_name = variant_type.value if variant_type is not None else None
    try:
        cache = _catalog_cache(
            source_name,
            variant_type=variant_name,
            cache_dir=cache_dir,
            refresh=refresh,
        )
        record = get_dataset_metadata(
            source_name,
            dataset_id,
            variant_type=variant_name,
            cache=cache,
            refresh=refresh,
        )
        rendered = (
            _render_metadata_text(record)
            if output_format is _OutputFormat.text
            else _render_json(_record_values(record))
        )
        _publish_output(rendered, output)
    except (DMSParserError, requests.RequestException, OSError) as exc:
        logger.error("Catalog metadata lookup failed: %s", exc)
        raise typer.Exit(1) from None


def _download_options_help() -> str:
    """Return the shared MaveDB acquisition help text."""
    return (
        "MaveDB-only acquisition backend. Snapshot mode requires a prefetched "
        "concrete record and never downloads the approximately 1.9 GB archive "
        "implicitly."
    )


@app.command(
    "download",
    help="Download and standardize one dataset.",
    context_settings=_HELP_CONTEXT,
)
def _download_command(
    source: Annotated[_CatalogSource, typer.Option("--source")],
    dataset_id: Annotated[
        str,
        typer.Option(
            "--dataset-id",
            parser=_non_empty_dataset_id,
            metavar="DATASET_ID",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            parser=_non_empty_path,
            metavar="OUTPUT_DIR",
        ),
    ],
    variant_type: Annotated[
        _VariantType,
        typer.Option(
            "--variant-type",
            help=(
                "Select the ProteinGym dataset family; this does not filter "
                "MaveDB score-set rows."
            ),
        ),
    ] = _VariantType.substitutions,
    cache_dir: Annotated[
        Path | None,
        typer.Option(
            "--cache-dir",
            parser=_non_empty_path,
            metavar="CACHE_DIR",
        ),
    ] = None,
    refresh: Annotated[bool, typer.Option("--refresh")] = False,
    acquisition: Annotated[
        _Acquisition | None,
        typer.Option("--acquisition", help=_download_options_help()),
    ] = None,
    snapshot_record: Annotated[
        str | None,
        typer.Option(
            "--snapshot-record",
            parser=_positive_record_id,
            metavar="SNAPSHOT_RECORD",
            help="Concrete prefetched Zenodo record ID for snapshot acquisition.",
        ),
    ] = None,
    include_superseded: Annotated[
        bool,
        typer.Option(
            "--include-superseded",
            help="Allow superseded score sets in MaveDB snapshot mode.",
        ),
    ] = False,
    drop_failed: Annotated[bool, typer.Option("--drop-failed")] = False,
    add_wildtype_row: Annotated[
        bool,
        typer.Option("--add-wildtype-row"),
    ] = False,
    wt_sequence: Annotated[
        str | None,
        typer.Option(
            "--wt-sequence",
            metavar="SEQUENCE",
            help="Fallback WT protein sequence used only when source evidence is absent.",
        ),
    ] = None,
    wt_score: Annotated[
        float | None,
        typer.Option(
            "--wt-score",
            metavar="SCORE",
            help=(
                "Explicit fallback reference for --add-relative-score; it never "
                "searches for or selects mutations. A valid observed WT row takes "
                "priority, and no WT value of 0 or 1 is assumed. Downloads without "
                "relative-score calculation do not need it."
            ),
        ),
    ] = None,
    add_relative_score: Annotated[
        bool,
        typer.Option(
            "--add-relative-score",
            help="Add an explicitly requested WT-relative regression score.",
        ),
    ] = False,
    relative_method: Annotated[
        _RelativeMethod,
        typer.Option("--relative-method"),
    ] = _RelativeMethod.log_ratio,
    relative_output_col: Annotated[
        str,
        typer.Option("--relative-output-col", metavar="COLUMN"),
    ] = "score_log_ratio",
    add_binary_label: Annotated[
        bool,
        typer.Option("--add-binary-label"),
    ] = False,
    delta: Annotated[float, typer.Option("--delta")] = 0.1,
    higher_is_better: Annotated[
        _BooleanValue,
        typer.Option("--higher-is-better"),
    ] = _BooleanValue.true,
    binary_output_col: Annotated[
        str,
        typer.Option("--binary-output-col", metavar="COLUMN"),
    ] = "score_binary_like",
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    log_level: Annotated[
        _LogLevel,
        typer.Option(
            "--log-level",
            help="Set the process logging level (default: INFO).",
        ),
    ] = _LogLevel.info,
) -> None:
    """Download and standardize one source dataset."""
    _start_command(log_level)
    source_name = source.value
    variant_type_name = variant_type.value
    acquisition_name = acquisition.value if acquisition is not None else None
    try:
        validate_source_dataset_id(source_name, dataset_id)
        _validate_download_variant_type(source_name, variant_type_name)
        _validate_download_acquisition(
            source_name,
            acquisition=acquisition_name,
            snapshot_record_id=snapshot_record,
            include_superseded=include_superseded,
        )
        _standardization_options(
            wt_sequence=wt_sequence,
            wt_score=wt_score,
            add_relative_score=add_relative_score,
            relative_method=relative_method.value,
            relative_output_col=relative_output_col,
            add_binary_label=add_binary_label,
            delta=delta,
            higher_is_better=higher_is_better is _BooleanValue.true,
            binary_output_col=binary_output_col,
        )
    except (SourceConfigurationError, InvalidPipelineOptionError) as exc:
        _usage_error(str(exc))

    try:
        result = download_and_standardize_dataset(
            source_name,
            dataset_id,
            output_dir=output_dir.expanduser(),
            cache=FilesystemCache(_cache_root(cache_dir)),
            refresh=refresh,
            drop_failed=drop_failed,
            add_wildtype_row=add_wildtype_row,
            wt_sequence=wt_sequence,
            wt_score=wt_score,
            add_relative_score=add_relative_score,
            relative_method=relative_method.value,
            relative_output_col=relative_output_col,
            add_binary_label=add_binary_label,
            delta=delta,
            higher_is_better=higher_is_better is _BooleanValue.true,
            binary_output_col=binary_output_col,
            overwrite=overwrite,
            acquisition=acquisition_name,
            snapshot_record_id=snapshot_record,
            include_superseded=include_superseded,
            variant_type=variant_type_name,
        )
    except (DMSParserError, requests.RequestException, OSError) as exc:
        logger.error("Dataset download failed: %s", exc)
        raise typer.Exit(1) from None

    logger.info(
        "Dataset download completed dataset=%s summary_csv=%s summary_json=%s",
        result.dataset_path,
        result.summary_csv_path,
        result.summary_json_path,
    )


@app.command(
    "download-many",
    help="Download and standardize several datasets.",
    context_settings=_HELP_CONTEXT,
)
def _download_many_command(
    source: Annotated[_CatalogSource, typer.Option("--source")],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            parser=_non_empty_path,
            metavar="OUTPUT_DIR",
        ),
    ],
    dataset_ids: Annotated[
        list[str] | None,
        typer.Option(
            "--dataset-id",
            parser=_batch_dataset_id,
            metavar="DATASET_ID",
        ),
    ] = None,
    dataset_id_file: Annotated[
        Path | None,
        typer.Option(
            "--dataset-id-file",
            parser=_non_empty_path,
            metavar="PATH",
            help=(
                "UTF-8 text file containing one dataset identifier per line; "
                "blank lines are ignored."
            ),
        ),
    ] = None,
    variant_type: Annotated[
        _VariantType,
        typer.Option(
            "--variant-type",
            help=(
                "Select the ProteinGym dataset family; this does not filter "
                "MaveDB score-set rows."
            ),
        ),
    ] = _VariantType.substitutions,
    cache_dir: Annotated[
        Path | None,
        typer.Option(
            "--cache-dir",
            parser=_non_empty_path,
            metavar="CACHE_DIR",
        ),
    ] = None,
    refresh: Annotated[bool, typer.Option("--refresh")] = False,
    acquisition: Annotated[
        _Acquisition | None,
        typer.Option("--acquisition", help=_download_options_help()),
    ] = None,
    snapshot_record: Annotated[
        str | None,
        typer.Option(
            "--snapshot-record",
            parser=_positive_record_id,
            metavar="SNAPSHOT_RECORD",
            help="Concrete prefetched Zenodo record ID for snapshot acquisition.",
        ),
    ] = None,
    include_superseded: Annotated[
        bool,
        typer.Option(
            "--include-superseded",
            help="Allow superseded score sets in MaveDB snapshot mode.",
        ),
    ] = False,
    drop_failed: Annotated[bool, typer.Option("--drop-failed")] = False,
    add_wildtype_row: Annotated[
        bool,
        typer.Option("--add-wildtype-row"),
    ] = False,
    wt_sequence_entries: Annotated[
        list[str] | None,
        typer.Option(
            "--wt-sequence",
            metavar="DATASET_ID=SEQUENCE",
            help="Repeatable per-dataset fallback WT protein sequence.",
        ),
    ] = None,
    wt_score_entries: Annotated[
        list[str] | None,
        typer.Option(
            "--wt-score",
            metavar="DATASET_ID=SCORE",
            help=(
                "Repeatable per-dataset fallback reference for "
                "--add-relative-score; it never searches for or selects mutations. "
                "A valid observed WT row takes priority, and no WT value of 0 or 1 "
                "is assumed. Batches without relative-score calculation do not "
                "need it."
            ),
        ),
    ] = None,
    add_relative_score: Annotated[
        bool,
        typer.Option("--add-relative-score"),
    ] = False,
    relative_method: Annotated[
        _RelativeMethod,
        typer.Option("--relative-method"),
    ] = _RelativeMethod.log_ratio,
    relative_output_col: Annotated[
        str,
        typer.Option("--relative-output-col", metavar="COLUMN"),
    ] = "score_log_ratio",
    add_binary_label: Annotated[
        bool,
        typer.Option("--add-binary-label"),
    ] = False,
    delta: Annotated[float, typer.Option("--delta")] = 0.1,
    higher_is_better: Annotated[
        _BooleanValue,
        typer.Option("--higher-is-better"),
    ] = _BooleanValue.true,
    binary_output_col: Annotated[
        str,
        typer.Option("--binary-output-col", metavar="COLUMN"),
    ] = "score_binary_like",
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    log_level: Annotated[
        _LogLevel,
        typer.Option(
            "--log-level",
            help="Set the process logging level (default: INFO).",
        ),
    ] = _LogLevel.info,
) -> None:
    """Download and standardize an ordered batch from one source."""
    _start_command(log_level)
    source_name = source.value
    variant_type_name = variant_type.value
    acquisition_name = acquisition.value if acquisition is not None else None
    resolved_output_dir = output_dir.expanduser()
    resolved_dataset_id_file = (
        dataset_id_file.expanduser()
        if dataset_id_file is not None
        else None
    )
    try:
        file_dataset_ids = (
            _read_dataset_id_file(resolved_dataset_id_file)
            if resolved_dataset_id_file is not None
            else []
        )
    except InvalidPipelineOptionError as exc:
        _usage_error(str(exc))
    except UnicodeError as exc:
        _usage_error(
            f"Dataset ID file {resolved_dataset_id_file} is not valid "
            f"UTF-8: {exc}"
        )
    except OSError as exc:
        logger.error(
            "Dataset ID file preflight failed path=%s: %s",
            resolved_dataset_id_file,
            exc,
        )
        raise typer.Exit(1) from None

    resolved_dataset_ids = [*(dataset_ids or []), *file_dataset_ids]
    try:
        sequence_fallbacks = _batch_fallback_mapping(
            wt_sequence_entries,
            resolved_dataset_ids,
            field="wt_sequence",
        )
        score_fallbacks = _batch_fallback_mapping(
            wt_score_entries,
            resolved_dataset_ids,
            field="wt_score",
        )
        _validate_download_variant_type(source_name, variant_type_name)
        _validate_download_acquisition(
            source_name,
            acquisition=acquisition_name,
            snapshot_record_id=snapshot_record,
            include_superseded=include_superseded,
        )
        _validate_dataset_batch_request(
            source_name,
            resolved_dataset_ids,
            output_dir=resolved_output_dir,
            refresh=refresh,
            drop_failed=drop_failed,
            add_wildtype_row=add_wildtype_row,
            overwrite=overwrite,
        )
        _validate_batch_fallbacks(
            tuple(resolved_dataset_ids),
            wt_sequence=sequence_fallbacks,
            wt_score=score_fallbacks,
        )
        _standardization_options(
            wt_sequence=None,
            wt_score=None,
            add_relative_score=add_relative_score,
            relative_method=relative_method.value,
            relative_output_col=relative_output_col,
            add_binary_label=add_binary_label,
            delta=delta,
            higher_is_better=higher_is_better is _BooleanValue.true,
            binary_output_col=binary_output_col,
        )
    except (SourceConfigurationError, InvalidPipelineOptionError) as exc:
        _usage_error(str(exc))
    except OSError as exc:
        logger.error("Dataset batch preflight failed: %s", exc)
        raise typer.Exit(1) from None

    try:
        result = download_and_standardize_datasets(
            source_name,
            resolved_dataset_ids,
            output_dir=resolved_output_dir,
            cache=FilesystemCache(_cache_root(cache_dir)),
            refresh=refresh,
            drop_failed=drop_failed,
            add_wildtype_row=add_wildtype_row,
            wt_sequence=sequence_fallbacks,
            wt_score=score_fallbacks,
            add_relative_score=add_relative_score,
            relative_method=relative_method.value,
            relative_output_col=relative_output_col,
            add_binary_label=add_binary_label,
            delta=delta,
            higher_is_better=higher_is_better is _BooleanValue.true,
            binary_output_col=binary_output_col,
            overwrite=overwrite,
            acquisition=acquisition_name,
            snapshot_record_id=snapshot_record,
            include_superseded=include_superseded,
            variant_type=variant_type_name,
        )
    except (DMSParserError, requests.RequestException, OSError) as exc:
        logger.error("Dataset batch download failed: %s", exc)
        raise typer.Exit(1) from None

    logger.info(
        "Dataset batch completed successes=%d failures=%d summary_csv=%s "
        "summary_json=%s",
        result.success_count,
        result.failure_count,
        result.summary_csv_path,
        result.summary_json_path,
    )
    _finish(result.exit_code)


app.add_typer(
    snapshot_app,
    name="snapshot",
    help="Manage official MaveDB bulk snapshots.",
)


@snapshot_app.command(
    "fetch",
    help="Resolve, verify, and cache a MaveDB bulk snapshot.",
    context_settings=_HELP_CONTEXT,
)
def _snapshot_fetch_command(
    latest: Annotated[
        bool,
        typer.Option(
            "--latest",
            help="Resolve the latest concrete MaveDB snapshot.",
        ),
    ] = False,
    record: Annotated[
        str | None,
        typer.Option(
            "--record",
            parser=_positive_record_id,
            metavar="RECORD",
            help="Use one fixed concrete Zenodo record ID.",
        ),
    ] = None,
    cache_dir: Annotated[
        Path | None,
        typer.Option(
            "--cache-dir",
            parser=_non_empty_path,
            metavar="CACHE_DIR",
        ),
    ] = None,
    refresh: Annotated[bool, typer.Option("--refresh")] = False,
    output_format: Annotated[
        _OutputFormat,
        typer.Option("--format"),
    ] = _OutputFormat.text,
    log_level: Annotated[
        _LogLevel,
        typer.Option(
            "--log-level",
            help="Set the process logging level (default: INFO).",
        ),
    ] = _LogLevel.info,
) -> None:
    """Resolve and prepare one managed MaveDB bulk snapshot."""
    _start_command(log_level)
    selector = _snapshot_selection(latest, record)
    try:
        result = fetch_mavedb_snapshot(
            selector,
            cache=FilesystemCache(_cache_root(cache_dir)),
            refresh=refresh,
        )
        rendered = (
            _render_snapshot_text(result)
            if output_format is _OutputFormat.text
            else _render_json(_snapshot_values(result))
        )
        _publish_output(rendered, None)
    except (DMSParserError, requests.RequestException, OSError) as exc:
        logger.error("MaveDB snapshot fetch failed: %s", exc)
        raise typer.Exit(1) from None


@snapshot_app.command(
    "extract",
    help=(
        "Extract selected raw score/count CSVs from a managed MaveDB snapshot. "
        "An uncached snapshot may download approximately 1.9 GB."
    ),
    short_help="Extract selected raw score/count CSVs from a managed snapshot.",
    context_settings=_HELP_CONTEXT,
)
def _snapshot_extract_command(
    dataset_ids: Annotated[
        list[str],
        typer.Option(
            "--dataset-id",
            parser=_snapshot_table_dataset_id,
            metavar="DATASET_ID",
            help="Canonical MaveDB score-set URN; repeat for multiple tables.",
        ),
    ],
    latest: Annotated[
        bool,
        typer.Option(
            "--latest",
            help="Resolve the latest concrete MaveDB snapshot.",
        ),
    ] = False,
    record: Annotated[
        str | None,
        typer.Option(
            "--record",
            parser=_positive_record_id,
            metavar="RECORD",
            help="Use one fixed concrete Zenodo record ID.",
        ),
    ] = None,
    include_superseded: Annotated[
        bool,
        typer.Option("--include-superseded"),
    ] = False,
    cache_dir: Annotated[
        Path | None,
        typer.Option(
            "--cache-dir",
            parser=_non_empty_path,
            metavar="CACHE_DIR",
        ),
    ] = None,
    refresh: Annotated[bool, typer.Option("--refresh")] = False,
    output_format: Annotated[
        _OutputFormat,
        typer.Option("--format"),
    ] = _OutputFormat.text,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            parser=_non_empty_path,
            metavar="OUTPUT",
        ),
    ] = None,
    log_level: Annotated[
        _LogLevel,
        typer.Option(
            "--log-level",
            help="Set the process logging level (default: INFO).",
        ),
    ] = _LogLevel.info,
) -> None:
    """Resolve one managed snapshot and extract explicit raw score tables."""
    _start_command(log_level)
    selector = _snapshot_selection(latest, record)
    try:
        validated_ids = _validate_snapshot_table_dataset_ids(dataset_ids)
    except MaveDBSnapshotTableError as exc:
        _usage_error(str(exc))

    cache = FilesystemCache(_cache_root(cache_dir))
    try:
        snapshot = fetch_mavedb_snapshot(
            selector,
            cache=cache,
            refresh=refresh,
        )
        result = extract_mavedb_snapshot_tables(
            snapshot,
            validated_ids,
            cache=cache,
            include_superseded=include_superseded,
            refresh=refresh,
        )
        rendered = (
            _render_snapshot_table_text(result)
            if output_format is _OutputFormat.text
            else _render_json(_snapshot_table_extraction_values(result))
        )
        _publish_output(rendered, output)
    except (DMSParserError, requests.RequestException, OSError) as exc:
        logger.error("MaveDB snapshot table extraction failed: %s", exc)
        raise typer.Exit(1) from None


@app.command(
    "discover",
    help=(
        "Search score-set metadata in one fixed local MaveDB snapshot, not the "
        "active API."
    ),
    context_settings=_HELP_CONTEXT,
)
def _discover_command(
    query: Annotated[
        str,
        typer.Option(
            "--query",
            "-q",
            parser=_non_empty_query,
            metavar="QUERY",
            help=(
                "Match target names by case-insensitive substring and target "
                "identifiers exactly; superseded score sets are excluded by "
                "default."
            ),
        ),
    ],
    main_json: Annotated[
        Path | None,
        typer.Option(
            "--main-json",
            parser=_non_empty_path,
            metavar="MAIN_JSON",
            help="Use a local extracted MaveDB main.json.",
        ),
    ] = None,
    snapshot: Annotated[
        str | None,
        typer.Option(
            "--snapshot",
            parser=_snapshot_selector,
            metavar="SNAPSHOT",
            help="Use 'latest' or one concrete managed snapshot record ID.",
        ),
    ] = None,
    include_superseded: Annotated[
        bool,
        typer.Option("--include-superseded"),
    ] = False,
    cache_dir: Annotated[
        Path | None,
        typer.Option(
            "--cache-dir",
            parser=_non_empty_path,
            metavar="CACHE_DIR",
        ),
    ] = None,
    refresh: Annotated[bool, typer.Option("--refresh")] = False,
    output_format: Annotated[
        _OutputFormat,
        typer.Option("--format"),
    ] = _OutputFormat.text,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            parser=_non_empty_path,
            metavar="OUTPUT",
        ),
    ] = None,
    log_level: Annotated[
        _LogLevel,
        typer.Option(
            "--log-level",
            help="Set the process logging level (default: INFO).",
        ),
    ] = _LogLevel.info,
) -> None:
    """Search one local or managed MaveDB bulk catalog."""
    _start_command(log_level)
    if (main_json is None) == (snapshot is None):
        _usage_error("exactly one of --main-json or --snapshot is required")
    if main_json is not None:
        unsupported = [
            option
            for option, supplied in (
                ("--cache-dir", cache_dir is not None),
                ("--refresh", refresh),
            )
            if supplied
        ]
        if unsupported:
            _usage_error(
                "--main-json does not support " + ", ".join(unsupported)
            )

    try:
        if main_json is not None:
            catalog_path = main_json.expanduser()
            source = {"kind": "local"}
        else:
            assert snapshot is not None
            managed_snapshot = fetch_mavedb_snapshot(
                snapshot,
                cache=FilesystemCache(_cache_root(cache_dir)),
                refresh=refresh,
            )
            catalog_path = managed_snapshot.main_json_path
            source = _discovery_snapshot_source(managed_snapshot)

        catalog = MaveDBBulkCatalog.from_file(catalog_path)
        result = catalog.search_by_gene(
            query,
            include_superseded=include_superseded,
        )
        rendered = (
            _render_discovery_text(result, source)
            if output_format is _OutputFormat.text
            else _render_json(_discovery_values(result, source))
        )
        _publish_output(rendered, output)
    except (DMSParserError, requests.RequestException, OSError) as exc:
        logger.error("MaveDB bulk discovery failed: %s", exc)
        raise typer.Exit(1) from None


def _snapshot_selection(latest: bool, record: str | None) -> str:
    """Return exactly one validated snapshot selector."""
    if latest == (record is not None):
        _usage_error("exactly one of --latest or --record is required")
    return "latest" if latest else record or ""


def _catalog_cache(
    source: str,
    *,
    variant_type: str | None,
    cache_dir: Path | None,
    refresh: bool,
) -> FilesystemCache | None:
    """Validate source-specific options and lazily create a ProteinGym cache."""
    if source == "mavedb":
        unsupported = [
            option
            for option, supplied in (
                ("--variant-type", variant_type is not None),
                ("--cache-dir", cache_dir is not None),
                ("--refresh", refresh),
            )
            if supplied
        ]
        if unsupported:
            _usage_error(
                "source mavedb does not support " + ", ".join(unsupported)
            )
        return None

    return FilesystemCache(_cache_root(cache_dir))


def _default_cache_root() -> Path:
    """Return the shared default cache root without creating it."""
    return Path.home() / ".cache" / "dmsroute"


def _cache_root(cache_dir: Path | None) -> Path:
    """Resolve an optional CLI cache directory without creating it."""
    return cache_dir.expanduser() if cache_dir is not None else _default_cache_root()


def main(argv: list[str] | None = None) -> int:
    """Invoke the Typer application with the historical compatibility contract."""
    global _COMPATIBILITY_OUTCOME
    _COMPATIBILITY_OUTCOME = None
    try:
        app(args=argv, prog_name="dmsroute", standalone_mode=True)
    except SystemExit as exc:
        outcome = _COMPATIBILITY_OUTCOME
        _COMPATIBILITY_OUTCOME = None
        if outcome == "command":
            return int(exc.code or 0)
        raise
    _COMPATIBILITY_OUTCOME = None
    return 0


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


def _cache_inventory_artifact_values(
    artifact: CacheInventoryArtifact,
) -> dict[str, Any]:
    """Serialize one present cache payload."""
    return {
        "role": artifact.role,
        "path": str(artifact.path),
        "size_bytes": artifact.size_bytes,
    }


def _cache_inventory_entry_values(
    entry: CacheInventoryEntry,
) -> dict[str, Any]:
    """Serialize one cache inventory entry with aggregate payload fields."""
    return {
        "representation": entry.representation,
        "source": entry.source,
        "dataset_id": entry.dataset_id,
        "snapshot_record_id": entry.snapshot_record_id,
        "state": entry.state,
        "artifact_count": len(entry.artifacts),
        "size_bytes": sum(
            artifact.size_bytes for artifact in entry.artifacts
        ),
        "artifacts": [
            _cache_inventory_artifact_values(artifact)
            for artifact in entry.artifacts
        ],
        "error": entry.error,
    }


def _cache_inventory_values(
    cache_root: Path,
    result: CacheInventoryResult,
) -> dict[str, Any]:
    """Return deterministic machine-readable cache inventory values."""
    return {
        "cache_root": str(cache_root),
        "entries": [
            _cache_inventory_entry_values(entry)
            for entry in result.entries
        ],
        "issues": [
            {
                "path": str(issue.path),
                "state": issue.state,
                "error": issue.error,
            }
            for issue in result.issues
        ],
        "summary": _cache_inventory_summary_values(result),
    }


def _cache_inventory_summary_values(
    result: CacheInventoryResult,
) -> dict[str, int]:
    """Return stable cache inventory aggregate fields."""
    return {
        "entry_count": result.entry_count,
        "artifact_count": result.artifact_count,
        "total_size_bytes": result.total_size_bytes,
        "complete_entry_count": result.complete_entry_count,
        "invalid_entry_count": result.invalid_entry_count,
        "issue_count": result.issue_count,
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


def _render_cache_inventory_text(
    cache_root: Path,
    result: CacheInventoryResult,
) -> str:
    """Render cache entries, incomplete objects, and aggregate counts."""
    lines = [f"cache_root: {cache_root}", "entries:"]
    if not result.entries:
        lines.append("  (none)")
    for entry in result.entries:
        entry_size = sum(
            artifact.size_bytes for artifact in entry.artifacts
        )
        lines.extend(
            [
                f"  - representation: {entry.representation}",
                f"    source: {entry.source}",
                f"    dataset_id: {_display_value(entry.dataset_id)}",
                "    snapshot_record_id: "
                f"{_display_value(entry.snapshot_record_id)}",
                f"    state: {entry.state}",
                f"    artifact_count: {len(entry.artifacts)}",
                f"    size_bytes: {entry_size}",
                "    artifacts:",
            ]
        )
        if not entry.artifacts:
            lines.append("      (none)")
        for artifact in entry.artifacts:
            lines.extend(
                [
                    f"      - role: {artifact.role}",
                    f"        path: {artifact.path}",
                    f"        size_bytes: {artifact.size_bytes}",
                ]
            )
        lines.append(f"    error: {_display_value(entry.error)}")

    lines.append("issues:")
    if not result.issues:
        lines.append("  (none)")
    for issue in result.issues:
        lines.extend(
            [
                f"  - path: {issue.path}",
                f"    state: {issue.state}",
                f"    error: {issue.error}",
            ]
        )

    lines.append("summary:")
    lines.extend(
        f"  {name}: {value}"
        for name, value in _cache_inventory_summary_values(result).items()
    )
    return "\n".join(lines) + "\n"


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
