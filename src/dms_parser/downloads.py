"""Standardized single- and multi-dataset download orchestration."""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests

from dms_parser._dataset import (
    _completed_dataset_counts,
    _extract_wt_from_metadata,
    _metadata_text,
    _mavedb_gene,
    _mavedb_target_protein,
    _mavedb_uniprot_id,
)
from dms_parser.builders import build_mavedb_dataset, build_proteingym_dataset
from dms_parser.cache import FilesystemCache
from dms_parser.catalog import DatasetRecord, get_dataset_metadata, list_datasets
from dms_parser.config import validate_source_dataset_id
from dms_parser.exceptions import (
    DMSParserError,
    DatasetNotFoundError,
    InvalidDatasetError,
    InvalidPipelineOptionError,
    MissingWildTypeError,
    SourceConfigurationError,
)
from dms_parser.io import (
    _preflight_download_batch,
    _preflight_dataset_bundle,
    _publish_download_summary,
    _publish_dataset_bundle,
    read_table,
    write_table,
)
from dms_parser.sources.mavedb import download_mavedb_dataset
from dms_parser.sources.mavedb_catalog import MAVEDB_API_URL
from dms_parser.sources._mavedb_snapshot_dataset import (
    MaveDBSnapshotDataset,
    acquire_cached_mavedb_snapshot_datasets,
)
from dms_parser.sources.proteingym import download_proteingym_dataset
from dms_parser.sources.proteingym_resources import get_proteingym_resource

logger = logging.getLogger(__name__)

_PROTEINGYM_SUBSTITUTIONS_BENCHMARK_CACHE_ID = (
    "resource-data-dms-substitutions"
)
_PORTABLE_DATASET_COMPONENT_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")
_EXPECTED_DOWNLOAD_ERRORS = (
    DMSParserError,
    requests.RequestException,
    OSError,
)


@dataclass(frozen=True)
class DatasetDownloadResult:
    """Stable output paths and summary for one standardized download."""

    dataset_path: Path
    summary_csv_path: Path
    summary_json_path: Path
    summary: dict[str, Any]


@dataclass(frozen=True)
class DatasetBatchDownloadEntry:
    """One ordered success or expected failure from a batch download."""

    source: str
    dataset_id: str
    output_dir: Path
    result: DatasetDownloadResult | None
    error_type: str | None
    error: str | None


@dataclass(frozen=True)
class DatasetBatchDownloadResult:
    """Aggregate result returned by :func:`download_and_standardize_datasets`."""

    entries: tuple[DatasetBatchDownloadEntry, ...]
    summary_csv_path: Path
    summary_json_path: Path

    @property
    def successful_entries(self) -> tuple[DatasetBatchDownloadEntry, ...]:
        """Return entries whose standardized bundles were published."""
        return tuple(entry for entry in self.entries if entry.result is not None)

    @property
    def failed_entries(self) -> tuple[DatasetBatchDownloadEntry, ...]:
        """Return entries that ended with an expected operational failure."""
        return tuple(entry for entry in self.entries if entry.result is None)

    @property
    def success_count(self) -> int:
        """Return the number of successful entries."""
        return len(self.successful_entries)

    @property
    def failure_count(self) -> int:
        """Return the number of failed entries."""
        return len(self.failed_entries)

    @property
    def has_errors(self) -> bool:
        """Return whether any requested dataset failed."""
        return self.failure_count > 0

    @property
    def exit_code(self) -> int:
        """Return the process exit code suitable for the CLI adapter."""
        return 1 if self.has_errors else 0


@dataclass(frozen=True)
class _DatasetBatchPlan:
    """Validated dataset IDs and deterministic output paths for one batch."""

    source: str
    dataset_ids: tuple[str, ...]
    output_root: Path
    output_dirs: tuple[Path, ...]


@dataclass(frozen=True)
class _ProteinGymDownloadMetadata:
    """Resolved ProteinGym fields needed by the existing builder."""

    wt_sequence: str
    protein_id: str | None
    gene: str | None
    uniprot_id: str | None


@dataclass(frozen=True)
class _MaveDBDownloadInput:
    """Source-specific inputs for the common MaveDB build path."""

    scores_path: Path
    metadata: dict[str, Any]
    provenance: Mapping[str, Any]


def download_and_standardize_dataset(
    source: str,
    dataset_id: str,
    *,
    output_dir: str | Path,
    cache: FilesystemCache,
    refresh: bool = False,
    drop_failed: bool = False,
    add_wildtype_row: bool = False,
    overwrite: bool = False,
    acquisition: str | None = None,
    snapshot_record_id: str | None = None,
    include_superseded: bool = False,
) -> DatasetDownloadResult:
    """Download and standardize one substitutions dataset without YAML config."""
    validate_source_dataset_id(source, dataset_id)
    acquisition_method = _validate_download_acquisition(
        source,
        acquisition=acquisition,
        snapshot_record_id=snapshot_record_id,
        include_superseded=include_superseded,
    )
    if not isinstance(output_dir, (str, Path)) or not str(output_dir).strip():
        raise InvalidPipelineOptionError("output_dir must be a non-empty path.")
    if not isinstance(cache, FilesystemCache):
        raise InvalidPipelineOptionError("cache must be a FilesystemCache.")
    for option_name, option_value in (
        ("refresh", refresh),
        ("drop_failed", drop_failed),
        ("add_wildtype_row", add_wildtype_row),
        ("overwrite", overwrite),
    ):
        if not isinstance(option_value, bool):
            raise InvalidPipelineOptionError(
                f"{option_name} must be a boolean."
            )

    resolved_output_dir = Path(output_dir).expanduser()
    dataset_path, summary_csv_path, summary_json_path = (
        _preflight_dataset_bundle(
            resolved_output_dir,
            overwrite=overwrite,
        )
    )

    if source == "proteingym":
        built_table, summary = _download_proteingym_dataset(
            dataset_id,
            cache=cache,
            refresh=refresh,
            drop_failed=drop_failed,
            add_wildtype_row=add_wildtype_row,
        )
    elif acquisition_method == "snapshot":
        assert snapshot_record_id is not None
        built_table, summary = _download_mavedb_snapshot_dataset(
            dataset_id,
            snapshot_record_id=snapshot_record_id,
            cache=cache,
            refresh=refresh,
            drop_failed=drop_failed,
            add_wildtype_row=add_wildtype_row,
            include_superseded=include_superseded,
        )
    else:
        built_table, summary = _download_mavedb_dataset(
            dataset_id,
            cache=cache,
            refresh=refresh,
            drop_failed=drop_failed,
            add_wildtype_row=add_wildtype_row,
        )

    return _publish_download_result(
        built_table,
        summary,
        output_dir=resolved_output_dir,
        output_paths=(dataset_path, summary_csv_path, summary_json_path),
        overwrite=overwrite,
    )


def download_and_standardize_datasets(
    source: str,
    dataset_ids: Sequence[str],
    *,
    output_dir: str | Path,
    cache: FilesystemCache,
    refresh: bool = False,
    drop_failed: bool = False,
    add_wildtype_row: bool = False,
    overwrite: bool = False,
    acquisition: str | None = None,
    snapshot_record_id: str | None = None,
    include_superseded: bool = False,
) -> DatasetBatchDownloadResult:
    """Download and standardize an ordered batch from one substitutions source."""
    acquisition_method = _validate_download_acquisition(
        source,
        acquisition=acquisition,
        snapshot_record_id=snapshot_record_id,
        include_superseded=include_superseded,
    )
    plan = _validate_dataset_batch_request(
        source,
        dataset_ids,
        output_dir=output_dir,
        refresh=refresh,
        drop_failed=drop_failed,
        add_wildtype_row=add_wildtype_row,
        overwrite=overwrite,
    )
    if not isinstance(cache, FilesystemCache):
        raise InvalidPipelineOptionError("cache must be a FilesystemCache.")

    batch_provenance: dict[str, Mapping[str, Any]] | None = None
    if source == "proteingym":
        entries = _download_proteingym_batch(
            plan,
            cache=cache,
            refresh=refresh,
            drop_failed=drop_failed,
            add_wildtype_row=add_wildtype_row,
            overwrite=overwrite,
        )
    elif acquisition_method == "snapshot":
        assert snapshot_record_id is not None
        entries, batch_provenance = _download_mavedb_snapshot_batch(
            plan,
            snapshot_record_id=snapshot_record_id,
            cache=cache,
            refresh=refresh,
            drop_failed=drop_failed,
            add_wildtype_row=add_wildtype_row,
            overwrite=overwrite,
            include_superseded=include_superseded,
        )
    else:
        entries = _download_mavedb_batch(
            plan,
            cache=cache,
            refresh=refresh,
            drop_failed=drop_failed,
            add_wildtype_row=add_wildtype_row,
            overwrite=overwrite,
        )

    records = [
        _batch_summary_record(
            entry,
            batch_root=plan.output_root,
            provenance=(
                batch_provenance.get(entry.dataset_id)
                if batch_provenance is not None
                else None
            ),
        )
        for entry in entries
    ]
    summary_csv_path, summary_json_path = _publish_download_summary(
        records,
        plan.output_root,
        overwrite=overwrite,
    )
    logger.info("Batch CSV summary saved at: %s", summary_csv_path)
    logger.info("Batch JSON summary saved at: %s", summary_json_path)
    return DatasetBatchDownloadResult(
        entries=entries,
        summary_csv_path=summary_csv_path,
        summary_json_path=summary_json_path,
    )


def _validate_download_acquisition(
    source: str,
    *,
    acquisition: str | None,
    snapshot_record_id: str | None,
    include_superseded: bool,
) -> str:
    """Validate acquisition compatibility before any download side effect."""
    if not isinstance(include_superseded, bool):
        raise InvalidPipelineOptionError(
            "include_superseded must be a boolean."
        )
    if acquisition is not None and not isinstance(acquisition, str):
        raise InvalidPipelineOptionError(
            "acquisition must be None, 'api', or 'snapshot'."
        )
    if acquisition not in {None, "api", "snapshot"}:
        raise InvalidPipelineOptionError(
            "acquisition must be None, 'api', or 'snapshot'."
        )
    if snapshot_record_id is not None and not isinstance(
        snapshot_record_id,
        str,
    ):
        raise InvalidPipelineOptionError(
            "snapshot_record_id must be a concrete positive record ID or None."
        )

    if source != "mavedb":
        if (
            acquisition is not None
            or snapshot_record_id is not None
            or include_superseded
        ):
            raise InvalidPipelineOptionError(
                "Acquisition options are supported only for MaveDB."
            )
        return "native"

    method = acquisition or "api"
    if method == "snapshot":
        if (
            snapshot_record_id is None
            or re.fullmatch(r"[1-9][0-9]*", snapshot_record_id) is None
        ):
            raise InvalidPipelineOptionError(
                "Snapshot acquisition requires a concrete positive "
                "snapshot_record_id; 'latest' is not allowed."
            )
        return method

    if snapshot_record_id is not None:
        raise InvalidPipelineOptionError(
            "snapshot_record_id requires acquisition='snapshot'."
        )
    if include_superseded:
        raise InvalidPipelineOptionError(
            "include_superseded requires acquisition='snapshot'."
        )
    return method


def _validate_dataset_batch_request(
    source: str,
    dataset_ids: Sequence[str],
    *,
    output_dir: str | Path,
    refresh: bool,
    drop_failed: bool,
    add_wildtype_row: bool,
    overwrite: bool,
) -> _DatasetBatchPlan:
    """Validate a complete batch and preflight every deterministic target."""
    if isinstance(dataset_ids, (str, bytes)) or not isinstance(
        dataset_ids,
        Sequence,
    ):
        raise InvalidPipelineOptionError(
            "dataset_ids must be a non-string sequence."
        )
    if not dataset_ids:
        raise InvalidPipelineOptionError(
            "dataset_ids must contain at least one dataset identifier."
        )
    if not isinstance(output_dir, (str, Path)) or not str(output_dir).strip():
        raise InvalidPipelineOptionError("output_dir must be a non-empty path.")
    for option_name, option_value in (
        ("refresh", refresh),
        ("drop_failed", drop_failed),
        ("add_wildtype_row", add_wildtype_row),
        ("overwrite", overwrite),
    ):
        if not isinstance(option_value, bool):
            raise InvalidPipelineOptionError(
                f"{option_name} must be a boolean."
            )

    validated_ids: list[str] = []
    seen: set[str] = set()
    components: list[str] = []
    component_ids: dict[str, str] = {}
    for dataset_id in dataset_ids:
        if isinstance(dataset_id, str):
            if dataset_id != dataset_id.strip():
                raise SourceConfigurationError(
                    "Batch dataset identifiers must not have leading or "
                    "trailing whitespace."
                )
            if any(
                unicodedata.category(character) == "Cc"
                for character in dataset_id
            ):
                raise SourceConfigurationError(
                    "Batch dataset identifiers must not contain control "
                    "characters."
                )
        validate_source_dataset_id(source, dataset_id)
        if dataset_id in seen:
            raise SourceConfigurationError(
                f"Duplicate {source} dataset_id {dataset_id!r}."
            )
        component = _portable_dataset_component(dataset_id)
        folded_component = component.casefold()
        previous_id = component_ids.get(folded_component)
        if previous_id is not None:
            raise SourceConfigurationError(
                "Dataset identifiers produce the same portable output "
                f"directory: {previous_id!r} and {dataset_id!r}."
            )
        seen.add(dataset_id)
        component_ids[folded_component] = dataset_id
        validated_ids.append(dataset_id)
        components.append(component)

    output_root = Path(output_dir).expanduser()
    output_dirs = tuple(
        output_root / source / component
        for component in components
    )
    _preflight_download_batch(
        output_root,
        output_dirs,
        overwrite=overwrite,
    )
    return _DatasetBatchPlan(
        source=source,
        dataset_ids=tuple(validated_ids),
        output_root=output_root,
        output_dirs=output_dirs,
    )


def _portable_dataset_component(dataset_id: str) -> str:
    """Return a bounded portable directory component for one dataset ID."""
    digest = hashlib.sha256(dataset_id.encode("utf-8")).hexdigest()[:16]
    slug = _PORTABLE_DATASET_COMPONENT_PATTERN.sub("-", dataset_id)
    slug = slug.strip("._-")[:48].strip("._-") or "dataset"
    return f"id-{slug}--{digest}"


def _download_mavedb_batch(
    plan: _DatasetBatchPlan,
    *,
    cache: FilesystemCache,
    refresh: bool,
    drop_failed: bool,
    add_wildtype_row: bool,
    overwrite: bool,
) -> tuple[DatasetBatchDownloadEntry, ...]:
    """Process independent MaveDB URNs through the single-download API."""
    entries: list[DatasetBatchDownloadEntry] = []
    for dataset_id, dataset_output_dir in zip(
        plan.dataset_ids,
        plan.output_dirs,
        strict=True,
    ):
        try:
            result = download_and_standardize_dataset(
                "mavedb",
                dataset_id,
                output_dir=dataset_output_dir,
                cache=cache,
                refresh=refresh,
                drop_failed=drop_failed,
                add_wildtype_row=add_wildtype_row,
                overwrite=overwrite,
            )
        except _EXPECTED_DOWNLOAD_ERRORS as exc:
            entries.append(
                _failed_batch_entry(
                    "mavedb",
                    dataset_id,
                    dataset_output_dir,
                    exc,
                )
            )
            continue
        entries.append(
            _successful_batch_entry(
                "mavedb",
                dataset_id,
                dataset_output_dir,
                result,
            )
        )
    return tuple(entries)


def _download_mavedb_snapshot_batch(
    plan: _DatasetBatchPlan,
    *,
    snapshot_record_id: str,
    cache: FilesystemCache,
    refresh: bool,
    drop_failed: bool,
    add_wildtype_row: bool,
    overwrite: bool,
    include_superseded: bool,
) -> tuple[
    tuple[DatasetBatchDownloadEntry, ...],
    dict[str, Mapping[str, Any]],
]:
    """Acquire snapshot tables once and independently build each dataset."""
    try:
        acquisition = acquire_cached_mavedb_snapshot_datasets(
            snapshot_record_id,
            plan.dataset_ids,
            cache=cache,
            include_superseded=include_superseded,
            refresh=refresh,
        )
    except _EXPECTED_DOWNLOAD_ERRORS as exc:
        return (
            tuple(
                _failed_batch_entry(
                    "mavedb",
                    dataset_id,
                    output_dir,
                    exc,
                )
                for dataset_id, output_dir in zip(
                    plan.dataset_ids,
                    plan.output_dirs,
                    strict=True,
                )
            ),
            {
                dataset_id: _empty_snapshot_provenance(snapshot_record_id)
                for dataset_id in plan.dataset_ids
            },
        )

    datasets = {
        dataset.dataset_id: dataset
        for dataset in acquisition.datasets
    }
    errors = dict(acquisition.errors)
    provenance = dict(acquisition.provenance)
    entries: list[DatasetBatchDownloadEntry] = []
    for dataset_id, output_dir in zip(
        plan.dataset_ids,
        plan.output_dirs,
        strict=True,
    ):
        error = errors.get(dataset_id)
        if error is not None:
            entries.append(
                _failed_batch_entry("mavedb", dataset_id, output_dir, error)
            )
            continue
        try:
            built_table, summary = _build_mavedb_download(
                dataset_id,
                _snapshot_download_input(datasets[dataset_id]),
                drop_failed=drop_failed,
                add_wildtype_row=add_wildtype_row,
            )
            result = _publish_download_result(
                built_table,
                summary,
                output_dir=output_dir,
                output_paths=_preflight_dataset_bundle(
                    output_dir,
                    overwrite=overwrite,
                ),
                overwrite=overwrite,
            )
        except _EXPECTED_DOWNLOAD_ERRORS as exc:
            entries.append(
                _failed_batch_entry("mavedb", dataset_id, output_dir, exc)
            )
            continue
        entries.append(
            _successful_batch_entry(
                "mavedb",
                dataset_id,
                output_dir,
                result,
            )
        )
    return tuple(entries), provenance


def _download_proteingym_batch(
    plan: _DatasetBatchPlan,
    *,
    cache: FilesystemCache,
    refresh: bool,
    drop_failed: bool,
    add_wildtype_row: bool,
    overwrite: bool,
) -> tuple[DatasetBatchDownloadEntry, ...]:
    """Acquire shared ProteinGym artifacts once and process requested assays."""
    try:
        records = list_datasets(
            "proteingym",
            variant_type="substitutions",
            cache=cache,
            refresh=refresh,
        )
    except _EXPECTED_DOWNLOAD_ERRORS as exc:
        return tuple(
            _failed_batch_entry(
                "proteingym",
                dataset_id,
                dataset_output_dir,
                exc,
            )
            for dataset_id, dataset_output_dir in zip(
                plan.dataset_ids,
                plan.output_dirs,
                strict=True,
            )
        )

    records_by_id: dict[str, list[DatasetRecord]] = {}
    for record in records:
        records_by_id.setdefault(record.dataset_id, []).append(record)

    outcomes: dict[str, DatasetBatchDownloadEntry] = {}
    resolved_metadata: dict[str, _ProteinGymDownloadMetadata] = {}
    for dataset_id, dataset_output_dir in zip(
        plan.dataset_ids,
        plan.output_dirs,
        strict=True,
    ):
        matches = records_by_id.get(dataset_id, [])
        if not matches:
            error = DatasetNotFoundError(
                f"ProteinGym dataset {dataset_id!r} was not found."
            )
            outcomes[dataset_id] = _failed_batch_entry(
                "proteingym",
                dataset_id,
                dataset_output_dir,
                error,
            )
            continue
        if len(matches) > 1:
            error = InvalidDatasetError(
                f"ProteinGym dataset {dataset_id!r} is ambiguous."
            )
            outcomes[dataset_id] = _failed_batch_entry(
                "proteingym",
                dataset_id,
                dataset_output_dir,
                error,
            )
            continue
        try:
            resolved_metadata[dataset_id] = (
                _resolve_proteingym_download_metadata(matches[0], dataset_id)
            )
        except _EXPECTED_DOWNLOAD_ERRORS as exc:
            outcomes[dataset_id] = _failed_batch_entry(
                "proteingym",
                dataset_id,
                dataset_output_dir,
                exc,
            )

    if resolved_metadata:
        try:
            benchmark_table = _acquire_proteingym_benchmark(
                cache=cache,
                refresh=refresh,
            )
        except _EXPECTED_DOWNLOAD_ERRORS as exc:
            for dataset_id, dataset_output_dir in zip(
                plan.dataset_ids,
                plan.output_dirs,
                strict=True,
            ):
                if dataset_id in resolved_metadata:
                    outcomes[dataset_id] = _failed_batch_entry(
                        "proteingym",
                        dataset_id,
                        dataset_output_dir,
                        exc,
                    )
        else:
            grouped_indices = benchmark_table.groupby(
                "DMS_id",
                sort=False,
            ).groups
            for dataset_id, dataset_output_dir in zip(
                plan.dataset_ids,
                plan.output_dirs,
                strict=True,
            ):
                if dataset_id not in resolved_metadata:
                    continue
                indices = grouped_indices.get(dataset_id)
                if indices is None or len(indices) == 0:
                    error = DatasetNotFoundError(
                        f"ProteinGym dataset {dataset_id!r} was not found "
                        "in the benchmark."
                    )
                    outcomes[dataset_id] = _failed_batch_entry(
                        "proteingym",
                        dataset_id,
                        dataset_output_dir,
                        error,
                    )
                    continue
                experiment_table = benchmark_table.loc[indices].copy()
                try:
                    built_table, summary = _build_proteingym_download(
                        dataset_id,
                        resolved_metadata[dataset_id],
                        experiment_table,
                        drop_failed=drop_failed,
                        add_wildtype_row=add_wildtype_row,
                    )
                    result = _publish_download_result(
                        built_table,
                        summary,
                        output_dir=dataset_output_dir,
                        output_paths=_preflight_dataset_bundle(
                            dataset_output_dir,
                            overwrite=overwrite,
                        ),
                        overwrite=overwrite,
                    )
                except _EXPECTED_DOWNLOAD_ERRORS as exc:
                    outcomes[dataset_id] = _failed_batch_entry(
                        "proteingym",
                        dataset_id,
                        dataset_output_dir,
                        exc,
                    )
                    continue
                outcomes[dataset_id] = _successful_batch_entry(
                    "proteingym",
                    dataset_id,
                    dataset_output_dir,
                    result,
                )

    return tuple(outcomes[dataset_id] for dataset_id in plan.dataset_ids)


def _download_proteingym_dataset(
    dataset_id: str,
    *,
    cache: FilesystemCache,
    refresh: bool,
    drop_failed: bool,
    add_wildtype_row: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Acquire and build one ProteinGym substitutions assay."""
    metadata_record = get_dataset_metadata(
        "proteingym",
        dataset_id,
        variant_type="substitutions",
        cache=cache,
        refresh=refresh,
    )
    metadata = _resolve_proteingym_download_metadata(
        metadata_record,
        dataset_id,
    )
    benchmark_table = _acquire_proteingym_benchmark(
        cache=cache,
        refresh=refresh,
    )
    experiment_table = benchmark_table[
        benchmark_table["DMS_id"] == dataset_id
    ].copy()
    if experiment_table.empty:
        raise DatasetNotFoundError(
            f"ProteinGym dataset {dataset_id!r} was not found in the benchmark."
        )
    return _build_proteingym_download(
        dataset_id,
        metadata,
        experiment_table,
        drop_failed=drop_failed,
        add_wildtype_row=add_wildtype_row,
    )


def _resolve_proteingym_download_metadata(
    metadata_record: DatasetRecord,
    dataset_id: str,
) -> _ProteinGymDownloadMetadata:
    """Resolve and validate ProteinGym builder metadata without acquisition."""
    metadata = pd.Series(metadata_record.raw_metadata)
    wt_sequence = _metadata_text(metadata, "target_seq")
    if wt_sequence is None:
        raise MissingWildTypeError(
            f"No WT sequence found in ProteinGym metadata for {dataset_id}."
        )
    return _ProteinGymDownloadMetadata(
        wt_sequence=wt_sequence,
        protein_id=_metadata_text(metadata, "molecule_name"),
        gene=_metadata_text(metadata, "gene", "Gene", "gene_name"),
        uniprot_id=_metadata_text(metadata, "UniProt_ID"),
    )


def _acquire_proteingym_benchmark(
    *,
    cache: FilesystemCache,
    refresh: bool,
) -> pd.DataFrame:
    """Acquire and validate the shared ProteinGym substitutions benchmark."""
    resource = get_proteingym_resource(
        "dms_substitutions",
        require_processing=True,
    )
    benchmark_path = download_proteingym_dataset(
        resource.data_url,
        cache=cache,
        dataset_id=_PROTEINGYM_SUBSTITUTIONS_BENCHMARK_CACHE_ID,
        refresh=refresh,
    )
    benchmark_table = read_table(benchmark_path)
    if "DMS_id" not in benchmark_table.columns:
        raise InvalidDatasetError(
            "ProteinGym substitutions benchmark is missing the 'DMS_id' column."
        )
    return benchmark_table


def _build_proteingym_download(
    dataset_id: str,
    metadata: _ProteinGymDownloadMetadata,
    experiment_table: pd.DataFrame,
    *,
    drop_failed: bool,
    add_wildtype_row: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build one selected ProteinGym assay through the existing builder."""
    with TemporaryDirectory(prefix="dms-parser-download-") as temporary_dir:
        selected_path = Path(temporary_dir) / "selected.csv"
        write_table(experiment_table, selected_path, index=False)
        built_table = build_proteingym_dataset(
            input_path=selected_path,
            score_col="DMS_score",
            variant_col="mutant",
            dataset_id=dataset_id,
            protein_id=metadata.protein_id,
            gene=metadata.gene,
            uniprot_id=metadata.uniprot_id,
            wt_sequence=metadata.wt_sequence,
            add_relative_score=False,
            add_binary_label=False,
            add_wildtype_row=add_wildtype_row,
            drop_failed=drop_failed,
            validate_output=True,
            require_wt_for_transforms=False,
        )

    return built_table, _successful_download_summary(
        source="proteingym",
        dataset_id=dataset_id,
        target_protein=(
            metadata.uniprot_id or metadata.protein_id or "Unknown"
        ),
        wt_sequence=metadata.wt_sequence,
        raw_rows=len(experiment_table),
        built_table=built_table,
    )


def _download_mavedb_dataset(
    dataset_id: str,
    *,
    cache: FilesystemCache,
    refresh: bool,
    drop_failed: bool,
    add_wildtype_row: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Acquire one MaveDB score set from the API and build it."""
    acquired = _acquire_mavedb_api_dataset(
        dataset_id,
        cache=cache,
        refresh=refresh,
    )
    return _build_mavedb_download(
        dataset_id,
        acquired,
        drop_failed=drop_failed,
        add_wildtype_row=add_wildtype_row,
    )


def _download_mavedb_snapshot_dataset(
    dataset_id: str,
    *,
    snapshot_record_id: str,
    cache: FilesystemCache,
    refresh: bool,
    drop_failed: bool,
    add_wildtype_row: bool,
    include_superseded: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Acquire one MaveDB score set from a cached snapshot and build it."""
    acquisition = acquire_cached_mavedb_snapshot_datasets(
        snapshot_record_id,
        (dataset_id,),
        cache=cache,
        include_superseded=include_superseded,
        refresh=refresh,
    )
    if acquisition.errors:
        raise acquisition.errors[0][1]
    return _build_mavedb_download(
        dataset_id,
        _snapshot_download_input(acquisition.datasets[0]),
        drop_failed=drop_failed,
        add_wildtype_row=add_wildtype_row,
    )


def _acquire_mavedb_api_dataset(
    dataset_id: str,
    *,
    cache: FilesystemCache,
    refresh: bool,
) -> _MaveDBDownloadInput:
    """Return API metadata and score-table inputs without building them."""
    metadata_record = _get_mavedb_download_metadata(dataset_id)
    metadata = metadata_record.raw_metadata
    encoded_id = quote(dataset_id, safe=":")
    scores_url = (
        f"{MAVEDB_API_URL.rstrip('/')}/score-sets/{encoded_id}/scores"
    )
    scores_path = download_mavedb_dataset(
        scores_url,
        cache=cache,
        dataset_id=dataset_id,
        refresh=refresh,
    )
    return _MaveDBDownloadInput(
        scores_path=scores_path,
        metadata=metadata,
        provenance={},
    )


def _snapshot_download_input(
    dataset: MaveDBSnapshotDataset,
) -> _MaveDBDownloadInput:
    """Adapt private snapshot acquisition to the common builder input."""
    return _MaveDBDownloadInput(
        scores_path=dataset.scores_path,
        metadata=dataset.metadata,
        provenance=dataset.provenance,
    )


def _build_mavedb_download(
    dataset_id: str,
    acquired: _MaveDBDownloadInput,
    *,
    drop_failed: bool,
    add_wildtype_row: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build API or snapshot inputs through the existing MaveDB builder."""
    metadata = acquired.metadata
    wt_sequence = _extract_wt_from_metadata(metadata)
    if wt_sequence is None:
        raise MissingWildTypeError(
            f"No WT sequence found in MaveDB metadata for {dataset_id}."
        )

    raw_table = read_table(acquired.scores_path)
    hgvs_col = next(
        (column for column in ("hgvs_pro",) if column in raw_table.columns),
        None,
    )
    score_col = next(
        (
            column
            for column in ("score", "DMS_score", "fitness")
            if column in raw_table.columns
        ),
        None,
    )
    if hgvs_col is None or score_col is None:
        raise InvalidDatasetError(
            "Neither score nor HGVS columns were detected in the MaveDB table."
        )

    gene = _mavedb_gene(metadata)
    uniprot_id = _mavedb_uniprot_id(metadata)
    built_table = build_mavedb_dataset(
        input_path=acquired.scores_path,
        score_col=score_col,
        hgvs_col=hgvs_col,
        dataset_id=dataset_id,
        protein_id=None,
        gene=gene,
        uniprot_id=uniprot_id,
        wt_sequence=wt_sequence,
        add_relative_score=False,
        add_binary_label=False,
        add_wildtype_row=add_wildtype_row,
        drop_failed=drop_failed,
        validate_output=True,
        require_wt_for_transforms=False,
    )
    summary = _successful_download_summary(
        source="mavedb",
        dataset_id=dataset_id,
        target_protein=_mavedb_target_protein(metadata, gene),
        wt_sequence=wt_sequence,
        raw_rows=len(raw_table),
        built_table=built_table,
    )
    summary.update(acquired.provenance)
    return built_table, summary


def _get_mavedb_download_metadata(dataset_id: str) -> DatasetRecord:
    """Retrieve MaveDB metadata and normalize an HTTP 404 as not found."""
    try:
        return get_dataset_metadata("mavedb", dataset_id)
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            raise DatasetNotFoundError(
                f"MaveDB dataset {dataset_id!r} was not found."
            ) from exc
        raise


def _successful_download_summary(
    *,
    source: str,
    dataset_id: str,
    target_protein: str,
    wt_sequence: str,
    raw_rows: int,
    built_table: pd.DataFrame,
) -> dict[str, Any]:
    """Build the existing successful summary representation for one dataset."""
    return {
        "source": source,
        "input": dataset_id,
        "status": "OK",
        "dataset_id": dataset_id,
        "target_protein": target_protein,
        "wt_length": len(wt_sequence),
        "raw_rows": raw_rows,
        **_completed_dataset_counts(built_table, raw_rows=raw_rows),
    }


def _publish_download_result(
    built_table: pd.DataFrame,
    summary: dict[str, Any],
    *,
    output_dir: Path,
    output_paths: tuple[Path, Path, Path],
    overwrite: bool,
) -> DatasetDownloadResult:
    """Publish one unchanged three-file download bundle and its public result."""
    dataset_path, summary_csv_path, summary_json_path = output_paths
    summary["output_file"] = str(dataset_path)
    _publish_dataset_bundle(
        built_table,
        summary,
        output_dir,
        overwrite=overwrite,
    )
    logger.info("Standardized dataset saved at: %s", dataset_path)
    logger.info("CSV summary saved at: %s", summary_csv_path)
    logger.info("JSON summary saved at: %s", summary_json_path)
    return DatasetDownloadResult(
        dataset_path=dataset_path,
        summary_csv_path=summary_csv_path,
        summary_json_path=summary_json_path,
        summary=summary,
    )


def _successful_batch_entry(
    source: str,
    dataset_id: str,
    output_dir: Path,
    result: DatasetDownloadResult,
) -> DatasetBatchDownloadEntry:
    """Return one successful ordered batch entry."""
    return DatasetBatchDownloadEntry(
        source=source,
        dataset_id=dataset_id,
        output_dir=output_dir,
        result=result,
        error_type=None,
        error=None,
    )


def _failed_batch_entry(
    source: str,
    dataset_id: str,
    output_dir: Path,
    error: Exception,
) -> DatasetBatchDownloadEntry:
    """Return and log one expected batch failure."""
    logger.error(
        "Dataset batch entry failed source=%s dataset_id=%s error_type=%s "
        "error=%s",
        source,
        dataset_id,
        type(error).__name__,
        error,
    )
    return DatasetBatchDownloadEntry(
        source=source,
        dataset_id=dataset_id,
        output_dir=output_dir,
        result=None,
        error_type=type(error).__name__,
        error=str(error),
    )


def _empty_snapshot_provenance(record_id: str) -> dict[str, Any]:
    """Return stable null provenance when a cached snapshot cannot load."""
    return {
        "acquisition_method": "snapshot",
        "snapshot_record_id": record_id,
        "snapshot_doi": None,
        "snapshot_concept_doi": None,
        "snapshot_publication_date": None,
        "snapshot_archive_filename": None,
        "snapshot_archive_size": None,
        "snapshot_archive_checksum": None,
        "snapshot_catalog_as_of": None,
        "snapshot_score_set_superseded": None,
    }


def _batch_summary_record(
    entry: DatasetBatchDownloadEntry,
    *,
    batch_root: Path,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one stable aggregate record with batch-root-relative paths."""
    relative_output_dir = entry.output_dir.relative_to(batch_root).as_posix()
    if entry.result is None:
        result_paths: dict[str, str | None] = {
            "dataset_path": None,
            "summary_csv_path": None,
            "summary_json_path": None,
        }
        summary: dict[str, Any] = {}
    else:
        result_paths = {
            "dataset_path": entry.result.dataset_path.relative_to(
                batch_root
            ).as_posix(),
            "summary_csv_path": entry.result.summary_csv_path.relative_to(
                batch_root
            ).as_posix(),
            "summary_json_path": entry.result.summary_json_path.relative_to(
                batch_root
            ).as_posix(),
        }
        summary = entry.result.summary

    return {
        "source": entry.source,
        "dataset_id": entry.dataset_id,
        "status": "SUCCESS" if entry.result is not None else "ERROR",
        "output_dir": relative_output_dir,
        **result_paths,
        "target_protein": summary.get("target_protein"),
        "wt_length": summary.get("wt_length"),
        "raw_rows": summary.get("raw_rows"),
        "validated_rows": summary.get("validated_rows"),
        "discarded_rows": summary.get("discarded_rows"),
        "output_rows": summary.get("output_rows"),
        "wildtype_rows": summary.get("wildtype_rows"),
        "synthetic_wildtype_rows": summary.get(
            "synthetic_wildtype_rows"
        ),
        **(provenance or {}),
        "error_type": entry.error_type,
        "error": entry.error,
    }
