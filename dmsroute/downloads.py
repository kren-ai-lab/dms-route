"""Standardized single- and multi-dataset download orchestration."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests

from dmsroute.core._dataset import (
    _completed_dataset_counts,
    _mavedb_wt_sequence_evidence,
    _metadata_text,
    _mavedb_gene,
    _mavedb_target_protein,
    _mavedb_uniprot_id,
    _wt_summary_fields,
)
from dmsroute.core._wildtype import (
    replace_wt_sequence_provenance,
    resolve_wt_sequence,
    validate_standardization_options,
)
from dmsroute.builders import (
    build_mavedb_dataset,
    build_proteingym_dataset,
    build_proteingym_indel_dataset,
)
from dmsroute.acquisition.cache import FilesystemCache
from dmsroute.catalog import (
    DatasetRecord,
    get_dataset_metadata,
    list_datasets,
    normalize_raw_metadata,
)
from dmsroute.config import validate_source_dataset_id
from dmsroute.core.exceptions import (
    DMSParserError,
    DatasetNotFoundError,
    InvalidDatasetError,
    InvalidPipelineOptionError,
    SourceConfigurationError,
)
from dmsroute.acquisition.io import (
    _preflight_download_batch,
    _preflight_dataset_bundle,
    _publish_download_summary,
    _publish_dataset_bundle,
    read_table,
    write_table,
)
from dmsroute.sources.mavedb import download_mavedb_dataset
from dmsroute.sources.mavedb_catalog import MAVEDB_API_URL
from dmsroute.sources._mavedb_snapshot_dataset import (
    MaveDBSnapshotDataset,
    acquire_cached_mavedb_snapshot_datasets,
)
from dmsroute.sources.proteingym import download_proteingym_dataset
from dmsroute.sources.proteingym_resources import get_proteingym_resource

logger = logging.getLogger(__name__)

_PROTEINGYM_BENCHMARK_CACHE_IDS = {
    "substitutions": "resource-data-dms-substitutions",
    "indels": "resource-data-dms-indels",
}
_PROTEINGYM_SUBSTITUTIONS_BENCHMARK_CACHE_ID = (
    _PROTEINGYM_BENCHMARK_CACHE_IDS["substitutions"]
)
_MAVEDB_METADATA_CACHE_SOURCE = "mavedb-metadata"
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
    wt_sequence_provenance: str
    protein_id: str | None
    gene: str | None
    uniprot_id: str | None


@dataclass(frozen=True)
class _MaveDBDownloadInput:
    """Source-specific inputs for the common MaveDB build path."""

    scores_path: Path
    metadata: dict[str, Any]
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class _StandardizationOptions:
    """Validated WT and transformation options for one dataset."""

    wt_sequence: str | None = None
    wt_score: float | None = None
    add_relative_score: bool = False
    relative_method: str = "log_ratio"
    relative_output_col: str = "score_log_ratio"
    add_binary_label: bool = False
    delta: float = 0.1
    higher_is_better: bool = True
    binary_output_col: str = "score_binary_like"


def download_and_standardize_dataset(
    source: str,
    dataset_id: str,
    *,
    output_dir: str | Path,
    cache: FilesystemCache,
    refresh: bool = False,
    drop_failed: bool = False,
    add_wildtype_row: bool = False,
    wt_sequence: str | None = None,
    wt_score: float | None = None,
    add_relative_score: bool = False,
    relative_method: str = "log_ratio",
    relative_output_col: str = "score_log_ratio",
    add_binary_label: bool = False,
    delta: float = 0.1,
    higher_is_better: bool = True,
    binary_output_col: str = "score_binary_like",
    overwrite: bool = False,
    acquisition: str | None = None,
    snapshot_record_id: str | None = None,
    include_superseded: bool = False,
    variant_type: str = "substitutions",
) -> DatasetDownloadResult:
    """Download and standardize one dataset without YAML configuration."""
    validate_source_dataset_id(source, dataset_id)
    selected_variant_type = _validate_download_variant_type(
        source,
        variant_type,
    )
    acquisition_method = _validate_download_acquisition(
        source,
        acquisition=acquisition,
        snapshot_record_id=snapshot_record_id,
        include_superseded=include_superseded,
    )
    standardization = _standardization_options(
        wt_sequence=wt_sequence,
        wt_score=wt_score,
        add_relative_score=add_relative_score,
        relative_method=relative_method,
        relative_output_col=relative_output_col,
        add_binary_label=add_binary_label,
        delta=delta,
        higher_is_better=higher_is_better,
        binary_output_col=binary_output_col,
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
            variant_type=selected_variant_type,
            cache=cache,
            refresh=refresh,
            drop_failed=drop_failed,
            add_wildtype_row=add_wildtype_row,
            standardization=standardization,
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
            standardization=standardization,
        )
    else:
        built_table, summary = _download_mavedb_dataset(
            dataset_id,
            cache=cache,
            refresh=refresh,
            drop_failed=drop_failed,
            add_wildtype_row=add_wildtype_row,
            standardization=standardization,
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
    wt_sequence: Mapping[str, str] | None = None,
    wt_score: Mapping[str, float] | None = None,
    add_relative_score: bool = False,
    relative_method: str = "log_ratio",
    relative_output_col: str = "score_log_ratio",
    add_binary_label: bool = False,
    delta: float = 0.1,
    higher_is_better: bool = True,
    binary_output_col: str = "score_binary_like",
    overwrite: bool = False,
    acquisition: str | None = None,
    snapshot_record_id: str | None = None,
    include_superseded: bool = False,
    variant_type: str = "substitutions",
) -> DatasetBatchDownloadResult:
    """Download and standardize an ordered batch from one source."""
    selected_variant_type = _validate_download_variant_type(
        source,
        variant_type,
    )
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
    sequence_fallbacks, score_fallbacks = _validate_batch_fallbacks(
        plan.dataset_ids,
        wt_sequence=wt_sequence,
        wt_score=wt_score,
    )
    standardization = _standardization_options(
        wt_sequence=None,
        wt_score=None,
        add_relative_score=add_relative_score,
        relative_method=relative_method,
        relative_output_col=relative_output_col,
        add_binary_label=add_binary_label,
        delta=delta,
        higher_is_better=higher_is_better,
        binary_output_col=binary_output_col,
    )

    batch_provenance: dict[str, Mapping[str, Any]] | None = None
    if source == "proteingym":
        entries = _download_proteingym_batch(
            plan,
            variant_type=selected_variant_type,
            cache=cache,
            refresh=refresh,
            drop_failed=drop_failed,
            add_wildtype_row=add_wildtype_row,
            overwrite=overwrite,
            standardization=standardization,
            sequence_fallbacks=sequence_fallbacks,
            score_fallbacks=score_fallbacks,
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
            standardization=standardization,
            sequence_fallbacks=sequence_fallbacks,
            score_fallbacks=score_fallbacks,
        )
    else:
        entries = _download_mavedb_batch(
            plan,
            cache=cache,
            refresh=refresh,
            drop_failed=drop_failed,
            add_wildtype_row=add_wildtype_row,
            overwrite=overwrite,
            standardization=standardization,
            sequence_fallbacks=sequence_fallbacks,
            score_fallbacks=score_fallbacks,
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


def _validate_download_variant_type(source: str, variant_type: str) -> str:
    """Validate direct-download ProteinGym resource selection."""
    if (
        not isinstance(variant_type, str)
        or variant_type not in _PROTEINGYM_BENCHMARK_CACHE_IDS
    ):
        raise InvalidPipelineOptionError(
            "variant_type must be 'substitutions' or 'indels'."
        )
    if source == "mavedb" and variant_type == "indels":
        raise InvalidPipelineOptionError(
            "variant_type='indels' is supported only for ProteinGym."
        )
    return variant_type


def _standardization_options(
    *,
    wt_sequence: str | None,
    wt_score: float | None,
    add_relative_score: bool,
    relative_method: str,
    relative_output_col: str,
    add_binary_label: bool,
    delta: float,
    higher_is_better: bool,
    binary_output_col: str,
) -> _StandardizationOptions:
    """Validate and collect options before source or output side effects."""
    validate_standardization_options(
        wt_sequence=wt_sequence,
        wt_score=wt_score,
        add_relative_score=add_relative_score,
        relative_method=relative_method,
        relative_output_col=relative_output_col,
        add_binary_label=add_binary_label,
        delta=delta,
        higher_is_better=higher_is_better,
        binary_output_col=binary_output_col,
    )
    return _StandardizationOptions(
        wt_sequence=wt_sequence,
        wt_score=wt_score,
        add_relative_score=add_relative_score,
        relative_method=relative_method,
        relative_output_col=relative_output_col,
        add_binary_label=add_binary_label,
        delta=float(delta),
        higher_is_better=higher_is_better,
        binary_output_col=binary_output_col,
    )


def _validate_batch_fallbacks(
    dataset_ids: tuple[str, ...],
    *,
    wt_sequence: Mapping[str, str] | None,
    wt_score: Mapping[str, float] | None,
) -> tuple[dict[str, str], dict[str, float]]:
    """Validate per-dataset fallback mappings before acquisition."""
    requested = set(dataset_ids)
    sequences = _validate_fallback_mapping(
        wt_sequence,
        requested=requested,
        field="wt_sequence",
    )
    scores = _validate_fallback_mapping(
        wt_score,
        requested=requested,
        field="wt_score",
    )
    for dataset_id, sequence in sequences.items():
        resolve_wt_sequence((), sequence, dataset_id=dataset_id)
    for dataset_id, score in scores.items():
        validate_standardization_options(
            wt_sequence=None,
            wt_score=score,
            add_relative_score=False,
            relative_method="log_ratio",
            relative_output_col="score_log_ratio",
            add_binary_label=False,
            delta=0.1,
            higher_is_better=True,
            binary_output_col="score_binary_like",
        )
    return sequences, scores


def _validate_fallback_mapping(
    value: Mapping[str, Any] | None,
    *,
    requested: set[str],
    field: str,
) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise InvalidPipelineOptionError(
            f"{field} must be a mapping from dataset ID to fallback value."
        )
    resolved = dict(value)
    invalid_keys = [key for key in resolved if key not in requested]
    if invalid_keys:
        joined = ", ".join(repr(key) for key in invalid_keys)
        raise InvalidPipelineOptionError(
            f"{field} contains mappings for unrequested datasets: {joined}."
        )
    return resolved


def _options_for_dataset(
    options: _StandardizationOptions,
    dataset_id: str,
    sequence_fallbacks: Mapping[str, str],
    score_fallbacks: Mapping[str, float],
) -> _StandardizationOptions:
    """Return common options with one dataset's independent WT fallbacks."""
    return replace(
        options,
        wt_sequence=sequence_fallbacks.get(dataset_id),
        wt_score=score_fallbacks.get(dataset_id),
    )


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
    standardization: _StandardizationOptions,
    sequence_fallbacks: Mapping[str, str],
    score_fallbacks: Mapping[str, float],
) -> tuple[DatasetBatchDownloadEntry, ...]:
    """Process independent MaveDB URNs through the single-download API."""
    entries: list[DatasetBatchDownloadEntry] = []
    for dataset_id, dataset_output_dir in zip(
        plan.dataset_ids,
        plan.output_dirs,
        strict=True,
    ):
        options = _options_for_dataset(
            standardization,
            dataset_id,
            sequence_fallbacks,
            score_fallbacks,
        )
        try:
            result = download_and_standardize_dataset(
                "mavedb",
                dataset_id,
                output_dir=dataset_output_dir,
                cache=cache,
                refresh=refresh,
                drop_failed=drop_failed,
                add_wildtype_row=add_wildtype_row,
                wt_sequence=options.wt_sequence,
                wt_score=options.wt_score,
                add_relative_score=options.add_relative_score,
                relative_method=options.relative_method,
                relative_output_col=options.relative_output_col,
                add_binary_label=options.add_binary_label,
                delta=options.delta,
                higher_is_better=options.higher_is_better,
                binary_output_col=options.binary_output_col,
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
    standardization: _StandardizationOptions,
    sequence_fallbacks: Mapping[str, str],
    score_fallbacks: Mapping[str, float],
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
            options = _options_for_dataset(
                standardization,
                dataset_id,
                sequence_fallbacks,
                score_fallbacks,
            )
            built_table, summary = _build_mavedb_download(
                dataset_id,
                _snapshot_download_input(datasets[dataset_id]),
                drop_failed=drop_failed,
                add_wildtype_row=add_wildtype_row,
                standardization=options,
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
    variant_type: str,
    cache: FilesystemCache,
    refresh: bool,
    drop_failed: bool,
    add_wildtype_row: bool,
    overwrite: bool,
    standardization: _StandardizationOptions,
    sequence_fallbacks: Mapping[str, str],
    score_fallbacks: Mapping[str, float],
) -> tuple[DatasetBatchDownloadEntry, ...]:
    """Acquire shared ProteinGym artifacts once and process requested assays."""
    try:
        records = list_datasets(
            "proteingym",
            variant_type=variant_type,
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
                _resolve_proteingym_download_metadata(
                    matches[0],
                    dataset_id,
                    wt_sequence=sequence_fallbacks.get(dataset_id),
                )
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
                variant_type=variant_type,
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
                        variant_type=variant_type,
                        drop_failed=drop_failed,
                        add_wildtype_row=add_wildtype_row,
                        standardization=_options_for_dataset(
                            standardization,
                            dataset_id,
                            sequence_fallbacks,
                            score_fallbacks,
                        ),
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
    variant_type: str,
    cache: FilesystemCache,
    refresh: bool,
    drop_failed: bool,
    add_wildtype_row: bool,
    standardization: _StandardizationOptions,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Acquire and build one assay from the selected ProteinGym variant resource."""
    metadata_record = get_dataset_metadata(
        "proteingym",
        dataset_id,
        variant_type=variant_type,
        cache=cache,
        refresh=refresh,
    )
    metadata = _resolve_proteingym_download_metadata(
        metadata_record,
        dataset_id,
        wt_sequence=standardization.wt_sequence,
    )
    benchmark_table = _acquire_proteingym_benchmark(
        variant_type=variant_type,
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
        variant_type=variant_type,
        drop_failed=drop_failed,
        add_wildtype_row=add_wildtype_row,
        standardization=standardization,
    )


def _resolve_proteingym_download_metadata(
    metadata_record: DatasetRecord,
    dataset_id: str,
    *,
    wt_sequence: str | None,
) -> _ProteinGymDownloadMetadata:
    """Resolve and validate ProteinGym builder metadata without acquisition."""
    metadata = pd.Series(metadata_record.raw_metadata)
    metadata_sequence = _metadata_text(metadata, "target_seq")
    automatic = (
        (("proteingym_reference_metadata", metadata_sequence),)
        if metadata_sequence is not None
        else ()
    )
    resolved_sequence, sequence_provenance = resolve_wt_sequence(
        automatic,
        wt_sequence,
        dataset_id=dataset_id,
    )
    return _ProteinGymDownloadMetadata(
        wt_sequence=resolved_sequence,
        wt_sequence_provenance=sequence_provenance,
        protein_id=_metadata_text(metadata, "molecule_name"),
        gene=_metadata_text(metadata, "gene", "Gene", "gene_name"),
        uniprot_id=_metadata_text(metadata, "UniProt_ID"),
    )


def _acquire_proteingym_benchmark(
    *,
    variant_type: str,
    cache: FilesystemCache,
    refresh: bool,
) -> pd.DataFrame:
    """Acquire and validate the selected shared ProteinGym benchmark."""
    resource = get_proteingym_resource(
        f"dms_{variant_type}",
        require_processing=True,
    )
    benchmark_path = download_proteingym_dataset(
        resource.data_url,
        cache=cache,
        dataset_id=_PROTEINGYM_BENCHMARK_CACHE_IDS[variant_type],
        refresh=refresh,
    )
    benchmark_table = read_table(benchmark_path)
    if "DMS_id" not in benchmark_table.columns:
        raise InvalidDatasetError(
            f"ProteinGym {variant_type} benchmark is missing the 'DMS_id' column."
        )
    return benchmark_table


def _build_proteingym_download(
    dataset_id: str,
    metadata: _ProteinGymDownloadMetadata,
    experiment_table: pd.DataFrame,
    *,
    variant_type: str,
    drop_failed: bool,
    add_wildtype_row: bool,
    standardization: _StandardizationOptions,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build one selected ProteinGym assay through the existing builder."""
    with TemporaryDirectory(prefix="dmsroute-download-") as temporary_dir:
        selected_path = Path(temporary_dir) / "selected.csv"
        write_table(experiment_table, selected_path, index=False)
        builder = (
            build_proteingym_dataset
            if variant_type == "substitutions"
            else build_proteingym_indel_dataset
        )
        builder_kwargs = {
            "variant_col": "mutant"
        } if variant_type == "substitutions" else {
            "mutated_sequence_col": "mutated_sequence",
            "target_sequence_col": "target_seq",
        }
        built_table = builder(
            input_path=selected_path,
            score_col="DMS_score",
            **builder_kwargs,
            dataset_id=dataset_id,
            protein_id=metadata.protein_id,
            gene=metadata.gene,
            uniprot_id=metadata.uniprot_id,
            wt_sequence=metadata.wt_sequence,
            wt_score=standardization.wt_score,
            add_relative_score=standardization.add_relative_score,
            relative_method=standardization.relative_method,
            relative_output_col=standardization.relative_output_col,
            add_binary_label=standardization.add_binary_label,
            delta=standardization.delta,
            higher_is_better=standardization.higher_is_better,
            binary_output_col=standardization.binary_output_col,
            add_wildtype_row=add_wildtype_row,
            drop_failed=drop_failed,
            validate_output=True,
            require_wt_for_transforms=standardization.add_relative_score,
        )

    replace_wt_sequence_provenance(
        built_table,
        metadata.wt_sequence_provenance,
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
        requested_transformation=(
            standardization.relative_method
            if standardization.add_relative_score
            else None
        ),
        transformed_output_column=(
            standardization.relative_output_col
            if standardization.add_relative_score
            else None
        ),
    )


def _download_mavedb_dataset(
    dataset_id: str,
    *,
    cache: FilesystemCache,
    refresh: bool,
    drop_failed: bool,
    add_wildtype_row: bool,
    standardization: _StandardizationOptions,
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
        standardization=standardization,
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
    standardization: _StandardizationOptions,
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
        standardization=standardization,
    )


def _acquire_mavedb_api_dataset(
    dataset_id: str,
    *,
    cache: FilesystemCache,
    refresh: bool,
) -> _MaveDBDownloadInput:
    """Return API metadata and score-table inputs without building them."""
    metadata = _acquire_mavedb_api_metadata(
        dataset_id,
        cache=cache,
        refresh=refresh,
    )
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


def _acquire_mavedb_api_metadata(
    dataset_id: str,
    *,
    cache: FilesystemCache,
    refresh: bool,
) -> dict[str, Any]:
    """Resolve deterministic MaveDB score-set metadata through the cache."""
    encoded_id = quote(dataset_id, safe=":")
    metadata_url = f"{MAVEDB_API_URL.rstrip('/')}/score-sets/{encoded_id}"
    metadata_path = cache.resolve(
        _MAVEDB_METADATA_CACHE_SOURCE,
        dataset_id,
        refresh=refresh,
    )
    if metadata_path is None:
        metadata_record = _get_mavedb_download_metadata(dataset_id)
        _, serialized = _validated_mavedb_metadata_payload(
            metadata_record.raw_metadata,
            dataset_id,
        )
        metadata_path = cache.store_bytes(
            _MAVEDB_METADATA_CACHE_SOURCE,
            dataset_id,
            metadata_url,
            serialized,
            refresh=refresh,
        )

    try:
        values = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InvalidDatasetError(
            f"Cached MaveDB metadata for {dataset_id!r} is invalid: {exc}"
        ) from exc
    return _validate_mavedb_metadata(values, dataset_id, cached=True)


def _validated_mavedb_metadata_payload(
    values: Any,
    dataset_id: str,
) -> tuple[dict[str, Any], bytes]:
    """Validate and serialize fresh MaveDB metadata before cache replacement."""
    metadata = _validate_mavedb_metadata(values, dataset_id, cached=False)
    try:
        serialized = (
            json.dumps(
                metadata,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise InvalidDatasetError(
            f"MaveDB metadata for {dataset_id!r} cannot be serialized: {exc}"
        ) from exc
    return metadata, serialized


def _validate_mavedb_metadata(
    values: Any,
    dataset_id: str,
    *,
    cached: bool,
) -> dict[str, Any]:
    """Validate source-derived MaveDB metadata without using user fallbacks."""
    description = "Cached MaveDB metadata" if cached else "MaveDB metadata"
    if not isinstance(values, Mapping):
        raise InvalidDatasetError(
            f"{description} for {dataset_id!r} must be a JSON object."
        )
    metadata = normalize_raw_metadata(values)
    if not isinstance(metadata, Mapping):
        raise InvalidDatasetError(
            f"Normalized {description.lower()} for {dataset_id!r} must be "
            "a JSON object."
        )
    metadata = dict(metadata)
    if metadata.get("urn") != dataset_id:
        raise InvalidDatasetError(
            f"{description} does not match {dataset_id!r}."
        )
    evidence = _mavedb_wt_sequence_evidence(metadata)
    if evidence:
        resolve_wt_sequence(evidence, None, dataset_id=dataset_id)
    return metadata


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
    standardization: _StandardizationOptions,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build API or snapshot inputs through the existing MaveDB builder."""
    metadata = acquired.metadata
    wt_sequence, sequence_provenance = resolve_wt_sequence(
        _mavedb_wt_sequence_evidence(metadata),
        standardization.wt_sequence,
        dataset_id=dataset_id,
    )
    if sequence_provenance.startswith("mavedb_score_set_metadata:"):
        sequence_provenance = "mavedb_score_set_metadata"

    raw_table = read_table(acquired.scores_path)
    hgvs_col = next(
        (column for column in ("hgvs_pro",) if column in raw_table.columns),
        None,
    )
    score_col = next(
        (
            column
            for column in ("score", "scores.score", "DMS_score", "fitness")
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
        wt_score=standardization.wt_score,
        add_relative_score=standardization.add_relative_score,
        relative_method=standardization.relative_method,
        relative_output_col=standardization.relative_output_col,
        add_binary_label=standardization.add_binary_label,
        delta=standardization.delta,
        higher_is_better=standardization.higher_is_better,
        binary_output_col=standardization.binary_output_col,
        add_wildtype_row=add_wildtype_row,
        drop_failed=drop_failed,
        validate_output=True,
        require_wt_for_transforms=standardization.add_relative_score,
    )
    replace_wt_sequence_provenance(built_table, sequence_provenance)
    summary = _successful_download_summary(
        source="mavedb",
        dataset_id=dataset_id,
        target_protein=_mavedb_target_protein(metadata, gene),
        wt_sequence=wt_sequence,
        raw_rows=len(raw_table),
        built_table=built_table,
        requested_transformation=(
            standardization.relative_method
            if standardization.add_relative_score
            else None
        ),
        transformed_output_column=(
            standardization.relative_output_col
            if standardization.add_relative_score
            else None
        ),
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
    requested_transformation: str | None,
    transformed_output_column: str | None,
) -> dict[str, Any]:
    """Build the existing successful summary representation for one dataset."""
    return {
        "source": source,
        "input": dataset_id,
        "status": "OK",
        "dataset_id": dataset_id,
        "target_protein": target_protein,
        "wt_length": len(wt_sequence),
        "wt_sequence_sha256": hashlib.sha256(
            wt_sequence.encode("utf-8")
        ).hexdigest(),
        **_wt_summary_fields(
            built_table,
            requested_transformation=requested_transformation,
            transformed_output_column=transformed_output_column,
        ),
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
        "wt_sequence_sha256": summary.get("wt_sequence_sha256"),
        "wt_sequence_provenance": summary.get("wt_sequence_provenance"),
        "wt_score": summary.get("wt_score"),
        "wt_score_provenance": summary.get("wt_score_provenance"),
        "observed_wildtype_row": summary.get("observed_wildtype_row"),
        "synthetic_wildtype_inserted": summary.get(
            "synthetic_wildtype_inserted"
        ),
        "requested_transformation": summary.get("requested_transformation"),
        "transformed_output_column": summary.get(
            "transformed_output_column"
        ),
        "wt_score_unavailable_reason": summary.get(
            "wt_score_unavailable_reason"
        ),
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
