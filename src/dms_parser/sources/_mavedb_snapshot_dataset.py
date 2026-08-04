"""Private coordination for standardized datasets from cached snapshots."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from dms_parser.cache import FilesystemCache
from dms_parser.exceptions import (
    MaveDBSnapshotTableError,
    SnapshotDatasetNotFoundError,
    SupersededSnapshotDatasetError,
)
from dms_parser.sources.mavedb_bulk_catalog import MaveDBBulkCatalog
from dms_parser.sources.mavedb_snapshot_tables import (
    extract_mavedb_snapshot_tables,
)
from dms_parser.sources.mavedb_snapshots import (
    MaveDBSnapshot,
    load_cached_mavedb_snapshot,
)


@dataclass(frozen=True)
class MaveDBSnapshotDataset:
    """Snapshot inputs required by the existing MaveDB builder."""

    dataset_id: str
    scores_path: Path
    metadata: dict[str, Any]
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class MaveDBSnapshotDatasetAcquisition:
    """Ordered snapshot acquisition outcomes before dataset construction."""

    datasets: tuple[MaveDBSnapshotDataset, ...]
    errors: tuple[tuple[str, Exception], ...]
    provenance: tuple[tuple[str, Mapping[str, Any]], ...]


def acquire_cached_mavedb_snapshot_datasets(
    record_id: str,
    dataset_ids: tuple[str, ...],
    *,
    cache: FilesystemCache,
    include_superseded: bool,
    refresh: bool,
) -> MaveDBSnapshotDatasetAcquisition:
    """Acquire selected score tables without resolving or fetching a snapshot."""
    snapshot = load_cached_mavedb_snapshot(record_id, cache=cache)
    catalog = MaveDBBulkCatalog.from_file(snapshot.main_json_path)
    metadata_by_id: dict[str, dict[str, Any]] = {}
    superseded_by_id: dict[str, bool | None] = {}
    errors: dict[str, Exception] = {}
    valid_ids: list[str] = []

    for dataset_id in dataset_ids:
        record = catalog.get_score_set(dataset_id)
        if record is None:
            superseded_by_id[dataset_id] = None
            errors[dataset_id] = SnapshotDatasetNotFoundError(
                f"MaveDB score set {dataset_id!r} is absent from snapshot "
                f"{snapshot.record.record_id}."
            )
            continue
        superseded_by_id[dataset_id] = record.is_superseded
        if record.is_superseded and not include_superseded:
            errors[dataset_id] = SupersededSnapshotDatasetError(
                f"MaveDB score set {dataset_id!r} is superseded; pass "
                "include_superseded=True to standardize it."
            )
            continue
        metadata = catalog.get_score_set_metadata(dataset_id)
        if metadata is None:
            errors[dataset_id] = SnapshotDatasetNotFoundError(
                f"MaveDB score set {dataset_id!r} has no snapshot metadata."
            )
            continue
        metadata_by_id[dataset_id] = metadata
        valid_ids.append(dataset_id)

    tables_by_id = {}
    if valid_ids:
        try:
            extraction = extract_mavedb_snapshot_tables(
                snapshot,
                valid_ids,
                cache=cache,
                include_superseded=include_superseded,
                refresh=refresh,
            )
        except (MaveDBSnapshotTableError, OSError) as exc:
            errors.update((dataset_id, exc) for dataset_id in valid_ids)
        else:
            tables_by_id = {
                table.dataset_id: table
                for table in extraction.tables
            }

    provenance = tuple(
        (
            dataset_id,
            _snapshot_provenance(
                snapshot,
                catalog_as_of=catalog.as_of,
                is_superseded=superseded_by_id.get(dataset_id),
            ),
        )
        for dataset_id in dataset_ids
    )
    datasets = tuple(
        MaveDBSnapshotDataset(
            dataset_id=dataset_id,
            scores_path=tables_by_id[dataset_id].scores_path,
            metadata=metadata_by_id[dataset_id],
            provenance=provenance_value,
        )
        for dataset_id, provenance_value in provenance
        if dataset_id in tables_by_id
    )
    return MaveDBSnapshotDatasetAcquisition(
        datasets=datasets,
        errors=tuple(
            (dataset_id, errors[dataset_id])
            for dataset_id in dataset_ids
            if dataset_id in errors
        ),
        provenance=provenance,
    )


def _snapshot_provenance(
    snapshot: MaveDBSnapshot,
    *,
    catalog_as_of: str | None,
    is_superseded: bool | None,
) -> Mapping[str, Any]:
    """Return deterministic scalar provenance in stable output order."""
    record = snapshot.record
    return MappingProxyType({
        "acquisition_method": "snapshot",
        "snapshot_record_id": record.record_id,
        "snapshot_doi": record.doi,
        "snapshot_concept_doi": record.concept_doi,
        "snapshot_publication_date": record.publication_date,
        "snapshot_archive_filename": record.filename,
        "snapshot_archive_size": record.size,
        "snapshot_archive_checksum": record.checksum,
        "snapshot_catalog_as_of": catalog_as_of,
        "snapshot_score_set_superseded": is_superseded,
    })
