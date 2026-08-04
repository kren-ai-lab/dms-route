"""Source-specific dataset loading utilities."""

from __future__ import annotations

from dms_parser.sources.mavedb import (
    download_mavedb_dataset,
    ensure_local_copy as ensure_local_mavedb_copy,
    load_mavedb_dataset,
    load_mavedb_from_url,
)
from dms_parser.sources.mavedb_catalog import MaveDBCatalog
from dms_parser.sources.mavedb_bulk_catalog import (
    MaveDBBulkCatalog,
    MaveDBDiscoveredExperiment,
    MaveDBDiscoveredScoreSet,
    MaveDBDiscoveryResult,
)
from dms_parser.sources.mavedb_snapshots import (
    MaveDBSnapshot,
    MaveDBSnapshotRecord,
    fetch_mavedb_snapshot,
    resolve_mavedb_snapshot,
)
from dms_parser.sources.mavedb_snapshot_tables import (
    MaveDBSnapshotTable,
    MaveDBSnapshotTableExtractionResult,
    extract_mavedb_snapshot_tables,
)
from dms_parser.sources.proteingym import (
    download_proteingym_dataset,
    ensure_local_copy as ensure_local_proteingym_copy,
    load_proteingym_dataset,
    load_proteingym_from_url,
)
from dms_parser.sources.proteingym_catalog import ProteinGymCatalog
from dms_parser.sources.proteingym_resources import (
    PROTEINGYM_RESOURCES,
    ProteinGymResource,
    get_proteingym_resource,
    list_proteingym_resources,
)

__all__ = [
    "MaveDBCatalog",
    "MaveDBSnapshot",
    "MaveDBSnapshotRecord",
    "fetch_mavedb_snapshot",
    "resolve_mavedb_snapshot",
    "MaveDBBulkCatalog",
    "MaveDBDiscoveredExperiment",
    "MaveDBDiscoveredScoreSet",
    "MaveDBDiscoveryResult",
    "MaveDBSnapshotTable",
    "MaveDBSnapshotTableExtractionResult",
    "extract_mavedb_snapshot_tables",
    "download_mavedb_dataset",
    "load_mavedb_dataset",
    "load_mavedb_from_url",
    "ensure_local_mavedb_copy",
    "ProteinGymCatalog",
    "ProteinGymResource",
    "PROTEINGYM_RESOURCES",
    "get_proteingym_resource",
    "list_proteingym_resources",
    "download_proteingym_dataset",
    "load_proteingym_dataset",
    "load_proteingym_from_url",
    "ensure_local_proteingym_copy",
]
