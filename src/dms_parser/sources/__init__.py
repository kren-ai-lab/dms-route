"""Source-specific dataset loading utilities."""

from __future__ import annotations

import logging

from dms_parser.sources.mavedb import (
    download_mavedb_dataset,
    ensure_local_copy as ensure_local_mavedb_copy,
    load_mavedb_dataset,
    load_mavedb_from_url,
)
from dms_parser.sources.mavedb_catalog import MaveDBCatalog
from dms_parser.sources.proteingym import (
    download_proteingym_dataset,
    ensure_local_copy as ensure_local_proteingym_copy,
    load_proteingym_dataset,
    load_proteingym_from_url,
)
from dms_parser.sources.proteingym_catalog import ProteinGymCatalog

logger = logging.getLogger(__name__)

__all__ = [
    "MaveDBCatalog",
    "download_mavedb_dataset",
    "load_mavedb_dataset",
    "load_mavedb_from_url",
    "ensure_local_mavedb_copy",
    "ProteinGymCatalog",
    "download_proteingym_dataset",
    "load_proteingym_dataset",
    "load_proteingym_from_url",
    "ensure_local_proteingym_copy",
]
