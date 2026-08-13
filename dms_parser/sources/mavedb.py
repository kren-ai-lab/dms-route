"""Utilities for downloading and loading datasets from MaveDB."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from dms_parser.cache import FilesystemCache
from dms_parser.io import (
    ensure_local_copy as ensure_local_dataset_copy,
    read_table,
)
from dms_parser.sources._dataset_adapter import download_source_dataset

logger = logging.getLogger(__name__)


def download_mavedb_dataset(
    url: str,
    output_dir: str | Path = "./data/mavedb",
    filename: Optional[str] = None,
    *,
    overwrite: bool = False,
    cache: FilesystemCache | None = None,
    dataset_id: str | None = None,
    refresh: bool = False,
) -> Path:
    """Download a MaveDB dataset directly or through a filesystem cache.

    When ``cache`` is provided, ``dataset_id`` is required and the cached
    artifact path is returned. ``output_dir`` and ``filename`` apply only to
    uncached downloads.
    """
    return download_source_dataset(
        url,
        output_dir,
        filename,
        overwrite=overwrite,
        cache=cache,
        dataset_id=dataset_id,
        refresh=refresh,
        source="mavedb",
        source_label="MaveDB",
        default_filename="mavedb_dataset.csv",
        logger=logger,
    )


def load_mavedb_dataset(
    path: str | Path,
    *,
    sep: str | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Load a MaveDB dataset from disk."""
    logger.debug("Loading source dataset source=mavedb path=%s", path)
    return read_table(path, sep=sep, **kwargs)


def load_mavedb_from_url(
    url: str,
    *,
    output_dir: str | Path = "./data/mavedb",
    filename: Optional[str] = None,
    overwrite: bool = False,
    cache: FilesystemCache | None = None,
    dataset_id: str | None = None,
    refresh: bool = False,
    sep: str | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Download and load a MaveDB dataset in one step."""
    path = download_mavedb_dataset(
        url=url,
        output_dir=output_dir,
        filename=filename,
        overwrite=overwrite,
        cache=cache,
        dataset_id=dataset_id,
        refresh=refresh,
    )
    return load_mavedb_dataset(path, sep=sep, **kwargs)


def ensure_local_copy(
    path_or_url: str | Path,
    *,
    output_dir: str | Path = "./data/mavedb",
    overwrite: bool = False,
) -> Path:
    """Ensure a MaveDB dataset is available locally."""
    return ensure_local_dataset_copy(
        path_or_url,
        output_dir=output_dir,
        default_name="mavedb_dataset.csv",
        overwrite=overwrite,
    )
