"""Utilities for downloading and loading datasets from MaveDB."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from dms_parser.cache import FilesystemCache
from dms_parser.exceptions import InvalidCacheEntryError
from dms_parser.fetch import fetch_to_cache
from dms_parser.io import (
    download_file,
    ensure_local_copy as ensure_local_dataset_copy,
    infer_filename_from_url,
    read_table,
)

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
    logger.info(
        "Starting source download source=mavedb dataset_id=%s cache_enabled=%s",
        dataset_id,
        cache is not None,
    )
    logger.debug(
        "Source download options source=mavedb dataset_id=%s overwrite=%s "
        "refresh=%s",
        dataset_id,
        overwrite,
        refresh,
    )
    if cache is not None:
        if dataset_id is None:
            raise InvalidCacheEntryError(
                "dataset_id is required when caching a MaveDB dataset."
            )
        if overwrite:
            raise InvalidCacheEntryError(
                "overwrite cannot be used with a cache; use refresh=True."
            )
        path = fetch_to_cache(
            url,
            source="mavedb",
            dataset_id=dataset_id,
            cache=cache,
            refresh=refresh,
        )
        logger.info(
            "Completed source download source=mavedb dataset_id=%s",
            dataset_id,
        )
        logger.debug(
            "Resolved source artifact source=mavedb dataset_id=%s path=%s",
            dataset_id,
            path,
        )
        return path

    if refresh:
        raise InvalidCacheEntryError(
            "refresh requires a cache; use overwrite=True for an uncached download."
        )

    output_dir = Path(output_dir)

    if filename is None:
        filename = infer_filename_from_url(url, default_name="mavedb_dataset.csv")

    output_path = output_dir / filename
    logger.debug(
        "Resolved source output source=mavedb dataset_id=%s path=%s",
        dataset_id,
        output_path,
    )
    path = download_file(url, output_path, overwrite=overwrite)
    logger.info(
        "Completed source download source=mavedb dataset_id=%s",
        dataset_id,
    )
    return path


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
