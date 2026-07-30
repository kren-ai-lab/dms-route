"""Reusable operations for downloading artifacts into a filesystem cache."""

from __future__ import annotations

import logging
from pathlib import Path
from tempfile import TemporaryDirectory

from dms_parser.cache import FilesystemCache
from dms_parser.io import download_file

logger = logging.getLogger(__name__)


def fetch_to_cache(
    url: str,
    *,
    source: str,
    dataset_id: str,
    cache: FilesystemCache,
    refresh: bool = False,
    chunk_size: int = 8192,
    timeout: int = 60,
) -> Path:
    """Resolve a cached artifact or download and publish it atomically.

    Cache keys are validated by the cache lookup before any staging resource
    or network request is created. A miss is downloaded into an isolated
    temporary directory, then published through the cache's crash-safe store.
    """
    logger.info(
        "Starting fetch source=%s dataset_id=%s",
        source,
        dataset_id,
    )
    if refresh:
        logger.info(
            "Explicit cache refresh requested source=%s dataset_id=%s",
            source,
            dataset_id,
        )
    cached_path = cache.resolve(
        source,
        dataset_id,
        refresh=refresh,
    )
    if cached_path is not None:
        logger.info(
            "Cache hit source=%s dataset_id=%s",
            source,
            dataset_id,
        )
        logger.debug(
            "Resolved cached artifact source=%s dataset_id=%s path=%s",
            source,
            dataset_id,
            cached_path,
        )
        return cached_path

    logger.info(
        "Cache miss source=%s dataset_id=%s",
        source,
        dataset_id,
    )
    logger.debug(
        "Cache identity source=%s dataset_id=%s path=%s",
        source,
        dataset_id,
        cache.entry_path(source, dataset_id),
    )
    with TemporaryDirectory(prefix="dms-parser-fetch-") as staging_directory:
        staging_path = Path(staging_directory) / "artifact.download"
        logger.debug(
            "Created fetch staging path source=%s dataset_id=%s path=%s",
            source,
            dataset_id,
            staging_path,
        )
        download_file(
            url,
            staging_path,
            chunk_size=chunk_size,
            timeout=timeout,
        )
        cached_path = cache.store_file(
            source,
            dataset_id,
            url,
            staging_path,
            refresh=refresh,
        )
        logger.info(
            "Completed cache publication source=%s dataset_id=%s",
            source,
            dataset_id,
        )
        logger.debug(
            "Published cached artifact source=%s dataset_id=%s path=%s",
            source,
            dataset_id,
            cached_path,
        )
        return cached_path
