"""Reusable operations for downloading artifacts into a filesystem cache."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from dms_parser.cache import FilesystemCache
from dms_parser.io import download_file


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
    cached_path = cache.resolve(
        source,
        dataset_id,
        refresh=refresh,
    )
    if cached_path is not None:
        return cached_path

    with TemporaryDirectory(prefix="dms-parser-fetch-") as staging_directory:
        staging_path = Path(staging_directory) / "artifact.download"
        download_file(
            url,
            staging_path,
            chunk_size=chunk_size,
            timeout=timeout,
        )
        return cache.store_file(
            source,
            dataset_id,
            url,
            staging_path,
            refresh=refresh,
        )
