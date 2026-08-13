"""Shared implementation for thin source dataset adapters."""

from __future__ import annotations

import logging
from pathlib import Path

from dms_parser.acquisition.cache import FilesystemCache
from dms_parser.core.exceptions import InvalidCacheEntryError
from dms_parser.acquisition.fetch import fetch_to_cache
from dms_parser.acquisition.io import download_file, infer_filename_from_url


def download_source_dataset(
    url: str,
    output_dir: str | Path,
    filename: str | None,
    *,
    overwrite: bool, cache: FilesystemCache | None,
    dataset_id: str | None, refresh: bool,
    source: str, source_label: str,
    default_filename: str, logger: logging.Logger,
) -> Path:
    """Download one source artifact directly or through the shared cache."""
    logger.info("Starting source download source=%s dataset_id=%s cache_enabled=%s", source, dataset_id, cache is not None)
    logger.debug(
        "Source download options source=%s dataset_id=%s overwrite=%s "
        "refresh=%s",
        source, dataset_id, overwrite, refresh,
    )
    if cache is not None:
        if dataset_id is None:
            raise InvalidCacheEntryError(
                f"dataset_id is required when caching a {source_label} dataset."
            )
        if overwrite:
            raise InvalidCacheEntryError(
                "overwrite cannot be used with a cache; use refresh=True."
            )
        path = fetch_to_cache(
            url,
            source=source,
            dataset_id=dataset_id,
            cache=cache,
            refresh=refresh,
        )
        logger.info("Completed source download source=%s dataset_id=%s", source, dataset_id)
        logger.debug("Resolved source artifact source=%s dataset_id=%s path=%s", source, dataset_id, path)
        return path

    if refresh:
        raise InvalidCacheEntryError(
            "refresh requires a cache; use overwrite=True for an uncached download."
        )

    resolved_filename = filename or infer_filename_from_url(
        url, default_name=default_filename
    )
    output_path = Path(output_dir) / resolved_filename
    logger.debug("Resolved source output source=%s dataset_id=%s path=%s", source, dataset_id, output_path)
    path = download_file(url, output_path, overwrite=overwrite)
    logger.info("Completed source download source=%s dataset_id=%s", source, dataset_id)
    return path
