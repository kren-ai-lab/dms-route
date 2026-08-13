"""Unified metadata-only dataset catalog interfaces."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
import pandas as pd

from dmsroute.acquisition.cache import FilesystemCache
from dmsroute.core.exceptions import InvalidCatalogQueryError

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from dmsroute.sources.mavedb_catalog import MaveDBCatalog
    from dmsroute.sources.proteingym_catalog import ProteinGymCatalog

    CatalogAdapter = MaveDBCatalog | ProteinGymCatalog
else:
    CatalogAdapter = Any

_SUPPORTED_SOURCES = frozenset({"mavedb", "proteingym"})


@dataclass(frozen=True)
class DatasetRecord:
    """Common metadata representation for one source dataset."""

    source: str
    dataset_id: str
    title: str | None
    target_id: str | None
    variant_type: str | None
    n_variants: int | None
    raw_metadata: dict[str, Any]


def list_datasets(
    source: str,
    *,
    query: str | None = None,
    limit: int | None = None,
    offset: int = 0,
    variant_type: str | None = None,
    cache: FilesystemCache | None = None,
    refresh: bool = False,
    catalog: CatalogAdapter | None = None,
) -> list[DatasetRecord]:
    """List datasets from one supported source through a common interface.

    ``source="all"`` is intentionally unsupported because MaveDB and
    ProteinGym have different pagination models.
    """
    source_name = _validate_source(source)
    adapter = _resolve_catalog(
        source_name,
        cache=cache,
        refresh=refresh,
        catalog=catalog,
    )

    if source_name == "mavedb":
        if variant_type is not None:
            raise InvalidCatalogQueryError(
                "variant_type is only supported for the ProteinGym catalog."
            )
        mavedb_limit = 100 if limit is None else limit
        return adapter.list_datasets(
            query=query,
            limit=mavedb_limit,
            offset=offset,
        )

    return adapter.list_datasets(
        query=query,
        variant_type=variant_type,
        limit=limit,
        offset=offset,
    )


def get_dataset_metadata(
    source: str,
    dataset_id: str,
    *,
    variant_type: str | None = None,
    cache: FilesystemCache | None = None,
    refresh: bool = False,
    catalog: CatalogAdapter | None = None,
) -> DatasetRecord:
    """Retrieve metadata for one dataset from a supported source."""
    source_name = _validate_source(source)
    adapter = _resolve_catalog(
        source_name,
        cache=cache,
        refresh=refresh,
        catalog=catalog,
    )

    if source_name == "mavedb":
        if variant_type is not None:
            raise InvalidCatalogQueryError(
                "variant_type is only supported for the ProteinGym catalog."
            )
        return adapter.get_metadata(dataset_id)

    return adapter.get_metadata(dataset_id, variant_type=variant_type)


def normalize_raw_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe copy of source metadata without missing sentinels."""
    return {
        str(key): _normalize_metadata_value(value)
        for key, value in metadata.items()
    }


def optional_text(value: Any) -> str | None:
    """Return a non-empty text value or ``None`` for missing metadata."""
    value = _normalize_metadata_value(value)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def optional_int(value: Any) -> int | None:
    """Return an integer value or ``None`` for missing metadata."""
    value = _normalize_metadata_value(value)
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not number.is_integer():
        return None
    return int(number)


def validate_query(query: str | None) -> None:
    """Validate an optional catalog search query."""
    if query is not None and not isinstance(query, str):
        raise InvalidCatalogQueryError("query must be a string or None.")


def validate_limit(limit: int | None, *, allow_none: bool) -> None:
    """Validate a catalog result limit."""
    if limit is None and allow_none:
        return
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        expectation = "a positive integer or None" if allow_none else "a positive integer"
        raise InvalidCatalogQueryError(f"limit must be {expectation}.")


def validate_offset(offset: int) -> None:
    """Validate a non-negative catalog result offset."""
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise InvalidCatalogQueryError("offset must be a non-negative integer.")


def validate_dataset_id(dataset_id: str) -> None:
    """Validate a non-empty source dataset identifier."""
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise InvalidCatalogQueryError("dataset_id must be a non-empty string.")


def _validate_source(source: str) -> str:
    """Return a normalized supported source name."""
    if not isinstance(source, str):
        raise InvalidCatalogQueryError("source must be 'mavedb' or 'proteingym'.")
    source_name = source.strip().lower()
    if source_name not in _SUPPORTED_SOURCES:
        raise InvalidCatalogQueryError(
            f"Unsupported catalog source {source!r}; expected 'mavedb' or "
            "'proteingym'."
        )
    return source_name


def _resolve_catalog(
    source: str,
    *,
    cache: FilesystemCache | None,
    refresh: bool,
    catalog: CatalogAdapter | None,
) -> CatalogAdapter:
    """Return a matching injected or default source catalog."""
    from dmsroute.sources.mavedb_catalog import MaveDBCatalog
    from dmsroute.sources.proteingym_catalog import ProteinGymCatalog

    expected_type = MaveDBCatalog if source == "mavedb" else ProteinGymCatalog
    if catalog is not None:
        if cache is not None or refresh:
            raise InvalidCatalogQueryError(
                "cache and refresh cannot be combined with an injected catalog."
            )
        if not isinstance(catalog, expected_type):
            raise InvalidCatalogQueryError(
                f"The injected catalog does not match source {source!r}."
            )
        logger.debug("Selected injected catalog adapter source=%s adapter=%s", source, type(catalog).__name__)
        return catalog

    if source == "mavedb":
        if cache is not None:
            raise InvalidCatalogQueryError(
                "cache is not supported for MaveDB catalog requests."
            )
        if refresh:
            raise InvalidCatalogQueryError(
                "refresh is only supported for the ProteinGym catalog."
            )
        adapter = MaveDBCatalog()
        logger.debug("Selected default catalog adapter source=%s adapter=%s", source, type(adapter).__name__)
        return adapter

    adapter = ProteinGymCatalog(cache=cache, refresh=refresh)
    logger.debug("Selected default catalog adapter source=%s adapter=%s", source, type(adapter).__name__)
    return adapter


def _normalize_metadata_value(value: Any) -> Any:
    """Recursively convert source values into JSON-safe Python values."""
    if isinstance(value, Mapping):
        return normalize_raw_metadata(value)
    if isinstance(value, (list, tuple, set)):
        return [_normalize_metadata_value(item) for item in value]
    if value is None:
        return None

    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and missing:
        return None

    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None

    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            scalar = item_method()
        except (TypeError, ValueError):
            scalar = value
        if scalar is not value:
            return _normalize_metadata_value(scalar)

    return value
