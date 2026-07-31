"""Metadata-only catalog access for ProteinGym reference files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal, Mapping

import pandas as pd

from dms_parser.cache import FilesystemCache
from dms_parser.catalog import (
    DatasetRecord,
    normalize_raw_metadata,
    optional_int,
    optional_text,
    validate_dataset_id,
    validate_limit,
    validate_offset,
    validate_query,
)
from dms_parser.exceptions import (
    CatalogError,
    DatasetNotFoundError,
    InvalidCatalogQueryError,
)
from dms_parser.fetch import fetch_to_cache
from dms_parser.sources.proteingym_resources import get_proteingym_resource

logger = logging.getLogger(__name__)

PROTEINGYM_SUBSTITUTIONS_URL = get_proteingym_resource(
    "dms_substitutions"
).metadata_url
PROTEINGYM_INDELS_URL = get_proteingym_resource("dms_indels").metadata_url
PROTEINGYM_SUBSTITUTIONS_CACHE_ID = "reference-files-dms-substitutions"
PROTEINGYM_INDELS_CACHE_ID = "reference-files-dms-indels"

VariantType = Literal["substitutions", "indels"]
_VARIANT_TYPES: tuple[VariantType, ...] = ("substitutions", "indels")


class ProteinGymCatalog:
    """List and retrieve ProteinGym metadata from lightweight reference files."""

    source = "proteingym"

    def __init__(
        self,
        *,
        cache: FilesystemCache | None = None,
        substitutions_url: str = PROTEINGYM_SUBSTITUTIONS_URL,
        indels_url: str = PROTEINGYM_INDELS_URL,
        substitutions_path: str | Path | None = None,
        indels_path: str | Path | None = None,
        refresh: bool = False,
    ) -> None:
        """Configure cached remote or injected local reference files."""
        self.cache = cache
        self.urls = {
            "substitutions": _validate_url(substitutions_url, "substitutions_url"),
            "indels": _validate_url(indels_url, "indels_url"),
        }
        self.paths = {
            "substitutions": (
                Path(substitutions_path) if substitutions_path is not None else None
            ),
            "indels": Path(indels_path) if indels_path is not None else None,
        }
        if not isinstance(refresh, bool):
            raise InvalidCatalogQueryError("refresh must be a boolean.")
        self.refresh = refresh

    def list_datasets(
        self,
        query: str | None = None,
        variant_type: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[DatasetRecord]:
        """List ProteinGym assays with deterministic filtering and pagination."""
        validate_query(query)
        selected_types = _validate_variant_type(variant_type)
        validate_limit(limit, allow_none=True)
        validate_offset(offset)
        logger.info("Starting catalog listing source=proteingym")
        logger.debug(
            "Validated catalog filters source=proteingym query_supplied=%s "
            "variant_types=%s limit=%s offset=%d",
            query is not None,
            selected_types,
            limit,
            offset,
        )

        records = self._load_records(selected_types)
        if query is not None:
            needle = query.casefold()
            records = [
                record
                for record in records
                if needle in record.dataset_id.casefold()
                or (
                    record.target_id is not None
                    and needle in record.target_id.casefold()
                )
            ]

        records.sort(
            key=lambda record: (
                record.dataset_id.casefold(),
                record.variant_type or "",
            )
        )
        end = None if limit is None else offset + limit
        records = records[offset:end]
        if not records:
            logger.warning("Catalog listing returned no datasets source=proteingym")
        logger.info(
            "Completed catalog listing source=proteingym result_count=%d",
            len(records),
        )
        return records

    def get_metadata(
        self,
        dataset_id: str,
        variant_type: str | None = None,
    ) -> DatasetRecord:
        """Retrieve one ProteinGym assay by its exact DMS identifier."""
        validate_dataset_id(dataset_id)
        selected_types = _validate_variant_type(variant_type)
        logger.info(
            "Retrieving dataset metadata source=proteingym dataset_id=%s",
            dataset_id,
        )
        logger.debug(
            "Validated metadata filter source=proteingym dataset_id=%s "
            "variant_types=%s",
            dataset_id,
            selected_types,
        )
        matches = [
            record
            for record in self._load_records(selected_types)
            if record.dataset_id == dataset_id
        ]
        if not matches:
            raise DatasetNotFoundError(
                f"ProteinGym dataset {dataset_id!r} was not found."
            )
        if len(matches) > 1:
            raise CatalogError(
                f"ProteinGym dataset {dataset_id!r} is ambiguous; specify "
                "variant_type."
            )
        logger.info(
            "Completed metadata retrieval source=proteingym dataset_id=%s",
            dataset_id,
        )
        return matches[0]

    def _load_records(
        self,
        variant_types: tuple[VariantType, ...],
    ) -> list[DatasetRecord]:
        """Load and map records for the selected lightweight reference files."""
        records: list[DatasetRecord] = []
        for variant_type in variant_types:
            path = self._resolve_reference_path(variant_type)
            logger.debug(
                "Loading ProteinGym reference source=proteingym "
                "variant_type=%s path=%s",
                variant_type,
                path,
            )
            frame = pd.read_csv(path)
            if "DMS_id" not in frame.columns:
                raise CatalogError(
                    f"ProteinGym {variant_type} reference file is missing "
                    "the 'DMS_id' column."
                )
            records.extend(
                _proteingym_record(row, variant_type)
                for row in frame.to_dict(orient="records")
            )
        return records

    def _resolve_reference_path(self, variant_type: VariantType) -> Path:
        """Resolve one local reference file, fetching it through the cache if needed."""
        local_path = self.paths[variant_type]
        if local_path is not None:
            if not local_path.is_file():
                raise FileNotFoundError(
                    f"ProteinGym {variant_type} reference file not found: "
                    f"{local_path}"
                )
            logger.debug(
                "Resolved local catalog reference source=proteingym "
                "variant_type=%s path=%s",
                variant_type,
                local_path,
            )
            return local_path

        if self.cache is None:
            raise InvalidCatalogQueryError(
                "A FilesystemCache is required for remote ProteinGym catalog files."
            )

        cache_id = (
            PROTEINGYM_SUBSTITUTIONS_CACHE_ID
            if variant_type == "substitutions"
            else PROTEINGYM_INDELS_CACHE_ID
        )
        logger.debug(
            "Resolved catalog cache identity source=proteingym "
            "variant_type=%s dataset_id=%s",
            variant_type,
            cache_id,
        )
        return fetch_to_cache(
            self.urls[variant_type],
            source="proteingym",
            dataset_id=cache_id,
            cache=self.cache,
            refresh=self.refresh,
        )


def _proteingym_record(
    value: Mapping[str, Any],
    variant_type: VariantType,
) -> DatasetRecord:
    """Map one ProteinGym CSV row into the public catalog model."""
    metadata = normalize_raw_metadata(value)
    dataset_id = optional_text(metadata.get("DMS_id"))
    if dataset_id is None:
        raise CatalogError("ProteinGym reference row is missing its DMS_id.")

    return DatasetRecord(
        source="proteingym",
        dataset_id=dataset_id,
        title=optional_text(metadata.get("title")),
        target_id=optional_text(metadata.get("UniProt_ID")),
        variant_type=variant_type,
        n_variants=optional_int(metadata.get("DMS_total_number_mutants")),
        raw_metadata=metadata,
    )


def _validate_variant_type(
    variant_type: str | None,
) -> tuple[VariantType, ...]:
    """Validate a ProteinGym variant-type filter."""
    if variant_type is None:
        return _VARIANT_TYPES
    if variant_type not in _VARIANT_TYPES:
        raise InvalidCatalogQueryError(
            "variant_type must be None, 'substitutions', or 'indels'."
        )
    return (variant_type,)


def _validate_url(value: str, field: str) -> str:
    """Validate a non-empty reference file URL."""
    if not isinstance(value, str) or not value.strip():
        raise InvalidCatalogQueryError(f"{field} must be a non-empty string.")
    return value
