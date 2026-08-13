"""Metadata-only catalog access for public MaveDB score sets."""

from __future__ import annotations

import logging
from typing import Any, Mapping
from urllib.parse import quote

import requests

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
from dms_parser.exceptions import CatalogError

logger = logging.getLogger(__name__)

MAVEDB_API_URL = "https://api.mavedb.org/api/v1/"


class MaveDBCatalog:
    """List and retrieve public MaveDB score-set metadata."""

    source = "mavedb"

    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        base_url: str = MAVEDB_API_URL,
        timeout: int = 60,
    ) -> None:
        """Configure MaveDB HTTP access without making a request."""
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be a non-empty string.")
        if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
            raise ValueError("timeout must be a positive integer.")
        self.session = session if session is not None else requests.Session()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def list_datasets(
        self,
        query: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[DatasetRecord]:
        """Search public score sets while preserving server pagination."""
        validate_query(query)
        validate_limit(limit, allow_none=False)
        validate_offset(offset)
        logger.info("Starting catalog listing source=mavedb")
        logger.debug(
            "Validated catalog filters source=mavedb query_supplied=%s "
            "limit=%d offset=%d",
            query is not None, limit, offset,
        )

        payload: dict[str, Any] = {"limit": limit, "offset": offset}
        if query is not None:
            payload["text"] = query

        logger.debug("Requesting catalog endpoint source=mavedb endpoint=/score-sets/search")
        response = self.session.post(
            f"{self.base_url}/score-sets/search",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        values = response.json()
        score_sets = _mavedb_score_sets(values)

        records = [_mavedb_record(value) for value in score_sets]
        records = sorted(records, key=lambda record: record.dataset_id)
        if not records:
            logger.warning("Catalog listing returned no datasets source=mavedb")
        logger.info("Completed catalog listing source=mavedb result_count=%d", len(records))
        return records

    def get_metadata(self, dataset_id: str) -> DatasetRecord:
        """Retrieve one public MaveDB score-set metadata object."""
        validate_dataset_id(dataset_id)
        logger.info("Retrieving dataset metadata source=mavedb dataset_id=%s", dataset_id)
        encoded_id = quote(dataset_id, safe=":")
        logger.debug("Requesting metadata endpoint source=mavedb endpoint=/score-sets/<dataset_id>")
        response = self.session.get(
            f"{self.base_url}/score-sets/{encoded_id}",
            timeout=self.timeout,
        )
        response.raise_for_status()
        record = _mavedb_record(response.json())
        logger.info("Completed metadata retrieval source=mavedb dataset_id=%s", dataset_id)
        return record


def _mavedb_score_sets(value: object) -> list[object]:
    """Extract score-set items from supported MaveDB search responses."""
    if isinstance(value, list):
        return value
    if not isinstance(value, Mapping):
        raise CatalogError(
            "MaveDB search response must be a JSON object containing "
            "'scoreSets' or a JSON list."
        )
    if "scoreSets" not in value:
        raise CatalogError(
            "MaveDB search response is missing the 'scoreSets' field."
        )
    score_sets = value["scoreSets"]
    if not isinstance(score_sets, list):
        raise CatalogError(
            "MaveDB search response field 'scoreSets' must contain a list."
        )
    return score_sets


def _mavedb_record(value: object) -> DatasetRecord:
    """Map one MaveDB score-set object into the public catalog model."""
    if not isinstance(value, Mapping):
        raise CatalogError("MaveDB score-set metadata must be a JSON object.")

    metadata = normalize_raw_metadata(value)
    dataset_id = optional_text(metadata.get("urn"))
    if dataset_id is None:
        raise CatalogError("MaveDB score-set metadata is missing its URN.")

    return DatasetRecord(
        source="mavedb",
        dataset_id=dataset_id,
        title=optional_text(metadata.get("title")),
        target_id=_mavedb_target_id(metadata),
        variant_type=None,
        n_variants=optional_int(metadata.get("numVariants")),
        raw_metadata=metadata,
    )


def _mavedb_target_id(metadata: Mapping[str, Any]) -> str | None:
    """Return a target identifier only when MaveDB provides one unambiguously."""
    for field in ("targetId", "targetIdentifier"):
        target_id = optional_text(metadata.get(field))
        if target_id is not None:
            return target_id

    target_genes = metadata.get("targetGenes")
    if not isinstance(target_genes, list) or len(target_genes) != 1:
        return None
    target_gene = target_genes[0]
    if not isinstance(target_gene, Mapping):
        return None
    return optional_text(target_gene.get("name"))
