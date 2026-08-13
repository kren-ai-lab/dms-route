"""Shared dataset metadata and summary helpers for application workflows."""

from __future__ import annotations

from typing import Any

import pandas as pd

from dmsroute.core._wildtype import get_wt_resolution
from dmsroute.core.parsing import translate_dna


def _completed_dataset_counts(
    built_table: pd.DataFrame,
    *,
    raw_rows: int,
) -> dict[str, int]:
    """Return row accounting for one completed dataset."""
    status_ok = built_table["status"] == "OK"
    is_wildtype = built_table["is_wildtype"].eq(True)
    is_synthetic_wildtype = is_wildtype & built_table["is_synthetic"].eq(True)
    output_rows = len(built_table)
    synthetic_wildtype_rows = int(is_synthetic_wildtype.sum())
    source_output_rows = output_rows - synthetic_wildtype_rows

    return {
        "validated_rows": int(status_ok.sum()),
        "discarded_rows": raw_rows - source_output_rows,
        "output_rows": output_rows,
        "wildtype_rows": int((status_ok & is_wildtype).sum()),
        "synthetic_wildtype_rows": synthetic_wildtype_rows,
    }


def _metadata_text(row: pd.Series, *columns: str) -> str | None:
    """Return the first non-empty text value from source metadata columns."""
    for column in columns:
        if column not in row:
            continue
        value = row[column]
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _mavedb_uniprot_id(metadata: dict[str, Any]) -> str | None:
    """Return a UniProt identifier from the first current MaveDB target."""
    targets = metadata.get("targetGenes")
    if not isinstance(targets, list) or not targets:
        return None
    target = targets[0]
    if not isinstance(target, dict):
        return None
    identifiers = target.get("externalIdentifiers")
    if isinstance(identifiers, list):
        for external in identifiers:
            if not isinstance(external, dict):
                continue
            nested_identifier = external.get("identifier")
            identifier = (
                nested_identifier
                if isinstance(nested_identifier, dict)
                else external
            )
            db_name = str(identifier.get("dbName", "")).casefold()
            if db_name == "uniprot":
                value = (
                    identifier.get("identifier")
                    if identifier is not external
                    else nested_identifier
                )
                if isinstance(value, str) and value.strip():
                    return value.strip()
    mapped_identifier = target.get("uniprotIdFromMappedMetadata")
    if isinstance(mapped_identifier, str) and mapped_identifier.strip():
        return mapped_identifier.strip()
    return None


def _mavedb_gene(metadata: dict[str, Any]) -> str | None:
    """Return the non-empty gene name from the first MaveDB target."""
    targets = metadata.get("targetGenes")
    if not isinstance(targets, list) or not targets:
        return None
    target = targets[0]
    if not isinstance(target, dict):
        return None
    name = target.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    return name.strip()


def _mavedb_target_protein(
    metadata: dict[str, Any],
    gene: str | None,
) -> str:
    """Return the summary display target without fabricating gene metadata."""
    if gene is not None:
        return gene
    title = metadata.get("title")
    if isinstance(title, str) and title.strip():
        return title.split()[0]
    return "Unknown"


def _mavedb_wt_sequence_evidence(
    metadata: dict[str, Any],
) -> tuple[tuple[str, str], ...]:
    """Return only explicit MaveDB targetSequence sequence evidence."""
    hits: list[tuple[str, str]] = []

    def walk(value: Any, path: str = "root") -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                walk(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")
        elif isinstance(value, str) and path.lower().endswith(
            "targetsequence.sequence"
        ):
            hits.append((path, value.strip()))

    walk(metadata)
    evidence: list[tuple[str, str]] = []
    for path, sequence in hits:
        normalized = sequence.upper()
        if set(normalized) <= set("ACGTN"):
            normalized = translate_dna(
                normalized,
                frame=1,
                stop_at_stop=True,
            )
        evidence.append((f"mavedb_score_set_metadata:{path}", normalized))
    return tuple(evidence)


def _wt_summary_fields(
    built_table: pd.DataFrame,
    *,
    requested_transformation: str | None,
    transformed_output_column: str | None,
) -> dict[str, Any]:
    """Return stable WT provenance fields for one summary record."""
    resolution = get_wt_resolution(built_table)
    return {
        "wt_sequence_provenance": resolution.sequence_provenance,
        "wt_score": resolution.score,
        "wt_score_provenance": resolution.score_provenance,
        "observed_wildtype_row": resolution.observed_wildtype_row,
        "synthetic_wildtype_inserted": resolution.synthetic_wildtype_inserted,
        "requested_transformation": requested_transformation,
        "transformed_output_column": transformed_output_column,
        "wt_score_unavailable_reason": resolution.score_unavailable_reason,
    }
