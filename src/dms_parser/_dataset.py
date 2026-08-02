"""Shared dataset metadata and summary helpers for application workflows."""

from __future__ import annotations

from typing import Any

import pandas as pd

from dms_parser.parsing import translate_dna


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


def _extract_wt_from_metadata(metadata: dict[str, Any]) -> str | None:
    """Resolve a MaveDB WT sequence from score-set metadata."""
    hits: list[tuple[str, str]] = []

    def walk(value: Any, path: str = "root") -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                walk(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")
        elif isinstance(value, str):
            hits.append((path, value.strip()))

    walk(metadata)
    sequence_fields = [
        (path, sequence)
        for path, sequence in hits
        if "sequence" in path.lower()
    ]

    for path, sequence in sequence_fields:
        normalized = sequence.upper()
        if path.lower().endswith("targetsequence.sequence"):
            if set(normalized) <= set("ACGTN"):
                return translate_dna(
                    normalized,
                    frame=1,
                    stop_at_stop=True,
                )
            return normalized

    for _, sequence in sequence_fields:
        normalized = sequence.upper()
        if set(normalized) <= set("ACDEFGHIKLMNPQRSTVWYBXZJUO*"):
            return normalized
        if set(normalized) <= set("ACGTN"):
            return translate_dna(
                normalized,
                frame=1,
                stop_at_stop=True,
            )
    return None
