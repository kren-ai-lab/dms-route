"""Configuration-driven orchestration for ProteinGym and MaveDB datasets."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from dms_parser._dataset import (
    _completed_dataset_counts,
    _mavedb_wt_sequence_evidence,
    _metadata_text,
    _mavedb_gene,
    _mavedb_target_protein,
    _mavedb_uniprot_id,
    _wt_summary_fields,
)
from dms_parser._wildtype import (
    replace_wt_sequence_provenance,
    resolve_wt_sequence,
)
from dms_parser.builders import build_mavedb_dataset, build_proteingym_dataset
from dms_parser.config import validate_pipeline_config
from dms_parser.downloads import (
    DatasetBatchDownloadEntry,
    DatasetBatchDownloadResult,
    DatasetDownloadResult,
    download_and_standardize_dataset,
    download_and_standardize_datasets,
)
from dms_parser.exceptions import (
    DMSParserError,
    DatasetNotFoundError,
    DownloadError,
    InvalidDatasetError,
    InvalidPipelineOptionError,
)
from dms_parser.io import download_file, read_table, write_table
from dms_parser.sources.mavedb_catalog import MAVEDB_API_URL
from dms_parser.sources.proteingym_resources import get_proteingym_resource

logger = logging.getLogger(__name__)

_PIPELINE_SOURCES = ("all", "proteingym", "mavedb")


@dataclass(frozen=True)
class PipelineResult:
    """Result returned by :func:`run_pipeline`."""

    summary: list[dict[str, Any]]
    summary_path: Path | None = None

    @property
    def has_errors(self) -> bool:
        """Return whether any dataset ended with an ``ERROR`` status."""
        return any(row.get("status") == "ERROR" for row in self.summary)

    @property
    def exit_code(self) -> int:
        """Return the process exit code suitable for a command-line wrapper."""
        return 1 if self.has_errors else 0


def run_pipeline(
    config: object,
    *,
    only: str = "all",
    dry_run: bool = False,
) -> PipelineResult:
    """Run configured sources and return their combined pipeline result.

    This function does not configure logging, parse command-line arguments, or
    terminate the process. Structural configuration and programmatic options
    are validated before source I/O or dataset directory creation.
    """
    if only not in _PIPELINE_SOURCES:
        choices = ", ".join(_PIPELINE_SOURCES)
        raise InvalidPipelineOptionError(
            f"only must be one of: {choices}."
        )
    if not isinstance(dry_run, bool):
        raise InvalidPipelineOptionError("dry_run must be a boolean.")

    validated = validate_pipeline_config(config)
    summary: list[dict[str, Any]] = []
    summary_path: Path | None = None

    try:
        if only in ("all", "proteingym") and "proteingym" in validated:
            summary.extend(
                process_proteingym(
                    validated["proteingym"],
                    dry_run=dry_run,
                )
            )

        if only in ("all", "mavedb") and "mavedb" in validated:
            summary.extend(
                process_mavedb(
                    validated["mavedb"],
                    dry_run=dry_run,
                )
            )
    finally:
        if summary:
            summary_path = write_summary(summary, validated.get("output", {}))
            logger.info("Summary saved at: %s", summary_path)
        log_summary(summary)

    return PipelineResult(summary=summary, summary_path=summary_path)


def process_proteingym(
    config: object,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Process configured ProteinGym datasets or return dry-run summaries."""
    validated = validate_pipeline_config({"proteingym": config})
    source_config = validated["proteingym"]
    resource = get_proteingym_resource(
        source_config["resource"],
        require_processing=True,
    )
    dir_base = Path(source_config["dir_base"])
    data_dir = dir_base / "raw"
    output_dir = dir_base / "processed"

    metadata_path = dir_base / resource.metadata_filename
    benchmark_path = dir_base / resource.data_filename
    logger.debug(
        "[proteingym] Resolved paths metadata=%s benchmark=%s output=%s dry_run=%s",
        metadata_path, benchmark_path, output_dir, dry_run,
    )

    if not dry_run:
        _ensure_dirs(data_dir, output_dir)

    logger.info("[proteingym] Downloading/verifying metadata...")
    download_file(resource.metadata_url, metadata_path, overwrite=False)

    if not dry_run:
        logger.info("[proteingym] Downloading/verifying benchmark data...")
        download_file(resource.data_url, benchmark_path, overwrite=False)

    metadata_table = read_table(metadata_path)
    benchmark_table: pd.DataFrame | None = None
    if not dry_run:
        benchmark_table = read_table(benchmark_path)
        logger.info(
            "[proteingym] Metadata: %d available experiments. "
            "Base table: %d mutations.",
            metadata_table.shape[0], benchmark_table.shape[0],
        )
    else:
        logger.info("[proteingym] Metadata: %d available experiments.", metadata_table.shape[0])

    default_build_kwargs = source_config.get("default_build_kwargs", {})
    entries = source_config.get("datasets", [])
    summary: list[dict[str, Any]] = []

    for index, entry in enumerate(entries):
        dataset_id = entry["dataset_id"]
        logger.info("[proteingym] %d/%d Processing: %s", index + 1, len(entries), dataset_id)
        row: dict[str, Any] = {
            "source": "proteingym",
            "input": dataset_id,
            "status": "ERROR",
        }

        try:
            metadata_matches = metadata_table[
                metadata_table["DMS_id"] == dataset_id
            ]
            if metadata_matches.empty:
                raise DatasetNotFoundError(
                    f"No information found in metadata for {dataset_id}."
                )

            selected_row = metadata_matches.iloc[0]
            resolved_dataset_id = selected_row["DMS_id"]
            metadata_sequence = _metadata_text(selected_row, "target_seq")
            automatic_sequence = (
                (("proteingym_reference_metadata", metadata_sequence),)
                if metadata_sequence is not None
                else ()
            )
            wt_sequence, wt_sequence_provenance = resolve_wt_sequence(
                automatic_sequence,
                entry.get("wt_sequence"),
                dataset_id=dataset_id,
            )
            uniprot_id = _metadata_text(selected_row, "UniProt_ID")
            protein_id = _metadata_text(selected_row, "molecule_name")
            gene = _metadata_text(selected_row, "gene", "Gene", "gene_name")
            source_filename = _metadata_text(selected_row, "DMS_filename")
            if source_filename is None:
                source_filename = f"{dataset_id}.csv"
            logger.debug(
                "[proteingym] Resolved WT source=reference_metadata dataset_id=%s length=%d",
                resolved_dataset_id, len(wt_sequence),
            )

            row.update(
                {
                    "dataset_id": resolved_dataset_id,
                    "target_protein": uniprot_id or protein_id or "Unknown",
                    "wt_length": len(wt_sequence),
                    "wt_sequence_sha256": hashlib.sha256(
                        wt_sequence.encode("utf-8")
                    ).hexdigest(),
                    "wt_sequence_provenance": wt_sequence_provenance,
                    "raw_rows": None,
                }
            )

            if dry_run:
                row["status"] = "DRY_RUN"
                summary.append(row)
                logger.info("[proteingym] (dry-run) metadata resolved for %s", resolved_dataset_id)
                continue

            if benchmark_table is None:
                raise RuntimeError("ProteinGym benchmark data was not loaded.")

            experiment_table = benchmark_table[
                benchmark_table["DMS_id"] == resolved_dataset_id
            ].copy()
            initial_rows = len(experiment_table)
            row["raw_rows"] = initial_rows
            raw_path = data_dir / source_filename

            build_kwargs = _deep_merge(
                default_build_kwargs,
                entry.get("build_kwargs", {}),
            )
            build_kwargs.setdefault("score_col", "DMS_score")
            build_kwargs.setdefault("variant_col", "mutant")
            build_kwargs.setdefault("add_relative_score", False)
            build_kwargs.setdefault("add_binary_label", False)
            build_kwargs.setdefault("add_wildtype_row", False)
            build_kwargs.setdefault("drop_failed", False)
            if entry.get("wt_score") is not None:
                build_kwargs["wt_score"] = entry["wt_score"]
            build_kwargs["dataset_id"] = resolved_dataset_id
            build_kwargs["protein_id"] = protein_id
            build_kwargs["gene"] = gene
            build_kwargs["uniprot_id"] = uniprot_id
            logger.debug(
                "[proteingym] Detected columns dataset_id=%s variant_col=%s "
                "score_col=%s",
                resolved_dataset_id, build_kwargs["variant_col"], build_kwargs["score_col"],
            )

            write_table(experiment_table, raw_path, index=False)
            built_table = build_proteingym_dataset(
                input_path=raw_path,
                wt_sequence=wt_sequence,
                **build_kwargs,
            )
            replace_wt_sequence_provenance(
                built_table,
                wt_sequence_provenance,
            )
            counts = _completed_dataset_counts(built_table, raw_rows=initial_rows)

            output_file = (
                output_dir / f"{resolved_dataset_id}_processed.csv"
            )
            built_table.to_csv(output_file, index=False)
            logger.debug("[proteingym] Resolved output dataset_id=%s path=%s", resolved_dataset_id, output_file)

            row.update(
                {
                    "status": "OK",
                    **_wt_summary_fields(
                        built_table,
                        requested_transformation=(
                            build_kwargs.get("relative_method", "log_ratio")
                            if build_kwargs["add_relative_score"]
                            else None
                        ),
                        transformed_output_column=(
                            build_kwargs.get(
                                "relative_output_col",
                                "score_log_ratio",
                            )
                            if build_kwargs["add_relative_score"]
                            else None
                        ),
                    ),
                    **counts,
                    "output_file": str(output_file),
                }
            )
            logger.info("[proteingym] Complete. Retained rows: %d/%d", counts["output_rows"], initial_rows)
        except (DMSParserError, requests.RequestException, OSError) as exc:
            row["error"] = str(exc)
            logger.exception("[proteingym] Error processing %s", dataset_id)

        summary.append(row)

    return summary


def process_mavedb(
    config: object,
    dry_run: bool = False,
    *,
    base_url: str = MAVEDB_API_URL,
) -> list[dict[str, Any]]:
    """Process configured MaveDB datasets or return dry-run summaries."""
    validated = validate_pipeline_config({"mavedb": config})
    source_config = validated["mavedb"]
    dir_base = Path(source_config["dir_base"])
    data_dir = dir_base / "raw"
    output_dir = dir_base / "processed"
    logger.debug("[mavedb] Resolved paths raw=%s output=%s dry_run=%s", data_dir, output_dir, dry_run)
    if not dry_run:
        _ensure_dirs(data_dir, output_dir)

    resolved_base_url = base_url.rstrip("/")
    default_build_kwargs = source_config.get("default_build_kwargs", {})
    entries = source_config.get("datasets", [])
    summary: list[dict[str, Any]] = []

    for index, entry in enumerate(entries):
        dataset_id = entry["dataset_id"]
        logger.info("[mavedb] %d/%d Processing URN: %s", index + 1, len(entries), dataset_id)
        row: dict[str, Any] = {
            "source": "mavedb",
            "input": dataset_id,
            "status": "ERROR",
        }

        try:
            metadata_response = requests.get(
                f"{resolved_base_url}/score-sets/{dataset_id}",
                timeout=60,
            )
            if metadata_response.status_code != 200:
                raise DownloadError(
                    "Error while downloading metadata "
                    f"(HTTP {metadata_response.status_code})."
                )
            metadata = metadata_response.json()

            gene = _mavedb_gene(metadata)
            target_name = _mavedb_target_protein(metadata, gene)
            uniprot_id = _mavedb_uniprot_id(metadata)
            wt_sequence, wt_sequence_provenance = resolve_wt_sequence(
                _mavedb_wt_sequence_evidence(metadata),
                entry.get("wt_sequence"),
                dataset_id=dataset_id,
            )
            if wt_sequence_provenance.startswith("mavedb_score_set_metadata:"):
                wt_sequence_provenance = "mavedb_score_set_metadata"
            logger.debug(
                "[mavedb] Resolved WT source=score_set_metadata dataset_id=%s length=%d",
                dataset_id, len(wt_sequence),
            )

            row.update(
                {
                    "dataset_id": dataset_id,
                    "target_protein": target_name,
                    "wt_length": len(wt_sequence),
                    "wt_sequence_sha256": hashlib.sha256(
                        wt_sequence.encode("utf-8")
                    ).hexdigest(),
                    "wt_sequence_provenance": wt_sequence_provenance,
                    "raw_rows": None,
                }
            )

            if dry_run:
                row["status"] = "DRY_RUN"
                summary.append(row)
                logger.info("[mavedb] (dry-run) metadata resolved for %s", dataset_id)
                continue

            scores_response = requests.get(
                f"{resolved_base_url}/score-sets/{dataset_id}/scores",
                timeout=60,
            )
            scores_response.raise_for_status()
            scores_path = (
                data_dir / f"{dataset_id.replace(':', '_')}_scores.csv"
            )
            scores_path.write_text(scores_response.text, encoding="utf-8")

            raw_table = pd.read_csv(scores_path)
            initial_rows = len(raw_table)
            hgvs_col = entry.get("hgvs_col") or next(
                (column for column in ["hgvs_pro"] if column in raw_table.columns),
                None,
            )
            score_col = entry.get("score_col") or next(
                (
                    column
                    for column in [
                        "score",
                        "scores.score",
                        "DMS_score",
                        "fitness",
                    ]
                    if column in raw_table.columns
                ),
                None,
            )
            if not hgvs_col or not score_col:
                raise InvalidDatasetError(
                    "Neither score nor HGVS columns were detected."
                )
            logger.debug(
                "[mavedb] Detected columns dataset_id=%s variant_col=%s score_col=%s",
                dataset_id, hgvs_col, score_col,
            )
            row["raw_rows"] = initial_rows

            build_kwargs = _deep_merge(
                default_build_kwargs,
                entry.get("build_kwargs", {}),
            )
            build_kwargs.setdefault("add_relative_score", False)
            build_kwargs.setdefault("add_binary_label", False)
            build_kwargs.setdefault("add_wildtype_row", False)
            build_kwargs.setdefault("drop_failed", False)
            if entry.get("wt_score") is not None:
                build_kwargs["wt_score"] = entry["wt_score"]
            build_kwargs["score_col"] = score_col
            build_kwargs["hgvs_col"] = hgvs_col
            build_kwargs["dataset_id"] = dataset_id
            build_kwargs["protein_id"] = None
            build_kwargs["gene"] = gene
            build_kwargs["uniprot_id"] = uniprot_id

            built_table = build_mavedb_dataset(
                input_path=scores_path,
                wt_sequence=wt_sequence,
                **build_kwargs,
            )
            replace_wt_sequence_provenance(
                built_table,
                wt_sequence_provenance,
            )
            counts = _completed_dataset_counts(built_table, raw_rows=initial_rows)

            output_file = (
                output_dir / f"{dataset_id.replace(':', '_')}_processed.csv"
            )
            built_table.to_csv(output_file, index=False)
            logger.debug("[mavedb] Resolved output dataset_id=%s path=%s", dataset_id, output_file)

            row.update(
                {
                    "status": "OK",
                    **_wt_summary_fields(
                        built_table,
                        requested_transformation=(
                            build_kwargs.get("relative_method", "log_ratio")
                            if build_kwargs["add_relative_score"]
                            else None
                        ),
                        transformed_output_column=(
                            build_kwargs.get(
                                "relative_output_col",
                                "score_log_ratio",
                            )
                            if build_kwargs["add_relative_score"]
                            else None
                        ),
                    ),
                    **counts,
                    "output_file": str(output_file),
                }
            )
            logger.info("[mavedb] Complete. Retained: %d/%d", counts["output_rows"], initial_rows)
        except (DMSParserError, requests.RequestException, OSError) as exc:
            row["error"] = str(exc)
            logger.exception("[mavedb] Error processing %s", dataset_id)

        summary.append(row)

    return summary


def write_summary(
    summary: list[dict[str, Any]],
    output_config: object,
) -> Path:
    """Write timestamped CSV and JSON summaries and return the CSV path."""
    if not isinstance(output_config, dict):
        raise TypeError("output_config must be a mapping.")
    summary_dir = Path(
        output_config.get("summary_dir", "datasets/summaries")
    )
    _ensure_dirs(summary_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    summary_table = pd.DataFrame(summary)
    csv_path = summary_dir / f"summary_{timestamp}.csv"
    json_path = summary_dir / f"summary_{timestamp}.json"
    summary_table.to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            summary,
            handle,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    return csv_path


def log_summary(summary: list[dict[str, Any]]) -> None:
    """Log the final combined pipeline summary."""
    if not summary:
        logger.warning("No dataset processed.")
        return

    ok_count = sum(1 for row in summary if row["status"] == "OK")
    error_count = sum(1 for row in summary if row["status"] == "ERROR")
    dry_run_count = sum(1 for row in summary if row["status"] == "DRY_RUN")

    logger.info("=" * 60)
    logger.info("FINAL SUMMARY: %d processed datasets", len(summary))
    logger.info("  OK: %d | ERROR: %d | DRY_RUN: %d", ok_count, error_count, dry_run_count)
    for row in summary:
        if row["status"] == "OK":
            logger.info(
                "  [OK]    %-12s %-30s %s/%s rows",
                row["source"], row.get("dataset_id", row["input"]),
                row.get("validated_rows"), row.get("raw_rows"),
            )
        elif row["status"] == "ERROR":
            logger.info("  [ERROR] %-12s %-30s %s", row["source"], row["input"], row.get("error", ""))
    logger.info("=" * 60)


def _deep_merge(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged

def _ensure_dirs(*directories: Path) -> None:
    """Create output directories when pipeline execution requires them."""
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
