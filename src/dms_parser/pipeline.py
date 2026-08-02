"""Configuration-driven orchestration for ProteinGym and MaveDB datasets."""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests

from dms_parser.builders import build_mavedb_dataset, build_proteingym_dataset
from dms_parser.cache import FilesystemCache
from dms_parser.catalog import DatasetRecord, get_dataset_metadata
from dms_parser.config import validate_pipeline_config, validate_source_dataset_id
from dms_parser.exceptions import (
    DatasetNotFoundError,
    InvalidDatasetError,
    InvalidPipelineOptionError,
    MissingWildTypeError,
)
from dms_parser.io import (
    _preflight_dataset_bundle,
    _publish_dataset_bundle,
    download_file,
    read_table,
    write_table,
)
from dms_parser.parsing import translate_dna
from dms_parser.sources.mavedb import download_mavedb_dataset
from dms_parser.sources.mavedb_catalog import MAVEDB_API_URL
from dms_parser.sources.proteingym import download_proteingym_dataset
from dms_parser.sources.proteingym_resources import get_proteingym_resource

logger = logging.getLogger(__name__)

_PIPELINE_SOURCES = ("all", "proteingym", "mavedb")
_PROTEINGYM_SUBSTITUTIONS_BENCHMARK_CACHE_ID = (
    "resource-data-dms-substitutions"
)


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


@dataclass(frozen=True)
class DatasetDownloadResult:
    """Stable output paths and summary for one standardized download."""

    dataset_path: Path
    summary_csv_path: Path
    summary_json_path: Path
    summary: dict[str, Any]


def download_and_standardize_dataset(
    source: str,
    dataset_id: str,
    *,
    output_dir: str | Path,
    cache: FilesystemCache,
    refresh: bool = False,
    drop_failed: bool = False,
    add_wildtype_row: bool = False,
    overwrite: bool = False,
) -> DatasetDownloadResult:
    """Download and standardize one substitutions dataset without YAML config."""
    validate_source_dataset_id(source, dataset_id)
    if not isinstance(output_dir, (str, Path)) or not str(output_dir).strip():
        raise InvalidPipelineOptionError("output_dir must be a non-empty path.")
    if not isinstance(cache, FilesystemCache):
        raise InvalidPipelineOptionError("cache must be a FilesystemCache.")
    for option_name, option_value in (
        ("refresh", refresh),
        ("drop_failed", drop_failed),
        ("add_wildtype_row", add_wildtype_row),
        ("overwrite", overwrite),
    ):
        if not isinstance(option_value, bool):
            raise InvalidPipelineOptionError(
                f"{option_name} must be a boolean."
            )

    resolved_output_dir = Path(output_dir).expanduser()
    dataset_path, summary_csv_path, summary_json_path = (
        _preflight_dataset_bundle(
            resolved_output_dir,
            overwrite=overwrite,
        )
    )

    if source == "proteingym":
        built_table, summary = _download_proteingym_dataset(
            dataset_id,
            cache=cache,
            refresh=refresh,
            drop_failed=drop_failed,
            add_wildtype_row=add_wildtype_row,
        )
    else:
        built_table, summary = _download_mavedb_dataset(
            dataset_id,
            cache=cache,
            refresh=refresh,
            drop_failed=drop_failed,
            add_wildtype_row=add_wildtype_row,
        )

    summary["output_file"] = str(dataset_path)
    _publish_dataset_bundle(
        built_table,
        summary,
        resolved_output_dir,
        overwrite=overwrite,
    )
    logger.info("Standardized dataset saved at: %s", dataset_path)
    logger.info("CSV summary saved at: %s", summary_csv_path)
    logger.info("JSON summary saved at: %s", summary_json_path)
    return DatasetDownloadResult(
        dataset_path=dataset_path,
        summary_csv_path=summary_csv_path,
        summary_json_path=summary_json_path,
        summary=summary,
    )


def _download_proteingym_dataset(
    dataset_id: str,
    *,
    cache: FilesystemCache,
    refresh: bool,
    drop_failed: bool,
    add_wildtype_row: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Acquire and build one ProteinGym substitutions assay."""
    metadata_record = get_dataset_metadata(
        "proteingym",
        dataset_id,
        variant_type="substitutions",
        cache=cache,
        refresh=refresh,
    )
    metadata = pd.Series(metadata_record.raw_metadata)
    wt_sequence = _metadata_text(metadata, "target_seq")
    if wt_sequence is None:
        raise MissingWildTypeError(
            f"No WT sequence found in ProteinGym metadata for {dataset_id}."
        )

    resource = get_proteingym_resource(
        "dms_substitutions",
        require_processing=True,
    )
    benchmark_path = download_proteingym_dataset(
        resource.data_url,
        cache=cache,
        dataset_id=_PROTEINGYM_SUBSTITUTIONS_BENCHMARK_CACHE_ID,
        refresh=refresh,
    )
    benchmark_table = read_table(benchmark_path)
    if "DMS_id" not in benchmark_table.columns:
        raise InvalidDatasetError(
            "ProteinGym substitutions benchmark is missing the 'DMS_id' column."
        )
    experiment_table = benchmark_table[
        benchmark_table["DMS_id"] == dataset_id
    ].copy()
    if experiment_table.empty:
        raise DatasetNotFoundError(
            f"ProteinGym dataset {dataset_id!r} was not found in the benchmark."
        )

    protein_id = _metadata_text(metadata, "molecule_name")
    gene = _metadata_text(metadata, "gene", "Gene", "gene_name")
    uniprot_id = _metadata_text(metadata, "UniProt_ID")
    with TemporaryDirectory(prefix="dms-parser-download-") as temporary_dir:
        selected_path = Path(temporary_dir) / "selected.csv"
        write_table(experiment_table, selected_path, index=False)
        built_table = build_proteingym_dataset(
            input_path=selected_path,
            score_col="DMS_score",
            variant_col="mutant",
            dataset_id=dataset_id,
            protein_id=protein_id,
            gene=gene,
            uniprot_id=uniprot_id,
            wt_sequence=wt_sequence,
            add_relative_score=False,
            add_binary_label=False,
            add_wildtype_row=add_wildtype_row,
            drop_failed=drop_failed,
            validate_output=True,
            require_wt_for_transforms=False,
        )

    return built_table, _successful_download_summary(
        source="proteingym",
        dataset_id=dataset_id,
        target_protein=uniprot_id or protein_id or "Unknown",
        wt_sequence=wt_sequence,
        raw_rows=len(experiment_table),
        built_table=built_table,
    )


def _download_mavedb_dataset(
    dataset_id: str,
    *,
    cache: FilesystemCache,
    refresh: bool,
    drop_failed: bool,
    add_wildtype_row: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Acquire and build one MaveDB substitutions score set."""
    metadata_record = _get_mavedb_download_metadata(dataset_id)
    metadata = metadata_record.raw_metadata
    wt_sequence = _extract_wt_from_metadata(metadata)
    if wt_sequence is None:
        raise MissingWildTypeError(
            f"No WT sequence found in MaveDB metadata for {dataset_id}."
        )

    encoded_id = quote(dataset_id, safe=":")
    scores_url = (
        f"{MAVEDB_API_URL.rstrip('/')}/score-sets/{encoded_id}/scores"
    )
    scores_path = download_mavedb_dataset(
        scores_url,
        cache=cache,
        dataset_id=dataset_id,
        refresh=refresh,
    )
    raw_table = read_table(scores_path)
    hgvs_col = next(
        (column for column in ("hgvs_pro",) if column in raw_table.columns),
        None,
    )
    score_col = next(
        (
            column
            for column in ("score", "DMS_score", "fitness")
            if column in raw_table.columns
        ),
        None,
    )
    if hgvs_col is None or score_col is None:
        raise InvalidDatasetError(
            "Neither score nor HGVS columns were detected in the MaveDB table."
        )

    gene = _mavedb_gene(metadata)
    uniprot_id = _mavedb_uniprot_id(metadata)
    built_table = build_mavedb_dataset(
        input_path=scores_path,
        score_col=score_col,
        hgvs_col=hgvs_col,
        dataset_id=dataset_id,
        protein_id=None,
        gene=gene,
        uniprot_id=uniprot_id,
        wt_sequence=wt_sequence,
        add_relative_score=False,
        add_binary_label=False,
        add_wildtype_row=add_wildtype_row,
        drop_failed=drop_failed,
        validate_output=True,
        require_wt_for_transforms=False,
    )
    return built_table, _successful_download_summary(
        source="mavedb",
        dataset_id=dataset_id,
        target_protein=_mavedb_target_protein(metadata, gene),
        wt_sequence=wt_sequence,
        raw_rows=len(raw_table),
        built_table=built_table,
    )


def _get_mavedb_download_metadata(dataset_id: str) -> DatasetRecord:
    """Retrieve MaveDB metadata and normalize an HTTP 404 as not found."""
    try:
        return get_dataset_metadata("mavedb", dataset_id)
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            raise DatasetNotFoundError(
                f"MaveDB dataset {dataset_id!r} was not found."
            ) from exc
        raise


def _successful_download_summary(
    *,
    source: str,
    dataset_id: str,
    target_protein: str,
    wt_sequence: str,
    raw_rows: int,
    built_table: pd.DataFrame,
) -> dict[str, Any]:
    """Build the existing successful summary representation for one dataset."""
    return {
        "source": source,
        "input": dataset_id,
        "status": "OK",
        "dataset_id": dataset_id,
        "target_protein": target_protein,
        "wt_length": len(wt_sequence),
        "raw_rows": raw_rows,
        **_completed_dataset_counts(built_table, raw_rows=raw_rows),
    }


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
                raise ValueError(
                    f"No information found in metadata for {dataset_id}."
                )

            selected_row = metadata_matches.iloc[0]
            resolved_dataset_id = selected_row["DMS_id"]
            wt_sequence = selected_row["target_seq"]
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
            counts = _completed_dataset_counts(built_table, raw_rows=initial_rows)

            output_file = (
                output_dir / f"{resolved_dataset_id}_processed.csv"
            )
            built_table.to_csv(output_file, index=False)
            logger.debug("[proteingym] Resolved output dataset_id=%s path=%s", resolved_dataset_id, output_file)

            row.update(
                {
                    "status": "OK",
                    **counts,
                    "output_file": str(output_file),
                }
            )
            logger.info("[proteingym] Complete. Retained rows: %d/%d", counts["output_rows"], initial_rows)
        except Exception as exc:  # noqa: BLE001 - isolate dataset failures
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
                raise ValueError(
                    "Error while downloading metadata "
                    f"(HTTP {metadata_response.status_code})."
                )
            metadata = metadata_response.json()

            gene = _mavedb_gene(metadata)
            target_name = _mavedb_target_protein(metadata, gene)
            uniprot_id = _mavedb_uniprot_id(metadata)
            wt_sequence = _extract_wt_from_metadata(metadata)
            if wt_sequence is None:
                raise ValueError("No WT found in metadata.")
            logger.debug(
                "[mavedb] Resolved WT source=score_set_metadata dataset_id=%s length=%d",
                dataset_id, len(wt_sequence),
            )

            row.update(
                {
                    "dataset_id": dataset_id,
                    "target_protein": target_name,
                    "wt_length": len(wt_sequence),
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
                    for column in ["score", "DMS_score", "fitness"]
                    if column in raw_table.columns
                ),
                None,
            )
            if not hgvs_col or not score_col:
                raise ValueError("Neither score nor HGVS columns were detected.")
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
            counts = _completed_dataset_counts(built_table, raw_rows=initial_rows)

            output_file = (
                output_dir / f"{dataset_id.replace(':', '_')}_processed.csv"
            )
            built_table.to_csv(output_file, index=False)
            logger.debug("[mavedb] Resolved output dataset_id=%s path=%s", dataset_id, output_file)

            row.update(
                {
                    "status": "OK",
                    **counts,
                    "output_file": str(output_file),
                }
            )
            logger.info("[mavedb] Complete. Retained: %d/%d", counts["output_rows"], initial_rows)
        except Exception as exc:  # noqa: BLE001 - isolate dataset failures
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


def _ensure_dirs(*directories: Path) -> None:
    """Create output directories when pipeline execution requires them."""
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


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
