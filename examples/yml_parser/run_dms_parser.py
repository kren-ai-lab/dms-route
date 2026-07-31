#!/usr/bin/env python3
"""
run_dms_parser.py
====================

Configuration-driven console runner for processing ProteinGym and MaveDB
datasets with `dms_parser`.

Usage
-----
    python run_dms_parser.py --config config.yml
    python run_dms_parser.py --config config.yml --only proteingym
    python run_dms_parser.py --config config.yml --only mavedb --dry-run

Design notes
------------
- All dataset selection, paths and `build_*_dataset(...)` keyword arguments
  live in the YAML config.
- Each source section supports a `default_build_kwargs` block that is
  deep-merged with an optional per-dataset `build_kwargs` override, to 
  process many datasets with shared settings while tweaking a handful.
- A single failing dataset never aborts the run. Errors are caught, logged,
  and recorded in the summary with an "ERROR" status.
- A combined summary (one row per dataset, across both sources) is written
  as CSV/JSON at the end, mirroring the `processing_summary` list from the
  original notebooks.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml

from dms_parser import (
    SourceConfigurationError,
    build_mavedb_dataset,
    build_proteingym_dataset,
    get_proteingym_resource,
    read_table,
    translate_dna,
    write_table,
)
from dms_parser.io import download_file
from dms_parser.sources.mavedb_catalog import MAVEDB_API_URL
from dms_parser.sources.proteingym_resources import ProteinGymResource

logger = logging.getLogger("dms_parser.example_runner")

_MAVEDB_SCORE_SET_URN_PATTERN = re.compile(
    r"urn:mavedb:[0-9]{8}-(?:[a-z]+|0)-[1-9][0-9]*"
)


# --------------------------------------------------------------------------- #
# Helpers shared by both sources
# --------------------------------------------------------------------------- #

def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into a copy of `base` (override wins)."""
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def validate_config(config: object) -> dict[str, Any]:
    """Validate the example runner configuration without performing I/O."""
    if not isinstance(config, dict):
        raise SourceConfigurationError(
            "The configuration root must be a YAML mapping."
        )
    if "proteingym" in config:
        validate_proteingym_config(config["proteingym"])
    if "mavedb" in config:
        validate_mavedb_config(config["mavedb"])
    output = config.get("output")
    if output is not None and not isinstance(output, dict):
        raise SourceConfigurationError("'output' must be a mapping when provided.")
    return config


def validate_proteingym_config(config: object) -> ProteinGymResource:
    """Validate ProteinGym configuration and return its selected resource."""
    cfg = _validate_source_mapping(config, "proteingym")
    _reject_legacy_keys(
        cfg,
        "proteingym",
        {"enabled", "metadata_url", "benchmark_url"},
    )
    _validate_dir_base(cfg, "proteingym")
    if "resource" not in cfg:
        raise SourceConfigurationError(
            "ProteinGym configuration requires 'resource'."
        )
    resource_id = cfg["resource"]
    if not isinstance(resource_id, str) or not resource_id.strip():
        raise SourceConfigurationError(
            "ProteinGym 'resource' must be a non-empty string."
        )
    resource = get_proteingym_resource(
        resource_id,
        require_processing=True,
    )
    _validate_dataset_entries(cfg, "proteingym")
    return resource


def validate_mavedb_config(config: object) -> None:
    """Validate MaveDB configuration and canonical score-set identifiers."""
    cfg = _validate_source_mapping(config, "mavedb")
    _reject_legacy_keys(cfg, "mavedb", {"enabled", "base_url"})
    _validate_dir_base(cfg, "mavedb")
    entries = _validate_dataset_entries(cfg, "mavedb")
    for entry in entries:
        dataset_id = entry["dataset_id"]
        if _MAVEDB_SCORE_SET_URN_PATTERN.fullmatch(dataset_id) is None:
            raise SourceConfigurationError(
                "MaveDB dataset_id must be a complete permanent score-set URN "
                "matching 'urn:mavedb:<8 digits>-<lowercase experiment letters "
                "or 0>-<positive index>'."
            )


def _validate_source_mapping(
    config: object,
    source: str,
) -> dict[str, Any]:
    """Return a validated source mapping."""
    if not isinstance(config, dict):
        raise SourceConfigurationError(
            f"The {source!r} source section must be a mapping."
        )
    return config


def _reject_legacy_keys(
    config: dict[str, Any],
    source: str,
    keys: set[str],
) -> None:
    """Reject legacy source keys that are no longer part of the YAML contract."""
    present = sorted(keys.intersection(config))
    if present:
        joined = ", ".join(present)
        raise SourceConfigurationError(
            f"The {source!r} source section contains unsupported legacy "
            f"configuration keys: {joined}."
        )


def _validate_dir_base(config: dict[str, Any], source: str) -> None:
    """Validate a source output base directory."""
    if "dir_base" not in config:
        raise SourceConfigurationError(
            f"The {source!r} source section requires 'dir_base'."
        )
    value = config["dir_base"]
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise SourceConfigurationError(
            f"The {source!r} 'dir_base' must be a non-empty path."
        )


def _validate_dataset_entries(
    config: dict[str, Any],
    source: str,
) -> list[dict[str, Any]]:
    """Validate canonical dataset entries and reject duplicates."""
    if "datasets" not in config:
        raise SourceConfigurationError(
            f"The {source!r} source section requires 'datasets'."
        )
    entries = config["datasets"]
    if not isinstance(entries, list) or not entries:
        raise SourceConfigurationError(
            f"The {source!r} 'datasets' value must be a non-empty list."
        )
    default_build_kwargs = config.get("default_build_kwargs", {})
    if not isinstance(default_build_kwargs, dict):
        raise SourceConfigurationError(
            f"The {source!r} 'default_build_kwargs' value must be a mapping."
        )

    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise SourceConfigurationError(
                f"The {source!r} dataset entry at index {index} must be a mapping."
            )
        legacy_key = "filename" if source == "proteingym" else "urn"
        if legacy_key in entry:
            raise SourceConfigurationError(
                f"The {source!r} dataset key {legacy_key!r} is unsupported; "
                "use 'dataset_id'."
            )
        if "dataset_id" not in entry:
            raise SourceConfigurationError(
                f"The {source!r} dataset entry at index {index} requires "
                "'dataset_id'."
            )
        dataset_id = entry["dataset_id"]
        if not isinstance(dataset_id, str) or not dataset_id.strip():
            raise SourceConfigurationError(
                f"The {source!r} dataset_id at index {index} must be a "
                "non-empty string."
            )
        if source == "proteingym" and dataset_id.lower().endswith(".csv"):
            raise SourceConfigurationError(
                "ProteinGym dataset_id must be the canonical DMS_id, not a "
                "source filename."
            )
        if dataset_id in seen:
            raise SourceConfigurationError(
                f"Duplicate {source} dataset_id {dataset_id!r}."
            )
        build_kwargs = entry.get("build_kwargs", {})
        if not isinstance(build_kwargs, dict):
            raise SourceConfigurationError(
                f"The {source!r} build_kwargs at index {index} must be a mapping."
            )
        seen.add(dataset_id)
        validated.append(entry)
    return validated


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


def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def extract_wt_from_metadata(metadata: dict) -> str | None:
    """
    Walk a MaveDB score-set metadata JSON looking for any string field whose
    key path contains 'sequence'. Prefers `targetSequence.sequence`, and
    translates DNA to protein when needed.
    """
    hits: list[tuple[str, str]] = []

    def _walk(x: Any, path: str = "root") -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                _walk(v, f"{path}.{k}")
        elif isinstance(x, list):
            for i, v in enumerate(x):
                _walk(v, f"{path}[{i}]")
        elif isinstance(x, str):
            hits.append((path, x.strip()))

    _walk(metadata)
    sequence_fields = [(p, v) for p, v in hits if "sequence" in p.lower()]

    for path, seq in sequence_fields:
        seq_clean = seq.upper()
        if path.lower().endswith("targetsequence.sequence"):
            if set(seq_clean) <= set("ACGTN"):
                return translate_dna(seq_clean, frame=1, stop_at_stop=True)
            return seq_clean

    for path, seq in sequence_fields:
        seq_clean = seq.upper()
        if set(seq_clean) <= set("ACDEFGHIKLMNPQRSTVWYBXZJUO*"):
            return seq_clean
        if set(seq_clean) <= set("ACGTN"):
            return translate_dna(seq_clean, frame=1, stop_at_stop=True)

    return None


# --------------------------------------------------------------------------- #
# ProteinGym source
# --------------------------------------------------------------------------- #

def process_proteingym(cfg: dict, dry_run: bool = False) -> list[dict]:
    """Process configured ProteinGym datasets or return metadata-only summaries."""
    resource = validate_proteingym_config(cfg)
    dir_base = Path(cfg["dir_base"])
    data_dir = dir_base / "raw"
    output_dir = dir_base / "processed"

    metadata_path = dir_base / resource.metadata_filename
    benchmark_path = dir_base / resource.data_filename
    logger.debug(
        "[proteingym] Resolved paths metadata=%s benchmark=%s output=%s "
        "dry_run=%s",
        metadata_path,
        benchmark_path,
        output_dir,
        dry_run,
    )

    if not dry_run:
        ensure_dirs(data_dir, output_dir)

    logger.info("[proteingym] Downloading/verifying metadata...")
    download_file(resource.metadata_url, metadata_path, overwrite=False)

    if not dry_run:
        logger.info("[proteingym] Downloading/verifying benchmark data...")
        download_file(resource.data_url, benchmark_path, overwrite=False)

    df_meta = read_table(metadata_path)
    df_all: pd.DataFrame | None = None
    if not dry_run:
        df_all = read_table(benchmark_path)
        logger.info(
            "[proteingym] Metadata: %d available experiments. Base table: %d mutations.",
            df_meta.shape[0],
            df_all.shape[0],
        )
    else:
        logger.info(
            "[proteingym] Metadata: %d available experiments.",
            df_meta.shape[0],
        )

    default_build_kwargs = cfg.get("default_build_kwargs", {})
    entries = cfg.get("datasets", [])
    summary: list[dict] = []

    for i, entry in enumerate(entries):
        dataset_id = entry["dataset_id"]
        logger.info(
            "[proteingym] %d/%d Processing: %s",
            i + 1,
            len(entries),
            dataset_id,
        )

        row = {
            "source": "proteingym",
            "input": dataset_id,
            "status": "ERROR",
        }

        try:
            meta_match = df_meta[df_meta["DMS_id"] == dataset_id]
            if meta_match.empty:
                raise ValueError(
                    f"No information found in metadata for {dataset_id}."
                )

            selected_row = meta_match.iloc[0]
            dms_id = selected_row["DMS_id"]
            wt_sequence = selected_row["target_seq"]
            uniprot_id = _metadata_text(selected_row, "UniProt_ID")
            protein_id = _metadata_text(selected_row, "molecule_name")
            gene = _metadata_text(selected_row, "gene", "Gene", "gene_name")
            source_filename = _metadata_text(selected_row, "DMS_filename")
            if source_filename is None:
                source_filename = f"{dataset_id}.csv"
            logger.debug(
                "[proteingym] Resolved WT source=reference_metadata "
                "dataset_id=%s length=%d",
                dms_id,
                len(wt_sequence),
            )

            row.update(
                {
                    "dataset_id": dms_id,
                    "target_protein": uniprot_id or protein_id or "Unknown",
                    "wt_length": len(wt_sequence),
                    "raw_rows": None,
                }
            )

            if dry_run:
                row["status"] = "DRY_RUN"
                summary.append(row)
                logger.info("[proteingym] (dry-run) metadata resolved for %s", dms_id)
                continue

            if df_all is None:
                raise RuntimeError("ProteinGym benchmark data was not loaded.")

            df_experiment = df_all[df_all["DMS_id"] == dms_id].copy()
            initial_rows = len(df_experiment)
            row["raw_rows"] = initial_rows

            temp_raw_path = data_dir / source_filename

            build_kwargs = deep_merge(default_build_kwargs, entry.get("build_kwargs", {}))
            build_kwargs.setdefault("score_col", "DMS_score")
            build_kwargs.setdefault("variant_col", "mutant")
            build_kwargs.setdefault("add_relative_score", False)
            build_kwargs.setdefault("add_binary_label", False)
            build_kwargs.setdefault("drop_failed", False)
            build_kwargs["dataset_id"] = dms_id
            build_kwargs["protein_id"] = protein_id
            build_kwargs["gene"] = gene
            build_kwargs["uniprot_id"] = uniprot_id
            logger.debug(
                "[proteingym] Detected columns dataset_id=%s variant_col=%s "
                "score_col=%s",
                dms_id,
                build_kwargs["variant_col"],
                build_kwargs["score_col"],
            )

            write_table(df_experiment, temp_raw_path, index=False)

            df_built = build_proteingym_dataset(
                input_path=temp_raw_path,
                wt_sequence=wt_sequence,
                **build_kwargs,
            )

            saved_rows = len(df_built)
            validated_rows = int((df_built["status"] == "OK").sum())

            output_file = output_dir / f"{dms_id}_processed.csv"
            df_built.to_csv(output_file, index=False)
            logger.debug(
                "[proteingym] Resolved output dataset_id=%s path=%s",
                dms_id,
                output_file,
            )

            row.update(
                {
                    "status": "OK",
                    "validated_rows": validated_rows,
                    "discarded_rows": initial_rows - saved_rows,
                    "output_file": str(output_file),
                }
            )
            logger.info(
                "[proteingym] Complete. Retained rows: %d/%d",
                saved_rows,
                initial_rows,
            )

        except Exception as exc:  # noqa: BLE001 - want to keep the batch going
            row["error"] = str(exc)
            logger.exception("[proteingym] Error processing %s", dataset_id)

        summary.append(row)

    return summary


# --------------------------------------------------------------------------- #
# MaveDB source
# --------------------------------------------------------------------------- #

def process_mavedb(
    cfg: dict,
    dry_run: bool = False,
    *,
    base_url: str = MAVEDB_API_URL,
) -> list[dict]:
    """Process configured MaveDB datasets or return metadata-only summaries."""
    validate_mavedb_config(cfg)
    dir_base = Path(cfg["dir_base"])
    data_dir = dir_base / "raw"
    output_dir = dir_base / "processed"
    logger.debug(
        "[mavedb] Resolved paths raw=%s output=%s dry_run=%s",
        data_dir,
        output_dir,
        dry_run,
    )
    if not dry_run:
        ensure_dirs(data_dir, output_dir)

    base_url = base_url.rstrip("/")
    default_build_kwargs = cfg.get("default_build_kwargs", {})
    entries = cfg.get("datasets", [])
    summary: list[dict] = []

    for i, entry in enumerate(entries):
        dataset_id = entry["dataset_id"]
        logger.info(
            "[mavedb] %d/%d Processing URN: %s",
            i + 1,
            len(entries),
            dataset_id,
        )

        row = {
            "source": "mavedb",
            "input": dataset_id,
            "status": "ERROR",
        }

        try:
            meta_response = requests.get(
                f"{base_url}/score-sets/{dataset_id}",
                timeout=60,
            )
            if meta_response.status_code != 200:
                raise ValueError(f"Error while downloading metadata (HTTP {meta_response.status_code}).")
            metadata = meta_response.json()

            gene = _mavedb_gene(metadata)
            target_name = _mavedb_target_protein(metadata, gene)
            uniprot_id = _mavedb_uniprot_id(metadata)

            wt_sequence = extract_wt_from_metadata(metadata)
            if wt_sequence is None:
                raise ValueError("No WT found in metadata.")
            logger.debug(
                "[mavedb] Resolved WT source=score_set_metadata "
                "dataset_id=%s length=%d",
                dataset_id,
                len(wt_sequence),
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
                logger.info(
                    "[mavedb] (dry-run) metadata resolved for %s",
                    dataset_id,
                )
                continue

            scores_response = requests.get(
                f"{base_url}/score-sets/{dataset_id}/scores",
                timeout=60,
            )
            scores_response.raise_for_status()
            scores_path = (
                data_dir / f"{dataset_id.replace(':', '_')}_scores.csv"
            )
            scores_path.write_text(scores_response.text, encoding="utf-8")

            df_raw = pd.read_csv(scores_path)
            initial_rows = len(df_raw)

            hgvs_col = entry.get("hgvs_col") or next(
                (c for c in ["hgvs_pro"] if c in df_raw.columns), None
            )
            score_col = entry.get("score_col") or next(
                (c for c in ["score", "DMS_score", "fitness"] if c in df_raw.columns), None
            )
            if not hgvs_col or not score_col:
                raise ValueError("Neither score nor HGVS columns were detected.")
            logger.debug(
                "[mavedb] Detected columns dataset_id=%s variant_col=%s "
                "score_col=%s",
                dataset_id,
                hgvs_col,
                score_col,
            )

            row["raw_rows"] = initial_rows

            build_kwargs = deep_merge(default_build_kwargs, entry.get("build_kwargs", {}))
            build_kwargs.setdefault("add_relative_score", False)
            build_kwargs.setdefault("add_binary_label", False)
            build_kwargs.setdefault("drop_failed", False)
            build_kwargs["score_col"] = score_col
            build_kwargs["hgvs_col"] = hgvs_col
            build_kwargs["dataset_id"] = dataset_id
            build_kwargs["protein_id"] = None
            build_kwargs["gene"] = gene
            build_kwargs["uniprot_id"] = uniprot_id

            df_built = build_mavedb_dataset(
                input_path=scores_path,
                wt_sequence=wt_sequence,
                **build_kwargs,
            )

            saved_rows = len(df_built)
            validated_rows = int((df_built["status"] == "OK").sum())

            output_file = (
                output_dir / f"{dataset_id.replace(':', '_')}_processed.csv"
            )
            df_built.to_csv(output_file, index=False)
            logger.debug(
                "[mavedb] Resolved output dataset_id=%s path=%s",
                dataset_id,
                output_file,
            )

            row.update(
                {
                    "status": "OK",
                    "validated_rows": validated_rows,
                    "discarded_rows": initial_rows - saved_rows,
                    "output_file": str(output_file),
                }
            )
            logger.info(
                "[mavedb] Complete. Retained: %d/%d",
                saved_rows,
                initial_rows,
            )

        except Exception as exc:  # noqa: BLE001 - want to keep the batch going
            row["error"] = str(exc)
            logger.exception("[mavedb] Error processing %s", dataset_id)

        summary.append(row)

    return summary


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not cfg:
        raise SourceConfigurationError(
            f"The configuration file {path} is invalid or empty."
        )
    return validate_config(cfg)


def write_summary(summary: list[dict], output_cfg: dict) -> Path:
    summary_dir = Path(output_cfg.get("summary_dir", "datasets/summaries"))
    ensure_dirs(summary_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    df_summary = pd.DataFrame(summary)
    csv_path = summary_dir / f"summary_{timestamp}.csv"
    json_path = summary_dir / f"summary_{timestamp}.json"
    df_summary.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    return csv_path


def print_report(summary: list[dict]) -> None:
    if not summary:
        logger.warning("No dataset processed.")
        return

    n_ok = sum(1 for r in summary if r["status"] == "OK")
    n_err = sum(1 for r in summary if r["status"] == "ERROR")
    n_dry = sum(1 for r in summary if r["status"] == "DRY_RUN")

    logger.info("=" * 60)
    logger.info("FINAL SUMMARY: %d processed datasets", len(summary))
    logger.info("  OK: %d | ERROR: %d | DRY_RUN: %d", n_ok, n_err, n_dry)
    for row in summary:
        if row["status"] == "OK":
            logger.info(
                "  [OK]    %-12s %-30s %s/%s rows",
                row["source"],
                row.get("dataset_id", row["input"]),
                row.get("validated_rows"),
                row.get("raw_rows"),
            )
        elif row["status"] == "ERROR":
            logger.info(
                "  [ERROR] %-12s %-30s %s",
                row["source"],
                row["input"],
                row.get("error", ""),
            )
    logger.info("=" * 60)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Processes ProteinGym and/or MaveDB DMS datasets using dms_parser."
    )
    parser.add_argument(
        "--config", "-c", required=True, type=Path, help="Configuration YAML file's location."
    )
    parser.add_argument(
        "--only",
        choices=["all", "proteingym", "mavedb"],
        default="all",
        help="Restrict run to a single database (default: all).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Resolve metadata and WT sequences without downloading scores "
            "or writing processed output."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(args.config)
    summary: list[dict] = []

    try:
        if args.only in ("all", "proteingym") and "proteingym" in cfg:
            summary += process_proteingym(cfg["proteingym"], dry_run=args.dry_run)

        if args.only in ("all", "mavedb") and "mavedb" in cfg:
            summary += process_mavedb(cfg["mavedb"], dry_run=args.dry_run)
    finally:
        if summary:
            summary_path = write_summary(summary, cfg.get("output", {}))
            logger.info("Summary saved at: %s", summary_path)
        print_report(summary)

    n_err = sum(1 for r in summary if r["status"] == "ERROR")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
