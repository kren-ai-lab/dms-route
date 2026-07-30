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
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml

from dms_parser import (
    build_mavedb_dataset,
    build_proteingym_dataset,
    read_table,
    translate_dna,
    write_table,
)
from dms_parser.io import download_file

logger = logging.getLogger("dms_parser.example_runner")


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
    dir_base = Path(cfg["dir_base"])
    data_dir = dir_base / "raw"
    output_dir = dir_base / "processed"

    metadata_path = dir_base / "DMS_substitutions.csv"
    benchmark_path = dir_base / "DMS_substitutions.parquet"
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
    download_file(cfg["metadata_url"], metadata_path, overwrite=False)

    if not dry_run:
        logger.info("[proteingym] Downloading/verifying benchmark data...")
        download_file(cfg["benchmark_url"], benchmark_path, overwrite=False)

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
        # Allow plain strings ("file.csv") or dicts ({"filename": ..., "build_kwargs": {...}})
        if isinstance(entry, str):
            entry = {"filename": entry}
        filename = entry["filename"]
        logger.info("[proteingym] %d/%d Processing: %s", i + 1, len(entries), filename)

        row = {
            "source": "proteingym",
            "input": filename,
            "status": "ERROR",
        }

        try:
            meta_match = df_meta[df_meta["DMS_filename"] == filename]
            if meta_match.empty:
                raise ValueError(f"No information found in metadata for {filename}.")

            selected_row = meta_match.iloc[0]
            dms_id = selected_row["DMS_id"]
            wt_sequence = selected_row["target_seq"]
            uniprot_id = selected_row["UniProt_ID"] if "UniProt_ID" in selected_row else "Unknown"
            logger.debug(
                "[proteingym] Resolved WT source=reference_metadata "
                "dataset_id=%s length=%d",
                dms_id,
                len(wt_sequence),
            )

            row.update(
                {
                    "dataset_id": dms_id,
                    "target_protein": uniprot_id,
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

            temp_raw_path = data_dir / filename

            build_kwargs = deep_merge(default_build_kwargs, entry.get("build_kwargs", {}))
            build_kwargs.setdefault("score_col", "DMS_score")
            build_kwargs.setdefault("variant_col", "mutant")
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

            df_ok = df_built[df_built["status"] == "OK"].copy()
            final_rows = len(df_ok)

            output_file = output_dir / f"{dms_id}_processed.csv"
            df_ok.to_csv(output_file, index=False)
            logger.debug(
                "[proteingym] Resolved output dataset_id=%s path=%s",
                dms_id,
                output_file,
            )

            row.update(
                {
                    "status": "OK",
                    "validated_rows": final_rows,
                    "discarded_rows": initial_rows - final_rows,
                    "output_file": str(output_file),
                }
            )
            logger.info("[proteingym] Complete. Retained rows: %d/%d", final_rows, initial_rows)

        except Exception as exc:  # noqa: BLE001 - want to keep the batch going
            row["error"] = str(exc)
            logger.exception("[proteingym] Error processing %s", filename)

        summary.append(row)

    return summary


# --------------------------------------------------------------------------- #
# MaveDB source
# --------------------------------------------------------------------------- #

def process_mavedb(cfg: dict, dry_run: bool = False) -> list[dict]:
    """Process configured MaveDB datasets or return metadata-only summaries."""
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

    base_url = cfg["base_url"]
    default_build_kwargs = cfg.get("default_build_kwargs", {})
    entries = cfg.get("datasets", [])
    summary: list[dict] = []

    for i, entry in enumerate(entries):
        if isinstance(entry, str):
            entry = {"urn": entry}
        urn = entry["urn"]
        logger.info("[mavedb] %d/%d Processing URN: %s", i + 1, len(entries), urn)

        row = {
            "source": "mavedb",
            "input": urn,
            "status": "ERROR",
        }

        try:
            meta_response = requests.get(f"{base_url}/score-sets/{urn}", timeout=60)
            if meta_response.status_code != 200:
                raise ValueError(f"Error while downloading metadata (HTTP {meta_response.status_code}).")
            metadata = meta_response.json()

            target_name = "Unknown"
            if "targetGenes" in metadata and isinstance(metadata["targetGenes"], list) and metadata["targetGenes"]:
                target_name = metadata["targetGenes"][0].get("name", "Unknown")
            elif "title" in metadata:
                target_name = metadata["title"].split(" ")[0]

            wt_sequence = extract_wt_from_metadata(metadata)
            if wt_sequence is None:
                raise ValueError("No WT found in metadata.")
            logger.debug(
                "[mavedb] Resolved WT source=score_set_metadata "
                "dataset_id=%s length=%d",
                urn,
                len(wt_sequence),
            )

            row.update(
                {
                    "dataset_id": urn,
                    "target_protein": target_name,
                    "wt_length": len(wt_sequence),
                    "raw_rows": None,
                }
            )

            if dry_run:
                row["status"] = "DRY_RUN"
                summary.append(row)
                logger.info("[mavedb] (dry-run) metadata resolved for %s", urn)
                continue

            scores_response = requests.get(f"{base_url}/score-sets/{urn}/scores", timeout=60)
            scores_response.raise_for_status()
            scores_path = data_dir / f"{urn.replace(':', '_')}_scores.csv"
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
                urn,
                hgvs_col,
                score_col,
            )

            row["raw_rows"] = initial_rows

            build_kwargs = deep_merge(default_build_kwargs, entry.get("build_kwargs", {}))

            df_built = build_mavedb_dataset(
                input_path=scores_path,
                score_col=score_col,
                hgvs_col=hgvs_col,
                dataset_id=urn,
                wt_sequence=wt_sequence,
                **build_kwargs,
            )

            df_ok = df_built[df_built["status"] == "OK"].copy()
            final_rows = len(df_ok)

            output_file = output_dir / f"{urn.replace(':', '_')}_processed.csv"
            df_ok.to_csv(output_file, index=False)
            logger.debug(
                "[mavedb] Resolved output dataset_id=%s path=%s",
                urn,
                output_file,
            )

            row.update(
                {
                    "status": "OK",
                    "validated_rows": final_rows,
                    "discarded_rows": initial_rows - final_rows,
                    "output_file": str(output_file),
                }
            )
            logger.info("[mavedb] Complete. Retained: %d/%d", final_rows, initial_rows)

        except Exception as exc:  # noqa: BLE001 - want to keep the batch going
            row["error"] = str(exc)
            logger.exception("[mavedb] Error processing %s", urn)

        summary.append(row)

    return summary


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not cfg:
        raise ValueError(f"The configuration file {path} is invalid or empty.")
    return cfg


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
        if args.only in ("all", "proteingym") and cfg.get("proteingym", {}).get("enabled", True):
            summary += process_proteingym(cfg["proteingym"], dry_run=args.dry_run)

        if args.only in ("all", "mavedb") and cfg.get("mavedb", {}).get("enabled", True):
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
