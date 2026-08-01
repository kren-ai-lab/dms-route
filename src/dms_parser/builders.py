"""High-level builders for DMS datasets."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from dms_parser.exceptions import MissingWildTypeError
from dms_parser.io import read_table
from dms_parser.parsing import (
    count_mutations,
    hgvs_pro_is_indel,
    hgvs_to_sequence,
    is_wildtype_variant,
    parse_variant_series,
    read_fasta_one,
    translate_dna,
    variant_to_sequence,
)
from dms_parser.transforms import add_pseudo_binary_label, add_wt_relative_score
from dms_parser.types import SequenceBuildResult
from dms_parser.validation import (
    has_wildtype_row,
    validate_consistent_sequence_lengths,
    validate_required_columns,
    validate_score_column,
    validate_standard_dataset,
    validate_variant_column,
    validate_wt_sequence,
)

logger = logging.getLogger(__name__)


def _resolve_wt_sequence(
    wt_sequence: str | None = None,
    wt_fasta_path: str | Path | None = None,
    wt_sequence_is_dna: bool = False,
    dna_frame: int = 1,
    stop_at_stop: bool = True,
) -> str:
    """Resolve WT sequence from direct string or FASTA."""
    if wt_sequence is None and wt_fasta_path is None:
        raise ValueError("Either wt_sequence or wt_fasta_path must be provided.")

    strategy = "provided_sequence" if wt_sequence is not None else "fasta_file"
    logger.debug("Resolving WT sequence strategy=%s dna_input=%s", strategy, wt_sequence_is_dna)
    if wt_sequence is None:
        _, wt_sequence = read_fasta_one(str(wt_fasta_path))

    wt_sequence = str(wt_sequence).strip().upper()

    if wt_sequence_is_dna:
        wt_sequence = translate_dna(
            wt_sequence,
            frame=dna_frame,
            stop_at_stop=stop_at_stop,
        )

    validate_wt_sequence(wt_sequence)
    logger.debug(
        "Resolved WT sequence strategy=%s translated_from_dna=%s length=%d",
        strategy, wt_sequence_is_dna, len(wt_sequence),
    )
    return wt_sequence


def _safe_hgvs_to_sequence(
    wt_sequence: str,
    hgvs_pro: Any,
) -> SequenceBuildResult:
    """Safely convert HGVS protein variant to mutated sequence."""
    hgvs_pro = str(hgvs_pro)

    out: SequenceBuildResult = {
        "variant": None,
        "mutated_sequence": None,
        "status": "OK",
        "error": "",
        "is_wildtype": False,
        "n_mutations": None,
    }

    try:
        if hgvs_pro_is_indel(hgvs_pro):
            out["status"] = "Unsupported"
            out["error"] = (
                "Protein-level indel/frameshift not supported in substitutions-only analysis."
            )
            return out

        mutated_sequence, variant = hgvs_to_sequence(wt_sequence, hgvs_pro)

        out["variant"] = variant
        out["mutated_sequence"] = mutated_sequence
        out["is_wildtype"] = is_wildtype_variant(variant)
        out["n_mutations"] = count_mutations(variant)

    except Exception as exc:
        out["status"] = "Error"
        out["error"] = str(exc)

    return out


def _safe_variant_to_sequence(
    wt_sequence: str,
    variant: Any,
    *,
    strict: bool = False,
) -> SequenceBuildResult:
    """Safely convert internal variant notation to mutated sequence."""
    out: SequenceBuildResult = {
        "variant": None,
        "mutated_sequence": None,
        "status": "OK",
        "error": "",
        "is_wildtype": False,
        "n_mutations": None,
    }

    try:
        mutated_sequence, normalized_variant = variant_to_sequence(
            wt_sequence,
            variant,
            strict=strict,
        )

        out["variant"] = normalized_variant
        out["mutated_sequence"] = mutated_sequence
        out["is_wildtype"] = is_wildtype_variant(normalized_variant)
        out["n_mutations"] = count_mutations(normalized_variant)

    except Exception as exc:
        out["status"] = "Error"
        out["error"] = str(exc)

    return out


def _maybe_add_wildtype_row(
    df: pd.DataFrame,
    *,
    add_wildtype_row: bool,
    wt_sequence: str,
    dataset_id: str | None,
    source: str,
    protein_id: str | None,
    gene: str | None,
    uniprot_id: str | None,
) -> pd.DataFrame:
    """Prepend a scoreless synthetic WT row when one was requested and is absent."""
    if not add_wildtype_row:
        return df

    if has_wildtype_row(df, wt_col="is_wildtype", status_col="status"):
        logger.debug("Synthetic wild-type row not needed source=%s dataset_id=%s", source, dataset_id)
        return df.reset_index(drop=True)

    row = {
        "dataset_id": dataset_id,
        "source": source,
        "protein_id": protein_id,
        "gene": gene,
        "uniprot_id": uniprot_id,
        "wt_sequence": wt_sequence,
        "variant": "",
        "mutated_sequence": wt_sequence,
        "is_wildtype": True,
        "is_synthetic": True,
        "n_mutations": 0,
        "score_raw": float("nan"),
        "status": "OK",
        "error": "",
    }
    original_dtypes = df.dtypes
    out = df.copy()
    out.index = pd.RangeIndex(1, len(out) + 1)
    out = out.reindex(pd.RangeIndex(len(out) + 1))
    for column, value in row.items():
        out.at[0, column] = value
        try:
            out[column] = out[column].astype(original_dtypes[column])
        except (TypeError, ValueError):
            pass

    logger.info("Added synthetic wild-type row source=%s dataset_id=%s", source, dataset_id)
    return out


def _maybe_add_transforms(
    df: pd.DataFrame,
    *,
    score_col: str,
    add_relative_score: bool = True,
    relative_method: str = "log_ratio",
    relative_output_col: str = "score_log_ratio",
    add_binary_label: bool = False,
    binary_score_col: str | None = None,
    delta: float = 0.1,
    higher_is_better: bool = True,
    binary_output_col: str = "score_binary_like",
) -> pd.DataFrame:
    """Optionally add WT-relative score and pseudo-binary label."""
    out = df.copy()

    if add_relative_score:
        out = add_wt_relative_score(
            out,
            score_col=score_col,
            wt_col="is_wildtype",
            method=relative_method,
            output_col=relative_output_col,
            status_col="status",
        )

    if add_binary_label:
        score_for_binary = binary_score_col or relative_output_col
        out = add_pseudo_binary_label(
            out,
            score_col=score_for_binary,
            delta=delta,
            higher_is_better=higher_is_better,
            output_col=binary_output_col,
        )

    return out


def build_mavedb_dataset(
    input_path: str | Path,
    score_col: str,
    hgvs_col: str = "hgvs_pro",
    dataset_id: str | None = None,
    protein_id: str | None = None,
    gene: str | None = None,
    uniprot_id: str | None = None,
    wt_sequence: str | None = None,
    wt_fasta_path: str | Path | None = None,
    wt_sequence_is_dna: bool = False,
    dna_frame: int = 1,
    stop_at_stop: bool = True,
    sep: str | None = None,
    add_relative_score: bool = False,
    relative_method: str = "log_ratio",
    relative_output_col: str = "score_log_ratio",
    add_binary_label: bool = False,
    delta: float = 0.1,
    higher_is_better: bool = True,
    binary_output_col: str = "score_binary_like",
    add_wildtype_row: bool = False,
    drop_failed: bool = False,
    validate_output: bool = True,
    require_wt_for_transforms: bool = False,
) -> pd.DataFrame:
    """Build a standardized MaveDB-like dataset with opt-in score transforms."""
    logger.info("Starting dataset build source=mavedb dataset_id=%s", dataset_id)
    logger.debug(
        "Builder options source=mavedb dataset_id=%s score_col=%s "
        "variant_col=%s add_relative_score=%s add_binary_label=%s "
        "drop_failed=%s validate_output=%s",
        dataset_id, score_col, hgvs_col, add_relative_score,
        add_binary_label, drop_failed, validate_output,
    )
    df = read_table(input_path, sep=sep)
    input_rows = len(df)

    validate_required_columns(df, [hgvs_col, score_col])
    validate_score_column(df, score_col, allow_na=True)
    logger.debug(
        "Detected dataset columns source=mavedb dataset_id=%s "
        "variant_col=%s score_col=%s input_rows=%d",
        dataset_id, hgvs_col, score_col, input_rows,
    )

    wt_seq = _resolve_wt_sequence(
        wt_sequence=wt_sequence,
        wt_fasta_path=wt_fasta_path,
        wt_sequence_is_dna=wt_sequence_is_dna,
        dna_frame=dna_frame,
        stop_at_stop=stop_at_stop,
    )

    parsed = df[hgvs_col].apply(lambda value: _safe_hgvs_to_sequence(wt_seq, value))
    parsed_df = pd.DataFrame(parsed.tolist(), index=df.index)

    out = df.copy()
    out["dataset_id"] = dataset_id
    out["source"] = "mavedb"
    out["protein_id"] = protein_id
    out["gene"] = gene
    out["uniprot_id"] = uniprot_id
    out["wt_sequence"] = wt_seq

    out["variant"] = parsed_df["variant"]
    out["mutated_sequence"] = parsed_df["mutated_sequence"]
    out["status"] = parsed_df["status"]
    out["error"] = parsed_df["error"]
    out["is_wildtype"] = parsed_df["is_wildtype"]
    out["is_synthetic"] = False
    out["n_mutations"] = parsed_df["n_mutations"]

    out["score_raw"] = pd.to_numeric(out[score_col], errors="coerce")
    validated_rows = int((parsed_df["status"] == "OK").sum())
    unsupported_rows = int((parsed_df["status"] == "Unsupported").sum())
    error_rows = int((parsed_df["status"] == "Error").sum())

    if drop_failed:
        out = out[out["status"] == "OK"].copy()

    out = _maybe_add_wildtype_row(
        out,
        add_wildtype_row=add_wildtype_row,
        wt_sequence=wt_seq,
        dataset_id=dataset_id,
        source="mavedb",
        protein_id=protein_id,
        gene=gene,
        uniprot_id=uniprot_id,
    )

    if add_relative_score:
        try:
            out = _maybe_add_transforms(
                out,
                score_col="score_raw",
                add_relative_score=True,
                relative_method=relative_method,
                relative_output_col=relative_output_col,
                add_binary_label=add_binary_label,
                delta=delta,
                higher_is_better=higher_is_better,
                binary_output_col=binary_output_col,
            )
        except MissingWildTypeError as exc:
            if require_wt_for_transforms:
                raise ValueError(
                    "WT-relative transforms were requested, but no valid numeric WT score was found."
                ) from exc
            logger.warning(
                "Skipped requested WT-relative transformation source=mavedb "
                "dataset_id=%s reason=no_valid_numeric_wild_type_score",
                dataset_id,
            )
        else:
            logger.info(
                "Applied score transformations source=mavedb dataset_id=%s "
                "relative_method=%s binary_label=%s",
                dataset_id, relative_method, add_binary_label,
            )
    elif add_binary_label:
        raise ValueError(
            "add_binary_label=True requires add_relative_score=True in this builder version."
        )

    if validate_output:
        validate_standard_dataset(
            out,
            require_wt=False,
            require_status=True,
            score_col="score_raw",
            variant_col="variant",
            wt_col="is_wildtype",
            n_mutations_col="n_mutations",
        )
        validate_consistent_sequence_lengths(
            out,
            wt_sequence_col="wt_sequence",
            mutated_sequence_col="mutated_sequence",
            only_status_ok=True,
            status_col="status",
        )

    logger.info(
        "Completed dataset build source=mavedb dataset_id=%s input_rows=%d "
        "output_rows=%d validated_rows=%d unsupported_rows=%d error_rows=%d",
        dataset_id, input_rows, len(out), validated_rows, unsupported_rows, error_rows,
    )
    return out


def build_proteingym_dataset(
    input_path: str | Path,
    score_col: str,
    variant_col: str = "variant",
    dataset_id: str | None = None,
    protein_id: str | None = None,
    gene: str | None = None,
    uniprot_id: str | None = None,
    wt_sequence: str | None = None,
    wt_fasta_path: str | Path | None = None,
    wt_sequence_is_dna: bool = False,
    dna_frame: int = 1,
    stop_at_stop: bool = True,
    sep: str | None = None,
    strict_variant_parsing: bool = False,
    add_relative_score: bool = False,
    relative_method: str = "log_ratio",
    relative_output_col: str = "score_log_ratio",
    add_binary_label: bool = False,
    delta: float = 0.1,
    higher_is_better: bool = True,
    binary_output_col: str = "score_binary_like",
    add_wildtype_row: bool = False,
    drop_failed: bool = False,
    validate_output: bool = True,
    require_wt_for_transforms: bool = False,
) -> pd.DataFrame:
    """Build a standardized ProteinGym-like dataset with opt-in score transforms."""
    logger.info("Starting dataset build source=proteingym dataset_id=%s", dataset_id)
    logger.debug(
        "Builder options source=proteingym dataset_id=%s score_col=%s "
        "variant_col=%s add_relative_score=%s add_binary_label=%s "
        "drop_failed=%s validate_output=%s strict_variant_parsing=%s",
        dataset_id, score_col, variant_col, add_relative_score,
        add_binary_label, drop_failed, validate_output, strict_variant_parsing,
    )
    df = read_table(input_path, sep=sep)
    input_rows = len(df)

    validate_required_columns(df, [variant_col, score_col])
    validate_variant_column(df, variant_col)
    validate_score_column(df, score_col, allow_na=True)
    logger.debug(
        "Detected dataset columns source=proteingym dataset_id=%s "
        "variant_col=%s score_col=%s input_rows=%d",
        dataset_id, variant_col, score_col, input_rows,
    )

    wt_seq = _resolve_wt_sequence(
        wt_sequence=wt_sequence,
        wt_fasta_path=wt_fasta_path,
        wt_sequence_is_dna=wt_sequence_is_dna,
        dna_frame=dna_frame,
        stop_at_stop=stop_at_stop,
    )

    parsed = df[variant_col].apply(
        lambda value: _safe_variant_to_sequence(
            wt_seq,
            value,
            strict=strict_variant_parsing,
        )
    )
    parsed_df = pd.DataFrame(parsed.tolist(), index=df.index)

    variant_info = parse_variant_series(
        df[variant_col],
        strict=strict_variant_parsing,
        prefix="parsed_",
    )

    out = df.copy()
    out["dataset_id"] = dataset_id
    out["source"] = "proteingym"
    out["protein_id"] = protein_id
    out["gene"] = gene
    out["uniprot_id"] = uniprot_id
    out["wt_sequence"] = wt_seq

    out["variant"] = parsed_df["variant"]
    out["mutated_sequence"] = parsed_df["mutated_sequence"]
    out["status"] = parsed_df["status"]
    out["error"] = parsed_df["error"]
    out["is_wildtype"] = parsed_df["is_wildtype"]
    out["is_synthetic"] = False
    out["n_mutations"] = parsed_df["n_mutations"]

    out["score_raw"] = pd.to_numeric(out[score_col], errors="coerce")
    out = pd.concat([out, variant_info], axis=1)
    validated_rows = int((parsed_df["status"] == "OK").sum())
    unsupported_rows = int((parsed_df["status"] == "Unsupported").sum())
    error_rows = int((parsed_df["status"] == "Error").sum())

    if drop_failed:
        out = out[out["status"] == "OK"].copy()

    out = _maybe_add_wildtype_row(
        out,
        add_wildtype_row=add_wildtype_row,
        wt_sequence=wt_seq,
        dataset_id=dataset_id,
        source="proteingym",
        protein_id=protein_id,
        gene=gene,
        uniprot_id=uniprot_id,
    )

    if add_relative_score:
        try:
            out = _maybe_add_transforms(
                out,
                score_col="score_raw",
                add_relative_score=True,
                relative_method=relative_method,
                relative_output_col=relative_output_col,
                add_binary_label=add_binary_label,
                delta=delta,
                higher_is_better=higher_is_better,
                binary_output_col=binary_output_col,
            )
        except MissingWildTypeError as exc:
            if require_wt_for_transforms:
                raise ValueError(
                    "WT-relative transforms were requested, but no valid numeric WT score was found."
                ) from exc
            logger.warning(
                "Skipped requested WT-relative transformation source=proteingym "
                "dataset_id=%s reason=no_valid_numeric_wild_type_score",
                dataset_id,
            )
        else:
            logger.info(
                "Applied score transformations source=proteingym dataset_id=%s "
                "relative_method=%s binary_label=%s",
                dataset_id, relative_method, add_binary_label,
            )
    elif add_binary_label:
        raise ValueError(
            "add_binary_label=True requires add_relative_score=True in this builder version."
        )

    if validate_output:
        validate_standard_dataset(
            out,
            require_wt=False,
            require_status=True,
            score_col="score_raw",
            variant_col="variant",
            wt_col="is_wildtype",
            n_mutations_col="n_mutations",
        )
        validate_consistent_sequence_lengths(
            out,
            wt_sequence_col="wt_sequence",
            mutated_sequence_col="mutated_sequence",
            only_status_ok=True,
            status_col="status",
        )

    logger.info(
        "Completed dataset build source=proteingym dataset_id=%s input_rows=%d "
        "output_rows=%d validated_rows=%d unsupported_rows=%d error_rows=%d",
        dataset_id, input_rows, len(out), validated_rows, unsupported_rows, error_rows,
    )
    return out
