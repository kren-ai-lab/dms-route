"""High-level builders for DMS datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

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
    add_relative_score: bool = True,
    relative_method: str = "log_ratio",
    relative_output_col: str = "score_log_ratio",
    add_binary_label: bool = False,
    delta: float = 0.1,
    higher_is_better: bool = True,
    binary_output_col: str = "score_binary_like",
    drop_failed: bool = False,
    validate_output: bool = True,
    require_wt_for_transforms: bool = False,
) -> pd.DataFrame:
    """Build a standardized DMS dataset from a local MaveDB-like table."""
    df = read_table(input_path, sep=sep)

    validate_required_columns(df, [hgvs_col, score_col])
    validate_score_column(df, score_col, allow_na=True)

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
    out["n_mutations"] = parsed_df["n_mutations"]

    out["score_raw"] = pd.to_numeric(out[score_col], errors="coerce")

    if drop_failed:
        out = out[out["status"] == "OK"].copy()

    if add_relative_score:
        wt_exists = has_wildtype_row(out, wt_col="is_wildtype", status_col="status")
        if wt_exists:
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
        elif require_wt_for_transforms:
            raise ValueError(
                "WT-relative transforms were requested, but no valid WT row was found."
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
    add_relative_score: bool = True,
    relative_method: str = "log_ratio",
    relative_output_col: str = "score_log_ratio",
    add_binary_label: bool = False,
    delta: float = 0.1,
    higher_is_better: bool = True,
    binary_output_col: str = "score_binary_like",
    drop_failed: bool = False,
    validate_output: bool = True,
    require_wt_for_transforms: bool = False,
) -> pd.DataFrame:
    """Build a standardized DMS dataset from a local ProteinGym-like table."""
    df = read_table(input_path, sep=sep)

    validate_required_columns(df, [variant_col, score_col])
    validate_variant_column(df, variant_col)
    validate_score_column(df, score_col, allow_na=True)

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
    out["n_mutations"] = parsed_df["n_mutations"]

    out["score_raw"] = pd.to_numeric(out[score_col], errors="coerce")
    out = pd.concat([out, variant_info], axis=1)

    if drop_failed:
        out = out[out["status"] == "OK"].copy()

    if add_relative_score:
        wt_exists = has_wildtype_row(out, wt_col="is_wildtype", status_col="status")
        if wt_exists:
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
        elif require_wt_for_transforms:
            raise ValueError(
                "WT-relative transforms were requested, but no valid WT row was found."
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

    return out