"""Validation utilities for DMS datasets."""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

from dmsroute.core.exceptions import InvalidDatasetError, MissingWildTypeError, SequenceValidationError

_SINGLE_DELETION_VARIANT_RE = re.compile(r"^[A-Z*X]\d+del$")
_SINGLE_INSERTION_VARIANT_RE = re.compile(
    r"^[A-Z*X]\d+_[A-Z*X]\d+ins[A-Z*X]$"
)


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: list[str],
) -> None:
    """Ensure that all required columns are present in a DataFrame."""
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise InvalidDatasetError(
            f"Missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def validate_column_non_empty(
    df: pd.DataFrame,
    column: str,
) -> None:
    """Ensure that a column exists and contains at least one non-null value."""
    validate_required_columns(df, [column])

    if df[column].dropna().empty:
        raise InvalidDatasetError(
            f"Column {column!r} exists but contains no non-null values."
        )


def validate_score_column(
    df: pd.DataFrame,
    score_col: str,
    *,
    allow_na: bool = True,
) -> None:
    """Validate that a score column exists and can be interpreted as numeric."""
    validate_required_columns(df, [score_col])

    coerced = pd.to_numeric(df[score_col], errors="coerce")

    if coerced.notna().sum() == 0:
        raise InvalidDatasetError(
            f"Column {score_col!r} does not contain any valid numeric values."
        )

    if not allow_na:
        invalid_count = coerced.isna().sum()
        if invalid_count > 0:
            raise InvalidDatasetError(
                f"Column {score_col!r} contains {invalid_count} non-numeric or missing values."
            )


def validate_variant_column(
    df: pd.DataFrame,
    variant_col: str,
) -> None:
    """Ensure that the variant column is present and not entirely empty."""
    validate_column_non_empty(df, variant_col)


def validate_status_column(
    df: pd.DataFrame,
    status_col: str = "status",
    allowed_values: Iterable[str] = ("OK", "Unsupported", "Error"),
) -> None:
    """Validate that status values belong to an allowed set."""
    validate_required_columns(df, [status_col])

    allowed = set(allowed_values)
    observed = set(df[status_col].dropna().astype(str).unique())
    invalid = observed - allowed

    if invalid:
        raise InvalidDatasetError(
            f"Column {status_col!r} contains invalid values: {sorted(invalid)}. "
            f"Allowed values are: {sorted(allowed)}"
        )


def validate_wt_sequence(
    wt_sequence: str,
    *,
    allow_stop: bool = True,
    allow_x: bool = True,
) -> None:
    """Validate a WT protein sequence."""
    sequence = str(wt_sequence).strip().upper()

    if not sequence:
        raise SequenceValidationError("WT sequence is empty.")

    allowed = set("ACDEFGHIKLMNPQRSTVWY")
    if allow_stop:
        allowed.add("*")
    if allow_x:
        allowed.add("X")

    invalid = sorted(set(sequence) - allowed)
    if invalid:
        raise SequenceValidationError(
            f"WT sequence contains invalid residue symbols: {invalid}"
        )


def validate_wt_presence(
    df: pd.DataFrame,
    wt_col: str = "is_wildtype",
    *,
    require_exactly_one: bool = False,
    status_col: str | None = None,
    accepted_status: str = "OK",
) -> None:
    """Validate that at least one WT row is present."""
    validate_required_columns(df, [wt_col])

    subset = df
    if status_col is not None:
        validate_required_columns(df, [status_col])
        subset = subset[subset[status_col] == accepted_status]

    wt_count = int((subset[wt_col] == True).sum())  # noqa: E712

    if wt_count == 0:
        raise MissingWildTypeError("No wild-type row found in dataset.")

    if require_exactly_one and wt_count != 1:
        raise InvalidDatasetError(
            f"Expected exactly one wild-type row, but found {wt_count}."
        )


def validate_n_mutations_column(
    df: pd.DataFrame,
    n_mutations_col: str = "n_mutations",
    *,
    allow_na: bool = True,
) -> None:
    """Validate that the mutation-count column is numeric and non-negative."""
    validate_required_columns(df, [n_mutations_col])

    values = pd.to_numeric(df[n_mutations_col], errors="coerce")

    if not allow_na and values.isna().any():
        raise InvalidDatasetError(
            f"Column {n_mutations_col!r} contains missing or non-numeric values."
        )

    negative_mask = values.dropna() < 0
    if negative_mask.any():
        raise InvalidDatasetError(
            f"Column {n_mutations_col!r} contains negative values."
        )


def validate_sequence_column(
    df: pd.DataFrame,
    sequence_col: str,
    *,
    allow_na: bool = True,
    allow_stop: bool = True,
    allow_x: bool = True,
) -> None:
    """Validate that a sequence column contains valid protein sequences."""
    validate_required_columns(df, [sequence_col])

    allowed = set("ACDEFGHIKLMNPQRSTVWY")
    if allow_stop:
        allowed.add("*")
    if allow_x:
        allowed.add("X")

    series = df[sequence_col]

    if not allow_na and series.isna().any():
        raise InvalidDatasetError(
            f"Column {sequence_col!r} contains missing sequence values."
        )

    invalid_rows: list[int] = []
    for idx, value in series.items():
        if pd.isna(value):
            continue

        seq = str(value).strip().upper()
        if not seq:
            invalid_rows.append(idx)
            continue

        if set(seq) - allowed:
            invalid_rows.append(idx)

    if invalid_rows:
        preview = invalid_rows[:10]
        raise InvalidDatasetError(
            f"Column {sequence_col!r} contains invalid protein sequences at rows {preview}."
        )


def validate_consistent_sequence_lengths(
    df: pd.DataFrame,
    wt_sequence_col: str = "wt_sequence",
    mutated_sequence_col: str = "mutated_sequence",
    *,
    only_status_ok: bool = True,
    status_col: str = "status",
) -> None:
    """Validate sequence lengths for substitutions and bounded protein indels."""
    validate_required_columns(df, [wt_sequence_col, mutated_sequence_col])

    subset = df.copy()
    if only_status_ok and status_col in subset.columns:
        subset = subset[subset[status_col] == "OK"]

    bad_rows: list[int] = []
    for idx, row in subset.iterrows():
        wt = row[wt_sequence_col]
        mut = row[mutated_sequence_col]

        if pd.isna(wt) or pd.isna(mut):
            continue

        variant = str(row.get("variant", ""))
        if _SINGLE_DELETION_VARIANT_RE.fullmatch(variant):
            expected_delta = -1
        elif _SINGLE_INSERTION_VARIANT_RE.fullmatch(variant):
            expected_delta = 1
        else:
            expected_delta = 0

        if len(str(mut)) != len(str(wt)) + expected_delta:
            bad_rows.append(idx)

    if bad_rows:
        preview = bad_rows[:10]
        raise InvalidDatasetError(
            f"Found sequence length mismatches between {wt_sequence_col!r} and "
            f"{mutated_sequence_col!r} at rows {preview}."
        )


def validate_standard_dataset(
    df: pd.DataFrame,
    *,
    require_wt: bool = False,
    require_status: bool = False,
    score_col: str = "score_raw",
    variant_col: str = "variant",
    wt_col: str = "is_wildtype",
    n_mutations_col: str = "n_mutations",
    authoritative_sequence_mode: bool = False,
) -> None:
    """Run a compact validation suite over a standardized DMS dataset."""
    validate_required_columns(df, [score_col, variant_col, wt_col, n_mutations_col])
    validate_score_column(df, score_col, allow_na=True)
    if authoritative_sequence_mode:
        validate_required_columns(
            df,
            [
                "dataset_id",
                "source",
                "protein_id",
                "gene",
                "uniprot_id",
                "wt_sequence",
                "mutated_sequence",
                "is_synthetic",
                "status",
                "error",
            ],
        )
    else:
        validate_variant_column(df, variant_col)
    validate_n_mutations_column(df, n_mutations_col, allow_na=True)

    if require_status or authoritative_sequence_mode:
        validate_status_column(df, status_col="status")

    if "wt_sequence" in df.columns:
        non_null_wt = df["wt_sequence"].dropna()
        if not non_null_wt.empty:
            validate_wt_sequence(non_null_wt.iloc[0])

    if authoritative_sequence_mode:
        notation_rows = df.index[df[variant_col].notna()].tolist()
        if notation_rows:
            raise InvalidDatasetError(
                "Authoritative sequence rows must not contain variant notation "
                f"at rows {notation_rows[:10]}."
            )

        ok_rows = df[df["status"] == "OK"]
        validate_sequence_column(
            ok_rows,
            "wt_sequence",
            allow_na=False,
        )
        validate_sequence_column(
            ok_rows,
            "mutated_sequence",
            allow_na=False,
        )
        wildtype_mismatch_rows: list[int] = []
        wt_count_rows: list[int] = []
        non_wt_count_rows: list[int] = []
        for idx, row in ok_rows.iterrows():
            sequence_is_wildtype = (
                row["mutated_sequence"] == row["wt_sequence"]
            )
            observed_is_wildtype = row[wt_col]
            if (
                pd.isna(observed_is_wildtype)
                or observed_is_wildtype not in (True, False)
                or bool(observed_is_wildtype) != sequence_is_wildtype
            ):
                wildtype_mismatch_rows.append(idx)

            mutation_count = row[n_mutations_col]
            if sequence_is_wildtype:
                numeric_count = pd.to_numeric(
                    pd.Series([mutation_count]),
                    errors="coerce",
                ).iloc[0]
                if pd.isna(numeric_count) or float(numeric_count) != 0:
                    wt_count_rows.append(idx)
            elif not pd.isna(mutation_count):
                non_wt_count_rows.append(idx)

        if wildtype_mismatch_rows:
            raise InvalidDatasetError(
                "Authoritative sequence rows have inconsistent is_wildtype "
                f"values at rows {wildtype_mismatch_rows[:10]}."
            )
        if wt_count_rows:
            raise InvalidDatasetError(
                "Authoritative WT rows must have n_mutations == 0 at rows "
                f"{wt_count_rows[:10]}."
            )
        if non_wt_count_rows:
            raise InvalidDatasetError(
                "Authoritative non-WT rows must have missing n_mutations at "
                f"rows {non_wt_count_rows[:10]}."
            )

    if require_wt:
        validate_wt_presence(
            df,
            wt_col=wt_col,
            require_exactly_one=False,
            status_col="status" if require_status and "status" in df.columns else None,
        )


def has_wildtype_row(
    df: pd.DataFrame,
    wt_col: str = "is_wildtype",
    *,
    status_col: str | None = None,
    accepted_status: str = "OK",
) -> bool:
    """Return True if the dataset contains at least one WT row."""
    validate_required_columns(df, [wt_col])

    subset = df
    if status_col is not None:
        validate_required_columns(df, [status_col])
        subset = subset[subset[status_col] == accepted_status]

    return bool((subset[wt_col] == True).any())  # noqa: E712


def count_invalid_sequences(
    df: pd.DataFrame,
    sequence_col: str,
    *,
    allow_stop: bool = True,
    allow_x: bool = True,
) -> int:
    """Count how many rows in a sequence column contain invalid protein strings."""
    validate_required_columns(df, [sequence_col])

    allowed = set("ACDEFGHIKLMNPQRSTVWY")
    if allow_stop:
        allowed.add("*")
    if allow_x:
        allowed.add("X")

    invalid_count = 0
    for value in df[sequence_col]:
        if pd.isna(value):
            continue
        seq = str(value).strip().upper()
        if not seq or (set(seq) - allowed):
            invalid_count += 1

    return invalid_count
