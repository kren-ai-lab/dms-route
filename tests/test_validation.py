from __future__ import annotations

import pandas as pd
import pytest

from dms_parser.exceptions import InvalidDatasetError, MissingWildTypeError, SequenceValidationError
from dms_parser.validation import (
    count_invalid_sequences,
    has_wildtype_row,
    validate_column_non_empty,
    validate_consistent_sequence_lengths,
    validate_n_mutations_column,
    validate_required_columns,
    validate_score_column,
    validate_sequence_column,
    validate_standard_dataset,
    validate_status_column,
    validate_variant_column,
    validate_wt_presence,
    validate_wt_sequence,
)


def test_validate_required_columns_passes(standardized_dataset: pd.DataFrame):
    validate_required_columns(standardized_dataset, ["variant", "score_raw"])


def test_validate_required_columns_raises(standardized_dataset: pd.DataFrame):
    with pytest.raises(InvalidDatasetError):
        validate_required_columns(standardized_dataset, ["missing_col"])


def test_validate_column_non_empty_passes(standardized_dataset: pd.DataFrame):
    validate_column_non_empty(standardized_dataset, "variant")


def test_validate_column_non_empty_raises():
    df = pd.DataFrame({"variant": [None, None]})
    with pytest.raises(InvalidDatasetError):
        validate_column_non_empty(df, "variant")


def test_validate_score_column_passes(standardized_dataset: pd.DataFrame):
    validate_score_column(standardized_dataset, "score_raw")


def test_validate_score_column_raises_for_all_invalid():
    df = pd.DataFrame({"score": ["a", "b"]})
    with pytest.raises(InvalidDatasetError):
        validate_score_column(df, "score")


def test_validate_variant_column_passes(standardized_dataset: pd.DataFrame):
    validate_variant_column(standardized_dataset, "variant")


def test_validate_status_column_passes(standardized_dataset: pd.DataFrame):
    validate_status_column(standardized_dataset)


def test_validate_status_column_raises():
    df = pd.DataFrame({"status": ["OK", "BAD"]})
    with pytest.raises(InvalidDatasetError):
        validate_status_column(df)


def test_validate_wt_sequence_passes(wt_sequence: str):
    validate_wt_sequence(wt_sequence)


def test_validate_wt_sequence_raises():
    with pytest.raises(SequenceValidationError):
        validate_wt_sequence("ABCJZ")


def test_validate_wt_presence_passes(standardized_dataset: pd.DataFrame):
    validate_wt_presence(standardized_dataset)


def test_validate_wt_presence_raises_when_missing(standardized_dataset: pd.DataFrame):
    df = standardized_dataset.copy()
    df["is_wildtype"] = False

    with pytest.raises(MissingWildTypeError):
        validate_wt_presence(df)


def test_validate_n_mutations_column_passes(standardized_dataset: pd.DataFrame):
    validate_n_mutations_column(standardized_dataset)


def test_validate_n_mutations_column_raises():
    df = pd.DataFrame({"n_mutations": [0, -1]})
    with pytest.raises(InvalidDatasetError):
        validate_n_mutations_column(df)


def test_validate_sequence_column_passes(standardized_dataset: pd.DataFrame):
    validate_sequence_column(standardized_dataset, "mutated_sequence")


def test_validate_sequence_column_raises():
    df = pd.DataFrame({"seq": ["MKT", "ABJ"]})
    with pytest.raises(InvalidDatasetError):
        validate_sequence_column(df, "seq")


def test_validate_consistent_sequence_lengths_passes(standardized_dataset: pd.DataFrame):
    validate_consistent_sequence_lengths(standardized_dataset)


def test_validate_consistent_sequence_lengths_raises(standardized_dataset: pd.DataFrame):
    df = standardized_dataset.copy()
    df.loc[1, "mutated_sequence"] = "SHORT"

    with pytest.raises(InvalidDatasetError):
        validate_consistent_sequence_lengths(df)


def test_validate_standard_dataset_passes(standardized_dataset: pd.DataFrame):
    validate_standard_dataset(
        standardized_dataset,
        require_wt=True,
        require_status=True,
    )


def test_has_wildtype_row(standardized_dataset: pd.DataFrame):
    assert has_wildtype_row(standardized_dataset) is True

    df = standardized_dataset.copy()
    df["is_wildtype"] = False
    assert has_wildtype_row(df) is False


def test_count_invalid_sequences():
    df = pd.DataFrame({"seq": ["MKT", "ABJ", None, ""]})
    assert count_invalid_sequences(df, "seq") == 2