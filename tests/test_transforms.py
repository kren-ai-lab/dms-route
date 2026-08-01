from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dms_parser.constants import NEUTRAL_LABEL
from dms_parser.exceptions import MissingWildTypeError
from dms_parser.transforms import (
    add_minmax,
    add_pseudo_binary_label,
    add_wt_relative_score,
    add_zscore,
    compute_wt_score,
)


def test_compute_wt_score(simple_variant_df: pd.DataFrame):
    wt_score = compute_wt_score(simple_variant_df, score_col="score_raw")
    assert wt_score == 1.0


def test_compute_wt_score_no_wt_raises(simple_variant_df: pd.DataFrame):
    df = simple_variant_df.copy()
    df["is_wildtype"] = False

    with pytest.raises(MissingWildTypeError):
        compute_wt_score(df, score_col="score_raw")


def test_compute_wt_score_ignores_scoreless_synthetic_wt():
    df = pd.DataFrame(
        {
            "score_raw": [np.nan, "not-numeric", 2.0, 4.0, 0.5],
            "is_wildtype": [True, True, True, True, False],
            "is_synthetic": [True, False, False, False, False],
            "status": ["OK", "OK", "OK", "OK", "OK"],
        }
    )

    assert compute_wt_score(df, "score_raw", status_col="status") == 3.0


def test_compute_wt_score_rejects_only_scoreless_wt():
    df = pd.DataFrame(
        {
            "score_raw": [np.nan, 0.5],
            "is_wildtype": [True, False],
            "status": ["OK", "OK"],
        }
    )

    with pytest.raises(MissingWildTypeError, match="No valid numeric wild-type score"):
        compute_wt_score(df, "score_raw", status_col="status")


def test_compute_wt_score_filters_status_before_numeric_scores():
    df = pd.DataFrame(
        {
            "score_raw": [10.0, 2.0],
            "is_wildtype": [True, True],
            "status": ["Error", "OK"],
        }
    )

    assert compute_wt_score(df, "score_raw", status_col="status") == 2.0


def test_scoreless_wt_does_not_create_relative_column():
    df = pd.DataFrame({"score_raw": [np.nan, 0.5], "is_wildtype": [True, False]})

    with pytest.raises(MissingWildTypeError, match="No valid numeric wild-type score"):
        add_wt_relative_score(df, "score_raw", output_col="relative")

    assert "relative" not in df.columns


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("ratio", [1.0, 0.8, 1.2, 0.1]),
        ("log_ratio", [0.0, np.log(0.8), np.log(1.2), np.log(0.1)]),
        ("log2_ratio", [0.0, np.log2(0.8), np.log2(1.2), np.log2(0.1)]),
        ("difference", [0.0, -0.2, 0.2, -0.9]),
    ],
)
def test_add_wt_relative_score_methods(
    method: str,
    expected: list[float],
    simple_variant_df: pd.DataFrame,
):
    result = add_wt_relative_score(
        simple_variant_df,
        score_col="score_raw",
        method=method,
        output_col="rel",
    )

    np.testing.assert_allclose(result["rel"], expected)


def test_add_wt_relative_score_invalid_method_raises(simple_variant_df: pd.DataFrame):
    with pytest.raises(ValueError):
        add_wt_relative_score(simple_variant_df, score_col="score_raw", method="bad")


def test_add_pseudo_binary_label_higher_is_better():
    df = pd.DataFrame({"score_log_ratio": [-0.5, -0.05, 0.0, 0.05, 0.5]})
    result = add_pseudo_binary_label(
        df,
        score_col="score_log_ratio",
        delta=0.1,
        output_col="label",
    )

    assert list(result["label"]) == [0, NEUTRAL_LABEL, NEUTRAL_LABEL, NEUTRAL_LABEL, 1]


def test_add_pseudo_binary_label_lower_is_better():
    df = pd.DataFrame({"score_log_ratio": [-0.5, 0.5]})
    result = add_pseudo_binary_label(
        df,
        score_col="score_log_ratio",
        delta=0.1,
        higher_is_better=False,
        output_col="label",
    )

    assert list(result["label"]) == [1, 0]


def test_add_zscore():
    df = pd.DataFrame({"score": [1.0, 2.0, 3.0]})
    result = add_zscore(df, "score", output_col="z")

    assert np.isclose(result["z"].mean(), 0.0)


def test_add_zscore_zero_std():
    df = pd.DataFrame({"score": [1.0, 1.0, 1.0]})
    result = add_zscore(df, "score", output_col="z")

    assert list(result["z"]) == [0.0, 0.0, 0.0]


def test_add_minmax():
    df = pd.DataFrame({"score": [1.0, 2.0, 3.0]})
    result = add_minmax(df, "score", output_col="mm")

    assert list(result["mm"]) == [0.0, 0.5, 1.0]


def test_add_minmax_constant_values():
    df = pd.DataFrame({"score": [2.0, 2.0, 2.0]})
    result = add_minmax(df, "score", output_col="mm")

    assert list(result["mm"]) == [0.0, 0.0, 0.0]
