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


def test_add_wt_relative_score_ratio(simple_variant_df: pd.DataFrame):
    result = add_wt_relative_score(
        simple_variant_df,
        score_col="score_raw",
        method="ratio",
        output_col="rel",
    )

    assert np.isclose(result.loc[0, "rel"], 1.0)
    assert np.isclose(result.loc[1, "rel"], 0.8)
    assert np.isclose(result.loc[2, "rel"], 1.2)


def test_add_wt_relative_score_log_ratio(simple_variant_df: pd.DataFrame):
    result = add_wt_relative_score(
        simple_variant_df,
        score_col="score_raw",
        method="log_ratio",
        output_col="rel",
    )

    assert np.isclose(result.loc[0, "rel"], 0.0)
    assert result.loc[1, "rel"] < 0
    assert result.loc[2, "rel"] > 0


def test_add_wt_relative_score_log2_ratio(simple_variant_df: pd.DataFrame):
    result = add_wt_relative_score(
        simple_variant_df,
        score_col="score_raw",
        method="log2_ratio",
        output_col="rel",
    )

    assert np.isclose(result.loc[0, "rel"], 0.0)
    assert result.loc[1, "rel"] < 0
    assert result.loc[2, "rel"] > 0


def test_add_wt_relative_score_difference(simple_variant_df: pd.DataFrame):
    result = add_wt_relative_score(
        simple_variant_df,
        score_col="score_raw",
        method="difference",
        output_col="rel",
    )

    assert np.isclose(result.loc[0, "rel"], 0.0)
    assert np.isclose(result.loc[1, "rel"], -0.2)
    assert np.isclose(result.loc[2, "rel"], 0.2)


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