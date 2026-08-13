from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dmsroute.core.constants import NEUTRAL_LABEL
from dmsroute.core.exceptions import (
    InvalidDatasetError,
    MissingWildTypeError,
    WildTypeConflictError,
)
from dmsroute.core.transforms import (
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
            "score_raw": [np.nan, "not-numeric", 2.0, 2.0, 0.5],
            "is_wildtype": [True, True, True, True, False],
            "is_synthetic": [True, False, False, False, False],
            "status": ["OK", "OK", "OK", "OK", "OK"],
        }
    )

    assert compute_wt_score(df, "score_raw", status_col="status") == 2.0


def test_compute_wt_score_rejects_conflicting_observed_rows():
    df = pd.DataFrame(
        {
            "score_raw": [2.0, 4.0],
            "is_wildtype": [True, True],
            "status": ["OK", "OK"],
        }
    )

    with pytest.raises(WildTypeConflictError, match="Conflicting WT scores"):
        compute_wt_score(df, "score_raw", status_col="status")


def test_direct_wt_transform_rejects_conflicting_observed_rows():
    df = pd.DataFrame(
        {
            "score_raw": [2.0, 4.0, 1.0],
            "is_wildtype": [True, True, False],
            "status": ["OK", "OK", "OK"],
        }
    )

    with pytest.raises(WildTypeConflictError, match="Conflicting WT scores"):
        add_wt_relative_score(
            df,
            "score_raw",
            method="difference",
            output_col="relative",
            status_col="status",
        )


def test_conflicting_observed_wt_message_is_compact() -> None:
    row_count = 10_000
    df = pd.DataFrame(
        {
            "score_raw": np.tile([2.0, 4.0], row_count // 2),
            "is_wildtype": [True] * row_count,
            "status": ["OK"] * row_count,
        }
    )

    with pytest.raises(WildTypeConflictError) as exc_info:
        compute_wt_score(df, "score_raw", status_col="status")

    message = str(exc_info.value)
    assert message.startswith("Conflicting WT scores for ")
    assert message.count("observed_wildtype_row") == 1
    assert len(message) < 500


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


@pytest.mark.parametrize("wt_score", [2.0, 0.0, -2.0])
def test_difference_accepts_any_finite_wt_score(wt_score: float):
    df = pd.DataFrame({"score_raw": [3.0], "is_wildtype": [False]})

    result = add_wt_relative_score(
        df,
        "score_raw",
        method="difference",
        output_col="difference",
        wt_score=wt_score,
    )

    assert result["difference"].tolist() == [3.0 - wt_score]


def test_ratio_rejects_zero_wt_score():
    df = pd.DataFrame({"score_raw": [1.0], "is_wildtype": [False]})

    with pytest.raises(InvalidDatasetError, match="non-zero WT score"):
        add_wt_relative_score(df, "score_raw", method="ratio", wt_score=0.0)


@pytest.mark.parametrize("method", ["log_ratio", "log2_ratio"])
def test_log_ratios_reject_non_positive_domains(method: str):
    df = pd.DataFrame({"score_raw": [-1.0], "is_wildtype": [False]})

    with pytest.raises(InvalidDatasetError, match="strictly positive"):
        add_wt_relative_score(df, "score_raw", method=method, wt_score=1.0)


def test_wt_relative_transform_rejects_non_finite_source_values():
    df = pd.DataFrame({"score_raw": [np.inf], "is_wildtype": [False]})

    with pytest.raises(InvalidDatasetError, match="finite values"):
        add_wt_relative_score(
            df,
            "score_raw",
            method="difference",
            wt_score=1.0,
        )


@pytest.mark.parametrize(
    ("method", "score", "wt_score"),
    [
        ("difference", np.finfo(float).max, -np.finfo(float).max),
        ("ratio", np.finfo(float).max, np.nextafter(0.0, 1.0)),
    ],
)
def test_wt_relative_transform_rejects_non_finite_results_without_assignment(
    method: str,
    score: float,
    wt_score: float,
) -> None:
    df = pd.DataFrame({"score_raw": [score], "is_wildtype": [False]})

    with pytest.raises(InvalidDatasetError, match="produced a non-finite result"):
        add_wt_relative_score(
            df,
            "score_raw",
            method=method,
            output_col="relative",
            wt_score=wt_score,
        )

    assert "relative" not in df.columns
    assert df["score_raw"].iloc[0] == score


@pytest.mark.parametrize("method", ["log_ratio", "log2_ratio"])
def test_log_ratios_avoid_extreme_intermediate_overflow(method: str) -> None:
    smallest = np.nextafter(0.0, 1.0)
    largest = np.finfo(float).max
    df = pd.DataFrame(
        {"score_raw": [smallest, largest], "is_wildtype": [False, False]}
    )

    result = add_wt_relative_score(
        df,
        "score_raw",
        method=method,
        output_col="relative",
        wt_score=smallest,
    )

    assert np.isfinite(result["relative"]).all()
    assert result["relative"].iloc[0] == 0.0


@pytest.mark.parametrize("method", ["log_ratio", "log2_ratio"])
def test_log_ratios_reject_zero_underflow_domain(method: str) -> None:
    df = pd.DataFrame({"score_raw": [0.0], "is_wildtype": [False]})

    with pytest.raises(InvalidDatasetError, match="strictly positive"):
        add_wt_relative_score(df, "score_raw", method=method, wt_score=1.0)


@pytest.mark.parametrize(
    ("operation", "kwargs"),
    [
        (
            add_wt_relative_score,
            {
                "score_col": "score_raw",
                "output_col": "score_raw",
                "wt_score": 1.0,
            },
        ),
        (
            add_pseudo_binary_label,
            {"score_col": "score_raw", "output_col": "score_raw"},
        ),
    ],
)
def test_generated_transform_cannot_overwrite_existing_column(
    operation,
    kwargs: dict[str, object],
) -> None:
    df = pd.DataFrame({"score_raw": [1.0], "is_wildtype": [False]})

    with pytest.raises(InvalidDatasetError, match="cannot overwrite"):
        operation(df, **kwargs)

    assert df["score_raw"].tolist() == [1.0]


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


def test_add_zscore_single_finite_value_is_zero():
    df = pd.DataFrame({"score_raw": [7.5]})

    result = add_zscore(df, "score_raw", output_col="z")

    assert result.columns.tolist() == ["score_raw", "z"]
    assert result["score_raw"].tolist() == [7.5]
    assert result["z"].tolist() == [0.0]


def test_add_minmax():
    df = pd.DataFrame({"score": [1.0, 2.0, 3.0]})
    result = add_minmax(df, "score", output_col="mm")

    assert list(result["mm"]) == [0.0, 0.5, 1.0]


def test_add_minmax_constant_values():
    df = pd.DataFrame({"score": [2.0, 2.0, 2.0]})
    result = add_minmax(df, "score", output_col="mm")

    assert list(result["mm"]) == [0.0, 0.0, 0.0]


@pytest.mark.parametrize("transform", [add_zscore, add_minmax])
def test_non_wt_transforms_reject_non_finite_values_without_requiring_wt(
    transform,
):
    valid = transform(pd.DataFrame({"score": [1.0, 2.0]}), "score")
    assert len(valid) == 2

    with pytest.raises(InvalidDatasetError, match="finite values"):
        transform(pd.DataFrame({"score": [1.0, np.nan]}), "score")
