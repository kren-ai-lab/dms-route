"""Transformations for DMS datasets."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from dms_parser.core._wildtype import resolve_wt_score
from dms_parser.core.constants import DEFAULT_EPSILON, NEUTRAL_LABEL
from dms_parser.core.exceptions import InvalidDatasetError, MissingWildTypeError

logger = logging.getLogger(__name__)


def _finite_values(df: pd.DataFrame, score_col: str, operation: str) -> pd.Series:
    """Return finite numeric values or reject an unsafe transformation domain."""
    values = pd.to_numeric(df[score_col], errors="coerce").astype(float)
    if not np.isfinite(values.to_numpy()).all():
        raise InvalidDatasetError(
            f"Transformation {operation!r} requires finite values in "
            f"{score_col!r}."
        )
    return values


def _validate_new_output_column(
    df: pd.DataFrame,
    output_col: str,
    operation: str,
) -> None:
    """Reject invalid or destructive generated-column assignments."""
    if not isinstance(output_col, str) or not output_col.strip():
        raise InvalidDatasetError(
            f"Transformation {operation!r} requires a non-empty output column."
        )
    if output_col in df.columns:
        raise InvalidDatasetError(
            f"Transformation {operation!r} cannot overwrite existing column "
            f"{output_col!r}."
        )


def _finite_result(values: pd.Series, operation: str) -> pd.Series:
    """Return a generated series only when every result is finite."""
    if not np.isfinite(values.to_numpy()).all():
        raise InvalidDatasetError(
            f"Transformation {operation!r} produced a non-finite result."
        )
    return values


def compute_wt_score(
    df: pd.DataFrame,
    score_col: str,
    wt_col: str = "is_wildtype",
    *,
    status_col: str | None = None,
    accepted_status: str = "OK",
) -> float:
    """Extract the wild-type reference score."""
    table = df.copy()
    if wt_col != "is_wildtype":
        table["is_wildtype"] = table[wt_col]
    if status_col is not None:
        table["status"] = table[status_col]
        table.loc[table["status"] != accepted_status, "is_wildtype"] = False
    table["score_raw"] = table[score_col]
    score, _, observed, _ = resolve_wt_score(
        table,
        None,
        dataset_id="dataset",
    )
    if score is None:
        if not observed:
            raise MissingWildTypeError("No wild-type row found in dataset.")
        raise MissingWildTypeError("No valid numeric wild-type score found in dataset.")
    return score


def add_wt_relative_score(
    df: pd.DataFrame,
    score_col: str,
    wt_col: str = "is_wildtype",
    method: str = "log_ratio",
    epsilon: float = DEFAULT_EPSILON,
    output_col: str | None = None,
    status_col: str | None = None,
    accepted_status: str = "OK",
    wt_score: float | None = None,
) -> pd.DataFrame:
    """Add WT-relative score to the dataset.

    Supported methods:
    - ratio: variant / wt
    - log_ratio: log(variant / wt)
    - log2_ratio: log2(variant / wt)
    - difference: variant - wt
    """
    df = df.copy()

    if method not in {"ratio", "log_ratio", "log2_ratio", "difference"}:
        raise ValueError(f"Unsupported method: {method}")
    if output_col is None:
        output_col = f"{score_col}_{method}"
    _validate_new_output_column(df, output_col, method)
    if wt_score is None:
        wt_score = compute_wt_score(
            df,
            score_col,
            wt_col,
            status_col=status_col,
            accepted_status=accepted_status,
        )
    try:
        resolved_wt_score = float(wt_score)
    except (TypeError, ValueError) as exc:
        raise InvalidDatasetError("WT score must be a finite number.") from exc
    if not np.isfinite(resolved_wt_score):
        raise InvalidDatasetError("WT score must be a finite number.")

    values = _finite_values(df, score_col, method)

    # ``epsilon`` remains in the public signature for compatibility. Domain
    # validation now makes numerical stabilization unnecessary, so the exact
    # documented formulas are used.
    _ = epsilon

    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        if method == "ratio":
            if resolved_wt_score == 0:
                raise InvalidDatasetError(
                    "Transformation 'ratio' requires a non-zero WT score."
                )
            rel = values / resolved_wt_score

        elif method in {"log_ratio", "log2_ratio"}:
            if resolved_wt_score == 0:
                raise InvalidDatasetError(
                    f"Transformation {method!r} requires a non-zero WT score."
                )
            same_sign = np.signbit(values.to_numpy()) == np.signbit(
                resolved_wt_score
            )
            if (values == 0).any() or not same_sign.all():
                raise InvalidDatasetError(
                    f"Transformation {method!r} requires strictly positive "
                    "score/WT ratios."
                )
            ratio = values / resolved_wt_score
            if np.isfinite(ratio.to_numpy()).all() and (ratio > 0).all():
                rel = np.log(ratio) if method == "log_ratio" else np.log2(ratio)
            else:
                absolute_values = np.abs(values)
                absolute_wt = abs(resolved_wt_score)
                if method == "log_ratio":
                    rel = np.log(absolute_values) - np.log(absolute_wt)
                else:
                    rel = np.log2(absolute_values) - np.log2(absolute_wt)

        elif method == "difference":
            rel = values - resolved_wt_score

    rel = _finite_result(rel, method)

    _validate_new_output_column(df, output_col, method)
    df[output_col] = rel
    logger.info(
        "Applied WT-relative transformation method=%s score_col=%s "
        "output_col=%s rows=%d",
        method, score_col, output_col, len(df),
    )

    return df


def add_pseudo_binary_label(
    df: pd.DataFrame,
    score_col: str,
    delta: float = 0.1,
    neutral_label: int = NEUTRAL_LABEL,
    higher_is_better: bool = True,
    output_col: str = "score_binary_like",
) -> pd.DataFrame:
    """Convert continuous score into pseudo-binary labels.

    Rules:
    - > +delta → 1
    - < -delta → 0
    - otherwise → neutral_label
    """
    df = df.copy()
    _validate_new_output_column(df, output_col, "pseudo_binary")
    values = df[score_col]

    labels = pd.Series(index=df.index, dtype="Int64")

    if higher_is_better:
        labels[values > delta] = 1
        labels[values < -delta] = 0
    else:
        labels[values < -delta] = 1
        labels[values > delta] = 0

    mask_neutral = (values >= -delta) & (values <= delta)
    labels[mask_neutral] = neutral_label

    _validate_new_output_column(df, output_col, "pseudo_binary")
    df[output_col] = labels
    logger.info(
        "Applied pseudo-binary transformation score_col=%s output_col=%s rows=%d",
        score_col, output_col, len(df),
    )

    return df


def add_zscore(
    df: pd.DataFrame,
    score_col: str,
    output_col: str | None = None,
) -> pd.DataFrame:
    """Apply z-score normalization."""
    df = df.copy()
    resolved_output_col = output_col or f"{score_col}_zscore"
    _validate_new_output_column(df, resolved_output_col, "zscore")

    values = _finite_values(df, score_col, "zscore")
    mean = values.mean()
    std = values.std()

    if len(values) <= 1 or std == 0:
        df[resolved_output_col] = 0.0
        logger.info(
            "Applied z-score transformation score_col=%s output_col=%s rows=%d",
            score_col, resolved_output_col, len(df),
        )
        return df

    z = (values - mean) / std

    df[resolved_output_col] = z
    logger.info("Applied z-score transformation score_col=%s output_col=%s rows=%d", score_col, resolved_output_col, len(df))

    return df


def add_minmax(
    df: pd.DataFrame,
    score_col: str,
    output_col: str | None = None,
) -> pd.DataFrame:
    """Apply min-max normalization."""
    df = df.copy()
    resolved_output_col = output_col or f"{score_col}_minmax"
    _validate_new_output_column(df, resolved_output_col, "minmax")

    values = _finite_values(df, score_col, "minmax")
    min_val = values.min()
    max_val = values.max()

    if max_val == min_val:
        df[resolved_output_col] = 0.0
        logger.info(
            "Applied min-max transformation score_col=%s output_col=%s rows=%d",
            score_col, resolved_output_col, len(df),
        )
        return df

    scaled = (values - min_val) / (max_val - min_val)

    df[resolved_output_col] = scaled
    logger.info("Applied min-max transformation score_col=%s output_col=%s rows=%d", score_col, resolved_output_col, len(df))

    return df
