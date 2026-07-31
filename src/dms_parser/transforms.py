"""Transformations for DMS datasets."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from dms_parser.constants import DEFAULT_EPSILON, NEUTRAL_LABEL
from dms_parser.exceptions import MissingWildTypeError

logger = logging.getLogger(__name__)


def compute_wt_score(
    df: pd.DataFrame,
    score_col: str,
    wt_col: str = "is_wildtype",
) -> float:
    """Extract the wild-type reference score."""
    wt_rows = df[df[wt_col] == True]

    if wt_rows.empty:
        raise MissingWildTypeError("No wild-type row found in dataset.")

    if len(wt_rows) > 1:
        # Average multiple WT measurements when a dataset provides them.
        return wt_rows[score_col].mean()

    return wt_rows[score_col].iloc[0]


def add_wt_relative_score(
    df: pd.DataFrame,
    score_col: str,
    wt_col: str = "is_wildtype",
    method: str = "log_ratio",
    epsilon: float = DEFAULT_EPSILON,
    output_col: str | None = None,
) -> pd.DataFrame:
    """Add WT-relative score to the dataset.

    Supported methods:
    - ratio: variant / wt
    - log_ratio: log(variant / wt)
    - log2_ratio: log2(variant / wt)
    - difference: variant - wt
    """
    df = df.copy()

    wt_score = compute_wt_score(df, score_col, wt_col)

    values = df[score_col].astype(float)

    if method == "ratio":
        rel = (values + epsilon) / (wt_score + epsilon)

    elif method == "log_ratio":
        rel = np.log((values + epsilon) / (wt_score + epsilon))

    elif method == "log2_ratio":
        rel = np.log2((values + epsilon) / (wt_score + epsilon))

    elif method == "difference":
        rel = values - wt_score

    else:
        raise ValueError(f"Unsupported method: {method}")

    if output_col is None:
        output_col = f"{score_col}_{method}"

    df[output_col] = rel
    logger.info(
        "Applied WT-relative transformation method=%s score_col=%s "
        "output_col=%s rows=%d",
        method,
        score_col,
        output_col,
        len(df),
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

    df[output_col] = labels
    logger.info(
        "Applied pseudo-binary transformation score_col=%s output_col=%s rows=%d",
        score_col,
        output_col,
        len(df),
    )

    return df


def add_zscore(
    df: pd.DataFrame,
    score_col: str,
    output_col: str | None = None,
) -> pd.DataFrame:
    """Apply z-score normalization."""
    df = df.copy()

    values = df[score_col].astype(float)
    mean = values.mean()
    std = values.std()

    if std == 0:
        resolved_output_col = output_col or f"{score_col}_zscore"
        df[resolved_output_col] = 0.0
        logger.info(
            "Applied z-score transformation score_col=%s output_col=%s rows=%d",
            score_col,
            resolved_output_col,
            len(df),
        )
        return df

    z = (values - mean) / std

    if output_col is None:
        output_col = f"{score_col}_zscore"

    df[output_col] = z
    logger.info(
        "Applied z-score transformation score_col=%s output_col=%s rows=%d",
        score_col,
        output_col,
        len(df),
    )

    return df


def add_minmax(
    df: pd.DataFrame,
    score_col: str,
    output_col: str | None = None,
) -> pd.DataFrame:
    """Apply min-max normalization."""
    df = df.copy()

    values = df[score_col].astype(float)
    min_val = values.min()
    max_val = values.max()

    if max_val == min_val:
        resolved_output_col = output_col or f"{score_col}_minmax"
        df[resolved_output_col] = 0.0
        logger.info(
            "Applied min-max transformation score_col=%s output_col=%s rows=%d",
            score_col,
            resolved_output_col,
            len(df),
        )
        return df

    scaled = (values - min_val) / (max_val - min_val)

    if output_col is None:
        output_col = f"{score_col}_minmax"

    df[output_col] = scaled
    logger.info(
        "Applied min-max transformation score_col=%s output_col=%s rows=%d",
        score_col,
        output_col,
        len(df),
    )

    return df
