import pandas as pd


def add_wt_relative_score(
    df: pd.DataFrame,
    score_col: str,
    wt_col: str = "is_wildtype",
    method: str = "log_ratio",
) -> pd.DataFrame:
    """Add a wild-type-relative score column."""
    raise NotImplementedError


def add_pseudo_binary_label(
    df: pd.DataFrame,
    score_col: str,
    threshold: float = 0.0,
    neutral_label: int = 999,
) -> pd.DataFrame:
    """Add a pseudo-binary label derived from a continuous score."""
    raise NotImplementedError