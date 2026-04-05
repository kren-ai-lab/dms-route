import pandas as pd


def validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Ensure that all required columns are present."""
    raise NotImplementedError


def validate_variant_column(df: pd.DataFrame, variant_col: str) -> None:
    """Ensure that the variant column is present and valid."""
    raise NotImplementedError