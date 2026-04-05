from pathlib import Path
import pandas as pd


def read_table(path: str | Path, sep: str | None = None) -> pd.DataFrame:
    """Read a tabular file into a pandas DataFrame."""
    raise NotImplementedError


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    """Write a DataFrame to disk."""
    raise NotImplementedError