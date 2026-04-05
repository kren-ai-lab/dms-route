from pathlib import Path
import pandas as pd


def download_mavedb_dataset(dataset_id: str, output_dir: str | Path | None = None) -> Path:
    """Download a MaveDB dataset and return the local file path."""
    raise NotImplementedError


def load_mavedb_dataset(path: str | Path) -> pd.DataFrame:
    """Load a raw MaveDB dataset."""
    raise NotImplementedError


def standardize_mavedb_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw MaveDB columns to the internal schema."""
    raise NotImplementedError