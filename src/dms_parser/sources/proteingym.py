from pathlib import Path
import pandas as pd


def download_proteingym_dataset(dataset_name: str, output_dir: str | Path | None = None) -> Path:
    """Download a ProteinGym dataset and return the local file path."""
    raise NotImplementedError


def load_proteingym_dataset(path: str | Path) -> pd.DataFrame:
    """Load a raw ProteinGym dataset."""
    raise NotImplementedError


def standardize_proteingym_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw ProteinGym columns to the internal schema."""
    raise NotImplementedError