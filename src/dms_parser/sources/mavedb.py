"""Utilities for downloading and loading datasets from MaveDB."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from dms_parser.io import (
    download_file,
    ensure_local_copy as ensure_local_dataset_copy,
    infer_filename_from_url,
    read_table,
)


def download_mavedb_dataset(
    url: str,
    output_dir: str | Path = "./data/mavedb",
    filename: Optional[str] = None,
    *,
    overwrite: bool = False,
) -> Path:
    """Download a MaveDB dataset file from a direct URL."""
    output_dir = Path(output_dir)

    if filename is None:
        filename = infer_filename_from_url(url, default_name="mavedb_dataset.csv")

    output_path = output_dir / filename

    return download_file(url, output_path, overwrite=overwrite)


def load_mavedb_dataset(
    path: str | Path,
    *,
    sep: str | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Load a MaveDB dataset from disk."""
    return read_table(path, sep=sep, **kwargs)


def load_mavedb_from_url(
    url: str,
    *,
    output_dir: str | Path = "./data/mavedb",
    filename: Optional[str] = None,
    overwrite: bool = False,
    sep: str | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Download and load a MaveDB dataset in one step."""
    path = download_mavedb_dataset(
        url=url,
        output_dir=output_dir,
        filename=filename,
        overwrite=overwrite,
    )
    return load_mavedb_dataset(path, sep=sep, **kwargs)


def ensure_local_copy(
    path_or_url: str | Path,
    *,
    output_dir: str | Path = "./data/mavedb",
    overwrite: bool = False,
) -> Path:
    """Ensure a MaveDB dataset is available locally."""
    return ensure_local_dataset_copy(
        path_or_url,
        output_dir=output_dir,
        default_name="mavedb_dataset.csv",
        overwrite=overwrite,
    )