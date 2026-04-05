"""Utilities for downloading and loading datasets from ProteinGym."""

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


def download_proteingym_dataset(
    url: str,
    output_dir: str | Path = "./data/proteingym",
    filename: Optional[str] = None,
    *,
    overwrite: bool = False,
) -> Path:
    """Download a ProteinGym dataset file from a direct URL."""
    output_dir = Path(output_dir)

    if filename is None:
        filename = infer_filename_from_url(url, default_name="proteingym_dataset.csv")

    output_path = output_dir / filename

    return download_file(url, output_path, overwrite=overwrite)


def load_proteingym_dataset(
    path: str | Path,
    *,
    sep: str | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Load a ProteinGym dataset from disk."""
    return read_table(path, sep=sep, **kwargs)


def load_proteingym_from_url(
    url: str,
    *,
    output_dir: str | Path = "./data/proteingym",
    filename: Optional[str] = None,
    overwrite: bool = False,
    sep: str | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Download and load a ProteinGym dataset in one step."""
    path = download_proteingym_dataset(
        url=url,
        output_dir=output_dir,
        filename=filename,
        overwrite=overwrite,
    )
    return load_proteingym_dataset(path, sep=sep, **kwargs)


def ensure_local_copy(
    path_or_url: str | Path,
    *,
    output_dir: str | Path = "./data/proteingym",
    overwrite: bool = False,
) -> Path:
    """Ensure a ProteinGym dataset is available locally."""
    return ensure_local_dataset_copy(
        path_or_url,
        output_dir=output_dir,
        default_name="proteingym_dataset.csv",
        overwrite=overwrite,
    )