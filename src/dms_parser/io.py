"""Input/output utilities for dms_parser."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from dms_parser.exceptions import DownloadError, FileFormatError


def download_file(
    url: str,
    output_path: str | Path,
    *,
    overwrite: bool = False,
    chunk_size: int = 8192,
    timeout: int = 60,
) -> Path:
    """Download a file from a URL to a local path."""
    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()

            with open(output_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        handle.write(chunk)
    except requests.RequestException as exc:
        raise DownloadError(f"Failed to download file from {url!r}: {exc}") from exc

    return output_path


def infer_filename_from_url(
    url: str,
    *,
    default_name: str = "dataset.csv",
) -> str:
    """Infer a reasonable filename from a URL."""
    name = Path(url).name

    if not name:
        return default_name

    return name.replace("?", "_").replace("&", "_")


def ensure_local_copy(
    path_or_url: str | Path,
    *,
    output_dir: str | Path,
    default_name: str = "dataset.csv",
    overwrite: bool = False,
) -> Path:
    """Ensure a dataset exists locally."""
    candidate = Path(path_or_url)

    if candidate.exists():
        return candidate

    filename = infer_filename_from_url(str(path_or_url), default_name=default_name)
    output_path = Path(output_dir) / filename

    return download_file(
        str(path_or_url),
        output_path,
        overwrite=overwrite,
    )


def read_table(
    path: str | Path,
    *,
    sep: str | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Read a tabular file into a pandas DataFrame."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    try:
        if suffix == ".parquet":
            return pd.read_parquet(path, **kwargs)

        if sep is not None:
            return pd.read_csv(path, sep=sep, **kwargs)

        if suffix in {".tsv", ".txt"}:
            return pd.read_csv(path, sep="\t", **kwargs)

        if suffix in {".csv", ""}:
            return pd.read_csv(path, **kwargs)

        raise FileFormatError(
            f"Unsupported file format for {path!s}. Supported: .csv, .tsv, .txt, .parquet"
        )
    except Exception as exc:
        if isinstance(exc, FileFormatError):
            raise
        raise FileFormatError(f"Failed to read table from {path!s}: {exc}") from exc


def write_table(
    df: pd.DataFrame,
    path: str | Path,
    *,
    index: bool = False,
    sep: str | None = None,
    **kwargs,
) -> Path:
    """Write a DataFrame to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    suffix = path.suffix.lower()

    try:
        if suffix == ".parquet":
            df.to_parquet(path, index=index, **kwargs)
            return path

        if sep is not None:
            df.to_csv(path, sep=sep, index=index, **kwargs)
            return path

        if suffix in {".tsv", ".txt"}:
            df.to_csv(path, sep="\t", index=index, **kwargs)
            return path

        if suffix in {".csv", ""}:
            df.to_csv(path, index=index, **kwargs)
            return path

        raise FileFormatError(
            f"Unsupported output format for {path!s}. Supported: .csv, .tsv, .txt, .parquet"
        )
    except Exception as exc:
        if isinstance(exc, FileFormatError):
            raise
        raise FileFormatError(f"Failed to write table to {path!s}: {exc}") from exc