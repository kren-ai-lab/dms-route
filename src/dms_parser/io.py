"""Input/output utilities for dms_parser."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
import requests

from dms_parser.exceptions import DownloadError, FileFormatError

logger = logging.getLogger(__name__)


def download_file(
    url: str,
    output_path: str | Path,
    *,
    overwrite: bool = False,
    chunk_size: int = 8192,
    timeout: int = 60,
) -> Path:
    """Download a URL and atomically publish it at a local path.

    The response is written to a temporary file in the destination directory.
    The destination is replaced only after response streaming and temporary
    file closure both complete successfully.
    """
    output_path = Path(output_path)
    logger.debug("Resolved download output path=%s", output_path)

    if output_path.exists() and not overwrite:
        logger.info("Using existing download destination path=%s", output_path)
        return output_path

    logger.info("Starting download destination=%s", output_path)
    logger.debug(
        "Requesting download endpoint=%s timeout=%s chunk_size=%s",
        _sanitize_url_for_logging(url),
        timeout,
        chunk_size,
    )
    temporary_path: Path | None = None
    failure: requests.RequestException | OSError | None = None
    downloaded_bytes = 0
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_path.parent,
            prefix=".download-",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            with requests.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        temporary_file.write(chunk)
                        downloaded_bytes += len(chunk)
        os.replace(temporary_path, output_path)
    except (requests.RequestException, OSError) as exc:
        failure = exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError as exc:
                if failure is None:
                    failure = exc

    if failure is not None:
        raise DownloadError(
            f"Failed to download file from {url!r}: {failure}"
        ) from failure

    logger.info("Completed download destination=%s", output_path)
    logger.debug(
        "Downloaded bytes=%d destination=%s",
        downloaded_bytes,
        output_path,
    )
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
    logger.debug("Reading table path=%s suffix=%s", path, suffix or "<none>")

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
    logger.debug(
        "Resolved table output path=%s suffix=%s rows=%d",
        path,
        suffix or "<none>",
        len(df),
    )

    try:
        if suffix == ".parquet":
            df.to_parquet(path, index=index, **kwargs)
            logger.info("Completed table write path=%s rows=%d", path, len(df))
            return path

        if sep is not None:
            df.to_csv(path, sep=sep, index=index, **kwargs)
            logger.info("Completed table write path=%s rows=%d", path, len(df))
            return path

        if suffix in {".tsv", ".txt"}:
            df.to_csv(path, sep="\t", index=index, **kwargs)
            logger.info("Completed table write path=%s rows=%d", path, len(df))
            return path

        if suffix in {".csv", ""}:
            df.to_csv(path, index=index, **kwargs)
            logger.info("Completed table write path=%s rows=%d", path, len(df))
            return path

        raise FileFormatError(
            f"Unsupported output format for {path!s}. Supported: .csv, .tsv, .txt, .parquet"
        )
    except Exception as exc:
        if isinstance(exc, FileFormatError):
            raise
        raise FileFormatError(f"Failed to write table to {path!s}: {exc}") from exc


def _sanitize_url_for_logging(url: str) -> str:
    """Return a URL endpoint without credentials, query parameters, or fragments."""
    try:
        parts = urlsplit(url)
        hostname = parts.hostname or ""
        if ":" in hostname:
            hostname = f"[{hostname}]"
        port = f":{parts.port}" if parts.port is not None else ""
        return urlunsplit((parts.scheme, f"{hostname}{port}", parts.path, "", ""))
    except (TypeError, ValueError):
        return "<invalid-url>"
