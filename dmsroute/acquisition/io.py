"""Input/output utilities for dmsroute."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
import requests

from dmsroute.core.exceptions import DownloadError, FileFormatError

logger = logging.getLogger(__name__)

_DATASET_FILENAME = "standardized.csv"
_SUMMARY_CSV_FILENAME = "summary.csv"
_SUMMARY_JSON_FILENAME = "summary.json"
_DOWNLOAD_SUMMARY_CSV_FILENAME = "download-summary.csv"
_DOWNLOAD_SUMMARY_JSON_FILENAME = "download-summary.json"


class _IncompletePublicationRollbackError(OSError):
    """Raised when recovery files must be retained after rollback failure."""


def _dataset_bundle_paths(output_dir: str | Path) -> tuple[Path, Path, Path]:
    """Return the deterministic paths for one standardized dataset bundle."""
    root = Path(output_dir)
    return (
        root / _DATASET_FILENAME,
        root / _SUMMARY_CSV_FILENAME,
        root / _SUMMARY_JSON_FILENAME,
    )


def _preflight_dataset_bundle(
    output_dir: str | Path,
    *,
    overwrite: bool,
) -> tuple[Path, Path, Path]:
    """Validate bundle destinations without creating or modifying anything."""
    root = Path(output_dir)
    if root.exists() and not root.is_dir():
        raise NotADirectoryError(
            f"Dataset output directory is not a directory: {root}"
        )
    paths = _dataset_bundle_paths(root)
    if not overwrite:
        existing = [path for path in paths if path.exists()]
        if existing:
            joined = ", ".join(str(path) for path in existing)
            raise FileExistsError(
                f"Dataset output already exists; use overwrite=True: {joined}"
            )
    else:
        invalid = [path for path in paths if path.exists() and not path.is_file()]
        if invalid:
            joined = ", ".join(str(path) for path in invalid)
            raise IsADirectoryError(
                f"Dataset output targets must be regular files: {joined}"
            )
    return paths


def _preflight_download_batch(
    output_dir: str | Path,
    dataset_dirs: tuple[Path, ...],
    *,
    overwrite: bool,
) -> tuple[Path, Path]:
    """Preflight aggregate and per-dataset targets without side effects."""
    root = Path(output_dir)
    required_directories = {root}
    required_directories.update(path.parent for path in dataset_dirs)
    required_directories.update(dataset_dirs)
    for path in tuple(required_directories):
        required_directories.update(path.parents)
    invalid_directories = [
        path
        for path in required_directories
        if path.exists() and not path.is_dir()
    ]
    if invalid_directories:
        joined = ", ".join(
            str(path)
            for path in sorted(invalid_directories, key=str)
        )
        raise NotADirectoryError(
            f"Batch output paths must be directories: {joined}"
        )

    summary_paths = _download_summary_paths(root)
    _preflight_output_files(
        summary_paths,
        overwrite=overwrite,
        description="Batch summary output",
    )
    for dataset_dir in dataset_dirs:
        _preflight_dataset_bundle(dataset_dir, overwrite=overwrite)
    return summary_paths


def _download_summary_paths(output_dir: str | Path) -> tuple[Path, Path]:
    """Return deterministic aggregate summary paths for a download batch."""
    root = Path(output_dir)
    return (
        root / _DOWNLOAD_SUMMARY_CSV_FILENAME,
        root / _DOWNLOAD_SUMMARY_JSON_FILENAME,
    )


def _preflight_output_files(
    paths: tuple[Path, ...],
    *,
    overwrite: bool,
    description: str,
) -> None:
    """Validate deterministic output files without creating anything."""
    if not overwrite:
        existing = [path for path in paths if path.exists()]
        if existing:
            joined = ", ".join(str(path) for path in existing)
            raise FileExistsError(
                f"{description} already exists; use overwrite=True: {joined}"
            )
        return
    invalid = [path for path in paths if path.exists() and not path.is_file()]
    if invalid:
        joined = ", ".join(str(path) for path in invalid)
        raise IsADirectoryError(
            f"{description} targets must be regular files: {joined}"
        )


def _publish_dataset_bundle(
    dataset: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: str | Path,
    *,
    overwrite: bool,
) -> tuple[Path, Path, Path]:
    """Stage and publish one dataset and its summaries as a best-effort bundle."""
    output_dir = Path(output_dir)
    paths = _dataset_bundle_paths(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    staging_root = Path(
        tempfile.mkdtemp(
            prefix=".dmsroute-bundle-",
            dir=output_dir,
        )
    )
    cleanup_staging = True
    try:
        staged_paths = tuple(staging_root / path.name for path in paths)
        _write_csv_lf(dataset, staged_paths[0])
        _write_csv_lf(pd.DataFrame([summary]), staged_paths[1])
        with staged_paths[2].open("w", encoding="utf-8", newline="") as handle:
            handle.write(
                json.dumps(
                    [summary],
                    indent=2,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )

        _preflight_dataset_bundle(output_dir, overwrite=overwrite)
        try:
            _publish_staged_files(
                staged_paths,
                paths,
                staging_root=staging_root,
                overwrite=overwrite,
                description="Dataset output",
            )
        except _IncompletePublicationRollbackError as exc:
            cleanup_staging = False
            raise OSError(str(exc)) from exc.__cause__
    finally:
        if cleanup_staging:
            shutil.rmtree(staging_root, ignore_errors=True)

    return paths


def _publish_download_summary(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    overwrite: bool,
) -> tuple[Path, Path]:
    """Stage and publish the two aggregate download summary files."""
    output_dir = Path(output_dir)
    paths = _download_summary_paths(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=".dmsroute-download-summary-",
            dir=output_dir,
        )
    )
    cleanup_staging = True
    try:
        staged_paths = tuple(staging_root / path.name for path in paths)
        _write_csv_lf(pd.DataFrame(records), staged_paths[0])
        with staged_paths[1].open("w", encoding="utf-8", newline="") as handle:
            handle.write(
                json.dumps(
                    records,
                    indent=2,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
        _preflight_output_files(
            paths,
            overwrite=overwrite,
            description="Batch summary output",
        )
        try:
            _publish_staged_files(
                staged_paths,
                paths,
                staging_root=staging_root,
                overwrite=overwrite,
                description="Batch summary output",
            )
        except _IncompletePublicationRollbackError as exc:
            cleanup_staging = False
            raise OSError(str(exc)) from exc.__cause__
    finally:
        if cleanup_staging:
            shutil.rmtree(staging_root, ignore_errors=True)
    return paths


def _publish_staged_files(
    staged_paths: tuple[Path, ...],
    final_paths: tuple[Path, ...],
    *,
    staging_root: Path,
    overwrite: bool,
    description: str,
) -> None:
    """Publish staged files with backups and best-effort rollback."""
    backups: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for index, (staged_path, final_path) in enumerate(
            zip(staged_paths, final_paths, strict=True)
        ):
            if final_path.exists():
                if not overwrite:
                    raise FileExistsError(
                        f"Output appeared during publication: {final_path}"
                    )
                backup_path = staging_root / f"backup-{index}-{final_path.name}"
                os.replace(final_path, backup_path)
                backups.append((final_path, backup_path))
            os.replace(staged_path, final_path)
            published.append(final_path)
    except OSError as exc:
        rollback_errors = _rollback_dataset_bundle(backups, published)
        if rollback_errors:
            details = "; ".join(str(error) for error in rollback_errors)
            raise _IncompletePublicationRollbackError(
                f"{description} publication failed and rollback was incomplete; "
                f"recovery files remain in {staging_root}: {details}"
            ) from exc
        raise


def _write_csv_lf(table: pd.DataFrame, path: Path) -> None:
    """Write one UTF-8 CSV with LF line endings and no index."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        table.to_csv(handle, index=False, lineterminator="\n")


def _rollback_dataset_bundle(
    backups: list[tuple[Path, Path]],
    published: list[Path],
) -> list[OSError]:
    """Best-effort rollback after a multi-file bundle publication failure."""
    errors: list[OSError] = []
    for path in reversed(published):
        try:
            if path.exists():
                path.unlink()
        except OSError as exc:
            errors.append(exc)

    for final_path, backup_path in reversed(backups):
        try:
            if backup_path.exists():
                os.replace(backup_path, final_path)
        except OSError as exc:
            errors.append(exc)
    return errors


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
        _sanitize_url_for_logging(url), timeout, chunk_size,
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
    logger.debug("Downloaded bytes=%d destination=%s", downloaded_bytes, output_path)
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
    logger.debug("Resolved table output path=%s suffix=%s rows=%d", path, suffix or "<none>", len(df))

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
