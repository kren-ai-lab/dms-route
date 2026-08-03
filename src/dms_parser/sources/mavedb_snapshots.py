"""Managed MaveDB bulk snapshots published through Zenodo."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import stat
import tarfile
import tempfile
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import BinaryIO, Any
from urllib.parse import urlsplit

import requests

from dms_parser.cache import FilesystemCache
from dms_parser.catalog import optional_text
from dms_parser.exceptions import (
    InvalidSnapshotSelectorError,
    MaveDBSnapshotError,
)

logger = logging.getLogger(__name__)

MAVEDB_ZENODO_CONCEPT_RECORD_ID = "11201736"
MAVEDB_ZENODO_CONCEPT_DOI = "10.5281/zenodo.11201736"
MAVEDB_ZENODO_API_URL = (
    f"https://zenodo.org/api/records/{MAVEDB_ZENODO_CONCEPT_RECORD_ID}"
)

_ARCHIVE_PATTERN = re.compile(r"^mavedb-dump\..+\.(tar\.gz|zip)$")
_CHECKSUM_PATTERN = re.compile(r"^([A-Za-z0-9_-]+):([0-9A-Fa-f]+)$")
_CACHE_METADATA_FILENAME = "snapshot.json"
_MAIN_JSON_FILENAME = "main.json"
_DEFAULT_TIMEOUT = 60
_DEFAULT_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class MaveDBSnapshotRecord:
    """Immutable metadata for one concrete Zenodo snapshot record."""

    record_id: str
    doi: str
    concept_doi: str
    publication_date: str | None
    filename: str
    size: int
    checksum: str
    download_url: str


@dataclass(frozen=True)
class MaveDBSnapshot:
    """Local paths and metadata for one prepared MaveDB snapshot."""

    record: MaveDBSnapshotRecord
    archive_path: Path
    main_json_path: Path
    cache_hit: bool


def resolve_mavedb_snapshot(
    selector: str | int = "latest",
    *,
    session: requests.Session | None = None,
    timeout: int = _DEFAULT_TIMEOUT,
) -> MaveDBSnapshotRecord:
    """Resolve ``latest`` or one concrete MaveDB Zenodo record."""
    record_id = _validate_selector(selector)
    _validate_positive_integer(timeout, "timeout")
    endpoint = (
        MAVEDB_ZENODO_API_URL
        if record_id is None
        else f"https://zenodo.org/api/records/{record_id}"
    )
    http = session if session is not None else requests.Session()
    logger.info("Resolving MaveDB snapshot selector=%s", selector)
    response = http.get(endpoint, timeout=timeout, allow_redirects=True)
    response.raise_for_status()
    try:
        payload = response.json()
    except (ValueError, UnicodeError) as exc:
        raise MaveDBSnapshotError(
            "Zenodo snapshot metadata is not valid JSON."
        ) from exc
    record = _snapshot_record_from_zenodo(payload, expected_id=record_id)
    logger.info("Resolved MaveDB snapshot record_id=%s", record.record_id)
    return record


def fetch_mavedb_snapshot(
    selector: str | int = "latest",
    *,
    cache: FilesystemCache | None = None,
    refresh: bool = False,
    session: requests.Session | None = None,
    timeout: int = _DEFAULT_TIMEOUT,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> MaveDBSnapshot:
    """Resolve, download, verify, and safely prepare one MaveDB snapshot."""
    selected_record_id = _validate_selector(selector)
    if cache is not None and not isinstance(cache, FilesystemCache):
        raise MaveDBSnapshotError("cache must be a FilesystemCache or None.")
    if not isinstance(refresh, bool):
        raise MaveDBSnapshotError("refresh must be a boolean.")
    _validate_positive_integer(timeout, "timeout")
    _validate_positive_integer(chunk_size, "chunk_size")
    resolved_cache = cache or FilesystemCache(
        Path.home() / ".cache" / "dms-parser"
    )

    if selected_record_id is not None and not refresh:
        cached = _load_cached_snapshot(resolved_cache, selected_record_id)
        if cached is not None:
            return cached

    record = resolve_mavedb_snapshot(
        selector,
        session=session,
        timeout=timeout,
    )
    if not refresh:
        cached = _load_cached_snapshot(resolved_cache, record.record_id)
        if cached is not None:
            return cached

    return _acquire_snapshot(
        record,
        cache=resolved_cache,
        session=session,
        timeout=timeout,
        chunk_size=chunk_size,
    )


def _validate_selector(selector: object) -> str | None:
    """Return a concrete record ID, or ``None`` for the latest selector."""
    if isinstance(selector, bool):
        raise InvalidSnapshotSelectorError(
            "Snapshot selector must be 'latest' or a positive Zenodo record ID."
        )
    if isinstance(selector, int):
        if selector > 0:
            return str(selector)
        raise InvalidSnapshotSelectorError(
            "Zenodo record ID must be a positive integer."
        )
    if not isinstance(selector, str):
        raise InvalidSnapshotSelectorError(
            "Snapshot selector must be 'latest' or a positive Zenodo record ID."
        )
    if selector == "latest":
        return None
    if re.fullmatch(r"[1-9][0-9]*", selector) is None:
        raise InvalidSnapshotSelectorError(
            "Snapshot selector must be 'latest' or a positive Zenodo record ID."
        )
    return selector


def _validate_positive_integer(value: object, field: str) -> None:
    """Validate one positive non-Boolean integer option."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MaveDBSnapshotError(f"{field} must be a positive integer.")


def _snapshot_record_from_zenodo(
    payload: object,
    *,
    expected_id: str | None,
) -> MaveDBSnapshotRecord:
    """Validate Zenodo metadata and select exactly one supported archive."""
    if not isinstance(payload, Mapping):
        raise MaveDBSnapshotError("Zenodo snapshot metadata must be a JSON object.")

    record_id = _positive_record_id(payload.get("id"), "record id")
    if expected_id is not None and record_id != expected_id:
        raise MaveDBSnapshotError(
            f"Zenodo returned record {record_id!r} instead of {expected_id!r}."
        )
    concept_record_id = _positive_record_id(
        payload.get("conceptrecid"),
        "concept record id",
    )
    concept_doi = optional_text(payload.get("conceptdoi"))
    if (
        concept_record_id != MAVEDB_ZENODO_CONCEPT_RECORD_ID
        or concept_doi is None
        or concept_doi.casefold() != MAVEDB_ZENODO_CONCEPT_DOI.casefold()
    ):
        raise MaveDBSnapshotError(
            "Zenodo record does not belong to the official MaveDB concept record."
        )

    doi = optional_text(payload.get("doi"))
    if doi is None:
        raise MaveDBSnapshotError("Zenodo snapshot metadata is missing its DOI.")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise MaveDBSnapshotError(
            "Zenodo snapshot metadata field 'metadata' must be a JSON object."
        )
    publication_date = optional_text(metadata.get("publication_date"))
    archive = _select_archive(payload.get("files"))
    return MaveDBSnapshotRecord(
        record_id=record_id,
        doi=doi,
        concept_doi=concept_doi,
        publication_date=publication_date,
        filename=archive["filename"],
        size=archive["size"],
        checksum=archive["checksum"],
        download_url=archive["download_url"],
    )


def _positive_record_id(value: object, field: str) -> str:
    """Normalize one positive Zenodo record identifier."""
    if isinstance(value, bool):
        raise MaveDBSnapshotError(f"Zenodo {field} must be a positive integer.")
    if isinstance(value, int) and value > 0:
        return str(value)
    if isinstance(value, str) and re.fullmatch(r"[1-9][0-9]*", value):
        return value
    raise MaveDBSnapshotError(f"Zenodo {field} must be a positive integer.")


def _select_archive(files: object) -> dict[str, Any]:
    """Select and validate exactly one official MaveDB bulk archive."""
    if not isinstance(files, list):
        raise MaveDBSnapshotError("Zenodo snapshot field 'files' must be a list.")
    candidates: list[Mapping[str, Any]] = []
    for value in files:
        if not isinstance(value, Mapping):
            raise MaveDBSnapshotError(
                "Every Zenodo snapshot file entry must be a JSON object."
            )
        filename = optional_text(value.get("key"))
        if filename is None:
            raise MaveDBSnapshotError(
                "Zenodo snapshot file metadata is missing its filename."
            )
        if _ARCHIVE_PATTERN.fullmatch(filename):
            candidates.append(value)

    if not candidates:
        raise MaveDBSnapshotError(
            "Zenodo record contains no supported MaveDB bulk archive."
        )
    if len(candidates) != 1:
        raise MaveDBSnapshotError(
            "Zenodo record contains multiple supported MaveDB bulk archives."
        )

    selected = candidates[0]
    filename = optional_text(selected.get("key"))
    assert filename is not None
    _validate_safe_filename(filename)
    size = selected.get("size")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise MaveDBSnapshotError(
            "Zenodo archive size must be a positive integer."
        )
    checksum = _normalize_checksum(selected.get("checksum"))
    links = selected.get("links")
    if not isinstance(links, Mapping):
        raise MaveDBSnapshotError(
            "Zenodo archive links must be a JSON object."
        )
    download_url = optional_text(links.get("content"))
    if download_url is None or not _is_http_url(download_url):
        raise MaveDBSnapshotError(
            "Zenodo archive content URL is unavailable or invalid."
        )
    return {
        "filename": filename,
        "size": size,
        "checksum": checksum,
        "download_url": download_url,
    }


def _validate_safe_filename(filename: str) -> None:
    """Reject archive filenames that could escape their cache directory."""
    if (
        Path(filename).name != filename
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
    ):
        raise MaveDBSnapshotError("Zenodo archive filename is unsafe.")


def _normalize_checksum(value: object) -> str:
    """Validate a Zenodo checksum and return its canonical representation."""
    if not isinstance(value, str):
        raise MaveDBSnapshotError("Zenodo archive checksum is missing or malformed.")
    match = _CHECKSUM_PATTERN.fullmatch(value)
    if match is None:
        raise MaveDBSnapshotError("Zenodo archive checksum is missing or malformed.")
    algorithm = match.group(1).lower()
    digest = match.group(2).lower()
    try:
        checksum = hashlib.new(algorithm)
    except (ValueError, TypeError) as exc:
        raise MaveDBSnapshotError(
            f"Zenodo archive checksum algorithm {algorithm!r} is unsupported."
        ) from exc
    if checksum.digest_size <= 0 or len(digest) != checksum.digest_size * 2:
        raise MaveDBSnapshotError("Zenodo archive checksum digest is malformed.")
    return f"{algorithm}:{digest}"


def _is_http_url(value: str) -> bool:
    """Return whether a URL is an absolute HTTP(S) URL."""
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _snapshot_root(cache: FilesystemCache) -> Path:
    """Return the root for managed snapshot versions."""
    return cache.root / "mavedb" / "snapshots"


def _snapshot_entry(cache: FilesystemCache, record_id: str) -> Path:
    """Return the exact versioned cache directory for one record."""
    return _snapshot_root(cache) / record_id


def _load_cached_snapshot(
    cache: FilesystemCache,
    record_id: str,
) -> MaveDBSnapshot | None:
    """Load and fully validate one concrete cached snapshot."""
    entry = _snapshot_entry(cache, record_id)
    if not entry.exists():
        return None
    if not entry.is_dir() or entry.is_symlink():
        raise MaveDBSnapshotError(
            f"Cached MaveDB snapshot location is invalid: {entry}"
        )
    metadata_path = entry / _CACHE_METADATA_FILENAME
    if not metadata_path.is_file() or metadata_path.is_symlink():
        raise MaveDBSnapshotError(
            f"Cached MaveDB snapshot metadata is missing: {metadata_path}"
        )
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MaveDBSnapshotError(
            f"Could not load cached MaveDB snapshot metadata: {exc}"
        ) from exc
    record, archive_format, main_size, main_checksum = _record_from_cache_metadata(
        metadata,
        expected_id=record_id,
    )
    if archive_format != _archive_format(record.filename):
        raise MaveDBSnapshotError("Cached MaveDB archive format is inconsistent.")

    archive_path = entry / record.filename
    main_json_path = entry / _MAIN_JSON_FILENAME
    _validate_cached_file(
        archive_path,
        expected_size=record.size,
        checksum=record.checksum,
        description="archive",
    )
    _validate_cached_file(
        main_json_path,
        expected_size=main_size,
        checksum=f"sha256:{main_checksum}",
        description="main.json",
    )
    logger.info("Reusing cached MaveDB snapshot record_id=%s", record_id)
    return MaveDBSnapshot(
        record=record,
        archive_path=archive_path,
        main_json_path=main_json_path,
        cache_hit=True,
    )


def _record_from_cache_metadata(
    value: object,
    *,
    expected_id: str,
) -> tuple[MaveDBSnapshotRecord, str, int, str]:
    """Validate deterministic metadata stored beside a cached snapshot."""
    if not isinstance(value, Mapping):
        raise MaveDBSnapshotError(
            "Cached MaveDB snapshot metadata must be a JSON object."
        )
    required = {
        "record_id",
        "doi",
        "concept_doi",
        "publication_date",
        "archive_filename",
        "archive_size",
        "checksum",
        "download_url",
        "archive_format",
        "main_json_size",
        "main_json_sha256",
    }
    missing = required.difference(value)
    if missing:
        raise MaveDBSnapshotError(
            "Cached MaveDB snapshot metadata is missing fields: "
            + ", ".join(sorted(missing))
            + "."
        )
    record_id = _positive_record_id(value["record_id"], "record id")
    if record_id != expected_id:
        raise MaveDBSnapshotError(
            "Cached MaveDB snapshot record ID does not match its directory."
        )
    doi = optional_text(value["doi"])
    concept_doi = optional_text(value["concept_doi"])
    filename = optional_text(value["archive_filename"])
    download_url = optional_text(value["download_url"])
    if doi is None or concept_doi is None or filename is None:
        raise MaveDBSnapshotError(
            "Cached MaveDB snapshot contains empty required metadata."
        )
    if concept_doi.casefold() != MAVEDB_ZENODO_CONCEPT_DOI.casefold():
        raise MaveDBSnapshotError(
            "Cached snapshot does not belong to the official MaveDB concept."
        )
    _validate_safe_filename(filename)
    if download_url is None or not _is_http_url(download_url):
        raise MaveDBSnapshotError("Cached snapshot download URL is invalid.")
    size = value["archive_size"]
    main_size = value["main_json_size"]
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
        or isinstance(main_size, bool)
        or not isinstance(main_size, int)
        or main_size < 0
    ):
        raise MaveDBSnapshotError("Cached snapshot file size metadata is invalid.")
    checksum = _normalize_checksum(value["checksum"])
    main_checksum = value["main_json_sha256"]
    if (
        not isinstance(main_checksum, str)
        or re.fullmatch(r"[0-9a-f]{64}", main_checksum) is None
    ):
        raise MaveDBSnapshotError("Cached main.json checksum is invalid.")
    archive_format = value["archive_format"]
    if archive_format not in {"tar.gz", "zip"}:
        raise MaveDBSnapshotError("Cached MaveDB archive format is unsupported.")
    publication_date = value["publication_date"]
    if publication_date is not None:
        publication_date = optional_text(publication_date)
        if publication_date is None:
            raise MaveDBSnapshotError(
                "Cached snapshot publication date is invalid."
            )
    return (
        MaveDBSnapshotRecord(
            record_id=record_id,
            doi=doi,
            concept_doi=concept_doi,
            publication_date=publication_date,
            filename=filename,
            size=size,
            checksum=checksum,
            download_url=download_url,
        ),
        archive_format,
        main_size,
        main_checksum,
    )


def _validate_cached_file(
    path: Path,
    *,
    expected_size: int,
    checksum: str,
    description: str,
) -> None:
    """Validate one regular cached file by exact size and checksum."""
    if not path.is_file() or path.is_symlink():
        raise MaveDBSnapshotError(f"Cached MaveDB {description} is missing.")
    if path.stat().st_size != expected_size:
        raise MaveDBSnapshotError(
            f"Cached MaveDB {description} size does not match its metadata."
        )
    algorithm, expected_digest = checksum.split(":", maxsplit=1)
    if _checksum_file(path, algorithm) != expected_digest:
        raise MaveDBSnapshotError(
            f"Cached MaveDB {description} checksum does not match its metadata."
        )


def _acquire_snapshot(
    record: MaveDBSnapshotRecord,
    *,
    cache: FilesystemCache,
    session: requests.Session | None,
    timeout: int,
    chunk_size: int,
) -> MaveDBSnapshot:
    """Prepare a complete snapshot in staging and publish it as one version."""
    root = _snapshot_root(cache)
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{record.record_id}-", dir=root)
    )
    try:
        archive_path = staging / record.filename
        _download_archive(
            record,
            archive_path,
            session=session,
            timeout=timeout,
            chunk_size=chunk_size,
        )
        main_json_path = staging / _MAIN_JSON_FILENAME
        archive_format = _archive_format(record.filename)
        _extract_main_json(archive_path, main_json_path, archive_format)
        main_size = main_json_path.stat().st_size
        main_checksum = _checksum_file(main_json_path, "sha256")
        _write_snapshot_metadata(
            staging / _CACHE_METADATA_FILENAME,
            record,
            archive_format=archive_format,
            main_json_size=main_size,
            main_json_sha256=main_checksum,
        )
        final_entry = _snapshot_entry(cache, record.record_id)
        _publish_snapshot_directory(staging, final_entry)
        staging = final_entry
    finally:
        if staging.exists() and staging != _snapshot_entry(cache, record.record_id):
            shutil.rmtree(staging, ignore_errors=True)

    final_entry = _snapshot_entry(cache, record.record_id)
    logger.info("Prepared MaveDB snapshot record_id=%s", record.record_id)
    return MaveDBSnapshot(
        record=record,
        archive_path=final_entry / record.filename,
        main_json_path=final_entry / _MAIN_JSON_FILENAME,
        cache_hit=False,
    )


def _download_archive(
    record: MaveDBSnapshotRecord,
    output_path: Path,
    *,
    session: requests.Session | None,
    timeout: int,
    chunk_size: int,
) -> None:
    """Stream and verify an archive before publishing it in staging."""
    algorithm, expected_digest = record.checksum.split(":", maxsplit=1)
    checksum = hashlib.new(algorithm)
    downloaded = 0
    temporary_path: Path | None = None
    http = session if session is not None else requests.Session()
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_path.parent,
            prefix=".archive-",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            with http.get(
                record.download_url,
                stream=True,
                timeout=timeout,
            ) as response:
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    checksum.update(chunk)
                    downloaded += len(chunk)
        if downloaded != record.size:
            raise MaveDBSnapshotError(
                "Downloaded MaveDB archive size does not match Zenodo metadata."
            )
        if checksum.hexdigest().lower() != expected_digest:
            raise MaveDBSnapshotError(
                "Downloaded MaveDB archive checksum does not match Zenodo metadata."
            )
        os.replace(temporary_path, output_path)
    except requests.RequestException as exc:
        raise MaveDBSnapshotError(
            f"Failed to download MaveDB snapshot archive: {exc}"
        ) from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _archive_format(filename: str) -> str:
    """Return the supported archive format encoded in a filename."""
    if filename.endswith(".tar.gz"):
        return "tar.gz"
    if filename.endswith(".zip"):
        return "zip"
    raise MaveDBSnapshotError("MaveDB snapshot archive format is unsupported.")


def _extract_main_json(
    archive_path: Path,
    output_path: Path,
    archive_format: str,
) -> None:
    """Stream exactly one safe regular ``main.json`` member to staging."""
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_path.parent,
            prefix=".main-json-",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            if archive_format == "tar.gz":
                _copy_main_from_tar(archive_path, handle)
            elif archive_format == "zip":
                _copy_main_from_zip(archive_path, handle)
            else:
                raise MaveDBSnapshotError(
                    "MaveDB snapshot archive format is unsupported."
                )
        os.replace(temporary_path, output_path)
    except (tarfile.TarError, zipfile.BadZipFile) as exc:
        raise MaveDBSnapshotError(
            f"MaveDB snapshot archive is malformed: {exc}"
        ) from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _copy_main_from_tar(archive_path: Path, output: BinaryIO) -> None:
    """Copy a validated regular ``main.json`` member from TAR.GZ."""
    with tarfile.open(archive_path, mode="r:gz") as archive:
        candidates = [
            member
            for member in archive.getmembers()
            if _member_basename(member.name) == _MAIN_JSON_FILENAME
        ]
        member = _single_main_member(candidates)
        _validate_member_path(member.name)
        if not member.isfile() or member.issym() or member.islnk():
            raise MaveDBSnapshotError(
                "MaveDB snapshot main.json must be a regular TAR member."
            )
        source = archive.extractfile(member)
        if source is None:
            raise MaveDBSnapshotError(
                "MaveDB snapshot main.json could not be read from TAR."
            )
        with source:
            shutil.copyfileobj(source, output, length=_DEFAULT_CHUNK_SIZE)


def _copy_main_from_zip(archive_path: Path, output: BinaryIO) -> None:
    """Copy a validated regular unencrypted ``main.json`` member from ZIP."""
    with zipfile.ZipFile(archive_path, mode="r") as archive:
        candidates = [
            member
            for member in archive.infolist()
            if _member_basename(member.filename) == _MAIN_JSON_FILENAME
        ]
        member = _single_main_member(candidates)
        _validate_member_path(member.filename)
        mode = (member.external_attr >> 16) & 0xFFFF
        if (
            member.is_dir()
            or member.flag_bits & 0x1
            or stat.S_ISLNK(mode)
            or (stat.S_IFMT(mode) not in {0, stat.S_IFREG})
        ):
            raise MaveDBSnapshotError(
                "MaveDB snapshot main.json must be a regular unencrypted ZIP member."
            )
        with archive.open(member, mode="r") as source:
            shutil.copyfileobj(source, output, length=_DEFAULT_CHUNK_SIZE)


def _single_main_member(candidates: list[Any]) -> Any:
    """Return exactly one main.json archive member."""
    if not candidates:
        raise MaveDBSnapshotError("MaveDB snapshot archive is missing main.json.")
    if len(candidates) != 1:
        raise MaveDBSnapshotError(
            "MaveDB snapshot archive contains multiple main.json files."
        )
    return candidates[0]


def _member_basename(name: str) -> str:
    """Return a platform-independent archive member basename."""
    return PurePosixPath(name.replace("\\", "/")).name


def _validate_member_path(name: str) -> None:
    """Reject absolute, traversing, or drive-qualified archive member names."""
    normalized = name.replace("\\", "/")
    posix_path = PurePosixPath(normalized)
    windows_path = PureWindowsPath(name)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or ".." in posix_path.parts
    ):
        raise MaveDBSnapshotError(
            "MaveDB snapshot main.json member path is unsafe."
        )


def _write_snapshot_metadata(
    path: Path,
    record: MaveDBSnapshotRecord,
    *,
    archive_format: str,
    main_json_size: int,
    main_json_sha256: str,
) -> None:
    """Write deterministic snapshot metadata in the staged version directory."""
    values = {
        "record_id": record.record_id,
        "doi": record.doi,
        "concept_doi": record.concept_doi,
        "publication_date": record.publication_date,
        "archive_filename": record.filename,
        "archive_size": record.size,
        "checksum": record.checksum,
        "download_url": record.download_url,
        "archive_format": archive_format,
        "main_json_size": main_json_size,
        "main_json_sha256": main_json_sha256,
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        json.dump(values, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def _publish_snapshot_directory(staging: Path, final_entry: Path) -> None:
    """Publish one complete snapshot directory with rollback on replacement."""
    backup: Path | None = None
    if final_entry.exists():
        backup = Path(
            tempfile.mkdtemp(
                prefix=f".{final_entry.name}-backup-",
                dir=final_entry.parent,
            )
        )
        backup.rmdir()
        os.replace(final_entry, backup)
    try:
        os.replace(staging, final_entry)
    except OSError:
        if backup is not None and backup.exists():
            os.replace(backup, final_entry)
        raise
    if backup is not None and backup.exists():
        if backup.is_dir():
            shutil.rmtree(backup)
        else:
            backup.unlink()


def _checksum_file(path: Path, algorithm: str) -> str:
    """Return a streaming checksum for one local file."""
    checksum = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_DEFAULT_CHUNK_SIZE), b""):
            checksum.update(chunk)
    return checksum.hexdigest().lower()


__all__ = [
    "MaveDBSnapshot",
    "MaveDBSnapshotRecord",
    "fetch_mavedb_snapshot",
    "resolve_mavedb_snapshot",
]
