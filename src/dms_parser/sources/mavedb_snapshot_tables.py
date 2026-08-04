"""Selective raw-table extraction from managed MaveDB snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tarfile
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from dms_parser.cache import FilesystemCache
from dms_parser.exceptions import (
    MaveDBSnapshotError,
    MaveDBSnapshotTableError,
    SnapshotDatasetNotFoundError,
    SourceConfigurationError,
    SupersededSnapshotDatasetError,
)
from dms_parser.sources.mavedb_bulk_catalog import MaveDBBulkCatalog
from dms_parser.sources.mavedb_snapshots import (
    MAVEDB_ZENODO_CONCEPT_DOI,
    MaveDBSnapshot,
    MaveDBSnapshotRecord,
    _archive_format,
    _is_http_url,
    _normalize_checksum,
    _validate_safe_filename,
)

_CHUNK_SIZE = 1024 * 1024
_MANIFEST_FILENAME = "manifest.json"
_SCORES_FILENAME = "scores.csv"
_COUNTS_FILENAME = "counts.csv"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class MaveDBSnapshotTable:
    """One raw score-set table bundle extracted from a managed snapshot."""

    dataset_id: str
    scores_path: Path
    counts_path: Path | None
    is_superseded: bool
    cache_hit: bool


@dataclass(frozen=True)
class MaveDBSnapshotTableExtractionResult:
    """Deterministic result of one selective snapshot-table extraction."""

    record: MaveDBSnapshotRecord
    tables: tuple[MaveDBSnapshotTable, ...]

    @property
    def dataset_count(self) -> int:
        """Return the number of extracted or reused score-set bundles."""
        return len(self.tables)


@dataclass(frozen=True)
class _ExpectedTable:
    """Validated catalog record and deterministic extraction paths."""

    dataset_id: str
    is_superseded: bool
    archive_stem: str
    scores_member: str
    counts_member: str
    final_entry: Path


@dataclass(frozen=True)
class _ExtractedFile:
    """Streaming metadata for one staged archive member."""

    archive_member: str
    filename: str
    size: int
    sha256: str


class _InvalidTableCache(ValueError):
    """Internal signal that a table-cache bundle must be rebuilt."""


class _IncompleteTableRollbackError(OSError):
    """Internal failure retaining recovery data after incomplete rollback."""


def extract_mavedb_snapshot_tables(
    snapshot: MaveDBSnapshot,
    dataset_ids: Sequence[str],
    *,
    cache: FilesystemCache | None = None,
    include_superseded: bool = False,
    refresh: bool = False,
) -> MaveDBSnapshotTableExtractionResult:
    """Extract selected raw score/count CSVs from one prepared snapshot."""
    validated_ids = _validate_dataset_ids(dataset_ids)
    if not isinstance(include_superseded, bool):
        raise MaveDBSnapshotTableError(
            "include_superseded must be a boolean."
        )
    if not isinstance(refresh, bool):
        raise MaveDBSnapshotTableError("refresh must be a boolean.")
    if cache is not None and not isinstance(cache, FilesystemCache):
        raise MaveDBSnapshotTableError(
            "cache must be a FilesystemCache or None."
        )
    _validate_snapshot(snapshot)
    resolved_cache = cache or FilesystemCache(
        Path.home() / ".cache" / "dms-parser"
    )

    catalog = MaveDBBulkCatalog.from_file(snapshot.main_json_path)
    expected_tables = _resolve_expected_tables(
        snapshot,
        validated_ids,
        catalog=catalog,
        cache=resolved_cache,
        include_superseded=include_superseded,
    )

    results: dict[str, MaveDBSnapshotTable] = {}
    misses: list[_ExpectedTable] = []
    for expected in expected_tables:
        cached = None if refresh else _load_cached_table(snapshot, expected)
        if cached is None:
            misses.append(expected)
        else:
            results[expected.dataset_id] = cached

    if misses:
        rebuilt = _extract_and_publish(snapshot, tuple(misses))
        results.update((table.dataset_id, table) for table in rebuilt)

    return MaveDBSnapshotTableExtractionResult(
        record=snapshot.record,
        tables=tuple(results[dataset_id] for dataset_id in sorted(results)),
    )


def _validate_dataset_ids(dataset_ids: object) -> tuple[str, ...]:
    """Validate a non-empty sequence of unique canonical score-set URNs."""
    from dms_parser.config import validate_source_dataset_id

    if (
        isinstance(dataset_ids, (str, bytes, bytearray, Mapping, bool))
        or not isinstance(dataset_ids, Sequence)
    ):
        raise MaveDBSnapshotTableError(
            "dataset_ids must be a non-string sequence."
        )
    if not dataset_ids:
        raise MaveDBSnapshotTableError(
            "dataset_ids must contain at least one score-set URN."
        )

    validated: list[str] = []
    seen: set[str] = set()
    for dataset_id in dataset_ids:
        try:
            validate_source_dataset_id("mavedb", dataset_id)
        except SourceConfigurationError as exc:
            raise MaveDBSnapshotTableError(str(exc)) from exc
        assert isinstance(dataset_id, str)
        if dataset_id in seen:
            raise MaveDBSnapshotTableError(
                f"Duplicate MaveDB dataset_id {dataset_id!r}."
            )
        seen.add(dataset_id)
        validated.append(dataset_id)
    return tuple(validated)


def _validate_snapshot(snapshot: object) -> None:
    """Validate the prepared snapshot object and its required local files."""
    if not isinstance(snapshot, MaveDBSnapshot):
        raise MaveDBSnapshotTableError(
            "snapshot must be a MaveDBSnapshot."
        )
    record = snapshot.record
    if not isinstance(record, MaveDBSnapshotRecord):
        raise MaveDBSnapshotTableError(
            "snapshot.record must be a MaveDBSnapshotRecord."
        )
    if re.fullmatch(r"[1-9][0-9]*", record.record_id) is None:
        raise MaveDBSnapshotTableError("Snapshot record ID is invalid.")
    for field, value in (
        ("doi", record.doi),
        ("concept_doi", record.concept_doi),
        ("filename", record.filename),
        ("download_url", record.download_url),
    ):
        if not isinstance(value, str) or not value.strip():
            raise MaveDBSnapshotTableError(
                f"Snapshot {field} must be a non-empty string."
            )
    if record.concept_doi.casefold() != MAVEDB_ZENODO_CONCEPT_DOI.casefold():
        raise MaveDBSnapshotTableError(
            "Snapshot does not belong to the official MaveDB concept."
        )
    if not _is_http_url(record.download_url):
        raise MaveDBSnapshotTableError("Snapshot download URL is invalid.")
    if (
        isinstance(record.size, bool)
        or not isinstance(record.size, int)
        or record.size <= 0
    ):
        raise MaveDBSnapshotTableError(
            "Snapshot archive size must be a positive integer."
        )
    if record.publication_date is not None and (
        not isinstance(record.publication_date, str)
        or not record.publication_date.strip()
    ):
        raise MaveDBSnapshotTableError(
            "Snapshot publication date is invalid."
        )
    try:
        _validate_safe_filename(record.filename)
        _archive_format(record.filename)
        normalized_checksum = _normalize_checksum(record.checksum)
    except MaveDBSnapshotError as exc:
        raise MaveDBSnapshotTableError(str(exc)) from exc
    if normalized_checksum != record.checksum:
        raise MaveDBSnapshotTableError(
            "Snapshot archive checksum is not canonical."
        )
    if not isinstance(snapshot.archive_path, Path) or not isinstance(
        snapshot.main_json_path,
        Path,
    ):
        raise MaveDBSnapshotTableError(
            "Snapshot archive and main.json paths must be pathlib.Path values."
        )
    if snapshot.archive_path.name != record.filename:
        raise MaveDBSnapshotTableError(
            "Snapshot archive path does not match its record filename."
        )
    if snapshot.main_json_path.name != "main.json":
        raise MaveDBSnapshotTableError(
            "Snapshot main_json_path must identify main.json."
        )
    if snapshot.archive_path.parent != snapshot.main_json_path.parent:
        raise MaveDBSnapshotTableError(
            "Snapshot archive and main.json must share a version directory."
        )
    for path, description in (
        (snapshot.archive_path, "archive"),
        (snapshot.main_json_path, "main.json"),
    ):
        if not path.is_file() or path.is_symlink():
            raise MaveDBSnapshotTableError(
                f"Snapshot {description} is missing or not a regular file."
            )
    if snapshot.archive_path.stat().st_size != record.size:
        raise MaveDBSnapshotTableError(
            "Snapshot archive size does not match its record."
        )


def _resolve_expected_tables(
    snapshot: MaveDBSnapshot,
    dataset_ids: tuple[str, ...],
    *,
    catalog: MaveDBBulkCatalog,
    cache: FilesystemCache,
    include_superseded: bool,
) -> tuple[_ExpectedTable, ...]:
    """Resolve catalog membership and deterministic member/cache paths."""
    expected: list[_ExpectedTable] = []
    root = cache.root / "mavedb" / "snapshot_tables" / snapshot.record.record_id
    for dataset_id in dataset_ids:
        record = catalog.get_score_set(dataset_id)
        if record is None:
            raise SnapshotDatasetNotFoundError(
                f"MaveDB score set {dataset_id!r} is absent from snapshot "
                f"{snapshot.record.record_id}."
            )
        if record.is_superseded and not include_superseded:
            raise SupersededSnapshotDatasetError(
                f"MaveDB score set {dataset_id!r} is superseded; pass "
                "include_superseded=True to extract it."
            )
        stem = dataset_id.replace(":", "-")
        expected.append(
            _ExpectedTable(
                dataset_id=dataset_id,
                is_superseded=record.is_superseded,
                archive_stem=stem,
                scores_member=f"csv/{stem}.scores.csv",
                counts_member=f"csv/{stem}.counts.csv",
                final_entry=root / stem,
            )
        )
    return tuple(sorted(expected, key=lambda item: item.dataset_id))


def _load_cached_table(
    snapshot: MaveDBSnapshot,
    expected: _ExpectedTable,
) -> MaveDBSnapshotTable | None:
    """Return a fully validated table-cache hit or ``None`` for a miss."""
    try:
        counts_path = _validate_bundle(
            expected.final_entry,
            snapshot,
            expected,
        )
    except (_InvalidTableCache, OSError, UnicodeError, json.JSONDecodeError):
        return None
    return MaveDBSnapshotTable(
        dataset_id=expected.dataset_id,
        scores_path=expected.final_entry / _SCORES_FILENAME,
        counts_path=counts_path,
        is_superseded=expected.is_superseded,
        cache_hit=True,
    )


def _validate_bundle(
    entry: Path,
    snapshot: MaveDBSnapshot,
    expected: _ExpectedTable,
) -> Path | None:
    """Validate one complete staged or published extraction bundle."""
    if not entry.is_dir() or entry.is_symlink():
        raise _InvalidTableCache("Entry is not a regular directory.")
    manifest_path = entry / _MANIFEST_FILENAME
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise _InvalidTableCache("Manifest is missing or unsafe.")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, Mapping):
        raise _InvalidTableCache("Manifest must be a JSON object.")
    if set(manifest) != {
        "source",
        "record_id",
        "doi",
        "concept_doi",
        "archive",
        "dataset_id",
        "is_superseded",
        "files",
    }:
        raise _InvalidTableCache("Manifest fields are invalid.")
    record = snapshot.record
    if (
        manifest["source"] != "mavedb_snapshot"
        or manifest["record_id"] != record.record_id
        or manifest["doi"] != record.doi
        or manifest["concept_doi"] != record.concept_doi
        or manifest["dataset_id"] != expected.dataset_id
        or manifest["is_superseded"] is not expected.is_superseded
    ):
        raise _InvalidTableCache("Manifest provenance is inconsistent.")
    if manifest["archive"] != {
        "filename": record.filename,
        "size": record.size,
        "checksum": record.checksum,
    }:
        raise _InvalidTableCache("Manifest archive provenance is inconsistent.")
    files = manifest["files"]
    if not isinstance(files, Mapping) or set(files) != {"scores", "counts"}:
        raise _InvalidTableCache("Manifest file metadata is invalid.")

    scores_path = entry / _SCORES_FILENAME
    _validate_cached_file(
        scores_path,
        files["scores"],
        archive_member=expected.scores_member,
        filename=_SCORES_FILENAME,
    )
    counts_metadata = files["counts"]
    counts_path: Path | None = None
    if counts_metadata is None:
        if (entry / _COUNTS_FILENAME).exists():
            raise _InvalidTableCache(
                "Counts file exists but the manifest records no counts."
            )
    else:
        counts_path = entry / _COUNTS_FILENAME
        _validate_cached_file(
            counts_path,
            counts_metadata,
            archive_member=expected.counts_member,
            filename=_COUNTS_FILENAME,
        )

    expected_names = {_MANIFEST_FILENAME, _SCORES_FILENAME}
    if counts_path is not None:
        expected_names.add(_COUNTS_FILENAME)
    if {path.name for path in entry.iterdir()} != expected_names:
        raise _InvalidTableCache("Entry contains unexpected or partial files.")
    return counts_path


def _validate_cached_file(
    path: Path,
    metadata: object,
    *,
    archive_member: str,
    filename: str,
) -> None:
    """Validate one regular cached CSV against deterministic metadata."""
    if not isinstance(metadata, Mapping) or set(metadata) != {
        "archive_member",
        "filename",
        "size",
        "sha256",
    }:
        raise _InvalidTableCache("Cached file metadata is invalid.")
    size = metadata["size"]
    checksum = metadata["sha256"]
    if (
        metadata["archive_member"] != archive_member
        or metadata["filename"] != filename
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
        or not isinstance(checksum, str)
        or _SHA256_PATTERN.fullmatch(checksum) is None
    ):
        raise _InvalidTableCache("Cached file metadata is inconsistent.")
    if not path.is_file() or path.is_symlink():
        raise _InvalidTableCache("Cached file is missing or unsafe.")
    if path.stat().st_size != size:
        raise _InvalidTableCache("Cached file size is inconsistent.")
    if _sha256_file(path) != checksum:
        raise _InvalidTableCache("Cached file checksum is inconsistent.")


def _extract_and_publish(
    snapshot: MaveDBSnapshot,
    misses: tuple[_ExpectedTable, ...],
) -> tuple[MaveDBSnapshotTable, ...]:
    """Stage every miss, then transactionally publish all completed bundles."""
    record_root = misses[0].final_entry.parent
    record_root.mkdir(parents=True, exist_ok=True)
    transaction = Path(
        tempfile.mkdtemp(prefix=".extract-", dir=record_root)
    )
    cleanup_transaction = True
    try:
        staged_root = transaction / "staged"
        staged_root.mkdir()
        staged_entries = {
            expected.dataset_id: staged_root / expected.archive_stem
            for expected in misses
        }
        for entry in staged_entries.values():
            entry.mkdir()

        extracted = _extract_archive_members(
            snapshot.archive_path,
            snapshot.record.filename,
            misses,
            staged_entries,
        )
        for expected in misses:
            scores, counts = extracted[expected.dataset_id]
            _write_manifest(
                staged_entries[expected.dataset_id] / _MANIFEST_FILENAME,
                _manifest_values(snapshot, expected, scores, counts),
            )
            try:
                _validate_bundle(
                    staged_entries[expected.dataset_id],
                    snapshot,
                    expected,
                )
            except (_InvalidTableCache, OSError) as exc:
                raise MaveDBSnapshotTableError(
                    f"Staged snapshot table bundle for {expected.dataset_id!r} "
                    "failed validation."
                ) from exc

        try:
            _publish_staged_bundles(
                misses,
                staged_entries,
                transaction=transaction,
            )
        except _IncompleteTableRollbackError:
            cleanup_transaction = False
            raise
    finally:
        if cleanup_transaction:
            shutil.rmtree(transaction, ignore_errors=True)

    return tuple(
        MaveDBSnapshotTable(
            dataset_id=expected.dataset_id,
            scores_path=expected.final_entry / _SCORES_FILENAME,
            counts_path=(
                expected.final_entry / _COUNTS_FILENAME
                if (expected.final_entry / _COUNTS_FILENAME).is_file()
                else None
            ),
            is_superseded=expected.is_superseded,
            cache_hit=False,
        )
        for expected in misses
    )


def _extract_archive_members(
    archive_path: Path,
    archive_filename: str,
    expected_tables: tuple[_ExpectedTable, ...],
    staged_entries: Mapping[str, Path],
) -> dict[str, tuple[_ExtractedFile, _ExtractedFile | None]]:
    """Open and scan one supported archive once for all selected members."""
    archive_format = _archive_format(archive_filename)
    if archive_format == "zip":
        try:
            return _extract_from_zip(
                archive_path,
                expected_tables,
                staged_entries,
            )
        except (zipfile.BadZipFile, EOFError) as exc:
            raise MaveDBSnapshotTableError(
                f"MaveDB snapshot ZIP archive is malformed: {exc}"
            ) from exc
    if archive_format == "tar.gz":
        try:
            return _extract_from_tar(
                archive_path,
                expected_tables,
                staged_entries,
            )
        except (tarfile.TarError, EOFError) as exc:
            raise MaveDBSnapshotTableError(
                f"MaveDB snapshot TAR.GZ archive is malformed: {exc}"
            ) from exc
    raise MaveDBSnapshotTableError(
        "MaveDB snapshot archive format is unsupported."
    )


def _extract_from_zip(
    archive_path: Path,
    expected_tables: tuple[_ExpectedTable, ...],
    staged_entries: Mapping[str, Path],
) -> dict[str, tuple[_ExtractedFile, _ExtractedFile | None]]:
    """Scan one ZIP once and stream only exact requested regular members."""
    expected_names = {
        name
        for expected in expected_tables
        for name in (expected.scores_member, expected.counts_member)
    }
    with zipfile.ZipFile(archive_path, mode="r") as archive:
        selected: dict[str, list[zipfile.ZipInfo]] = {
            name: [] for name in expected_names
        }
        for member in archive.infolist():
            raw_name = getattr(member, "orig_filename", member.filename)
            if raw_name in selected:
                selected[raw_name].append(member)

        results: dict[
            str,
            tuple[_ExtractedFile, _ExtractedFile | None],
        ] = {}
        for expected in expected_tables:
            scores_member = _single_required_member(
                selected[expected.scores_member],
                expected.scores_member,
            )
            counts_member = _single_optional_member(
                selected[expected.counts_member],
                expected.counts_member,
            )
            _validate_zip_member(scores_member)
            if counts_member is not None:
                _validate_zip_member(counts_member)
            scores = _stream_zip_member(
                archive,
                scores_member,
                staged_entries[expected.dataset_id] / _SCORES_FILENAME,
            )
            counts = (
                _stream_zip_member(
                    archive,
                    counts_member,
                    staged_entries[expected.dataset_id] / _COUNTS_FILENAME,
                )
                if counts_member is not None
                else None
            )
            results[expected.dataset_id] = (scores, counts)
        return results


def _validate_zip_member(member: zipfile.ZipInfo) -> None:
    """Require one selected ZIP entry to be regular and unencrypted."""
    raw_name = getattr(member, "orig_filename", member.filename)
    mode = (member.external_attr >> 16) & 0xFFFF
    if (
        raw_name != member.filename
        or "\\" in raw_name
        or member.is_dir()
        or member.flag_bits & 0x1
        or stat.S_ISLNK(mode)
        or stat.S_IFMT(mode) not in {0, stat.S_IFREG}
    ):
        raise MaveDBSnapshotTableError(
            f"Selected ZIP member {member.filename!r} must be a regular "
            "unencrypted file."
        )


def _stream_zip_member(
    archive: zipfile.ZipFile,
    member: zipfile.ZipInfo,
    output_path: Path,
) -> _ExtractedFile:
    """Stream one ZIP member to its staged file with size and SHA-256."""
    with archive.open(member, mode="r") as source:
        size, checksum = _stream_to_staged_file(
            source,
            output_path,
            expected_size=member.file_size,
        )
    return _ExtractedFile(
        archive_member=getattr(member, "orig_filename", member.filename),
        filename=output_path.name,
        size=size,
        sha256=checksum,
    )


def _extract_from_tar(
    archive_path: Path,
    expected_tables: tuple[_ExpectedTable, ...],
    staged_entries: Mapping[str, Path],
) -> dict[str, tuple[_ExtractedFile, _ExtractedFile | None]]:
    """Scan one TAR.GZ once and stream exact requested regular members."""
    expected_names = {
        name
        for expected in expected_tables
        for name in (expected.scores_member, expected.counts_member)
    }
    with tarfile.open(archive_path, mode="r:gz") as archive:
        selected: dict[str, list[tarfile.TarInfo]] = {
            name: [] for name in expected_names
        }
        for member in archive.getmembers():
            if member.name in selected:
                selected[member.name].append(member)

        results: dict[
            str,
            tuple[_ExtractedFile, _ExtractedFile | None],
        ] = {}
        for expected in expected_tables:
            scores_member = _single_required_member(
                selected[expected.scores_member],
                expected.scores_member,
            )
            counts_member = _single_optional_member(
                selected[expected.counts_member],
                expected.counts_member,
            )
            _validate_tar_member(scores_member)
            if counts_member is not None:
                _validate_tar_member(counts_member)
            scores = _stream_tar_member(
                archive,
                scores_member,
                staged_entries[expected.dataset_id] / _SCORES_FILENAME,
            )
            counts = (
                _stream_tar_member(
                    archive,
                    counts_member,
                    staged_entries[expected.dataset_id] / _COUNTS_FILENAME,
                )
                if counts_member is not None
                else None
            )
            results[expected.dataset_id] = (scores, counts)
        return results


def _validate_tar_member(member: tarfile.TarInfo) -> None:
    """Require one selected TAR entry to be an ordinary regular file."""
    if not member.isfile() or member.issym() or member.islnk():
        raise MaveDBSnapshotTableError(
            f"Selected TAR member {member.name!r} must be a regular file."
        )


def _stream_tar_member(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    output_path: Path,
) -> _ExtractedFile:
    """Stream one TAR member to its staged file with size and SHA-256."""
    source = archive.extractfile(member)
    if source is None:
        raise MaveDBSnapshotTableError(
            f"Selected TAR member {member.name!r} could not be read."
        )
    with source:
        size, checksum = _stream_to_staged_file(
            source,
            output_path,
            expected_size=member.size,
        )
    return _ExtractedFile(
        archive_member=member.name,
        filename=output_path.name,
        size=size,
        sha256=checksum,
    )


def _single_required_member(members: list[Any], name: str) -> Any:
    """Return exactly one required archive member."""
    if not members:
        raise MaveDBSnapshotTableError(
            f"MaveDB snapshot archive is missing required scores member {name!r}."
        )
    if len(members) != 1:
        raise MaveDBSnapshotTableError(
            f"MaveDB snapshot archive contains duplicate member {name!r}."
        )
    return members[0]


def _single_optional_member(members: list[Any], name: str) -> Any | None:
    """Return zero or exactly one optional archive member."""
    if len(members) > 1:
        raise MaveDBSnapshotTableError(
            f"MaveDB snapshot archive contains duplicate member {name!r}."
        )
    return members[0] if members else None


def _stream_to_staged_file(
    source: BinaryIO,
    output_path: Path,
    *,
    expected_size: int,
) -> tuple[int, str]:
    """Stream one selected member into staging and verify declared size."""
    if (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size < 0
    ):
        raise MaveDBSnapshotTableError(
            "Selected archive member has an invalid declared size."
        )
    checksum = hashlib.sha256()
    size = 0
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_path.parent,
            prefix=f".{output_path.name}-",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            while chunk := source.read(_CHUNK_SIZE):
                handle.write(chunk)
                checksum.update(chunk)
                size += len(chunk)
        if size != expected_size:
            raise MaveDBSnapshotTableError(
                "Extracted archive member size does not match its metadata."
            )
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return size, checksum.hexdigest()


def _manifest_values(
    snapshot: MaveDBSnapshot,
    expected: _ExpectedTable,
    scores: _ExtractedFile,
    counts: _ExtractedFile | None,
) -> dict[str, Any]:
    """Return deterministic provenance for one raw-table cache bundle."""
    record = snapshot.record
    return {
        "source": "mavedb_snapshot",
        "record_id": record.record_id,
        "doi": record.doi,
        "concept_doi": record.concept_doi,
        "archive": {
            "filename": record.filename,
            "size": record.size,
            "checksum": record.checksum,
        },
        "dataset_id": expected.dataset_id,
        "is_superseded": expected.is_superseded,
        "files": {
            "scores": _extracted_file_values(scores),
            "counts": (
                _extracted_file_values(counts)
                if counts is not None
                else None
            ),
        },
    }


def _extracted_file_values(value: _ExtractedFile) -> dict[str, Any]:
    """Serialize deterministic metadata for one extracted CSV."""
    return {
        "archive_member": value.archive_member,
        "filename": value.filename,
        "size": value.size,
        "sha256": value.sha256,
    }


def _write_manifest(path: Path, values: Mapping[str, Any]) -> None:
    """Write the manifest last inside one staged dataset bundle."""
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=".manifest-",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(
                values,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _publish_staged_bundles(
    expected_tables: tuple[_ExpectedTable, ...],
    staged_entries: Mapping[str, Path],
    *,
    transaction: Path,
) -> None:
    """Publish several complete bundles with backups and rollback."""
    backups_root = transaction / "backups"
    backups_root.mkdir()
    backups: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for index, expected in enumerate(expected_tables):
            final_entry = expected.final_entry
            if final_entry.exists():
                backup = backups_root / f"{index}-{expected.archive_stem}"
                os.replace(final_entry, backup)
                backups.append((final_entry, backup))
            os.replace(staged_entries[expected.dataset_id], final_entry)
            published.append(final_entry)
    except OSError as exc:
        rollback_errors = _rollback_publication(backups, published)
        if rollback_errors:
            details = "; ".join(str(error) for error in rollback_errors)
            raise _IncompleteTableRollbackError(
                "Snapshot table publication failed and rollback was "
                f"incomplete; recovery files remain in {transaction}: {details}"
            ) from exc
        raise


def _rollback_publication(
    backups: list[tuple[Path, Path]],
    published: list[Path],
) -> list[OSError]:
    """Best-effort rollback of newly published raw-table bundles."""
    errors: list[OSError] = []
    for path in reversed(published):
        try:
            _remove_path(path)
        except OSError as exc:
            errors.append(exc)
    for final_entry, backup in reversed(backups):
        try:
            if backup.exists():
                os.replace(backup, final_entry)
        except OSError as exc:
            errors.append(exc)
    return errors


def _remove_path(path: Path) -> None:
    """Remove one known transaction-owned file or directory."""
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest for one regular local file."""
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


__all__ = [
    "MaveDBSnapshotTable",
    "MaveDBSnapshotTableExtractionResult",
    "extract_mavedb_snapshot_tables",
]
