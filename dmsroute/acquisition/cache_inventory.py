"""Read-only inventory of artifacts managed by the DMSRoute cache."""

from __future__ import annotations

import json
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dmsroute.acquisition.cache import CacheManifest, FilesystemCache
from dmsroute.core.exceptions import (
    CacheError,
    CacheInventoryError,
    MaveDBSnapshotError,
)
from dmsroute.sources.mavedb_snapshots import (
    _archive_format,
    _record_from_cache_metadata,
)
from dmsroute.sources.mavedb_snapshot_tables import (
    _InvalidTableCache,
    _SnapshotTableManifest,
    _snapshot_table_manifest_from_dict,
)

CacheEntryState = Literal["structurally_complete", "invalid"]

_GENERIC_COMPONENT_PATTERN = re.compile(
    r"[A-Za-z0-9._-]{1,48}-[0-9a-f]{16}"
)
_GENERIC_ARTIFACT_PATTERN = re.compile(
    r"artifact-[0-9a-f]{16}-[0-9a-f]{16}"
    r"(?:\.[A-Za-z0-9]+){0,2}"
)
_SNAPSHOT_RECORD_PATTERN = re.compile(r"[1-9][0-9]*")
_SNAPSHOT_TRANSIENT_PATTERN = re.compile(
    r"\.[1-9][0-9]*-(?:backup-)?[^/\\]+"
)
_GENERIC_TRANSIENT_PREFIXES = (".artifact-", ".manifest-")


@dataclass(frozen=True)
class CacheInventoryArtifact:
    """One present non-symlink regular payload in a managed cache entry."""

    role: str
    path: Path
    size_bytes: int


@dataclass(frozen=True)
class CacheInventoryEntry:
    """One recognized logical cache entry and its present payloads."""

    representation: str
    source: str
    dataset_id: str | None
    snapshot_record_id: str | None
    state: CacheEntryState
    artifacts: tuple[CacheInventoryArtifact, ...]
    error: str | None


@dataclass(frozen=True)
class CacheInventoryIssue:
    """One incomplete managed object without a reliable logical identity."""

    path: Path
    state: Literal["incomplete"]
    error: str


@dataclass(frozen=True)
class CacheInventoryResult:
    """Deterministic cache entries, incomplete objects, and aggregate counts."""

    entries: tuple[CacheInventoryEntry, ...]
    issues: tuple[CacheInventoryIssue, ...]

    @property
    def entry_count(self) -> int:
        """Return the number of recognized complete and invalid entries."""
        return len(self.entries)

    @property
    def artifact_count(self) -> int:
        """Return the number of present non-symlink regular payload files."""
        return sum(len(entry.artifacts) for entry in self.entries)

    @property
    def total_size_bytes(self) -> int:
        """Return the aggregate current size of present payload files."""
        return sum(
            artifact.size_bytes
            for entry in self.entries
            for artifact in entry.artifacts
        )

    @property
    def complete_entry_count(self) -> int:
        """Return the number of structurally complete entries."""
        return sum(
            entry.state == "structurally_complete"
            for entry in self.entries
        )

    @property
    def invalid_entry_count(self) -> int:
        """Return the number of structurally invalid entries."""
        return sum(entry.state == "invalid" for entry in self.entries)

    @property
    def issue_count(self) -> int:
        """Return the number of incomplete managed filesystem objects."""
        return len(self.issues)


@dataclass(frozen=True)
class _InventoryIssue:
    """Internal issue with optional identity for exact filtering."""

    path: Path
    error: str
    source: str | None = None
    dataset_id: str | None = None


def inventory_cache(
    cache: FilesystemCache,
    *,
    source: str | None = None,
    dataset_id: str | None = None,
) -> CacheInventoryResult:
    """Inspect managed cache entries without modifying or refreshing them."""
    if not isinstance(cache, FilesystemCache):
        raise TypeError("cache must be a FilesystemCache.")
    _validate_filter(source, "source")
    _validate_filter(dataset_id, "dataset_id")

    root = _authorized_root(cache.root)
    if root is None:
        return CacheInventoryResult(entries=(), issues=())

    entries: list[CacheInventoryEntry] = []
    issues: dict[str, _InventoryIssue] = {}
    try:
        children = _children(root, authorized_root=root)
    except OSError as exc:
        raise CacheInventoryError(
            f"Could not inspect cache root: {cache.root}."
        ) from exc

    for child in children:
        if child.name == "mavedb":
            _scan_mavedb_layout(child, root, entries, issues)
        elif _GENERIC_COMPONENT_PATTERN.fullmatch(child.name):
            _scan_generic_source(child, cache, root, entries, issues)

    filtered_entries = tuple(
        sorted(
            (
                _sorted_entry_artifacts(entry)
                for entry in entries
                if _matches_filters(
                    entry.source,
                    entry.dataset_id,
                    source=source,
                    dataset_id=dataset_id,
                )
            ),
            key=_entry_sort_key,
        )
    )
    filtered_issues = tuple(
        CacheInventoryIssue(
            path=issue.path,
            state="incomplete",
            error=issue.error,
        )
        for issue in sorted(issues.values(), key=_issue_sort_key)
        if _issue_matches_filters(
            issue,
            source=source,
            dataset_id=dataset_id,
        )
    )
    return CacheInventoryResult(
        entries=filtered_entries,
        issues=filtered_issues,
    )


def _authorized_root(path: Path) -> Path | None:
    """Return the resolved authorized root, allowing only the root symlink."""
    try:
        path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise CacheInventoryError(
            f"Could not inspect cache root: {path}."
        ) from exc
    try:
        root = path.resolve(strict=True)
        metadata = root.stat()
    except OSError as exc:
        raise CacheInventoryError(
            f"Could not inspect cache root: {path}."
        ) from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise CacheInventoryError(f"Cache root is not a directory: {path}.")
    return root


def _scan_generic_source(
    source_path: Path,
    cache: FilesystemCache,
    root: Path,
    entries: list[CacheInventoryEntry],
    issues: dict[str, _InventoryIssue],
) -> None:
    kind = _path_kind(source_path)
    if kind != "directory":
        _add_issue(
            issues,
            source_path,
            "Generic cache source location is incomplete or unsafe.",
        )
        return
    try:
        children = _children(source_path, authorized_root=root)
    except OSError:
        _add_issue(
            issues,
            source_path,
            "Generic cache source directory could not be inspected.",
        )
        return
    for entry_path in children:
        if not _GENERIC_COMPONENT_PATTERN.fullmatch(entry_path.name):
            continue
        if _path_kind(entry_path) != "directory":
            _add_issue(
                issues,
                entry_path,
                "Generic cache entry location is incomplete or unsafe.",
            )
            continue
        _scan_generic_entry(entry_path, cache, root, entries, issues)


def _scan_generic_entry(
    entry_path: Path,
    cache: FilesystemCache,
    root: Path,
    entries: list[CacheInventoryEntry],
    issues: dict[str, _InventoryIssue],
) -> None:
    try:
        children = _children(entry_path, authorized_root=root)
    except OSError:
        _add_issue(
            issues,
            entry_path,
            "Generic cache entry directory could not be inspected.",
        )
        return
    manifest_path = entry_path / "manifest.json"
    manifest_kind = _path_kind(manifest_path)
    if manifest_kind == "missing":
        _add_issue(
            issues,
            entry_path,
            "Generic cache entry is missing manifest.json.",
        )
        _record_generic_orphans(children, None, issues)
        return
    if manifest_kind != "file":
        _add_issue(
            issues,
            manifest_path,
            "Generic cache manifest is incomplete or unsafe.",
        )
        _record_generic_orphans(children, None, issues)
        return

    try:
        values = _read_json(manifest_path)
    except (OSError, UnicodeError, json.JSONDecodeError):
        _add_issue(
            issues,
            manifest_path,
            "Generic cache manifest could not be parsed.",
        )
        _record_generic_orphans(children, None, issues)
        return

    identity = _generic_manifest_identity(values)
    try:
        manifest = CacheManifest.from_dict(values)
    except CacheError:
        if identity is None:
            _add_issue(
                issues,
                manifest_path,
                "Generic cache manifest is structurally invalid.",
            )
            _record_generic_orphans(children, None, issues)
            return
        entries.append(
            CacheInventoryEntry(
                representation="generic",
                source=identity[0],
                dataset_id=identity[1],
                snapshot_record_id=None,
                state="invalid",
                artifacts=(),
                error="Generic cache manifest is structurally invalid.",
            )
        )
        _record_generic_orphans(
            children,
            None,
            issues,
            source=identity[0],
            dataset_id=identity[1],
        )
        return

    errors: list[str] = []
    expected_relative = cache.entry_path(
        manifest.source,
        manifest.dataset_id,
    ).relative_to(cache.root)
    expected_entry = root / expected_relative
    if _normalized_path(expected_entry) != _normalized_path(entry_path):
        errors.append("Generic cache manifest does not match its entry path.")

    artifact_path = entry_path / manifest.artifact_filename
    artifact_kind = _path_kind(artifact_path)
    artifacts: tuple[CacheInventoryArtifact, ...] = ()
    if artifact_kind == "file":
        try:
            size = artifact_path.stat().st_size
        except OSError:
            errors.append("Generic cache artifact size could not be read.")
        else:
            artifacts = (
                CacheInventoryArtifact(
                    role="artifact",
                    path=artifact_path,
                    size_bytes=size,
                ),
            )
            if size != manifest.file_size:
                errors.append(
                    "Generic cache artifact size does not match its manifest."
                )
    elif artifact_kind == "symlink":
        errors.append("Generic cache artifact is a symlink.")
    else:
        errors.append("Generic cache artifact is missing or not a regular file.")

    entries.append(
        CacheInventoryEntry(
            representation="generic",
            source=manifest.source,
            dataset_id=manifest.dataset_id,
            snapshot_record_id=None,
            state="invalid" if errors else "structurally_complete",
            artifacts=artifacts,
            error="; ".join(errors) if errors else None,
        )
    )
    _record_generic_orphans(
        children,
        manifest.artifact_filename,
        issues,
        source=manifest.source,
        dataset_id=manifest.dataset_id,
    )


def _record_generic_orphans(
    children: tuple[Path, ...],
    referenced_artifact: str | None,
    issues: dict[str, _InventoryIssue],
    *,
    source: str | None = None,
    dataset_id: str | None = None,
) -> None:
    """Report generic temporaries and unreferenced managed artifacts."""
    for child in children:
        if (
            child.name == "manifest.json"
            or child.name == referenced_artifact
        ):
            continue
        if child.name.startswith(_GENERIC_TRANSIENT_PREFIXES):
            error = "Incomplete generic cache temporary object."
        elif _GENERIC_ARTIFACT_PATTERN.fullmatch(child.name):
            error = "Unreferenced generic cache artifact."
        else:
            continue
        _add_issue(
            issues,
            child,
            error,
            source=source,
            dataset_id=dataset_id,
        )

def _scan_mavedb_layout(
    path: Path,
    root: Path,
    entries: list[CacheInventoryEntry],
    issues: dict[str, _InventoryIssue],
) -> None:
    if _path_kind(path) != "directory":
        _add_issue(
            issues,
            path,
            "Managed MaveDB cache location is incomplete or unsafe.",
            source="mavedb",
        )
        return
    try:
        children = _children(path, authorized_root=root)
    except OSError:
        _add_issue(
            issues,
            path,
            "Managed MaveDB cache directory could not be inspected.",
            source="mavedb",
        )
        return
    child_by_name = {child.name: child for child in children}
    snapshots = child_by_name.get("snapshots")
    if snapshots is not None:
        _scan_snapshot_root(snapshots, root, entries, issues)
    tables = child_by_name.get("snapshot_tables")
    if tables is not None:
        _scan_snapshot_table_root(tables, root, entries, issues)


def _scan_snapshot_root(
    path: Path,
    root: Path,
    entries: list[CacheInventoryEntry],
    issues: dict[str, _InventoryIssue],
) -> None:
    if _path_kind(path) != "directory":
        _add_issue(
            issues,
            path,
            "MaveDB snapshot root is incomplete or unsafe.",
            source="mavedb",
        )
        return
    try:
        children = _children(path, authorized_root=root)
    except OSError:
        _add_issue(
            issues,
            path,
            "MaveDB snapshot root could not be inspected.",
            source="mavedb",
        )
        return
    for child in children:
        if _SNAPSHOT_TRANSIENT_PATTERN.fullmatch(child.name):
            _add_issue(
                issues,
                child,
                "Incomplete MaveDB snapshot staging or backup object.",
                source="mavedb",
            )
        elif _SNAPSHOT_RECORD_PATTERN.fullmatch(child.name):
            if _path_kind(child) != "directory":
                entries.append(
                    _invalid_snapshot_entry(
                        child.name,
                        "MaveDB snapshot entry is not a safe directory.",
                    )
                )
            else:
                _scan_snapshot_entry(child, root, entries)


def _scan_snapshot_entry(
    entry_path: Path,
    root: Path,
    entries: list[CacheInventoryEntry],
) -> None:
    record_id = entry_path.name
    try:
        _children(entry_path, authorized_root=root)
    except OSError:
        entries.append(
            _invalid_snapshot_entry(
                record_id,
                "MaveDB snapshot entry could not be inspected.",
            )
        )
        return
    metadata_path = entry_path / "snapshot.json"
    if _path_kind(metadata_path) != "file":
        entries.append(
            _invalid_snapshot_entry(
                record_id,
                "MaveDB snapshot metadata is missing or unsafe.",
            )
        )
        return
    try:
        values = _read_json(metadata_path)
        record, archive_format, main_size, _ = _record_from_cache_metadata(
            values,
            expected_id=record_id,
        )
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        MaveDBSnapshotError,
    ):
        entries.append(
            _invalid_snapshot_entry(
                record_id,
                "MaveDB snapshot metadata is structurally invalid.",
            )
        )
        return

    errors: list[str] = []
    if archive_format != _archive_format(record.filename):
        errors.append("MaveDB snapshot archive format is inconsistent.")
    artifacts: list[CacheInventoryArtifact] = []
    for role, path, expected_size in (
        ("archive", entry_path / record.filename, record.size),
        ("main", entry_path / "main.json", main_size),
    ):
        kind = _path_kind(path)
        if kind == "file":
            try:
                size = path.stat().st_size
            except OSError:
                errors.append(f"MaveDB snapshot {role} size could not be read.")
                continue
            artifacts.append(CacheInventoryArtifact(role, path, size))
            if size != expected_size:
                errors.append(
                    f"MaveDB snapshot {role} size does not match metadata."
                )
        elif kind == "symlink":
            errors.append(f"MaveDB snapshot {role} is a symlink.")
        else:
            errors.append(
                f"MaveDB snapshot {role} is missing or not a regular file."
            )
    entries.append(
        CacheInventoryEntry(
            representation="mavedb_snapshot",
            source="mavedb",
            dataset_id=None,
            snapshot_record_id=record_id,
            state="invalid" if errors else "structurally_complete",
            artifacts=tuple(sorted(artifacts, key=_artifact_sort_key)),
            error="; ".join(errors) if errors else None,
        )
    )


def _invalid_snapshot_entry(
    record_id: str,
    error: str,
) -> CacheInventoryEntry:
    return CacheInventoryEntry(
        representation="mavedb_snapshot",
        source="mavedb",
        dataset_id=None,
        snapshot_record_id=record_id,
        state="invalid",
        artifacts=(),
        error=error,
    )


def _scan_snapshot_table_root(
    path: Path,
    root: Path,
    entries: list[CacheInventoryEntry],
    issues: dict[str, _InventoryIssue],
) -> None:
    if _path_kind(path) != "directory":
        _add_issue(
            issues,
            path,
            "MaveDB snapshot-table root is incomplete or unsafe.",
            source="mavedb_snapshot",
        )
        return
    try:
        record_paths = _children(path, authorized_root=root)
    except OSError:
        _add_issue(
            issues,
            path,
            "MaveDB snapshot-table root could not be inspected.",
            source="mavedb_snapshot",
        )
        return
    for record_path in record_paths:
        if not _SNAPSHOT_RECORD_PATTERN.fullmatch(record_path.name):
            continue
        if _path_kind(record_path) != "directory":
            _add_issue(
                issues,
                record_path,
                "MaveDB snapshot-table record location is incomplete or unsafe.",
                source="mavedb_snapshot",
            )
            continue
        try:
            children = _children(record_path, authorized_root=root)
        except OSError:
            _add_issue(
                issues,
                record_path,
                "MaveDB snapshot-table record directory could not be inspected.",
                source="mavedb_snapshot",
            )
            continue
        for child in children:
            if child.name.startswith(".extract-"):
                _add_issue(
                    issues,
                    child,
                    "Incomplete MaveDB snapshot-table extraction transaction.",
                    source="mavedb_snapshot",
                )
                continue
            if not child.name.startswith("urn-mavedb-"):
                continue
            if _path_kind(child) != "directory":
                _add_issue(
                    issues,
                    child,
                    "MaveDB snapshot-table entry is incomplete or unsafe.",
                    source="mavedb_snapshot",
                )
                continue
            _scan_snapshot_table_entry(
                child,
                record_path.name,
                root,
                entries,
                issues,
            )


def _scan_snapshot_table_entry(
    entry_path: Path,
    record_id: str,
    root: Path,
    entries: list[CacheInventoryEntry],
    issues: dict[str, _InventoryIssue],
) -> None:
    try:
        children = _children(entry_path, authorized_root=root)
    except OSError:
        _add_issue(
            issues,
            entry_path,
            "MaveDB snapshot-table entry could not be inspected.",
            source="mavedb_snapshot",
        )
        return
    manifest_path = entry_path / "manifest.json"
    if _path_kind(manifest_path) != "file":
        _add_issue(
            issues,
            manifest_path,
            "MaveDB snapshot-table manifest is missing or unsafe.",
            source="mavedb_snapshot",
        )
        return
    try:
        values = _read_json(manifest_path)
    except (OSError, UnicodeError, json.JSONDecodeError):
        _add_issue(
            issues,
            manifest_path,
            "MaveDB snapshot-table manifest could not be parsed.",
            source="mavedb_snapshot",
        )
        return

    dataset_id = _snapshot_table_dataset_identity(values)
    if dataset_id is None:
        _add_issue(
            issues,
            manifest_path,
            "MaveDB snapshot-table identity could not be recovered.",
            source="mavedb_snapshot",
        )
        return
    try:
        manifest = _snapshot_table_manifest_from_dict(
            values,
            expected_record_id=record_id,
            expected_dataset_id=dataset_id,
        )
    except _InvalidTableCache:
        entries.append(
            CacheInventoryEntry(
                representation="mavedb_snapshot_tables",
                source="mavedb_snapshot",
                dataset_id=dataset_id,
                snapshot_record_id=record_id,
                state="invalid",
                artifacts=_snapshot_table_present_artifacts(children),
                error="MaveDB snapshot-table manifest is structurally invalid.",
            )
        )
        return

    errors = _snapshot_table_errors(entry_path, children, manifest)
    entries.append(
        CacheInventoryEntry(
            representation="mavedb_snapshot_tables",
            source="mavedb_snapshot",
            dataset_id=manifest.dataset_id,
            snapshot_record_id=record_id,
            state="invalid" if errors else "structurally_complete",
            artifacts=_snapshot_table_present_artifacts(children),
            error="; ".join(errors) if errors else None,
        )
    )


def _snapshot_table_errors(
    entry_path: Path,
    children: tuple[Path, ...],
    manifest: _SnapshotTableManifest,
) -> list[str]:
    errors: list[str] = []
    stem = manifest.dataset_id.replace(":", "-")
    if entry_path.name != stem:
        errors.append("MaveDB snapshot-table manifest does not match its path.")
    expected_files = {"manifest.json", "scores.csv"}
    if manifest.counts is not None:
        expected_files.add("counts.csv")
    if {child.name for child in children} != expected_files:
        errors.append("MaveDB snapshot-table entry contains unexpected files.")
    for role, metadata in (
        ("scores", manifest.scores),
        ("counts", manifest.counts),
    ):
        path = entry_path / f"{role}.csv"
        if metadata is None:
            continue
        expected_member = f"csv/{stem}.{role}.csv"
        if metadata.filename != f"{role}.csv" or (
            metadata.archive_member != expected_member
        ):
            errors.append(
                f"MaveDB snapshot-table {role} metadata is inconsistent."
            )
        kind = _path_kind(path)
        if kind == "file":
            try:
                size = path.stat().st_size
            except OSError:
                errors.append(
                    f"MaveDB snapshot-table {role} size could not be read."
                )
            else:
                if size != metadata.size:
                    errors.append(
                        f"MaveDB snapshot-table {role} size does not match metadata."
                    )
        elif kind == "symlink":
            errors.append(f"MaveDB snapshot-table {role} is a symlink.")
        else:
            errors.append(
                f"MaveDB snapshot-table {role} is missing or not a regular file."
            )
    return errors


def _snapshot_table_present_artifacts(
    children: tuple[Path, ...],
) -> tuple[CacheInventoryArtifact, ...]:
    artifacts: list[CacheInventoryArtifact] = []
    for role in ("scores", "counts"):
        path = next(
            (child for child in children if child.name == f"{role}.csv"),
            None,
        )
        if path is None or _path_kind(path) != "file":
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        artifacts.append(CacheInventoryArtifact(role, path, size))
    return tuple(sorted(artifacts, key=_artifact_sort_key))


def _read_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _children(path: Path, *, authorized_root: Path) -> tuple[Path, ...]:
    """Return children without traversing symlinks or escaping the root."""
    if _path_kind(path) != "directory":
        raise OSError("managed path is not a regular directory")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(authorized_root)
    except ValueError as exc:
        raise OSError("managed path escapes the cache root") from exc
    return tuple(path.iterdir())


def _path_kind(path: Path) -> str:
    """Classify one path through lstat without following symlinks."""
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return "missing"
    except OSError:
        return "unreadable"
    mode = metadata.st_mode
    if stat.S_ISLNK(mode):
        return "symlink"
    if stat.S_ISREG(mode):
        return "file"
    if stat.S_ISDIR(mode):
        return "directory"
    return "other"


def _generic_manifest_identity(
    values: object,
) -> tuple[str, str] | None:
    if not isinstance(values, Mapping):
        return None
    source = values.get("source")
    dataset_id = values.get("dataset_id")
    if (
        not isinstance(source, str)
        or not source.strip()
        or not isinstance(dataset_id, str)
        or not dataset_id.strip()
    ):
        return None
    return source, dataset_id


def _snapshot_table_dataset_identity(values: object) -> str | None:
    if not isinstance(values, Mapping):
        return None
    dataset_id = values.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        return None
    return dataset_id


def _validate_filter(value: str | None, name: str) -> None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ValueError(f"{name} must be a non-empty string or None.")


def _matches_filters(
    entry_source: str,
    entry_dataset_id: str | None,
    *,
    source: str | None,
    dataset_id: str | None,
) -> bool:
    return (
        (source is None or entry_source == source)
        and (dataset_id is None or entry_dataset_id == dataset_id)
    )


def _issue_matches_filters(
    issue: _InventoryIssue,
    *,
    source: str | None,
    dataset_id: str | None,
) -> bool:
    if source is None and dataset_id is None:
        return True
    if source is not None and issue.source != source:
        return False
    if dataset_id is not None and issue.dataset_id != dataset_id:
        return False
    return True


def _add_issue(
    issues: dict[str, _InventoryIssue],
    path: Path,
    error: str,
    *,
    source: str | None = None,
    dataset_id: str | None = None,
) -> None:
    key = _normalized_path(path)
    issues.setdefault(
        key,
        _InventoryIssue(
            path=path,
            error=error,
            source=source,
            dataset_id=dataset_id,
        ),
    )


def _sorted_entry_artifacts(entry: CacheInventoryEntry) -> CacheInventoryEntry:
    return CacheInventoryEntry(
        representation=entry.representation,
        source=entry.source,
        dataset_id=entry.dataset_id,
        snapshot_record_id=entry.snapshot_record_id,
        state=entry.state,
        artifacts=tuple(sorted(entry.artifacts, key=_artifact_sort_key)),
        error=entry.error,
    )


def _entry_sort_key(entry: CacheInventoryEntry) -> tuple[object, ...]:
    primary = (
        entry.source,
        entry.snapshot_record_id or "",
        entry.dataset_id or "",
        entry.representation,
    )
    secondary = (
        entry.state,
        entry.error or "",
        tuple(
            (
                artifact.role,
                _normalized_path(artifact.path),
                artifact.size_bytes,
            )
            for artifact in entry.artifacts
        ),
    )
    return (*primary, *secondary)


def _artifact_sort_key(
    artifact: CacheInventoryArtifact,
) -> tuple[str, str]:
    return artifact.role, _normalized_path(artifact.path)


def _issue_sort_key(issue: _InventoryIssue) -> tuple[str, str]:
    return _normalized_path(issue.path), issue.error


def _normalized_path(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


__all__ = [
    "CacheEntryState",
    "CacheInventoryArtifact",
    "CacheInventoryEntry",
    "CacheInventoryIssue",
    "CacheInventoryResult",
    "inventory_cache",
]
