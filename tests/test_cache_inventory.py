from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest

import dms_parser.acquisition.cache as cache_module
import dms_parser.acquisition.cache_inventory as inventory_module
import dms_parser.sources.mavedb_snapshot_tables as table_module
import dms_parser.sources.mavedb_snapshots as snapshot_module
from dms_parser import (
    CacheInventoryArtifact,
    CacheInventoryEntry,
    CacheInventoryError,
    CacheInventoryIssue,
    CacheInventoryResult,
    FilesystemCache,
    inventory_cache,
)

RECORD_ID = "20840937"
DATASET_ID = "urn:mavedb:00000001-a-1"


def _write_json(path: Path, values: object) -> None:
    path.write_text(
        json.dumps(values, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _create_snapshot(
    cache: FilesystemCache,
    *,
    record_id: str = RECORD_ID,
    archive_content: bytes = b"archive",
    main_content: bytes = b"{}\n",
) -> Path:
    entry = cache.root / "mavedb" / "snapshots" / record_id
    entry.mkdir(parents=True)
    archive_name = "mavedb-snapshot.zip"
    (entry / archive_name).write_bytes(archive_content)
    (entry / "main.json").write_bytes(main_content)
    _write_json(
        entry / "snapshot.json",
        {
            "record_id": record_id,
            "doi": f"10.5281/zenodo.{record_id}",
            "concept_doi": snapshot_module.MAVEDB_ZENODO_CONCEPT_DOI,
            "publication_date": "2026-01-01",
            "archive_filename": archive_name,
            "archive_size": len(archive_content),
            "checksum": "md5:" + "0" * 32,
            "download_url": f"https://example.test/{archive_name}",
            "archive_format": "zip",
            "main_json_size": len(main_content),
            "main_json_sha256": "0" * 64,
        },
    )
    return entry


def _file_metadata(
    dataset_id: str,
    role: str,
    content: bytes,
) -> dict[str, Any]:
    stem = dataset_id.replace(":", "-")
    return {
        "archive_member": f"csv/{stem}.{role}.csv",
        "filename": f"{role}.csv",
        "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _create_snapshot_table(
    cache: FilesystemCache,
    *,
    dataset_id: str = DATASET_ID,
    record_id: str = RECORD_ID,
    counts_content: bytes | None = None,
) -> Path:
    entry = (
        cache.root
        / "mavedb"
        / "snapshot_tables"
        / record_id
        / dataset_id.replace(":", "-")
    )
    entry.mkdir(parents=True)
    scores_content = b"hgvs_pro,score\np.Ala1Val,1.0\n"
    (entry / "scores.csv").write_bytes(scores_content)
    if counts_content is not None:
        (entry / "counts.csv").write_bytes(counts_content)
    _write_json(
        entry / "manifest.json",
        {
            "source": "mavedb_snapshot",
            "record_id": record_id,
            "doi": f"10.5281/zenodo.{record_id}",
            "concept_doi": snapshot_module.MAVEDB_ZENODO_CONCEPT_DOI,
            "archive": {
                "filename": "mavedb-snapshot.zip",
                "size": 7,
                "checksum": "md5:" + "0" * 32,
            },
            "dataset_id": dataset_id,
            "is_superseded": False,
            "files": {
                "scores": _file_metadata(
                    dataset_id,
                    "scores",
                    scores_content,
                ),
                "counts": (
                    _file_metadata(dataset_id, "counts", counts_content)
                    if counts_content is not None
                    else None
                ),
            },
        },
    )
    return entry


def test_public_dataclass_field_order() -> None:
    assert [field.name for field in fields(CacheInventoryArtifact)] == [
        "role",
        "path",
        "size_bytes",
    ]
    assert [field.name for field in fields(CacheInventoryEntry)] == [
        "representation",
        "source",
        "dataset_id",
        "snapshot_record_id",
        "state",
        "artifacts",
        "error",
    ]
    assert [field.name for field in fields(CacheInventoryIssue)] == [
        "path",
        "state",
        "error",
    ]
    assert [field.name for field in fields(CacheInventoryResult)] == [
        "entries",
        "issues",
    ]


def test_nonexistent_root_is_empty_and_is_not_created(tmp_path: Path) -> None:
    root = tmp_path / "missing"

    result = inventory_cache(FilesystemCache(root))

    assert result == CacheInventoryResult(entries=(), issues=())
    assert not root.exists()


def test_root_lstat_permission_error_is_not_treated_as_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "cache"
    root.mkdir()
    original_lstat = Path.lstat

    def fail_root_lstat(path: Path):
        if path == root:
            raise PermissionError("denied")
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", fail_root_lstat)

    with pytest.raises(CacheInventoryError) as exc_info:
        inventory_cache(FilesystemCache(root))

    assert isinstance(exc_info.value.__cause__, PermissionError)


def test_empty_root_has_zero_aggregates(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    root.mkdir()

    result = inventory_cache(FilesystemCache(root))

    assert result.entry_count == 0
    assert result.artifact_count == 0
    assert result.total_size_bytes == 0
    assert result.complete_entry_count == 0
    assert result.invalid_entry_count == 0
    assert result.issue_count == 0


@pytest.mark.parametrize(
    ("source", "dataset_id", "url"),
    [
        (
            "proteingym",
            "reference-files-dms-substitutions",
            "https://example.test/reference.csv",
        ),
        (
            "proteingym",
            "reference-files-dms-indels",
            "https://example.test/indels.csv",
        ),
        (
            "proteingym",
            "resource-data-dms-substitutions",
            "https://example.test/data.parquet",
        ),
        ("mavedb", DATASET_ID, "https://example.test/scores"),
        ("mavedb-metadata", DATASET_ID, "https://example.test/metadata"),
        ("arbitrary source", "value/with:unsafe?text", "https://example.test/x"),
    ],
)
def test_generic_identities_are_recovered_exactly_from_manifests(
    tmp_path: Path,
    source: str,
    dataset_id: str,
    url: str,
) -> None:
    cache = FilesystemCache(tmp_path / "cache")
    path = cache.store_bytes(source, dataset_id, url, b"payload")

    result = inventory_cache(cache)

    assert result.entries == (
        CacheInventoryEntry(
            representation="generic",
            source=source,
            dataset_id=dataset_id,
            snapshot_record_id=None,
            state="structurally_complete",
            artifacts=(CacheInventoryArtifact("artifact", path, 7),),
            error=None,
        ),
    )


def test_store_file_entry_is_supported(tmp_path: Path) -> None:
    source_path = tmp_path / "source.bin"
    source_path.write_bytes(b"stored file")
    cache = FilesystemCache(tmp_path / "cache")
    cached = cache.store_file(
        "custom",
        "dataset",
        "https://example.test/source.bin",
        source_path,
    )

    result = inventory_cache(cache)

    assert result.entries[0].artifacts == (
        CacheInventoryArtifact("artifact", cached, len(b"stored file")),
    )


def test_generic_manifest_path_mismatch_is_invalid(tmp_path: Path) -> None:
    cache = FilesystemCache(tmp_path / "cache")
    artifact = cache.store_bytes(
        "source",
        "dataset",
        "https://example.test/file.bin",
        b"payload",
    )
    wrong_entry = cache.entry_path("source", "other")
    wrong_entry.parent.mkdir(parents=True, exist_ok=True)
    artifact.parent.rename(wrong_entry)

    result = inventory_cache(cache)

    assert result.entries[0].state == "invalid"
    assert "does not match its entry path" in (result.entries[0].error or "")


def test_expected_descendant_source_symlink_cannot_hide_path_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = FilesystemCache(tmp_path / "cache")
    artifact = cache.store_bytes(
        "source",
        "dataset",
        "https://example.test/file.bin",
        b"payload",
    )
    expected_source = artifact.parent.parent
    wrong_source = cache.entry_path("wrong-source", "placeholder").parent
    expected_source.rename(wrong_source)
    try:
        expected_source.symlink_to(wrong_source, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Directory symlinks are unavailable: {exc}")

    original_resolve = Path.resolve

    def guarded_resolve(
        path: Path,
        strict: bool = False,
    ) -> Path:
        if path == expected_source or expected_source in path.parents:
            raise AssertionError("descendant symlink path must not be resolved")
        return original_resolve(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", guarded_resolve)

    result = inventory_cache(cache)

    assert result.entry_count == 1
    assert result.entries[0].state == "invalid"
    assert result.entries[0].error == (
        "Generic cache manifest does not match its entry path."
    )
    assert result.issues[0].path == expected_source


@pytest.mark.parametrize("failure", ["missing", "size", "directory"])
def test_generic_payload_structural_failures_are_invalid(
    tmp_path: Path,
    failure: str,
) -> None:
    cache = FilesystemCache(tmp_path / "cache")
    artifact = cache.store_bytes(
        "source",
        "dataset",
        "https://example.test/file.bin",
        b"payload",
    )
    if failure == "missing":
        artifact.unlink()
    elif failure == "size":
        artifact.write_bytes(b"different size")
    else:
        artifact.unlink()
        artifact.mkdir()

    result = inventory_cache(cache)

    assert result.entries[0].state == "invalid"
    assert result.complete_entry_count == 0
    assert result.invalid_entry_count == 1
    assert result.artifact_count == (1 if failure == "size" else 0)


def test_malformed_generic_manifest_with_identity_is_invalid(
    tmp_path: Path,
) -> None:
    cache = FilesystemCache(tmp_path / "cache")
    artifact = cache.store_bytes(
        "source",
        "dataset",
        "https://example.test/file.bin",
        b"payload",
    )
    _write_json(
        artifact.parent / "manifest.json",
        {"source": "source", "dataset_id": "dataset"},
    )

    result = inventory_cache(cache)

    assert result.entries[0].state == "invalid"
    assert result.entries[0].source == "source"
    assert result.entries[0].dataset_id == "dataset"
    assert result.entries[0].artifacts == ()
    assert result.issues == (
        CacheInventoryIssue(
            path=artifact,
            state="incomplete",
            error="Unreferenced generic cache artifact.",
        ),
    )
    assert result.artifact_count == 0
    assert result.total_size_bytes == 0


def test_unrecoverable_manifest_and_generic_orphans_become_issues(
    tmp_path: Path,
) -> None:
    cache = FilesystemCache(tmp_path / "cache")
    artifact = cache.store_bytes(
        "source",
        "dataset",
        "https://example.test/file.bin",
        b"payload",
    )
    (artifact.parent / "manifest.json").write_text("{", encoding="utf-8")
    (artifact.parent / ".artifact-interrupted").write_bytes(b"partial")
    (artifact.parent / ".manifest-interrupted").write_bytes(b"partial")

    result = inventory_cache(cache)

    assert result.entries == ()
    assert {issue.path for issue in result.issues} == {
        artifact.parent / "manifest.json",
        artifact,
        artifact.parent / ".artifact-interrupted",
        artifact.parent / ".manifest-interrupted",
    }
    assert all(issue.state == "incomplete" for issue in result.issues)


def test_valid_snapshot_exposes_only_main_and_archive_payloads(
    tmp_path: Path,
) -> None:
    cache = FilesystemCache(tmp_path / "cache")
    entry = _create_snapshot(cache)
    (entry / "user-note.txt").write_text("ignored", encoding="utf-8")

    result = inventory_cache(cache)

    snapshot = result.entries[0]
    assert snapshot.representation == "mavedb_snapshot"
    assert snapshot.source == "mavedb"
    assert snapshot.dataset_id is None
    assert snapshot.snapshot_record_id == RECORD_ID
    assert snapshot.state == "structurally_complete"
    assert [(artifact.role, artifact.size_bytes) for artifact in snapshot.artifacts] == [
        ("archive", 7),
        ("main", 3),
    ]
    assert result.artifact_count == 2
    assert result.total_size_bytes == 10


@pytest.mark.parametrize("failure", ["metadata", "missing", "size"])
def test_snapshot_structural_failures_are_invalid(
    tmp_path: Path,
    failure: str,
) -> None:
    cache = FilesystemCache(tmp_path / "cache")
    entry = _create_snapshot(cache)
    if failure == "metadata":
        (entry / "snapshot.json").write_text("{}", encoding="utf-8")
    elif failure == "missing":
        (entry / "main.json").unlink()
    else:
        (entry / "main.json").write_bytes(b"wrong size")

    result = inventory_cache(cache)

    assert result.entries[0].state == "invalid"
    assert result.entries[0].snapshot_record_id == RECORD_ID


def test_snapshot_staging_and_backup_objects_are_incomplete(
    tmp_path: Path,
) -> None:
    cache = FilesystemCache(tmp_path / "cache")
    root = cache.root / "mavedb" / "snapshots"
    (root / ".20840937-random").mkdir(parents=True)
    (root / ".20840937-backup-random").mkdir()

    result = inventory_cache(cache)

    assert result.entry_count == 0
    assert [issue.path.name for issue in result.issues] == [
        ".20840937-backup-random",
        ".20840937-random",
    ]


def test_snapshot_payload_symlink_is_invalid_and_not_followed(
    tmp_path: Path,
) -> None:
    cache = FilesystemCache(tmp_path / "cache")
    entry = _create_snapshot(cache)
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"{}\n")
    main = entry / "main.json"
    main.unlink()
    try:
        main.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"File symlinks are unavailable: {exc}")

    result = inventory_cache(cache)

    assert result.entries[0].state == "invalid"
    assert [artifact.role for artifact in result.entries[0].artifacts] == [
        "archive"
    ]
    assert outside.read_bytes() == b"{}\n"


@pytest.mark.parametrize("with_counts", [False, True])
def test_snapshot_table_payloads_and_exact_identity(
    tmp_path: Path,
    with_counts: bool,
) -> None:
    cache = FilesystemCache(tmp_path / "cache")
    counts = b"count\n10\n" if with_counts else None
    _create_snapshot_table(cache, counts_content=counts)

    result = inventory_cache(cache)

    table = result.entries[0]
    assert table.representation == "mavedb_snapshot_tables"
    assert table.source == "mavedb_snapshot"
    assert table.dataset_id == DATASET_ID
    assert table.snapshot_record_id == RECORD_ID
    assert table.state == "structurally_complete"
    assert [artifact.role for artifact in table.artifacts] == (
        ["counts", "scores"] if with_counts else ["scores"]
    )


@pytest.mark.parametrize("failure", ["missing", "size", "extra", "manifest"])
def test_snapshot_table_structural_failures(
    tmp_path: Path,
    failure: str,
) -> None:
    cache = FilesystemCache(tmp_path / "cache")
    entry = _create_snapshot_table(cache)
    if failure == "missing":
        (entry / "scores.csv").unlink()
    elif failure == "size":
        (entry / "scores.csv").write_bytes(b"wrong")
    elif failure == "extra":
        (entry / "extra.txt").write_text("extra", encoding="utf-8")
    else:
        manifest = json.loads((entry / "manifest.json").read_text(encoding="utf-8"))
        manifest["source"] = "wrong"
        _write_json(entry / "manifest.json", manifest)

    result = inventory_cache(cache)

    assert result.entries[0].state == "invalid"
    assert result.entries[0].dataset_id == DATASET_ID


def test_snapshot_table_malformed_identity_is_an_issue(tmp_path: Path) -> None:
    cache = FilesystemCache(tmp_path / "cache")
    entry = _create_snapshot_table(cache)
    (entry / "manifest.json").write_text("{", encoding="utf-8")

    result = inventory_cache(cache)

    assert result.entries == ()
    assert result.issues[0].path == entry / "manifest.json"


def test_snapshot_table_payload_symlink_is_invalid_and_not_counted(
    tmp_path: Path,
) -> None:
    cache = FilesystemCache(tmp_path / "cache")
    entry = _create_snapshot_table(cache)
    outside = tmp_path / "outside.csv"
    outside.write_bytes((entry / "scores.csv").read_bytes())
    scores = entry / "scores.csv"
    scores.unlink()
    try:
        scores.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"File symlinks are unavailable: {exc}")

    result = inventory_cache(cache)

    assert result.entries[0].state == "invalid"
    assert result.artifact_count == 0
    assert outside.read_bytes().startswith(b"hgvs_pro")


def test_snapshot_table_extraction_transaction_is_an_issue(
    tmp_path: Path,
) -> None:
    cache = FilesystemCache(tmp_path / "cache")
    transaction = (
        cache.root
        / "mavedb"
        / "snapshot_tables"
        / RECORD_ID
        / ".extract-random"
    )
    transaction.mkdir(parents=True)

    result = inventory_cache(cache)

    assert result.issues == (
        CacheInventoryIssue(
            path=transaction,
            state="incomplete",
            error="Incomplete MaveDB snapshot-table extraction transaction.",
        ),
    )


def test_ordering_filters_and_aggregate_counts(tmp_path: Path) -> None:
    cache = FilesystemCache(tmp_path / "cache")
    cache.store_bytes(
        "z-source",
        "z-dataset",
        "https://example.test/z",
        b"z",
    )
    cache.store_bytes(
        "a-source",
        "a-dataset",
        "https://example.test/a",
        b"aa",
    )
    _create_snapshot(cache)
    _create_snapshot_table(cache, counts_content=b"count\n1\n")

    result = inventory_cache(cache)

    assert [entry.source for entry in result.entries] == [
        "a-source",
        "mavedb",
        "mavedb_snapshot",
        "z-source",
    ]
    assert result.entry_count == 4
    assert result.artifact_count == 6
    assert result.total_size_bytes == 50
    assert result.complete_entry_count == 4
    assert result.invalid_entry_count == 0

    assert inventory_cache(cache, source="A-source").entries == ()
    assert inventory_cache(cache, source="a-source").entries[0].dataset_id == (
        "a-dataset"
    )
    tables = inventory_cache(
        cache,
        source="mavedb_snapshot",
        dataset_id=DATASET_ID,
    )
    assert [entry.representation for entry in tables.entries] == [
        "mavedb_snapshot_tables"
    ]
    assert inventory_cache(cache, dataset_id=DATASET_ID).entries[0].source == (
        "mavedb_snapshot"
    )


def test_duplicate_logical_identities_have_stable_secondary_ordering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = FilesystemCache(tmp_path / "cache")
    artifact = cache.store_bytes(
        "source",
        "dataset",
        "https://example.test/file.bin",
        b"payload",
    )
    entry = artifact.parent
    manifest_bytes = (entry / "manifest.json").read_bytes()
    artifact_bytes = artifact.read_bytes()
    for physical_source in ("wrong-z", "wrong-a"):
        source_path = cache.entry_path(physical_source, "placeholder").parent
        duplicate = source_path / entry.name
        duplicate.mkdir(parents=True)
        (duplicate / "manifest.json").write_bytes(manifest_bytes)
        (duplicate / artifact.name).write_bytes(artifact_bytes)
    artifact.unlink()
    (entry / "manifest.json").unlink()
    entry.rmdir()
    entry.parent.rmdir()

    first = inventory_cache(cache)
    authorized_root = cache.root.resolve()
    original_children = inventory_module._children

    def reversed_root_children(
        path: Path,
        *,
        authorized_root: Path,
    ) -> tuple[Path, ...]:
        children = original_children(path, authorized_root=authorized_root)
        return tuple(reversed(children)) if path == authorized_root else children

    monkeypatch.setattr(
        inventory_module,
        "_children",
        reversed_root_children,
    )
    second = inventory_cache(cache)

    assert first.entries == second.entries
    assert len(first.entries) == 2
    assert all(entry.state == "invalid" for entry in first.entries)
    artifact_paths = [str(entry.artifacts[0].path).casefold() for entry in first.entries]
    assert artifact_paths == sorted(artifact_paths)


def test_identityless_issues_are_omitted_under_filters(tmp_path: Path) -> None:
    cache = FilesystemCache(tmp_path / "cache")
    entry = cache.entry_path("source", "dataset")
    entry.mkdir(parents=True)
    (entry / ".artifact-partial").write_bytes(b"partial")

    assert inventory_cache(cache).issue_count == 2
    assert inventory_cache(cache, source="source").issues == ()
    assert inventory_cache(cache, dataset_id="dataset").issues == ()


def test_inventory_does_not_recalculate_checksums(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = FilesystemCache(tmp_path / "cache")
    cache.store_bytes(
        "source",
        "dataset",
        "https://example.test/file.bin",
        b"payload",
    )
    _create_snapshot(cache)
    _create_snapshot_table(cache)

    def fail(*args: object, **kwargs: object) -> str:
        raise AssertionError("checksum verification must not run")

    monkeypatch.setattr(cache_module, "_sha256_file", fail)
    monkeypatch.setattr(snapshot_module, "_checksum_file", fail)
    monkeypatch.setattr(table_module, "_sha256_file", fail)

    result = inventory_cache(cache)

    assert result.complete_entry_count == 3


def test_inventory_invokes_no_cache_writes_acquisition_or_extraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = FilesystemCache(tmp_path / "cache")
    cache.store_bytes(
        "source",
        "dataset",
        "https://example.test/file.bin",
        b"payload",
    )
    _create_snapshot(cache)
    _create_snapshot_table(cache)

    def fail(*args: object, **kwargs: object) -> object:
        raise AssertionError("inventory invoked a mutating producer")

    monkeypatch.setattr(FilesystemCache, "resolve", fail)
    monkeypatch.setattr(FilesystemCache, "store_file", fail)
    monkeypatch.setattr(FilesystemCache, "store_bytes", fail)
    monkeypatch.setattr(snapshot_module, "fetch_mavedb_snapshot", fail)
    monkeypatch.setattr(table_module, "extract_mavedb_snapshot_tables", fail)

    assert inventory_cache(cache).complete_entry_count == 3


def test_malformed_entry_does_not_block_unrelated_valid_entry(
    tmp_path: Path,
) -> None:
    cache = FilesystemCache(tmp_path / "cache")
    valid = cache.store_bytes(
        "valid",
        "dataset",
        "https://example.test/valid.bin",
        b"valid",
    )
    malformed = cache.store_bytes(
        "broken",
        "dataset",
        "https://example.test/broken.bin",
        b"broken",
    )
    (malformed.parent / "manifest.json").write_text("{", encoding="utf-8")

    result = inventory_cache(cache)

    assert [entry.source for entry in result.entries] == ["valid"]
    assert result.entries[0].artifacts[0].path == valid
    assert result.issue_count == 2


def test_inventory_preserves_bytes_mtimes_and_directory_membership(
    tmp_path: Path,
) -> None:
    cache = FilesystemCache(tmp_path / "cache")
    artifact = cache.store_bytes(
        "source",
        "dataset",
        "https://example.test/file.bin",
        b"payload",
    )
    manifest = artifact.parent / "manifest.json"
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (artifact, manifest)
    }
    names_before = tuple(sorted(path.name for path in artifact.parent.iterdir()))

    inventory_cache(cache)

    assert tuple(sorted(path.name for path in artifact.parent.iterdir())) == names_before
    assert {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (artifact, manifest)
    } == before


def test_root_symlink_is_allowed_but_descendant_symlinks_are_not_followed(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    cache = FilesystemCache(target)
    artifact = cache.store_bytes(
        "source",
        "dataset",
        "https://example.test/file.bin",
        b"payload",
    )
    link = tmp_path / "cache-link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Directory symlinks are unavailable: {exc}")

    result = inventory_cache(FilesystemCache(link))
    assert result.complete_entry_count == 1

    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    artifact.unlink()
    artifact.symlink_to(outside)
    result = inventory_cache(FilesystemCache(link))

    assert result.entries[0].state == "invalid"
    assert result.artifact_count == 0
    assert outside.read_bytes() == b"outside"


def test_root_traversal_failure_is_chained(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "cache"
    root.mkdir()
    original = Path.iterdir

    def fail(path: Path):
        if path == root.resolve():
            raise PermissionError("denied")
        return original(path)

    monkeypatch.setattr(Path, "iterdir", fail)

    with pytest.raises(CacheInventoryError) as exc_info:
        inventory_cache(FilesystemCache(root))

    assert isinstance(exc_info.value.__cause__, PermissionError)


def test_unexpected_programming_error_is_not_normalized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "cache"
    root.mkdir()

    def fail(path: Path):
        raise RuntimeError("programming error")

    monkeypatch.setattr(Path, "iterdir", fail)

    with pytest.raises(RuntimeError, match="programming error"):
        inventory_cache(FilesystemCache(root))


def test_unrelated_content_is_ignored(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    (root / "user" / "nested").mkdir(parents=True)
    (root / "notes.txt").write_text("unmanaged", encoding="utf-8")

    assert inventory_cache(FilesystemCache(root)) == CacheInventoryResult(
        entries=(),
        issues=(),
    )
