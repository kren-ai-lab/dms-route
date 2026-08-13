from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import tarfile
import zipfile
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any

import pytest

import dms_parser
import dms_parser.sources as sources_module
import dms_parser.sources.mavedb_snapshot_tables as tables_module
from dms_parser import FilesystemCache
from dms_parser.core.exceptions import (
    MaveDBSnapshotTableError,
    SnapshotDatasetNotFoundError,
    SupersededSnapshotDatasetError,
)
from dms_parser.sources.mavedb_snapshots import (
    MAVEDB_ZENODO_CONCEPT_DOI,
    MaveDBSnapshot,
    MaveDBSnapshotRecord,
)
from dms_parser.sources.mavedb_snapshot_tables import (
    MaveDBSnapshotTable,
    MaveDBSnapshotTableExtractionResult,
    extract_mavedb_snapshot_tables,
)

CURRENT_ONE = "urn:mavedb:00000003-a-1"
CURRENT_TWO = "urn:mavedb:00000003-a-2"
SUPERSEDED = "urn:mavedb:00000003-a-3"
MISSING = "urn:mavedb:99999999-a-1"
CURRENT_ONE_SCORES = b"hgvs,score\np.A1V,1\n"
CURRENT_ONE_COUNTS = b"hgvs,count\np.A1V,4\n"


def _archive_member(dataset_id: str, kind: str) -> str:
    """Return the official exact member name for one raw table."""
    return f"csv/{dataset_id.replace(':', '-')}.{kind}.csv"


def _record_root(tmp_path: Path) -> Path:
    """Return the extraction-cache root for the fixture snapshot record."""
    return tmp_path / "cache" / "mavedb" / "snapshot_tables" / "20840937"


def _bundle_bytes(entry: Path) -> dict[str, bytes]:
    """Read one cache bundle for before/after preservation assertions."""
    return {path.name: path.read_bytes() for path in entry.iterdir()}


def _raiser(error: Exception):
    """Return a test double that raises one specific exception instance."""
    def fail(*args: Any, **kwargs: Any) -> None:
        raise error

    return fail


def _main_json() -> bytes:
    """Return one compact catalog with current and superseded score sets."""
    payload = {
        "title": "MaveDB public data dump test",
        "asOf": "2026-06-24T18:13:01Z",
        "experimentSets": [
            {
                "urn": "urn:mavedb:00000003",
                "experiments": [
                    {
                        "urn": "urn:mavedb:00000003-a",
                        "title": "Test experiment",
                        "scoreSetUrns": [CURRENT_ONE, CURRENT_TWO],
                        "scoreSets": [
                            {
                                "urn": CURRENT_ONE,
                                "title": "Current one",
                                "numVariants": 2,
                                "targetGenes": [{"name": "GENE1"}],
                            },
                            {
                                "urn": CURRENT_TWO,
                                "title": "Current two",
                                "numVariants": 1,
                                "targetGenes": [{"name": "GENE2"}],
                            },
                            {
                                "urn": SUPERSEDED,
                                "title": "Superseded",
                                "numVariants": 1,
                                "targetGenes": [{"name": "GENE1 legacy"}],
                            },
                        ],
                    }
                ],
            }
        ],
    }
    return (json.dumps(payload) + "\n").encode()


def _default_entries() -> list[tuple[str, bytes, str]]:
    """Return selected and unrelated archive fixtures."""
    return [
        ("main.json", _main_json(), "file"),
        (_archive_member(CURRENT_ONE, "scores"), CURRENT_ONE_SCORES, "file"),
        (_archive_member(CURRENT_ONE, "counts"), CURRENT_ONE_COUNTS, "file"),
        (_archive_member(CURRENT_TWO, "scores"), b"hgvs,score\np.A2V,2\n", "file"),
        (_archive_member(SUPERSEDED, "scores"), b"hgvs,score\np.A3V,3\n", "file"),
        ("csv/unrelated.scores.csv", b"must,not,extract\n", "file"),
    ]


def _under_top_level(
    entries: list[tuple[str, bytes, str]],
    top_level: str = "arbitrary-snapshot-version",
) -> list[tuple[str, bytes, str]]:
    """Place archive fixtures below one variable top-level directory."""
    return [
        (f"{top_level}/{name}", content, kind)
        for name, content, kind in entries
    ]


def _replace_scores_content(
    content: bytes,
) -> list[tuple[str, bytes, str]]:
    """Replace the primary fixture score table while preserving its path."""
    expected_name = _archive_member(CURRENT_ONE, "scores")
    return [
        (name, content if name == expected_name else value, kind)
        for name, value, kind in _default_entries()
    ]


def _write_zip(
    path: Path,
    entries: list[tuple[str, bytes, str]],
) -> None:
    """Write a ZIP fixture supporting selected special entry types."""
    raw_names: list[tuple[bytes, bytes]] = []
    with zipfile.ZipFile(path, mode="w") as archive:
        for name, content, kind in entries:
            stored_name = name.replace("\\", "/")
            if stored_name != name:
                raw_names.append((stored_name.encode(), name.encode()))
            member = zipfile.ZipInfo(stored_name)
            member.create_system = 3
            if kind == "symlink":
                member.external_attr = (stat.S_IFLNK | 0o777) << 16
            elif kind == "directory":
                member.external_attr = (stat.S_IFDIR | 0o755) << 16
            elif kind == "special":
                member.external_attr = (stat.S_IFIFO | 0o644) << 16
            else:
                member.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(member, content)
    if raw_names:
        content = path.read_bytes()
        for normalized, raw in raw_names:
            assert len(normalized) == len(raw)
            content = content.replace(normalized, raw)
        path.write_bytes(content)


def _write_tar(
    path: Path,
    entries: list[tuple[str, bytes, str]],
) -> None:
    """Write a TAR.GZ fixture supporting selected special entry types."""
    with tarfile.open(path, mode="w:gz") as archive:
        for name, content, kind in entries:
            member = tarfile.TarInfo(name)
            if kind == "symlink":
                member.type = tarfile.SYMTYPE
                member.linkname = "elsewhere"
                archive.addfile(member)
            elif kind == "hardlink":
                member.type = tarfile.LNKTYPE
                member.linkname = "elsewhere"
                archive.addfile(member)
            elif kind == "directory":
                member.type = tarfile.DIRTYPE
                archive.addfile(member)
            elif kind == "fifo":
                member.type = tarfile.FIFOTYPE
                archive.addfile(member)
            elif kind == "device":
                member.type = tarfile.CHRTYPE
                archive.addfile(member)
            else:
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))


def _snapshot(
    tmp_path: Path,
    archive_format: str = "zip",
    *,
    entries: list[tuple[str, bytes, str]] | None = None,
) -> MaveDBSnapshot:
    """Create one locally prepared managed snapshot fixture."""
    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir(parents=True, exist_ok=True)
    main_json_path = snapshot_root / "main.json"
    main_json_path.write_bytes(_main_json())
    filename = f"mavedb-dump.test.{archive_format}"
    archive_path = snapshot_root / filename
    selected_entries = _default_entries() if entries is None else entries
    if archive_format == "zip":
        _write_zip(archive_path, selected_entries)
    else:
        _write_tar(archive_path, selected_entries)
    archive_bytes = archive_path.read_bytes()
    return MaveDBSnapshot(
        record=MaveDBSnapshotRecord(
            record_id="20840937",
            doi="10.5281/zenodo.20840937",
            concept_doi=MAVEDB_ZENODO_CONCEPT_DOI,
            publication_date="2026-06-24",
            filename=filename,
            size=len(archive_bytes),
            checksum=f"md5:{hashlib.md5(archive_bytes).hexdigest()}",
            download_url="https://zenodo.org/content",
        ),
        archive_path=archive_path,
        main_json_path=main_json_path,
        cache_hit=True,
    )


def _extract(
    snapshot: MaveDBSnapshot,
    dataset_ids: object,
    tmp_path: Path,
    **kwargs: Any,
) -> MaveDBSnapshotTableExtractionResult:
    """Extract through a dedicated temporary FilesystemCache."""
    return extract_mavedb_snapshot_tables(
        snapshot,
        dataset_ids,  # type: ignore[arg-type]
        cache=FilesystemCache(tmp_path / "cache"),
        **kwargs,
    )


def test_public_exports_and_immutable_result_records(tmp_path: Path) -> None:
    names = (
        "MaveDBSnapshotTable",
        "MaveDBSnapshotTableExtractionResult",
        "extract_mavedb_snapshot_tables",
    )
    for name in names:
        assert getattr(dms_parser, name) is getattr(sources_module, name)
        assert getattr(dms_parser, name) is getattr(tables_module, name)

    table = MaveDBSnapshotTable(
        dataset_id=CURRENT_ONE,
        scores_path=tmp_path / "scores.csv",
        counts_path=None,
        is_superseded=False,
        cache_hit=False,
    )
    result = MaveDBSnapshotTableExtractionResult(
        record=_snapshot(tmp_path).record,
        tables=(table,),
    )
    assert result.dataset_count == 1
    with pytest.raises(FrozenInstanceError):
        table.cache_hit = True  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.tables = ()  # type: ignore[misc]


@pytest.mark.parametrize(
    "dataset_ids",
    [[], CURRENT_ONE, CURRENT_ONE.encode(), {}, {CURRENT_ONE}, True, None],
)
def test_invalid_dataset_sequences_are_rejected_before_io(
    dataset_ids: object,
    tmp_path: Path,
) -> None:
    with pytest.raises(MaveDBSnapshotTableError, match="dataset_ids"):
        _extract(_snapshot(tmp_path), dataset_ids, tmp_path)


@pytest.mark.parametrize(
    "dataset_ids",
    [
        [""],
        ["   "],
        ["invalid"],
        ["urn:mavedb:00000003-A-1"],
        [CURRENT_ONE, CURRENT_ONE],
        [True],
    ],
)
def test_invalid_or_duplicate_dataset_ids_are_rejected(
    dataset_ids: object,
    tmp_path: Path,
) -> None:
    with pytest.raises(MaveDBSnapshotTableError):
        _extract(_snapshot(tmp_path), dataset_ids, tmp_path)


def test_meta_analysis_score_set_urn_is_canonically_valid() -> None:
    assert tables_module._validate_dataset_ids(
        ["urn:mavedb:00000055-0-1"]
    ) == ("urn:mavedb:00000055-0-1",)


@pytest.mark.parametrize(
    ("option", "value"),
    [("include_superseded", 1), ("refresh", 1), ("cache", object())],
)
def test_invalid_options_are_rejected(
    option: str,
    value: object,
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(tmp_path)
    kwargs = {option: value}
    with pytest.raises(MaveDBSnapshotTableError, match=option):
        extract_mavedb_snapshot_tables(
            snapshot,
            [CURRENT_ONE],
            **kwargs,  # type: ignore[arg-type]
        )


def test_invalid_snapshot_object_is_rejected() -> None:
    with pytest.raises(MaveDBSnapshotTableError, match="snapshot"):
        extract_mavedb_snapshot_tables(object(), [CURRENT_ONE])  # type: ignore[arg-type]


@pytest.mark.parametrize("missing", ["archive", "main"])
def test_missing_snapshot_files_are_rejected(
    missing: str,
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(tmp_path)
    path = (
        snapshot.archive_path
        if missing == "archive"
        else snapshot.main_json_path
    )
    path.unlink()

    with pytest.raises(MaveDBSnapshotTableError, match=missing):
        _extract(snapshot, [CURRENT_ONE], tmp_path)


def test_snapshot_membership_and_superseded_policy(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)

    with pytest.raises(SnapshotDatasetNotFoundError, match=MISSING):
        _extract(snapshot, [MISSING], tmp_path)
    with pytest.raises(SupersededSnapshotDatasetError, match="superseded"):
        _extract(snapshot, [SUPERSEDED], tmp_path)

    result = _extract(
        snapshot,
        [SUPERSEDED, CURRENT_ONE],
        tmp_path,
        include_superseded=True,
    )
    assert [table.dataset_id for table in result.tables] == [
        CURRENT_ONE,
        SUPERSEDED,
    ]
    assert [table.is_superseded for table in result.tables] == [False, True]


def test_catalog_is_loaded_once_for_multiple_dataset_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = tables_module.MaveDBBulkCatalog.from_file

    def load(path: Path):
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(tables_module.MaveDBBulkCatalog, "from_file", load)

    _extract(_snapshot(tmp_path), [CURRENT_ONE, CURRENT_TWO], tmp_path)

    assert calls == 1


@pytest.mark.parametrize("archive_format", ["zip", "tar.gz"])
def test_root_level_selective_extraction_scores_counts_and_manifest(
    archive_format: str,
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(tmp_path, archive_format)
    result = _extract(snapshot, [CURRENT_TWO, CURRENT_ONE], tmp_path)

    assert result.record is snapshot.record
    assert result.dataset_count == 2
    assert [table.dataset_id for table in result.tables] == [
        CURRENT_ONE,
        CURRENT_TWO,
    ]
    first, second = result.tables
    expected_root = _record_root(tmp_path)
    assert first.scores_path == (
        expected_root / CURRENT_ONE.replace(":", "-") / "scores.csv"
    )
    assert first.scores_path.read_bytes() == CURRENT_ONE_SCORES
    assert first.counts_path is not None
    assert first.counts_path.read_bytes() == CURRENT_ONE_COUNTS
    assert second.counts_path is None
    assert first.cache_hit is second.cache_hit is False
    assert not (expected_root / "unrelated.scores.csv").exists()
    assert not any(
        path.name == "main.json" for path in expected_root.rglob("*")
    )

    manifest = json.loads(
        (first.scores_path.parent / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest == {
        "source": "mavedb_snapshot",
        "record_id": "20840937",
        "doi": "10.5281/zenodo.20840937",
        "concept_doi": MAVEDB_ZENODO_CONCEPT_DOI,
        "archive": {
            "filename": snapshot.record.filename,
            "size": snapshot.record.size,
            "checksum": snapshot.record.checksum,
        },
        "dataset_id": CURRENT_ONE,
        "is_superseded": False,
        "files": {
            "scores": {
                "archive_member": _archive_member(CURRENT_ONE, "scores"),
                "filename": "scores.csv",
                "size": len(CURRENT_ONE_SCORES),
                "sha256": hashlib.sha256(CURRENT_ONE_SCORES).hexdigest(),
            },
            "counts": {
                "archive_member": _archive_member(CURRENT_ONE, "counts"),
                "filename": "counts.csv",
                "size": len(CURRENT_ONE_COUNTS),
                "sha256": hashlib.sha256(CURRENT_ONE_COUNTS).hexdigest(),
            },
        },
    }
    manifest_text = (first.scores_path.parent / "manifest.json").read_text(
        encoding="utf-8"
    )
    for forbidden in ("retrieved_at", "extracted_at", "cache_hit", str(tmp_path)):
        assert forbidden not in manifest_text


@pytest.mark.parametrize("archive_format", ["zip", "tar.gz"])
def test_versioned_top_level_layout_resolves_scores_and_counts(
    archive_format: str,
    tmp_path: Path,
) -> None:
    top_level = "some-future-mavedb-release"
    snapshot = _snapshot(
        tmp_path,
        archive_format,
        entries=_under_top_level(_default_entries(), top_level),
    )

    first = _extract(snapshot, [CURRENT_ONE], tmp_path).tables[0]

    assert first.scores_path.read_bytes() == CURRENT_ONE_SCORES
    assert first.counts_path is not None
    assert first.counts_path.read_bytes() == CURRENT_ONE_COUNTS
    manifest = json.loads(
        (first.scores_path.parent / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["files"]["scores"]["archive_member"] == (
        f"{top_level}/{_archive_member(CURRENT_ONE, 'scores')}"
    )
    assert manifest["files"]["counts"]["archive_member"] == (
        f"{top_level}/{_archive_member(CURRENT_ONE, 'counts')}"
    )

    cached = _extract(snapshot, [CURRENT_ONE], tmp_path).tables[0]
    assert cached.cache_hit is True


@pytest.mark.parametrize("archive_format", ["zip", "tar.gz"])
def test_appledouble_sidecar_is_ignored_beside_valid_member(
    archive_format: str,
    tmp_path: Path,
) -> None:
    top_level = "versioned-dump"
    entries = _under_top_level(_default_entries(), top_level)
    filename = _archive_member(CURRENT_ONE, "scores").removeprefix("csv/")
    entries.append(
        (
            f"{top_level}/csv/._{filename}",
            b"appledouble metadata",
            "file",
        )
    )
    snapshot = _snapshot(tmp_path, archive_format, entries=entries)

    table = _extract(snapshot, [CURRENT_ONE], tmp_path).tables[0]

    assert table.scores_path.read_bytes() == CURRENT_ONE_SCORES


@pytest.mark.parametrize("archive_format", ["zip", "tar.gz"])
def test_current_bulk_score_headers_are_normalized_exactly(
    archive_format: str,
    tmp_path: Path,
) -> None:
    source = (
        b"accession,hgvs_pro,scores.score,scores.sd,scores.se,scores.df,"
        b"scores.score.extra\n"
        b"urn:mavedb:00000003-a-1#1,p.=,1.25,0.2,0.03,12,unchanged\n"
    )
    snapshot = _snapshot(
        tmp_path,
        archive_format,
        entries=_replace_scores_content(source),
    )

    table = _extract(snapshot, [CURRENT_ONE], tmp_path).tables[0]

    expected = (
        b"accession,hgvs_pro,score,sd,se,df,scores.score.extra\n"
        b"urn:mavedb:00000003-a-1#1,p.=,1.25,0.2,0.03,12,unchanged\n"
    )
    assert table.scores_path.read_bytes() == expected
    manifest = json.loads(
        (table.scores_path.parent / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["files"]["scores"]["size"] == len(expected)
    assert manifest["files"]["scores"]["sha256"] == hashlib.sha256(
        expected
    ).hexdigest()


@pytest.mark.parametrize("archive_format", ["zip", "tar.gz"])
def test_canonical_score_headers_are_preserved_byte_for_byte(
    archive_format: str,
    tmp_path: Path,
) -> None:
    source = b"hgvs_pro,score,sd,se,df\np.=,1.25,0.2,0.03,12\n"
    snapshot = _snapshot(
        tmp_path,
        archive_format,
        entries=_replace_scores_content(source),
    )

    table = _extract(snapshot, [CURRENT_ONE], tmp_path).tables[0]

    assert table.scores_path.read_bytes() == source


@pytest.mark.parametrize("archive_format", ["zip", "tar.gz"])
@pytest.mark.parametrize(
    ("source_name", "canonical_name"),
    [
        ("scores.score", "score"),
        ("scores.sd", "sd"),
        ("scores.se", "se"),
        ("scores.df", "df"),
    ],
)
def test_snapshot_score_header_collision_is_rejected(
    archive_format: str,
    source_name: str,
    canonical_name: str,
    tmp_path: Path,
) -> None:
    source = (
        f"hgvs_pro,{source_name},{canonical_name}\np.=,1.25,99\n".encode()
    )
    snapshot = _snapshot(
        tmp_path,
        archive_format,
        entries=_replace_scores_content(source),
    )

    with pytest.raises(MaveDBSnapshotTableError, match="already exists"):
        _extract(snapshot, [CURRENT_ONE], tmp_path)

    assert not (_record_root(tmp_path) / CURRENT_ONE.replace(":", "-")).exists()


@pytest.mark.parametrize("archive_format", ["zip", "tar.gz"])
def test_archive_is_opened_and_scanned_once_for_multiple_datasets(
    archive_format: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _snapshot(tmp_path, archive_format)
    if archive_format == "zip":
        original = tables_module.zipfile.ZipFile
        calls = 0

        def open_zip(*args: Any, **kwargs: Any):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(tables_module.zipfile, "ZipFile", open_zip)
    else:
        original = tables_module.tarfile.open
        calls = 0

        def open_tar(*args: Any, **kwargs: Any):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(tables_module.tarfile, "open", open_tar)

    _extract(snapshot, [CURRENT_ONE, CURRENT_TWO], tmp_path)

    assert calls == 1


@pytest.mark.parametrize("archive_format", ["zip", "tar.gz"])
def test_missing_scores_rejected_while_missing_counts_is_valid(
    archive_format: str,
    tmp_path: Path,
) -> None:
    entries = [
        entry
        for entry in _default_entries()
        if entry[0] != _archive_member(CURRENT_ONE, "scores")
    ]
    snapshot = _snapshot(tmp_path, archive_format, entries=entries)

    with pytest.raises(MaveDBSnapshotTableError, match="missing required scores"):
        _extract(snapshot, [CURRENT_ONE], tmp_path)

    assert not (_record_root(tmp_path) / CURRENT_ONE.replace(":", "-")).exists()


@pytest.mark.parametrize("archive_format", ["zip", "tar.gz"])
@pytest.mark.parametrize("kind", ["scores", "counts"])
@pytest.mark.filterwarnings("ignore:Duplicate name:UserWarning")
def test_duplicate_selected_members_are_rejected(
    archive_format: str,
    kind: str,
    tmp_path: Path,
) -> None:
    name = _archive_member(CURRENT_ONE, kind)
    entries = _default_entries() + [(name, b"duplicate", "file")]
    snapshot = _snapshot(tmp_path, archive_format, entries=entries)

    with pytest.raises(MaveDBSnapshotTableError, match="duplicate member"):
        _extract(snapshot, [CURRENT_ONE], tmp_path)


@pytest.mark.parametrize("archive_format", ["zip", "tar.gz"])
@pytest.mark.parametrize("kind", ["scores", "counts"])
def test_root_and_versioned_valid_members_are_ambiguous(
    archive_format: str,
    kind: str,
    tmp_path: Path,
) -> None:
    name = _archive_member(CURRENT_ONE, kind)
    entries = _default_entries() + [
        (f"another-release/{name}", b"ambiguous", "file")
    ]
    snapshot = _snapshot(tmp_path, archive_format, entries=entries)

    with pytest.raises(MaveDBSnapshotTableError, match="duplicate member"):
        _extract(snapshot, [CURRENT_ONE], tmp_path)


@pytest.mark.parametrize(
    ("archive_format", "special_kind"),
    [
        ("zip", "symlink"),
        ("zip", "directory"),
        ("zip", "special"),
        ("tar.gz", "symlink"),
        ("tar.gz", "hardlink"),
        ("tar.gz", "directory"),
        ("tar.gz", "fifo"),
        ("tar.gz", "device"),
    ],
)
def test_selected_non_regular_members_are_rejected(
    archive_format: str,
    special_kind: str,
    tmp_path: Path,
) -> None:
    expected_name = _archive_member(CURRENT_ONE, "scores")
    entries = [
        (name, content, special_kind if name == expected_name else kind)
        for name, content, kind in _default_entries()
    ]
    snapshot = _snapshot(tmp_path, archive_format, entries=entries)

    with pytest.raises(MaveDBSnapshotTableError, match="regular"):
        _extract(snapshot, [CURRENT_ONE], tmp_path)


def test_encrypted_zip_member_is_rejected() -> None:
    member = zipfile.ZipInfo(_archive_member(CURRENT_ONE, "scores"))
    member.flag_bits |= 0x1

    with pytest.raises(MaveDBSnapshotTableError, match="unencrypted"):
        tables_module._validate_zip_member(member)


@pytest.mark.parametrize(
    "lookalike",
    [
        "/csv/urn-mavedb-00000003-a-1.scores.csv",
        "../csv/urn-mavedb-00000003-a-1.scores.csv",
        "C:/csv/urn-mavedb-00000003-a-1.scores.csv",
        "release/../csv/urn-mavedb-00000003-a-1.scores.csv",
        r"csv\urn-mavedb-00000003-a-1.scores.csv",
        "nested/urn-mavedb-00000003-a-1.scores.csv",
        "one/two/csv/urn-mavedb-00000003-a-1.scores.csv",
        "release/not-csv/urn-mavedb-00000003-a-1.scores.csv",
        "prefix-csv/urn-mavedb-00000003-a-1.scores.csv",
        "csv/URN-mavedb-00000003-a-1.scores.csv",
        "csv/Current one.scores.csv",
        "csv/GENE1.scores.csv",
    ],
)
@pytest.mark.parametrize("archive_format", ["zip", "tar.gz"])
def test_only_exact_member_path_can_satisfy_scores(
    lookalike: str,
    archive_format: str,
    tmp_path: Path,
) -> None:
    expected_name = _archive_member(CURRENT_ONE, "scores")
    entries = [
        (lookalike, content, kind) if name == expected_name else (name, content, kind)
        for name, content, kind in _default_entries()
    ]
    snapshot = _snapshot(tmp_path, archive_format, entries=entries)

    with pytest.raises(MaveDBSnapshotTableError, match="missing required scores"):
        _extract(snapshot, [CURRENT_ONE], tmp_path)


def test_streaming_byte_count_must_match_declared_size(tmp_path: Path) -> None:
    output = tmp_path / "scores.csv"

    with pytest.raises(MaveDBSnapshotTableError, match="size"):
        tables_module._stream_to_staged_file(
            io.BytesIO(b"short"),
            output,
            expected_size=10,
        )

    assert not output.exists()
    assert list(tmp_path.glob(".scores.csv-*")) == []


@pytest.mark.parametrize("archive_format", ["zip", "tar.gz"])
def test_corrupt_archives_are_expected_errors(
    archive_format: str,
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(tmp_path, archive_format)
    snapshot.archive_path.write_bytes(b"not an archive")
    broken = replace(
        snapshot,
        record=replace(
            snapshot.record,
            size=len(b"not an archive"),
            checksum=f"md5:{hashlib.md5(b'not an archive').hexdigest()}",
        ),
    )

    with pytest.raises(MaveDBSnapshotTableError, match="malformed"):
        _extract(broken, [CURRENT_ONE], tmp_path)


def test_extraction_never_uses_broad_archive_extractors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        zipfile.ZipFile,
        "extract",
        _raiser(AssertionError("extract() used")),
    )
    monkeypatch.setattr(
        zipfile.ZipFile,
        "extractall",
        _raiser(AssertionError("extractall() used")),
    )

    result = _extract(_snapshot(tmp_path), [CURRENT_ONE], tmp_path)

    assert result.dataset_count == 1


def test_valid_cache_hit_avoids_opening_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _snapshot(tmp_path)
    first = _extract(snapshot, [CURRENT_ONE], tmp_path)
    monkeypatch.setattr(
        tables_module.zipfile,
        "ZipFile",
        _raiser(AssertionError("Cache hit opened archive")),
    )

    second = _extract(snapshot, [CURRENT_ONE], tmp_path)

    assert first.tables[0].cache_hit is False
    assert second.tables[0].cache_hit is True
    assert second.tables[0].scores_path == first.tables[0].scores_path


def test_partial_cache_hit_extracts_only_miss_and_opens_archive_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _snapshot(tmp_path)
    first = _extract(snapshot, [CURRENT_ONE], tmp_path).tables[0]
    original_scores = first.scores_path.read_bytes()
    opened = 0
    streamed: list[str] = []
    original_zip = tables_module.zipfile.ZipFile
    original_stream = tables_module._stream_zip_member

    def open_zip(*args: Any, **kwargs: Any):
        nonlocal opened
        opened += 1
        return original_zip(*args, **kwargs)

    def stream(*args: Any, **kwargs: Any):
        member = args[1]
        streamed.append(member.filename)
        return original_stream(*args, **kwargs)

    monkeypatch.setattr(tables_module.zipfile, "ZipFile", open_zip)
    monkeypatch.setattr(tables_module, "_stream_zip_member", stream)

    result = _extract(snapshot, [CURRENT_TWO, CURRENT_ONE], tmp_path)

    assert opened == 1
    assert streamed == [_archive_member(CURRENT_TWO, "scores")]
    assert [table.cache_hit for table in result.tables] == [True, False]
    assert first.scores_path.read_bytes() == original_scores


def test_refresh_rebuilds_every_requested_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _snapshot(tmp_path)
    _extract(snapshot, [CURRENT_ONE, CURRENT_TWO], tmp_path)
    opened = 0
    original_zip = tables_module.zipfile.ZipFile

    def open_zip(*args: Any, **kwargs: Any):
        nonlocal opened
        opened += 1
        return original_zip(*args, **kwargs)

    monkeypatch.setattr(tables_module.zipfile, "ZipFile", open_zip)

    result = _extract(
        snapshot,
        [CURRENT_ONE, CURRENT_TWO],
        tmp_path,
        refresh=True,
    )

    assert opened == 1
    assert [table.cache_hit for table in result.tables] == [False, False]


@pytest.mark.parametrize(
    "corruption",
    [
        "record_id",
        "archive_checksum",
        "missing_scores",
        "wrong_size",
        "wrong_sha256",
        "invalid_json",
        "unexpected_file",
    ],
)
def test_invalid_cache_entries_are_rebuilt_without_in_place_repair(
    corruption: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _snapshot(tmp_path)
    table = _extract(snapshot, [CURRENT_ONE], tmp_path).tables[0]
    entry = table.scores_path.parent
    manifest_path = entry / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if corruption == "record_id":
        manifest["record_id"] = "999"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif corruption == "archive_checksum":
        manifest["archive"]["checksum"] = "md5:" + "0" * 32
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif corruption == "missing_scores":
        table.scores_path.unlink()
    elif corruption == "wrong_size":
        manifest["files"]["scores"]["size"] += 1
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif corruption == "wrong_sha256":
        manifest["files"]["scores"]["sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif corruption == "invalid_json":
        manifest_path.write_text("{invalid", encoding="utf-8")
    else:
        (entry / "partial.tmp").write_text("partial", encoding="utf-8")

    opened = 0
    original_zip = tables_module.zipfile.ZipFile

    def open_zip(*args: Any, **kwargs: Any):
        nonlocal opened
        opened += 1
        return original_zip(*args, **kwargs)

    monkeypatch.setattr(tables_module.zipfile, "ZipFile", open_zip)

    rebuilt = _extract(snapshot, [CURRENT_ONE], tmp_path).tables[0]

    assert opened == 1
    assert rebuilt.cache_hit is False
    assert rebuilt.scores_path.read_bytes() == CURRENT_ONE_SCORES
    assert sorted(path.name for path in rebuilt.scores_path.parent.iterdir()) == [
        "counts.csv",
        "manifest.json",
        "scores.csv",
    ]


def test_unexpected_counts_presence_invalidates_counts_absent_entry(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(tmp_path)
    table = _extract(snapshot, [CURRENT_TWO], tmp_path).tables[0]
    assert table.counts_path is None
    (table.scores_path.parent / "counts.csv").write_text(
        "unexpected",
        encoding="utf-8",
    )

    rebuilt = _extract(snapshot, [CURRENT_TWO], tmp_path).tables[0]

    assert rebuilt.cache_hit is False
    assert rebuilt.counts_path is None
    assert not (rebuilt.scores_path.parent / "counts.csv").exists()


def test_changed_snapshot_archive_provenance_invalidates_cache(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(tmp_path)
    _extract(snapshot, [CURRENT_ONE], tmp_path)
    changed = replace(
        snapshot,
        record=replace(snapshot.record, checksum="md5:" + "0" * 32),
    )

    rebuilt = _extract(changed, [CURRENT_ONE], tmp_path).tables[0]

    assert rebuilt.cache_hit is False
    manifest = json.loads(
        (rebuilt.scores_path.parent / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["archive"]["checksum"] == "md5:" + "0" * 32


def test_symlinked_cached_scores_is_not_a_hit(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(tmp_path)
    table = _extract(snapshot, [CURRENT_ONE], tmp_path).tables[0]
    original = table.scores_path.read_bytes()
    target = tmp_path / "outside.csv"
    target.write_bytes(original)
    table.scores_path.unlink()
    try:
        table.scores_path.symlink_to(target)
    except OSError:
        pytest.skip("File symlinks are unavailable in this Windows environment")

    rebuilt = _extract(snapshot, [CURRENT_ONE], tmp_path).tables[0]

    assert rebuilt.cache_hit is False
    assert rebuilt.scores_path.is_symlink() is False
    assert target.read_bytes() == original


def test_failed_rebuild_does_not_modify_invalid_entry_in_place(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _snapshot(tmp_path)
    table = _extract(snapshot, [CURRENT_ONE], tmp_path).tables[0]
    table.scores_path.write_bytes(b"corrupted old entry")
    original_manifest = (table.scores_path.parent / "manifest.json").read_bytes()
    cause = RuntimeError("unexpected stream failure")
    monkeypatch.setattr(
        tables_module,
        "_extract_from_zip",
        _raiser(cause),
    )

    with pytest.raises(RuntimeError) as exc_info:
        _extract(snapshot, [CURRENT_ONE], tmp_path)

    assert exc_info.value is cause
    assert table.scores_path.read_bytes() == b"corrupted old entry"
    assert (table.scores_path.parent / "manifest.json").read_bytes() == original_manifest


def test_missing_member_failure_preserves_valid_hit_and_publishes_no_miss(
    tmp_path: Path,
) -> None:
    entries = [
        entry
        for entry in _default_entries()
        if entry[0] != _archive_member(CURRENT_TWO, "scores")
    ]
    snapshot = _snapshot(tmp_path, entries=entries)
    hit = _extract(snapshot, [CURRENT_ONE], tmp_path).tables[0]
    original = _bundle_bytes(hit.scores_path.parent)

    with pytest.raises(MaveDBSnapshotTableError, match="missing required scores"):
        _extract(snapshot, [CURRENT_ONE, CURRENT_TWO], tmp_path)

    assert _bundle_bytes(hit.scores_path.parent) == original
    assert not (
        hit.scores_path.parent.parent / CURRENT_TWO.replace(":", "-")
    ).exists()
    assert list(hit.scores_path.parent.parent.glob(".extract-*")) == []


def test_manifest_failure_preserves_existing_valid_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _snapshot(tmp_path)
    table = _extract(snapshot, [CURRENT_ONE], tmp_path).tables[0]
    originals = _bundle_bytes(table.scores_path.parent)
    monkeypatch.setattr(
        tables_module,
        "_write_manifest",
        _raiser(OSError("manifest failed")),
    )

    with pytest.raises(OSError, match="manifest failed"):
        _extract(snapshot, [CURRENT_ONE], tmp_path, refresh=True)

    assert _bundle_bytes(table.scores_path.parent) == originals
    assert list(table.scores_path.parent.parent.glob(".extract-*")) == []


def test_publication_failure_rolls_back_new_multiple_bundles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _snapshot(tmp_path)
    cache_root = _record_root(tmp_path)
    failed_final = cache_root / CURRENT_TWO.replace(":", "-")
    original_replace = tables_module.os.replace
    injected = False

    def fail_second(source: str | Path, destination: str | Path) -> None:
        nonlocal injected
        if Path(destination) == failed_final and not injected:
            injected = True
            raise OSError("second publication failed")
        original_replace(source, destination)

    monkeypatch.setattr(tables_module.os, "replace", fail_second)

    with pytest.raises(OSError, match="second publication failed"):
        _extract(snapshot, [CURRENT_ONE, CURRENT_TWO], tmp_path)

    assert injected is True
    assert not (cache_root / CURRENT_ONE.replace(":", "-")).exists()
    assert not failed_final.exists()
    assert list(cache_root.glob(".extract-*")) == []


def test_refresh_publication_failure_restores_previous_bundles_and_unrelated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _snapshot(tmp_path)
    result = _extract(snapshot, [CURRENT_ONE, CURRENT_TWO], tmp_path)
    originals = {
        path: _bundle_bytes(path)
        for path in (table.scores_path.parent for table in result.tables)
    }
    record_root = result.tables[0].scores_path.parent.parent
    unrelated = record_root / "unrelated-entry"
    unrelated.mkdir()
    (unrelated / "keep.txt").write_text("keep", encoding="utf-8")
    failed_final = result.tables[1].scores_path.parent
    original_replace = tables_module.os.replace
    injected = False

    def fail_second(source: str | Path, destination: str | Path) -> None:
        nonlocal injected
        if Path(destination) == failed_final and not injected:
            injected = True
            raise OSError("refresh publication failed")
        original_replace(source, destination)

    monkeypatch.setattr(tables_module.os, "replace", fail_second)

    with pytest.raises(OSError, match="refresh publication failed"):
        _extract(
            snapshot,
            [CURRENT_ONE, CURRENT_TWO],
            tmp_path,
            refresh=True,
        )

    assert injected is True
    for entry, expected_files in originals.items():
        assert _bundle_bytes(entry) == expected_files
    assert (unrelated / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert list(record_root.glob(".extract-*")) == []


@pytest.mark.parametrize(
    ("archive_format", "cause"),
    [
        ("zip", zipfile.BadZipFile("CRC failure")),
        ("tar.gz", tarfile.ReadError("short member")),
    ],
)
def test_archive_read_failures_are_expected_and_publish_nothing(
    archive_format: str,
    cause: Exception,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _snapshot(tmp_path, archive_format)
    target = (
        "_stream_zip_member" if archive_format == "zip" else "_stream_tar_member"
    )
    monkeypatch.setattr(
        tables_module,
        target,
        _raiser(cause),
    )

    with pytest.raises(MaveDBSnapshotTableError, match="malformed"):
        _extract(snapshot, [CURRENT_ONE], tmp_path)

    record_root = _record_root(tmp_path)
    assert not (record_root / CURRENT_ONE.replace(":", "-")).exists()
    assert list(record_root.glob(".extract-*")) == []


def test_unexpected_programming_error_propagates_and_cleans_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _snapshot(tmp_path)
    cause = RuntimeError("unexpected")
    monkeypatch.setattr(
        tables_module,
        "_extract_from_zip",
        _raiser(cause),
    )

    with pytest.raises(RuntimeError) as exc_info:
        _extract(snapshot, [CURRENT_ONE], tmp_path)

    assert exc_info.value is cause
    record_root = _record_root(tmp_path)
    assert list(record_root.glob(".extract-*")) == []
