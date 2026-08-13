from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import tarfile
import zipfile
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest
import requests

import dms_parser
import dms_parser.sources as sources_module
import dms_parser.sources.mavedb_snapshots as snapshots_module
from dms_parser import FilesystemCache
from dms_parser.core.exceptions import (
    InvalidSnapshotSelectorError,
    MaveDBSnapshotError,
)
from dms_parser.sources.mavedb_snapshots import (
    MAVEDB_ZENODO_API_URL,
    MAVEDB_ZENODO_CONCEPT_DOI,
    MaveDBSnapshot,
    MaveDBSnapshotRecord,
    fetch_mavedb_snapshot,
    load_cached_mavedb_snapshot,
    resolve_mavedb_snapshot,
)


class FakeResponse:
    """Minimal JSON or streaming response for offline snapshot tests."""

    def __init__(
        self,
        *,
        json_value: object | None = None,
        json_error: Exception | None = None,
        chunks: list[bytes] | None = None,
        stream_error: requests.RequestException | None = None,
    ) -> None:
        self.json_value = json_value
        self.json_error = json_error
        self.chunks = chunks or []
        self.stream_error = stream_error

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        if self.json_error is not None:
            raise self.json_error
        return self.json_value

    def iter_content(self, *, chunk_size: int):
        assert chunk_size > 0
        yield from self.chunks
        if self.stream_error is not None:
            raise self.stream_error


class FakeSession:
    """Ordered HTTP session double recording every request."""

    def __init__(self, *responses: FakeResponse) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append((url, kwargs))
        if not self.responses:
            raise AssertionError(f"Unexpected network request: {url}")
        return self.responses.pop(0)


def _tar_archive(
    entries: list[tuple[str, bytes]],
    *,
    symlink: str | None = None,
) -> bytes:
    """Return a tiny gzipped TAR fixture."""
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, content in entries:
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
        if symlink is not None:
            member = tarfile.TarInfo(symlink)
            member.type = tarfile.SYMTYPE
            member.linkname = "elsewhere.json"
            archive.addfile(member)
    return stream.getvalue()


def _zip_archive(
    entries: list[tuple[str, bytes]],
    *,
    symlink: str | None = None,
) -> bytes:
    """Return a tiny ZIP fixture."""
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, mode="w") as archive:
        for name, content in entries:
            archive.writestr(name, content)
        if symlink is not None:
            member = zipfile.ZipInfo(symlink)
            member.create_system = 3
            member.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(member, b"elsewhere.json")
    return stream.getvalue()


def _zenodo_payload(
    archive: bytes,
    *,
    record_id: str = "20840937",
    filename: str = "mavedb-dump.test.tar.gz",
    checksum: str | None = None,
) -> dict[str, Any]:
    """Return representative official Zenodo record metadata."""
    return {
        "id": int(record_id),
        "conceptrecid": "11201736",
        "doi": f"10.5281/zenodo.{record_id}",
        "conceptdoi": MAVEDB_ZENODO_CONCEPT_DOI,
        "metadata": {"publication_date": "2026-06-24"},
        "files": [
            {
                "key": filename,
                "size": len(archive),
                "checksum": checksum
                or f"md5:{hashlib.md5(archive).hexdigest()}",
                "links": {
                    "self": (
                        f"https://zenodo.org/api/records/{record_id}/files/"
                        f"{filename}/content"
                    )
                },
            }
        ],
    }


def _session_for_archive(
    archive: bytes,
    *,
    payload: dict[str, Any] | None = None,
) -> FakeSession:
    """Return a session serving metadata followed by streamed archive chunks."""
    return FakeSession(
        FakeResponse(json_value=payload or _zenodo_payload(archive)),
        FakeResponse(chunks=[archive[:7], b"", archive[7:]]),
    )


def _fetch_tar(tmp_path: Path) -> tuple[MaveDBSnapshot, FilesystemCache, bytes]:
    """Fetch one valid tiny TAR snapshot into a temporary cache."""
    archive = _tar_archive(
        [("snapshot/main.json", b'{"experimentSets": []}\n')]
    )
    cache = FilesystemCache(tmp_path / "cache")
    result = fetch_mavedb_snapshot(
        "20840937",
        cache=cache,
        session=_session_for_archive(archive),
    )
    return result, cache, archive


def test_resolve_latest_follows_redirects_and_maps_concrete_record() -> None:
    archive = _tar_archive([("main.json", b"{}")])
    session = FakeSession(FakeResponse(json_value=_zenodo_payload(archive)))

    record = resolve_mavedb_snapshot("latest", session=session, timeout=17)

    assert record == MaveDBSnapshotRecord(
        record_id="20840937",
        doi="10.5281/zenodo.20840937",
        concept_doi=MAVEDB_ZENODO_CONCEPT_DOI,
        publication_date="2026-06-24",
        filename="mavedb-dump.test.tar.gz",
        size=len(archive),
        checksum=f"md5:{hashlib.md5(archive).hexdigest()}",
        download_url=(
            "https://zenodo.org/api/records/20840937/files/"
            "mavedb-dump.test.tar.gz/content"
        ),
    )
    assert session.calls == [
        (MAVEDB_ZENODO_API_URL, {"timeout": 17, "allow_redirects": True})
    ]


def test_resolve_fixed_record_uses_concrete_endpoint() -> None:
    archive = _zip_archive([("main.json", b"{}")])
    payload = _zenodo_payload(archive, filename="mavedb-dump.test.zip")
    session = FakeSession(FakeResponse(json_value=payload))

    record = resolve_mavedb_snapshot(20840937, session=session)

    assert record.filename.endswith(".zip")
    assert session.calls[0][0].endswith("/20840937")


@pytest.mark.parametrize(
    "selector",
    ["", " ", True, False, 0, -1, "-1", "1.5", "latest ", object()],
)
def test_invalid_selectors_fail_before_network(selector: object) -> None:
    session = FakeSession()

    with pytest.raises(InvalidSnapshotSelectorError):
        resolve_mavedb_snapshot(selector, session=session)  # type: ignore[arg-type]

    assert session.calls == []


def test_record_must_belong_to_official_mavedb_concept() -> None:
    archive = _tar_archive([("main.json", b"{}")])
    payload = _zenodo_payload(archive)
    payload["conceptrecid"] = "999"

    with pytest.raises(MaveDBSnapshotError, match="official MaveDB concept"):
        resolve_mavedb_snapshot(
            "20840937",
            session=FakeSession(FakeResponse(json_value=payload)),
        )


@pytest.mark.parametrize(
    ("filename", "expected_format"),
    [
        ("mavedb-dump.test.tar.gz", "tar.gz"),
        ("mavedb-dump.test.zip", "zip"),
    ],
)
def test_supported_archive_forms_are_selected(
    filename: str,
    expected_format: str,
) -> None:
    archive = b"archive"
    payload = _zenodo_payload(archive, filename=filename)

    record = snapshots_module._snapshot_record_from_zenodo(
        payload,
        expected_id="20840937",
    )

    assert record.filename == filename
    assert snapshots_module._archive_format(record.filename) == expected_format


def test_current_zenodo_self_download_url_is_accepted() -> None:
    payload = _zenodo_payload(b"archive")

    record = snapshots_module._snapshot_record_from_zenodo(
        payload,
        expected_id="20840937",
    )

    assert record.download_url == (
        "https://zenodo.org/api/records/20840937/files/"
        "mavedb-dump.test.tar.gz/content"
    )


def test_legacy_content_download_url_is_accepted() -> None:
    payload = _zenodo_payload(b"archive")
    legacy_url = "https://zenodo.org/api/records/20840937/files/content"
    payload["files"][0]["links"] = {"content": legacy_url}

    record = snapshots_module._snapshot_record_from_zenodo(
        payload,
        expected_id="20840937",
    )

    assert record.download_url == legacy_url


@pytest.mark.parametrize(
    "invalid_content",
    ["", "relative/path", "ftp://example.test/archive"],
)
def test_valid_self_url_is_used_when_content_is_invalid(
    invalid_content: str,
) -> None:
    payload = _zenodo_payload(b"archive")
    self_url = payload["files"][0]["links"]["self"]
    payload["files"][0]["links"]["content"] = invalid_content

    record = snapshots_module._snapshot_record_from_zenodo(
        payload,
        expected_id="20840937",
    )

    assert record.download_url == self_url


def test_valid_content_url_is_preferred_over_self() -> None:
    payload = _zenodo_payload(b"archive")
    preferred_url = "https://zenodo.org/api/records/20840937/files/content"
    payload["files"][0]["links"]["content"] = preferred_url

    record = snapshots_module._snapshot_record_from_zenodo(
        payload,
        expected_id="20840937",
    )

    assert record.download_url == preferred_url


@pytest.mark.parametrize(
    "links",
    [
        {},
        {"self": ""},
        {"self": "relative/path"},
        {"self": "ftp://example.test/archive"},
        {"content": "relative/path", "self": "file:///tmp/archive"},
    ],
)
def test_invalid_or_missing_archive_download_urls_are_rejected(
    links: dict[str, str],
) -> None:
    payload = _zenodo_payload(b"archive")
    payload["files"][0]["links"] = links

    with pytest.raises(MaveDBSnapshotError, match="download URL"):
        snapshots_module._snapshot_record_from_zenodo(
            payload,
            expected_id="20840937",
        )


def test_unrelated_files_do_not_change_unique_archive_selection() -> None:
    archive = b"archive"
    payload = _zenodo_payload(archive)
    payload["files"].append(
        {
            "key": "README.txt",
            "size": 3,
            "checksum": "md5:900150983cd24fb0d6963f7d28e17f72",
            "links": {"content": "https://zenodo.org/readme"},
        }
    )

    record = snapshots_module._snapshot_record_from_zenodo(
        payload,
        expected_id=None,
    )

    assert record.filename == "mavedb-dump.test.tar.gz"


@pytest.mark.parametrize("files", [[], None, {}])
def test_missing_or_empty_archive_list_is_rejected(files: object) -> None:
    payload = _zenodo_payload(b"archive")
    payload["files"] = files

    with pytest.raises(MaveDBSnapshotError):
        snapshots_module._snapshot_record_from_zenodo(
            payload,
            expected_id=None,
        )


def test_multiple_supported_archives_are_rejected() -> None:
    payload = _zenodo_payload(b"archive")
    payload["files"].append(
        {
            "key": "mavedb-dump.second.zip",
            "size": 1,
            "checksum": "md5:0cc175b9c0f1b6a831c399e269772661",
            "links": {"content": "https://zenodo.org/second"},
        }
    )

    with pytest.raises(MaveDBSnapshotError, match="multiple"):
        snapshots_module._snapshot_record_from_zenodo(
            payload,
            expected_id=None,
        )


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"id": 20840937},
    ],
)
def test_malformed_zenodo_objects_are_rejected(payload: object) -> None:
    with pytest.raises(MaveDBSnapshotError):
        snapshots_module._snapshot_record_from_zenodo(
            payload,
            expected_id=None,
        )


def test_invalid_zenodo_json_is_wrapped() -> None:
    session = FakeSession(FakeResponse(json_error=ValueError("invalid JSON")))

    with pytest.raises(MaveDBSnapshotError, match="valid JSON"):
        resolve_mavedb_snapshot("latest", session=session)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("size", 0),
        ("size", True),
        ("checksum", "missing-colon"),
        ("checksum", "md5:short"),
        ("checksum", "unsupported:0011"),
    ],
)
def test_invalid_archive_metadata_is_rejected(field: str, value: object) -> None:
    payload = _zenodo_payload(b"archive")
    file_metadata = payload["files"][0]
    file_metadata[field] = value

    with pytest.raises(MaveDBSnapshotError):
        snapshots_module._snapshot_record_from_zenodo(
            payload,
            expected_id=None,
        )


def test_fetch_streams_verifies_and_publishes_deterministic_snapshot(
    tmp_path: Path,
) -> None:
    archive = _tar_archive(
        [
            ("nested/main.json", b'{"experimentSets": []}\n'),
            ("nested/scores.csv", b"score\n1\n"),
        ]
    )
    session = _session_for_archive(archive)
    cache = FilesystemCache(tmp_path / "cache")

    result = fetch_mavedb_snapshot(
        "latest",
        cache=cache,
        session=session,
        chunk_size=7,
    )

    expected_root = cache.root / "mavedb" / "snapshots" / "20840937"
    assert result.archive_path == expected_root / "mavedb-dump.test.tar.gz"
    assert result.main_json_path == expected_root / "main.json"
    assert result.cache_hit is False
    assert result.archive_path.read_bytes() == archive
    assert result.main_json_path.read_bytes() == b'{"experimentSets": []}\n'
    assert not (expected_root / "scores.csv").exists()
    assert session.calls[1] == (
        "https://zenodo.org/api/records/20840937/files/"
        "mavedb-dump.test.tar.gz/content",
        {"stream": True, "timeout": 60},
    )
    metadata = json.loads(
        (expected_root / "snapshot.json").read_text(encoding="utf-8")
    )
    assert metadata == {
        "archive_filename": "mavedb-dump.test.tar.gz",
        "archive_format": "tar.gz",
        "archive_size": len(archive),
        "checksum": f"md5:{hashlib.md5(archive).hexdigest()}",
        "concept_doi": MAVEDB_ZENODO_CONCEPT_DOI,
        "doi": "10.5281/zenodo.20840937",
        "download_url": (
            "https://zenodo.org/api/records/20840937/files/"
            "mavedb-dump.test.tar.gz/content"
        ),
        "main_json_sha256": hashlib.sha256(
            b'{"experimentSets": []}\n'
        ).hexdigest(),
        "main_json_size": len(b'{"experimentSets": []}\n'),
        "publication_date": "2026-06-24",
        "record_id": "20840937",
    }
    assert "downloaded_at" not in metadata
    assert list(expected_root.parent.glob(".20840937-*")) == []


@pytest.mark.parametrize("archive_format", ["tar.gz", "zip"])
def test_fetch_extracts_main_json_from_supported_archives(
    archive_format: str,
    tmp_path: Path,
) -> None:
    content = b'{"format": "ok"}'
    if archive_format == "tar.gz":
        archive = _tar_archive([("version/main.json", content)])
    else:
        archive = _zip_archive([("version/main.json", content)])
    filename = f"mavedb-dump.test.{archive_format}"
    payload = _zenodo_payload(archive, filename=filename)

    result = fetch_mavedb_snapshot(
        "20840937",
        cache=FilesystemCache(tmp_path / "cache"),
        session=_session_for_archive(archive, payload=payload),
    )

    assert result.main_json_path.read_bytes() == content


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("size", lambda value: value + 1),
        ("checksum", lambda value: "md5:" + "0" * 32),
    ],
)
def test_failed_verification_publishes_no_snapshot(
    field: str,
    replacement: Any,
    tmp_path: Path,
) -> None:
    archive = _tar_archive([("main.json", b"{}")])
    payload = _zenodo_payload(archive)
    file_metadata = payload["files"][0]
    file_metadata[field] = replacement(file_metadata[field])
    cache = FilesystemCache(tmp_path / "cache")

    with pytest.raises(MaveDBSnapshotError, match=field):
        fetch_mavedb_snapshot(
            "20840937",
            cache=cache,
            session=_session_for_archive(archive, payload=payload),
        )

    assert not (cache.root / "mavedb" / "snapshots" / "20840937").exists()
    assert list((cache.root / "mavedb" / "snapshots").glob(".20840937-*")) == []


def test_interrupted_download_leaves_no_partial_snapshot(tmp_path: Path) -> None:
    archive = _tar_archive([("main.json", b"{}")])
    session = FakeSession(
        FakeResponse(json_value=_zenodo_payload(archive)),
        FakeResponse(
            chunks=[archive[:5]],
            stream_error=requests.ConnectionError("interrupted"),
        ),
    )
    cache = FilesystemCache(tmp_path / "cache")

    with pytest.raises(MaveDBSnapshotError, match="Failed to download"):
        fetch_mavedb_snapshot("20840937", cache=cache, session=session)

    root = cache.root / "mavedb" / "snapshots"
    assert not (root / "20840937").exists()
    assert list(root.glob(".20840937-*")) == []


def test_pinned_valid_cache_is_reused_without_network(tmp_path: Path) -> None:
    first, cache, _ = _fetch_tar(tmp_path)
    session = FakeSession()

    second = fetch_mavedb_snapshot(
        20840937,
        cache=cache,
        session=session,
    )

    assert second.record == first.record
    assert second.archive_path == first.archive_path
    assert second.main_json_path == first.main_json_path
    assert second.cache_hit is True
    assert session.calls == []


def test_cache_only_snapshot_load_never_resolves_or_fetches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, cache, _ = _fetch_tar(tmp_path)
    monkeypatch.setattr(
        snapshots_module,
        "resolve_mavedb_snapshot",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("cache-only load resolved Zenodo")
        ),
    )
    monkeypatch.setattr(
        snapshots_module,
        "fetch_mavedb_snapshot",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("cache-only load fetched a snapshot")
        ),
    )

    loaded = load_cached_mavedb_snapshot(20840937, cache=cache)

    assert loaded.record == first.record
    assert loaded.archive_path == first.archive_path
    assert loaded.main_json_path == first.main_json_path
    assert loaded.cache_hit is True


def test_cache_only_snapshot_requires_cached_concrete_record(
    tmp_path: Path,
) -> None:
    cache = FilesystemCache(tmp_path / "cache")
    with pytest.raises(InvalidSnapshotSelectorError, match="concrete positive"):
        load_cached_mavedb_snapshot("latest", cache=cache)
    with pytest.raises(MaveDBSnapshotError, match="snapshot fetch --record 99"):
        load_cached_mavedb_snapshot("99", cache=cache)


@pytest.mark.parametrize("corrupt", ["metadata", "archive", "main"])
def test_cache_only_snapshot_rejects_corrupt_cached_files(
    corrupt: str,
    tmp_path: Path,
) -> None:
    snapshot, cache, _ = _fetch_tar(tmp_path)
    paths = {
        "metadata": snapshot.archive_path.parent / "snapshot.json",
        "archive": snapshot.archive_path,
        "main": snapshot.main_json_path,
    }
    paths[corrupt].write_bytes(b"corrupt")

    with pytest.raises(MaveDBSnapshotError):
        load_cached_mavedb_snapshot("20840937", cache=cache)


def test_latest_resolves_concrete_id_before_reusing_cache(tmp_path: Path) -> None:
    first, cache, archive = _fetch_tar(tmp_path)
    session = FakeSession(FakeResponse(json_value=_zenodo_payload(archive)))

    second = fetch_mavedb_snapshot("latest", cache=cache, session=session)

    assert second.record.record_id == first.record.record_id
    assert second.cache_hit is True
    assert len(session.calls) == 1
    assert session.calls[0][0] == MAVEDB_ZENODO_API_URL


def test_failed_refresh_preserves_valid_cached_snapshot(tmp_path: Path) -> None:
    first, cache, archive = _fetch_tar(tmp_path)
    original_archive = first.archive_path.read_bytes()
    original_main = first.main_json_path.read_bytes()
    original_metadata = (first.archive_path.parent / "snapshot.json").read_bytes()
    failed_session = FakeSession(
        FakeResponse(json_value=_zenodo_payload(archive)),
        FakeResponse(
            chunks=[archive[:4]],
            stream_error=requests.ConnectionError("interrupted"),
        ),
    )

    with pytest.raises(MaveDBSnapshotError):
        fetch_mavedb_snapshot(
            "20840937",
            cache=cache,
            refresh=True,
            session=failed_session,
        )

    assert first.archive_path.read_bytes() == original_archive
    assert first.main_json_path.read_bytes() == original_main
    assert (first.archive_path.parent / "snapshot.json").read_bytes() == original_metadata


def test_snapshot_directory_publication_is_atomic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _tar_archive([("main.json", b"{}")])
    cache = FilesystemCache(tmp_path / "cache")
    final_entry = cache.root / "mavedb" / "snapshots" / "20840937"
    original_replace = snapshots_module.os.replace

    def fail_final_publication(source: Path, destination: Path) -> None:
        if Path(destination) == final_entry:
            raise OSError("publication failed")
        original_replace(source, destination)

    monkeypatch.setattr(snapshots_module.os, "replace", fail_final_publication)

    with pytest.raises(OSError, match="publication failed"):
        fetch_mavedb_snapshot(
            "20840937",
            cache=cache,
            session=_session_for_archive(archive),
        )

    assert not final_entry.exists()
    assert list(final_entry.parent.glob(".20840937-*")) == []


@pytest.mark.parametrize("archive_format", ["tar.gz", "zip"])
def test_missing_main_json_is_rejected(
    archive_format: str,
    tmp_path: Path,
) -> None:
    archive = (
        _tar_archive([("scores.csv", b"score")])
        if archive_format == "tar.gz"
        else _zip_archive([("scores.csv", b"score")])
    )
    payload = _zenodo_payload(
        archive,
        filename=f"mavedb-dump.test.{archive_format}",
    )

    with pytest.raises(MaveDBSnapshotError, match="missing main.json"):
        fetch_mavedb_snapshot(
            "20840937",
            cache=FilesystemCache(tmp_path / "cache"),
            session=_session_for_archive(archive, payload=payload),
        )


@pytest.mark.parametrize("archive_format", ["tar.gz", "zip"])
def test_duplicate_main_json_is_rejected(
    archive_format: str,
    tmp_path: Path,
) -> None:
    entries = [("one/main.json", b"{}"), ("two/main.json", b"{}")]
    archive = (
        _tar_archive(entries)
        if archive_format == "tar.gz"
        else _zip_archive(entries)
    )
    payload = _zenodo_payload(
        archive,
        filename=f"mavedb-dump.test.{archive_format}",
    )

    with pytest.raises(MaveDBSnapshotError, match="multiple main.json"):
        fetch_mavedb_snapshot(
            "20840937",
            cache=FilesystemCache(tmp_path / "cache"),
            session=_session_for_archive(archive, payload=payload),
        )


@pytest.mark.parametrize("archive_format", ["tar.gz", "zip"])
def test_symlink_main_json_is_rejected(
    archive_format: str,
    tmp_path: Path,
) -> None:
    archive = (
        _tar_archive([], symlink="main.json")
        if archive_format == "tar.gz"
        else _zip_archive([], symlink="main.json")
    )
    payload = _zenodo_payload(
        archive,
        filename=f"mavedb-dump.test.{archive_format}",
    )

    with pytest.raises(MaveDBSnapshotError, match="regular"):
        fetch_mavedb_snapshot(
            "20840937",
            cache=FilesystemCache(tmp_path / "cache"),
            session=_session_for_archive(archive, payload=payload),
        )


@pytest.mark.parametrize("member_name", ["../main.json", "C:/main.json"])
def test_traversing_main_json_member_is_rejected(
    member_name: str,
    tmp_path: Path,
) -> None:
    archive = _tar_archive([(member_name, b"{}")])

    with pytest.raises(MaveDBSnapshotError, match="unsafe"):
        fetch_mavedb_snapshot(
            "20840937",
            cache=FilesystemCache(tmp_path / "cache"),
            session=_session_for_archive(archive),
        )


def test_public_snapshot_exports_are_identical_and_records_are_frozen(
    tmp_path: Path,
) -> None:
    names = (
        "MaveDBSnapshot",
        "MaveDBSnapshotRecord",
        "fetch_mavedb_snapshot",
        "load_cached_mavedb_snapshot",
        "resolve_mavedb_snapshot",
    )
    for name in names:
        assert getattr(dms_parser, name) is getattr(sources_module, name)
        assert getattr(dms_parser, name) is getattr(snapshots_module, name)

    record = MaveDBSnapshotRecord(
        record_id="1",
        doi="10.5281/zenodo.1",
        concept_doi=MAVEDB_ZENODO_CONCEPT_DOI,
        publication_date=None,
        filename="mavedb-dump.test.zip",
        size=1,
        checksum="md5:0cc175b9c0f1b6a831c399e269772661",
        download_url="https://zenodo.org/content",
    )
    snapshot = MaveDBSnapshot(
        record=record,
        archive_path=tmp_path / "archive.zip",
        main_json_path=tmp_path / "main.json",
        cache_hit=False,
    )
    with pytest.raises(FrozenInstanceError):
        record.record_id = "2"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        snapshot.cache_hit = True  # type: ignore[misc]
