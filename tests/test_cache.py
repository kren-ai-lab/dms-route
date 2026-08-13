from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pytest

from dms_parser.acquisition.cache import CacheManifest, FilesystemCache
from dms_parser.core.exceptions import (
    CorruptCacheManifestError,
    InvalidCacheEntryError,
)


def test_cache_path_is_deterministic(tmp_path):
    cache = FilesystemCache(tmp_path / "cache")

    first = cache.entry_path("source", "dataset-1")
    second = cache.entry_path("source", "dataset-1")

    assert first == second
    assert first.parent.parent == tmp_path / "cache"
    assert not cache.root.exists()


@pytest.mark.parametrize(
    "dataset_id",
    [
        "../outside",
        r"nested\windows:path",
        "query?id=1&format=csv",
        "unicode/数据",
    ],
)
def test_unsafe_dataset_identifiers_stay_within_cache(tmp_path, dataset_id):
    cache = FilesystemCache(tmp_path / "cache")

    entry_path = cache.entry_path("example", dataset_id)

    assert entry_path.parent.parent == cache.root
    assert "/" not in entry_path.name
    assert "\\" not in entry_path.name
    assert ".." not in entry_path.name


def test_different_unsafe_identifiers_have_different_paths(tmp_path):
    cache = FilesystemCache(tmp_path / "cache")

    assert cache.entry_path("source", "a/b") != cache.entry_path("source", "a\\b")


def test_cache_miss_does_not_create_directories(tmp_path):
    cache = FilesystemCache(tmp_path / "cache")

    assert cache.exists("source", "missing") is False
    assert cache.resolve("source", "missing") is None
    assert not cache.root.exists()


def test_valid_cache_hit_resolves_local_path(tmp_path):
    cache = FilesystemCache(tmp_path / "cache")
    content = b"variant,score\nA1V,0.5\n"

    stored_path = cache.store_bytes(
        "example",
        "dataset-1",
        "https://example.test/data.csv",
        content,
    )

    assert cache.exists("example", "dataset-1") is True
    assert cache.resolve("example", "dataset-1") == stored_path
    assert stored_path.read_bytes() == content


def test_manifest_creation_and_loading(tmp_path):
    cache = FilesystemCache(tmp_path / "cache")
    content = b"cached content"
    downloaded_at = datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)

    stored_path = cache.store_bytes(
        "example",
        "dataset-1",
        "https://example.test/archive.csv.gz",
        content,
        downloaded_at=downloaded_at,
    )
    manifest = cache.load_manifest("example", "dataset-1")

    assert manifest == CacheManifest(
        source="example",
        dataset_id="dataset-1",
        original_url="https://example.test/archive.csv.gz",
        downloaded_at="2025-01-02T03:04:05Z",
        file_size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        artifact_filename=stored_path.name,
    )
    assert stored_path.name.endswith(".csv.gz")


def test_checksum_validation_detects_changed_content_of_same_size(tmp_path):
    cache = FilesystemCache(tmp_path / "cache")
    stored_path = cache.store_bytes(
        "example",
        "dataset-1",
        "https://example.test/data.csv",
        b"original",
    )
    stored_path.write_bytes(b"modified")

    with pytest.raises(InvalidCacheEntryError, match="checksum"):
        cache.resolve("example", "dataset-1")


def test_size_validation_detects_truncated_artifact(tmp_path):
    cache = FilesystemCache(tmp_path / "cache")
    stored_path = cache.store_bytes(
        "example",
        "dataset-1",
        "https://example.test/data.csv",
        b"original",
    )
    stored_path.write_bytes(b"short")

    with pytest.raises(InvalidCacheEntryError, match="size"):
        cache.resolve("example", "dataset-1")


def test_store_returns_existing_hit_without_refresh(tmp_path):
    cache = FilesystemCache(tmp_path / "cache")
    original_path = cache.store_bytes(
        "example",
        "dataset-1",
        "https://example.test/data.csv",
        b"first",
    )

    returned_path = cache.store_bytes(
        "example",
        "dataset-1",
        "https://example.test/new.csv",
        b"second",
    )

    assert returned_path == original_path
    assert returned_path.read_bytes() == b"first"
    assert cache.load_manifest("example", "dataset-1").original_url.endswith(
        "/data.csv"
    )


def test_refresh_bypasses_and_replaces_existing_entry(tmp_path):
    cache = FilesystemCache(tmp_path / "cache")
    cache.store_bytes(
        "example",
        "dataset-1",
        "https://example.test/data.csv",
        b"first",
    )

    assert cache.resolve("example", "dataset-1", refresh=True) is None
    refreshed_path = cache.store_bytes(
        "example",
        "dataset-1",
        "https://example.test/new.tsv",
        b"second",
        refresh=True,
    )

    assert refreshed_path.read_bytes() == b"second"
    assert cache.resolve("example", "dataset-1") == refreshed_path
    assert cache.load_manifest("example", "dataset-1").original_url.endswith("/new.tsv")


@pytest.mark.parametrize(
    "manifest_content",
    [
        "{not-json",
        json.dumps({"source": "example"}),
        json.dumps(
            {
                "source": "example",
                "dataset_id": "dataset-1",
                "original_url": "https://example.test/data.csv",
                "downloaded_at": "not-a-timestamp",
                "file_size": 4,
                "sha256": "invalid",
                "artifact_filename": "../artifact.csv",
            }
        ),
    ],
)
def test_malformed_or_incomplete_manifest_raises(tmp_path, manifest_content):
    cache = FilesystemCache(tmp_path / "cache")
    manifest_path = cache.manifest_path("example", "dataset-1")
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(manifest_content, encoding="utf-8")

    with pytest.raises(CorruptCacheManifestError):
        cache.resolve("example", "dataset-1")


def test_manifest_key_must_match_lookup_key(tmp_path):
    cache = FilesystemCache(tmp_path / "cache")
    cache.store_bytes(
        "example",
        "dataset-1",
        "https://example.test/data.csv",
        b"data",
    )
    manifest_path = cache.manifest_path("example", "dataset-1")
    values = json.loads(manifest_path.read_text(encoding="utf-8"))
    values["dataset_id"] = "another-dataset"
    manifest_path.write_text(json.dumps(values), encoding="utf-8")

    with pytest.raises(CorruptCacheManifestError, match="does not match"):
        cache.resolve("example", "dataset-1")


def test_store_file_copies_content(tmp_path):
    source_path = tmp_path / "download.csv"
    source_path.write_bytes(b"a,b\n1,2\n")
    cache = FilesystemCache(tmp_path / "cache")

    stored_path = cache.store_file(
        "example",
        "dataset-1",
        "https://example.test/download.csv",
        source_path,
    )

    assert stored_path != source_path
    assert stored_path.read_bytes() == source_path.read_bytes()


def _fail_manifest_replace(monkeypatch):
    original_replace = __import__("os").replace

    def failing_replace(source, destination):
        if destination.name == "manifest.json":
            raise OSError("manifest publication failed")
        original_replace(source, destination)

    monkeypatch.setattr("dms_parser.acquisition.cache.os.replace", failing_replace)


def _assert_no_abandoned_cache_files(entry_path):
    if not entry_path.exists():
        return
    assert list(entry_path.glob(".artifact-*")) == []
    assert list(entry_path.glob(".manifest-*")) == []


def test_failed_initial_manifest_publication_leaves_no_entry(
    tmp_path,
    monkeypatch,
):
    cache = FilesystemCache(tmp_path / "cache")
    entry_path = cache.entry_path("example", "dataset-1")
    _fail_manifest_replace(monkeypatch)

    with pytest.raises(OSError, match="manifest publication failed"):
        cache.store_bytes(
            "example",
            "dataset-1",
            "https://example.test/data.csv",
            b"content",
        )

    assert cache.resolve("example", "dataset-1") is None
    assert list(entry_path.glob("artifact-*")) == []
    _assert_no_abandoned_cache_files(entry_path)


@pytest.mark.parametrize(
    "refreshed_url",
    [
        "https://example.test/data.csv",
        "https://example.test/replacement.tsv",
    ],
)
def test_failed_cache_refresh_preserves_previous_entry(
    tmp_path,
    monkeypatch,
    refreshed_url,
):
    cache = FilesystemCache(tmp_path / "cache")
    original_path = cache.store_bytes(
        "example",
        "dataset-1",
        "https://example.test/data.csv",
        b"original",
    )
    original_manifest = cache.manifest_path(
        "example",
        "dataset-1",
    ).read_bytes()
    _fail_manifest_replace(monkeypatch)

    with pytest.raises(OSError, match="manifest publication failed"):
        cache.store_bytes(
            "example",
            "dataset-1",
            refreshed_url,
            b"replacement",
            refresh=True,
        )

    assert cache.resolve("example", "dataset-1") == original_path
    assert original_path.read_bytes() == b"original"
    assert cache.manifest_path("example", "dataset-1").read_bytes() == original_manifest
    assert list(original_path.parent.glob("artifact-*")) == [original_path]
    _assert_no_abandoned_cache_files(original_path.parent)


def test_successful_cache_refresh_removes_superseded_artifact(tmp_path):
    cache = FilesystemCache(tmp_path / "cache")
    original_path = cache.store_bytes(
        "example",
        "dataset-1",
        "https://example.test/data.csv",
        b"original",
    )

    refreshed_path = cache.store_bytes(
        "example",
        "dataset-1",
        "https://example.test/replacement.tsv",
        b"replacement",
        refresh=True,
    )

    assert refreshed_path != original_path
    assert not original_path.exists()
    assert cache.resolve("example", "dataset-1") == refreshed_path
    assert list(refreshed_path.parent.glob("artifact-*")) == [refreshed_path]
    _assert_no_abandoned_cache_files(refreshed_path.parent)
