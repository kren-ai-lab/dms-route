from __future__ import annotations

from pathlib import Path

import pytest

import dmsroute.acquisition.fetch as fetch_module
from dmsroute import fetch_to_cache as public_fetch_to_cache
from dmsroute.acquisition.cache import FilesystemCache
from dmsroute.core.exceptions import DownloadError, InvalidCacheEntryError
from dmsroute.acquisition.fetch import fetch_to_cache


def test_fetch_to_cache_is_exported_from_package():
    assert public_fetch_to_cache is fetch_to_cache


@pytest.fixture
def recorded_staging_directories(tmp_path, monkeypatch):
    """Record fetch staging directories and place them under the test root."""
    staging_root = tmp_path / "staging"
    staging_root.mkdir()
    created_directories: list[Path] = []
    temporary_directory = fetch_module.TemporaryDirectory

    def create_temporary_directory(*, prefix: str):
        directory = temporary_directory(prefix=prefix, dir=staging_root)
        created_directories.append(Path(directory.name))
        return directory

    monkeypatch.setattr(
        fetch_module,
        "TemporaryDirectory",
        create_temporary_directory,
    )
    return created_directories, staging_root


def _install_successful_download(monkeypatch, content: bytes):
    """Install an offline download double and return its recorded calls."""
    calls: list[dict[str, object]] = []

    def successful_download(
        url: str,
        output_path: str | Path,
        *,
        overwrite: bool = False,
        chunk_size: int = 8192,
        timeout: int = 60,
    ) -> Path:
        path = Path(output_path)
        calls.append(
            {
                "url": url,
                "path": path,
                "overwrite": overwrite,
                "chunk_size": chunk_size,
                "timeout": timeout,
            }
        )
        path.write_bytes(content)
        return path

    monkeypatch.setattr(fetch_module, "download_file", successful_download)
    return calls


def _assert_staging_removed(recorded_staging_directories) -> None:
    """Assert that all recorded staging resources were removed."""
    directories, staging_root = recorded_staging_directories
    assert directories
    assert all(not directory.exists() for directory in directories)
    assert list(staging_root.iterdir()) == []


def test_cache_hit_performs_no_download_or_staging(tmp_path, monkeypatch):
    cache = FilesystemCache(tmp_path / "cache")
    cached_path = cache.store_bytes(
        "example",
        "dataset-1",
        "https://example.test/data.csv",
        b"cached",
    )

    def unexpected_operation(*args, **kwargs):
        raise AssertionError("A cache hit must not create staging or download.")

    monkeypatch.setattr(fetch_module, "TemporaryDirectory", unexpected_operation)
    monkeypatch.setattr(fetch_module, "download_file", unexpected_operation)

    result = fetch_to_cache(
        "https://example.test/data.csv",
        source="example",
        dataset_id="dataset-1",
        cache=cache,
    )

    assert result == cached_path


def test_cache_miss_downloads_once_and_stores_manifest(
    tmp_path,
    monkeypatch,
    recorded_staging_directories,
):
    cache = FilesystemCache(tmp_path / "cache")
    calls = _install_successful_download(monkeypatch, b"downloaded content")

    cached_path = fetch_to_cache(
        "https://example.test/data.csv",
        source="example",
        dataset_id="dataset-1",
        cache=cache,
        chunk_size=4096,
        timeout=12,
    )
    manifest = cache.load_manifest("example", "dataset-1")

    assert len(calls) == 1
    assert calls[0]["url"] == "https://example.test/data.csv"
    assert calls[0]["chunk_size"] == 4096
    assert calls[0]["timeout"] == 12
    assert cached_path.read_bytes() == b"downloaded content"
    assert cache.resolve("example", "dataset-1") == cached_path
    assert manifest.source == "example"
    assert manifest.dataset_id == "dataset-1"
    assert manifest.original_url == "https://example.test/data.csv"
    assert manifest.file_size == len(b"downloaded content")
    _assert_staging_removed(recorded_staging_directories)


def test_refresh_downloads_once_and_replaces_previous_entry(
    tmp_path,
    monkeypatch,
    recorded_staging_directories,
):
    cache = FilesystemCache(tmp_path / "cache")
    original_path = cache.store_bytes(
        "example",
        "dataset-1",
        "https://example.test/data.csv",
        b"original",
    )
    calls = _install_successful_download(monkeypatch, b"replacement")

    refreshed_path = fetch_to_cache(
        "https://example.test/data.csv",
        source="example",
        dataset_id="dataset-1",
        cache=cache,
        refresh=True,
    )

    assert len(calls) == 1
    assert refreshed_path != original_path
    assert refreshed_path.read_bytes() == b"replacement"
    assert not original_path.exists()
    assert cache.resolve("example", "dataset-1") == refreshed_path
    _assert_staging_removed(recorded_staging_directories)


def test_failed_initial_download_leaves_no_cache_entry(
    tmp_path,
    monkeypatch,
    recorded_staging_directories,
):
    cache = FilesystemCache(tmp_path / "cache")
    cause = ConnectionError("connection interrupted")

    def failed_download(*args, **kwargs):
        raise DownloadError("download failed") from cause

    monkeypatch.setattr(fetch_module, "download_file", failed_download)

    with pytest.raises(DownloadError) as exc_info:
        fetch_to_cache(
            "https://example.test/data.csv",
            source="example",
            dataset_id="dataset-1",
            cache=cache,
        )

    assert exc_info.value.__cause__ is cause
    assert cache.resolve("example", "dataset-1") is None
    _assert_staging_removed(recorded_staging_directories)


def test_failed_refresh_download_preserves_previous_entry(
    tmp_path,
    monkeypatch,
    recorded_staging_directories,
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

    def failed_download(*args, **kwargs):
        raise DownloadError("download failed") from ConnectionError(
            "connection interrupted"
        )

    monkeypatch.setattr(fetch_module, "download_file", failed_download)

    with pytest.raises(DownloadError):
        fetch_to_cache(
            "https://example.test/data.csv",
            source="example",
            dataset_id="dataset-1",
            cache=cache,
            refresh=True,
        )

    assert cache.resolve("example", "dataset-1") == original_path
    assert original_path.read_bytes() == b"original"
    assert cache.manifest_path("example", "dataset-1").read_bytes() == original_manifest
    _assert_staging_removed(recorded_staging_directories)


def test_cache_publication_failure_preserves_previous_entry(
    tmp_path,
    monkeypatch,
    recorded_staging_directories,
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
    calls = _install_successful_download(monkeypatch, b"replacement")

    def fail_manifest_publication(*args, **kwargs):
        raise OSError("manifest publication failed")

    monkeypatch.setattr(cache, "_write_manifest", fail_manifest_publication)

    with pytest.raises(OSError, match="manifest publication failed"):
        fetch_to_cache(
            "https://example.test/replacement.csv",
            source="example",
            dataset_id="dataset-1",
            cache=cache,
            refresh=True,
        )

    assert len(calls) == 1
    assert cache.resolve("example", "dataset-1") == original_path
    assert original_path.read_bytes() == b"original"
    assert cache.manifest_path("example", "dataset-1").read_bytes() == original_manifest
    assert list(original_path.parent.glob("artifact-*")) == [original_path]
    _assert_staging_removed(recorded_staging_directories)


@pytest.mark.parametrize(
    ("source", "dataset_id"),
    [
        ("", "dataset-1"),
        ("example", ""),
        (None, "dataset-1"),
        ("example", None),
    ],
)
def test_invalid_cache_key_fails_before_network_or_staging(
    tmp_path,
    monkeypatch,
    source,
    dataset_id,
):
    cache = FilesystemCache(tmp_path / "cache")

    def unexpected_operation(*args, **kwargs):
        raise AssertionError("Invalid cache keys must fail before network access.")

    monkeypatch.setattr(fetch_module, "TemporaryDirectory", unexpected_operation)
    monkeypatch.setattr(fetch_module, "download_file", unexpected_operation)

    with pytest.raises(InvalidCacheEntryError):
        fetch_to_cache(
            "https://example.test/data.csv",
            source=source,
            dataset_id=dataset_id,
            cache=cache,
        )

    assert not cache.root.exists()
