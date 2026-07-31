from __future__ import annotations

from pathlib import Path

import pytest

import dms_parser.fetch as fetch_module
import dms_parser.sources.mavedb as mavedb_module
import dms_parser.sources.proteingym as proteingym_module
from dms_parser.cache import FilesystemCache
from dms_parser.exceptions import DownloadError, InvalidCacheEntryError


@pytest.fixture
def recorded_staging_directories(tmp_path, monkeypatch):
    """Record source-fetch staging directories under the test directory."""
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


def _install_successful_cached_download(monkeypatch, content: bytes):
    """Install an offline fetch download and return its recorded calls."""
    calls: list[tuple[str, Path]] = []

    def successful_download(
        url: str,
        output_path: str | Path,
        *,
        overwrite: bool = False,
        chunk_size: int = 8192,
        timeout: int = 60,
    ) -> Path:
        del overwrite, chunk_size, timeout
        path = Path(output_path)
        calls.append((url, path))
        path.write_bytes(content)
        return path

    monkeypatch.setattr(fetch_module, "download_file", successful_download)
    return calls


def _assert_staging_removed(recorded_staging_directories) -> None:
    """Assert that every source-fetch staging directory was removed."""
    directories, staging_root = recorded_staging_directories
    assert directories
    assert all(not directory.exists() for directory in directories)
    assert list(staging_root.iterdir()) == []


@pytest.mark.parametrize(
    ("source_module", "download_name"),
    [
        (mavedb_module, "download_mavedb_dataset"),
        (proteingym_module, "download_proteingym_dataset"),
    ],
)
def test_source_helper_without_cache_preserves_download_behavior(
    tmp_path,
    monkeypatch,
    source_module,
    download_name,
):
    calls: list[tuple[str, Path, bool]] = []

    def offline_download(url, output_path, *, overwrite=False):
        path = Path(output_path)
        calls.append((url, path, overwrite))
        return path

    monkeypatch.setattr(source_module, "download_file", offline_download)
    download = getattr(source_module, download_name)

    result = download(
        "https://example.test/data.csv",
        output_dir=tmp_path,
        filename="chosen.csv",
        overwrite=True,
    )

    assert result == tmp_path / "chosen.csv"
    assert calls == [
        (
            "https://example.test/data.csv",
            tmp_path / "chosen.csv",
            True,
        )
    ]


@pytest.mark.parametrize(
    ("source_module", "download_name", "source_name"),
    [
        (mavedb_module, "download_mavedb_dataset", "mavedb"),
        (
            proteingym_module,
            "download_proteingym_dataset",
            "proteingym",
        ),
    ],
)
def test_source_cache_miss_stores_source_metadata(
    tmp_path,
    monkeypatch,
    recorded_staging_directories,
    source_module,
    download_name,
    source_name,
):
    cache = FilesystemCache(tmp_path / "cache")
    calls = _install_successful_cached_download(monkeypatch, b"source content")
    download = getattr(source_module, download_name)
    url = f"https://example.test/{source_name}.csv"

    cached_path = download(
        url,
        cache=cache,
        dataset_id="dataset-1",
    )
    manifest = cache.load_manifest(source_name, "dataset-1")

    assert len(calls) == 1
    assert calls[0][0] == url
    assert cached_path.read_bytes() == b"source content"
    assert cache.resolve(source_name, "dataset-1") == cached_path
    assert manifest.source == source_name
    assert manifest.dataset_id == "dataset-1"
    assert manifest.original_url == url
    _assert_staging_removed(recorded_staging_directories)


@pytest.mark.parametrize(
    ("source_module", "download_name", "source_name"),
    [
        (mavedb_module, "download_mavedb_dataset", "mavedb"),
        (
            proteingym_module,
            "download_proteingym_dataset",
            "proteingym",
        ),
    ],
)
def test_source_cache_hit_performs_no_network_or_staging(
    tmp_path,
    monkeypatch,
    source_module,
    download_name,
    source_name,
):
    cache = FilesystemCache(tmp_path / "cache")
    url = f"https://example.test/{source_name}.csv"
    cached_path = cache.store_bytes(
        source_name,
        "dataset-1",
        url,
        b"cached",
    )

    def unexpected_operation(*args, **kwargs):
        raise AssertionError("A source cache hit must not download or stage.")

    monkeypatch.setattr(fetch_module, "download_file", unexpected_operation)
    monkeypatch.setattr(fetch_module, "TemporaryDirectory", unexpected_operation)
    download = getattr(source_module, download_name)

    result = download(
        url,
        cache=cache,
        dataset_id="dataset-1",
    )

    assert result == cached_path


@pytest.mark.parametrize(
    ("source_module", "download_name", "source_name"),
    [
        (mavedb_module, "download_mavedb_dataset", "mavedb"),
        (
            proteingym_module,
            "download_proteingym_dataset",
            "proteingym",
        ),
    ],
)
def test_source_refresh_downloads_and_replaces_entry(
    tmp_path,
    monkeypatch,
    recorded_staging_directories,
    source_module,
    download_name,
    source_name,
):
    cache = FilesystemCache(tmp_path / "cache")
    url = f"https://example.test/{source_name}.csv"
    original_path = cache.store_bytes(
        source_name,
        "dataset-1",
        url,
        b"original",
    )
    calls = _install_successful_cached_download(monkeypatch, b"replacement")
    download = getattr(source_module, download_name)

    refreshed_path = download(
        url,
        cache=cache,
        dataset_id="dataset-1",
        refresh=True,
    )

    assert len(calls) == 1
    assert refreshed_path.read_bytes() == b"replacement"
    assert cache.resolve(source_name, "dataset-1") == refreshed_path
    assert not original_path.exists()
    _assert_staging_removed(recorded_staging_directories)


@pytest.mark.parametrize(
    ("source_module", "download_name", "source_name"),
    [
        (mavedb_module, "download_mavedb_dataset", "mavedb"),
        (
            proteingym_module,
            "download_proteingym_dataset",
            "proteingym",
        ),
    ],
)
def test_failed_source_refresh_preserves_previous_entry(
    tmp_path,
    monkeypatch,
    recorded_staging_directories,
    source_module,
    download_name,
    source_name,
):
    cache = FilesystemCache(tmp_path / "cache")
    url = f"https://example.test/{source_name}.csv"
    original_path = cache.store_bytes(
        source_name,
        "dataset-1",
        url,
        b"original",
    )
    original_manifest = cache.manifest_path(
        source_name,
        "dataset-1",
    ).read_bytes()

    def failed_download(*args, **kwargs):
        raise DownloadError("download failed") from ConnectionError(
            "connection interrupted"
        )

    monkeypatch.setattr(fetch_module, "download_file", failed_download)
    download = getattr(source_module, download_name)

    with pytest.raises(DownloadError):
        download(
            url,
            cache=cache,
            dataset_id="dataset-1",
            refresh=True,
        )

    assert cache.resolve(source_name, "dataset-1") == original_path
    assert original_path.read_bytes() == b"original"
    assert (
        cache.manifest_path(source_name, "dataset-1").read_bytes() == original_manifest
    )
    _assert_staging_removed(recorded_staging_directories)


@pytest.mark.parametrize(
    ("source_module", "download_name"),
    [
        (mavedb_module, "download_mavedb_dataset"),
        (proteingym_module, "download_proteingym_dataset"),
    ],
)
def test_missing_dataset_id_fails_before_request(
    tmp_path,
    monkeypatch,
    source_module,
    download_name,
):
    cache = FilesystemCache(tmp_path / "cache")

    def unexpected_operation(*args, **kwargs):
        raise AssertionError("A missing dataset ID must fail before downloading.")

    monkeypatch.setattr(fetch_module, "download_file", unexpected_operation)
    monkeypatch.setattr(fetch_module, "TemporaryDirectory", unexpected_operation)
    download = getattr(source_module, download_name)

    with pytest.raises(InvalidCacheEntryError, match="dataset_id is required"):
        download(
            "https://example.test/data.csv",
            cache=cache,
        )

    assert not cache.root.exists()


def test_cache_publication_failure_cleans_staging_and_preserves_entry(
    tmp_path,
    monkeypatch,
    recorded_staging_directories,
):
    cache = FilesystemCache(tmp_path / "cache")
    url = "https://example.test/mavedb.csv"
    original_path = cache.store_bytes(
        "mavedb",
        "dataset-1",
        url,
        b"original",
    )
    original_manifest = cache.manifest_path("mavedb", "dataset-1").read_bytes()
    _install_successful_cached_download(monkeypatch, b"replacement")

    def fail_manifest_publication(*args, **kwargs):
        raise OSError("manifest publication failed")

    monkeypatch.setattr(cache, "_write_manifest", fail_manifest_publication)

    with pytest.raises(OSError, match="manifest publication failed"):
        mavedb_module.download_mavedb_dataset(
            "https://example.test/replacement.csv",
            cache=cache,
            dataset_id="dataset-1",
            refresh=True,
        )

    assert cache.resolve("mavedb", "dataset-1") == original_path
    assert original_path.read_bytes() == b"original"
    assert cache.manifest_path("mavedb", "dataset-1").read_bytes() == original_manifest
    assert list(original_path.parent.glob("artifact-*")) == [original_path]
    _assert_staging_removed(recorded_staging_directories)
