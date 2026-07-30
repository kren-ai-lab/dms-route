from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import requests

from dms_parser.exceptions import DownloadError, FileFormatError
from dms_parser.io import (
    download_file,
    ensure_local_copy,
    infer_filename_from_url,
    read_table,
    write_table,
)


class FakeResponse:
    """Small context-managed response double for offline download tests."""

    def __init__(
        self,
        chunks: list[bytes],
        *,
        request_error: requests.RequestException | None = None,
        stream_error: requests.RequestException | None = None,
    ) -> None:
        self.chunks = chunks
        self.request_error = request_error
        self.stream_error = stream_error

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args) -> None:
        return None

    def raise_for_status(self) -> None:
        if self.request_error is not None:
            raise self.request_error

    def iter_content(self, *, chunk_size: int):
        del chunk_size
        for chunk in self.chunks:
            yield chunk
        if self.stream_error is not None:
            raise self.stream_error


def _temporary_downloads(directory: Path) -> list[Path]:
    """Return temporary files created by the download implementation."""
    return list(directory.glob(".download-*"))


def test_download_file_publishes_complete_content_atomically(tmp_path, monkeypatch):
    output_path = tmp_path / "data.csv"
    response = FakeResponse([b"a,b\n", b"1,2\n"])
    requested: list[tuple[str, bool, int]] = []

    def fake_get(url: str, *, stream: bool, timeout: int) -> FakeResponse:
        requested.append((url, stream, timeout))
        assert not output_path.exists()
        return response

    monkeypatch.setattr("dms_parser.io.requests.get", fake_get)

    result = download_file(
        "https://example.test/data.csv",
        output_path,
        chunk_size=2,
        timeout=15,
    )

    assert result == output_path
    assert output_path.read_bytes() == b"a,b\n1,2\n"
    assert requested == [("https://example.test/data.csv", True, 15)]
    assert _temporary_downloads(tmp_path) == []


def test_download_file_existing_destination_skips_request(tmp_path, monkeypatch):
    output_path = tmp_path / "data.csv"
    output_path.write_bytes(b"existing")

    def unexpected_request(*args, **kwargs):
        raise AssertionError("A cache hit must not perform a network request.")

    monkeypatch.setattr("dms_parser.io.requests.get", unexpected_request)

    assert download_file("https://example.test/data.csv", output_path) == output_path
    assert output_path.read_bytes() == b"existing"
    assert _temporary_downloads(tmp_path) == []


def test_failed_new_download_leaves_no_final_or_temporary_file(
    tmp_path,
    monkeypatch,
):
    output_path = tmp_path / "data.csv"
    error = requests.HTTPError("server error")
    monkeypatch.setattr(
        "dms_parser.io.requests.get",
        lambda *args, **kwargs: FakeResponse([], request_error=error),
    )

    with pytest.raises(DownloadError) as exc_info:
        download_file("https://example.test/data.csv", output_path)

    assert exc_info.value.__cause__ is error
    assert not output_path.exists()
    assert _temporary_downloads(tmp_path) == []


def test_interrupted_stream_leaves_no_partial_final_file(tmp_path, monkeypatch):
    output_path = tmp_path / "data.csv"
    error = requests.ConnectionError("stream interrupted")
    monkeypatch.setattr(
        "dms_parser.io.requests.get",
        lambda *args, **kwargs: FakeResponse(
            [b"partial"],
            stream_error=error,
        ),
    )

    with pytest.raises(DownloadError) as exc_info:
        download_file("https://example.test/data.csv", output_path)

    assert exc_info.value.__cause__ is error
    assert not output_path.exists()
    assert _temporary_downloads(tmp_path) == []


def test_failed_overwrite_preserves_previous_file(tmp_path, monkeypatch):
    output_path = tmp_path / "data.csv"
    output_path.write_bytes(b"previous")
    error = requests.ConnectionError("stream interrupted")
    monkeypatch.setattr(
        "dms_parser.io.requests.get",
        lambda *args, **kwargs: FakeResponse(
            [b"partial replacement"],
            stream_error=error,
        ),
    )

    with pytest.raises(DownloadError):
        download_file(
            "https://example.test/data.csv",
            output_path,
            overwrite=True,
        )

    assert output_path.read_bytes() == b"previous"
    assert _temporary_downloads(tmp_path) == []


def test_successful_overwrite_replaces_previous_file(tmp_path, monkeypatch):
    output_path = tmp_path / "data.csv"
    output_path.write_bytes(b"previous")
    monkeypatch.setattr(
        "dms_parser.io.requests.get",
        lambda *args, **kwargs: FakeResponse([b"replacement"]),
    )

    result = download_file(
        "https://example.test/data.csv",
        output_path,
        overwrite=True,
    )

    assert result == output_path
    assert output_path.read_bytes() == b"replacement"
    assert _temporary_downloads(tmp_path) == []


def test_download_filesystem_failure_raises_download_error(tmp_path, monkeypatch):
    output_path = tmp_path / "data.csv"
    error = PermissionError("cannot publish")
    monkeypatch.setattr(
        "dms_parser.io.requests.get",
        lambda *args, **kwargs: FakeResponse([b"content"]),
    )
    monkeypatch.setattr(
        "dms_parser.io.os.replace",
        lambda *args, **kwargs: (_ for _ in ()).throw(error),
    )

    with pytest.raises(DownloadError) as exc_info:
        download_file("https://example.test/data.csv", output_path)

    assert exc_info.value.__cause__ is error
    assert not output_path.exists()
    assert _temporary_downloads(tmp_path) == []


def test_infer_filename_from_url():
    name = infer_filename_from_url("https://example.com/data.csv?download=1")
    assert "data.csv" in name


def test_read_and_write_csv(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    path = tmp_path / "test.csv"

    write_table(df, path)
    loaded = read_table(path)

    pd.testing.assert_frame_equal(df, loaded)


def test_read_and_write_tsv(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    path = tmp_path / "test.tsv"

    write_table(df, path)
    loaded = read_table(path)

    pd.testing.assert_frame_equal(df, loaded)


def test_read_and_write_parquet(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    path = tmp_path / "test.parquet"

    write_table(df, path)
    loaded = read_table(path)

    pd.testing.assert_frame_equal(df, loaded)


def test_read_table_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_table(tmp_path / "missing.csv")


def test_read_table_unsupported_format_raises(tmp_path):
    path = tmp_path / "bad.xlsx"
    path.write_text("dummy", encoding="utf-8")

    with pytest.raises(FileFormatError):
        read_table(path)


def test_write_table_unsupported_format_raises(tmp_path):
    df = pd.DataFrame({"a": [1]})

    with pytest.raises(FileFormatError):
        write_table(df, tmp_path / "bad.xlsx")


def test_ensure_local_copy_with_existing_file(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")

    result = ensure_local_copy(path, output_dir=tmp_path)
    assert result == path
