from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import requests

import dmsroute.acquisition.io as io_module
from dmsroute.core.exceptions import DownloadError, FileFormatError
from dmsroute.acquisition.io import (
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

    monkeypatch.setattr("dmsroute.acquisition.io.requests.get", fake_get)

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

    monkeypatch.setattr("dmsroute.acquisition.io.requests.get", unexpected_request)

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
        "dmsroute.acquisition.io.requests.get",
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
        "dmsroute.acquisition.io.requests.get",
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
        "dmsroute.acquisition.io.requests.get",
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
        "dmsroute.acquisition.io.requests.get",
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
        "dmsroute.acquisition.io.requests.get",
        lambda *args, **kwargs: FakeResponse([b"content"]),
    )
    monkeypatch.setattr(
        "dmsroute.acquisition.io.os.replace",
        lambda *args, **kwargs: (_ for _ in ()).throw(error),
    )

    with pytest.raises(DownloadError) as exc_info:
        download_file("https://example.test/data.csv", output_path)

    assert exc_info.value.__cause__ is error
    assert not output_path.exists()
    assert _temporary_downloads(tmp_path) == []


def test_infer_filename_from_url():
    name = infer_filename_from_url("https://example.com/data.csv?download=1")
    assert name == "data.csv"


def test_infer_filename_from_url_does_not_include_query_secrets():
    name = infer_filename_from_url(
        "https://example.com/data%20table.csv?token=top-secret#fragment"
    )

    assert name == "data table.csv"
    assert "top-secret" not in name


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


def _bundle_table() -> pd.DataFrame:
    """Return a small standardized table for bundle publication tests."""
    return pd.DataFrame(
        {
            "variant": ["M1A"],
            "score_raw": [0.5],
            "source_note": ["café"],
        }
    )


def _bundle_summary(output_dir: Path) -> dict[str, object]:
    """Return one successful summary row."""
    return {
        "source": "proteingym",
        "input": "ASSAY_1",
        "status": "OK",
        "dataset_id": "ASSAY_1",
        "target_protein": "P12345",
        "wt_length": 3,
        "raw_rows": 1,
        "validated_rows": 1,
        "discarded_rows": 0,
        "output_rows": 1,
        "wildtype_rows": 0,
        "synthetic_wildtype_rows": 0,
        "output_file": str(output_dir / "standardized.csv"),
    }


def test_dataset_bundle_overwrite_replaces_targets_and_preserves_unrelated(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "nested" / "output"
    output_dir.mkdir(parents=True)
    unrelated = output_dir / "notes.txt"
    unrelated.write_text("preserve me", encoding="utf-8")
    for name in ("standardized.csv", "summary.csv", "summary.json"):
        (output_dir / name).write_text("obsolete", encoding="utf-8")

    paths = io_module._publish_dataset_bundle(
        _bundle_table(),
        _bundle_summary(output_dir),
        output_dir,
        overwrite=True,
    )

    assert tuple(path.name for path in paths) == (
        "standardized.csv",
        "summary.csv",
        "summary.json",
    )
    assert unrelated.read_text(encoding="utf-8") == "preserve me"
    assert pd.read_csv(paths[0])["score_raw"].tolist() == [0.5]
    assert pd.read_csv(paths[1])["status"].tolist() == ["OK"]
    assert json.loads(paths[2].read_text(encoding="utf-8"))[0]["status"] == "OK"
    for path in paths:
        content = path.read_bytes()
        assert content.endswith(b"\n")
        assert not content.endswith(b"\n\n")
        assert b"\r\n" not in content
    assert b"caf\xc3\xa9" in paths[0].read_bytes()
    assert list(output_dir.glob(".dmsroute-bundle-*")) == []


@pytest.mark.parametrize(
    "failed_target",
    ["standardized.csv", "summary.csv", "summary.json"],
)
def test_dataset_bundle_publication_failure_restores_all_previous_targets(
    failed_target: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    originals: dict[Path, bytes] = {}
    for name in ("standardized.csv", "summary.csv", "summary.json"):
        path = output_dir / name
        content = f"original {name}".encode()
        path.write_bytes(content)
        originals[path] = content
    unrelated = output_dir / "unrelated.txt"
    unrelated.write_text("untouched", encoding="utf-8")
    original_replace = io_module.os.replace
    failure_injected = False

    def fail_one_publication(source: str | Path, destination: str | Path) -> None:
        nonlocal failure_injected
        if Path(destination).name == failed_target and not failure_injected:
            failure_injected = True
            raise OSError(f"failed {failed_target}")
        original_replace(source, destination)

    monkeypatch.setattr(io_module.os, "replace", fail_one_publication)

    with pytest.raises(OSError, match=f"failed {failed_target}"):
        io_module._publish_dataset_bundle(
            _bundle_table(),
            _bundle_summary(output_dir),
            output_dir,
            overwrite=True,
        )

    assert failure_injected is True
    assert {path: path.read_bytes() for path in originals} == originals
    assert unrelated.read_text(encoding="utf-8") == "untouched"
    assert list(output_dir.glob(".dmsroute-bundle-*")) == []


def test_failed_new_bundle_removes_earlier_publications(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "output"
    original_replace = io_module.os.replace
    failure_injected = False

    def fail_summary_publication(source: str | Path, destination: str | Path) -> None:
        nonlocal failure_injected
        if Path(destination).name == "summary.csv" and not failure_injected:
            failure_injected = True
            raise OSError("summary publication failed")
        original_replace(source, destination)

    monkeypatch.setattr(io_module.os, "replace", fail_summary_publication)

    with pytest.raises(OSError, match="summary publication failed"):
        io_module._publish_dataset_bundle(
            _bundle_table(),
            _bundle_summary(output_dir),
            output_dir,
            overwrite=False,
        )

    assert failure_injected is True
    assert all(
        not (output_dir / name).exists()
        for name in ("standardized.csv", "summary.csv", "summary.json")
    )
    assert list(output_dir.glob(".dmsroute-bundle-*")) == []


def test_dataset_bundle_json_rejects_nan_before_publication(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    summary = _bundle_summary(output_dir)
    summary["raw_rows"] = float("nan")

    with pytest.raises(ValueError, match="Out of range float values"):
        io_module._publish_dataset_bundle(
            _bundle_table(),
            summary,
            output_dir,
            overwrite=False,
        )

    assert all(
        not (output_dir / name).exists()
        for name in ("standardized.csv", "summary.csv", "summary.json")
    )
    assert list(output_dir.glob(".dmsroute-bundle-*")) == []


def _aggregate_records() -> list[dict[str, object]]:
    """Return ordered success and failure records for aggregate tests."""
    columns = (
        "source",
        "dataset_id",
        "status",
        "output_dir",
        "dataset_path",
        "summary_csv_path",
        "summary_json_path",
        "target_protein",
        "wt_length",
        "raw_rows",
        "validated_rows",
        "discarded_rows",
        "output_rows",
        "wildtype_rows",
        "synthetic_wildtype_rows",
        "error_type",
        "error",
    )
    success = dict.fromkeys(columns)
    success.update(
        {
            "source": "proteingym",
            "dataset_id": "ASSAY_1",
            "status": "SUCCESS",
            "output_dir": "proteingym/id-ASSAY_1--digest",
            "dataset_path": "proteingym/id-ASSAY_1--digest/standardized.csv",
            "summary_csv_path": "proteingym/id-ASSAY_1--digest/summary.csv",
            "summary_json_path": "proteingym/id-ASSAY_1--digest/summary.json",
            "target_protein": "caf\u00e9",
            "wt_length": 3,
            "raw_rows": 1,
            "validated_rows": 1,
            "discarded_rows": 0,
            "output_rows": 1,
            "wildtype_rows": 0,
            "synthetic_wildtype_rows": 0,
        }
    )
    failure = dict.fromkeys(columns)
    failure.update(
        {
            "source": "proteingym",
            "dataset_id": "MISSING",
            "status": "ERROR",
            "output_dir": "proteingym/id-MISSING--digest",
            "error_type": "DatasetNotFoundError",
            "error": "missing",
        }
    )
    return [success, failure]


def test_download_summary_publication_contract_and_overwrite(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    unrelated = output_dir / "notes.txt"
    unrelated.write_text("preserve", encoding="utf-8")
    for name in ("download-summary.csv", "download-summary.json"):
        (output_dir / name).write_text("obsolete", encoding="utf-8")

    csv_path, json_path = io_module._publish_download_summary(
        _aggregate_records(),
        output_dir,
        overwrite=True,
    )

    csv_table = pd.read_csv(csv_path)
    assert csv_table.columns.tolist() == list(_aggregate_records()[0])
    assert csv_table["status"].tolist() == ["SUCCESS", "ERROR"]
    assert json.loads(json_path.read_text(encoding="utf-8")) == (
        _aggregate_records()
    )
    assert b"caf\xc3\xa9" in json_path.read_bytes()
    for path in (csv_path, json_path):
        content = path.read_bytes()
        assert content.endswith(b"\n")
        assert not content.endswith(b"\n\n")
        assert b"\r\n" not in content
    assert unrelated.read_text(encoding="utf-8") == "preserve"
    assert list(output_dir.glob(".dmsroute-download-summary-*")) == []


@pytest.mark.parametrize(
    "failed_target",
    ["download-summary.csv", "download-summary.json"],
)
def test_download_summary_failure_restores_previous_targets(
    failed_target: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    originals: dict[Path, bytes] = {}
    for name in ("download-summary.csv", "download-summary.json"):
        path = output_dir / name
        content = f"original {name}".encode()
        path.write_bytes(content)
        originals[path] = content
    original_replace = io_module.os.replace
    failure_injected = False

    def fail_one_publication(source: str | Path, destination: str | Path) -> None:
        nonlocal failure_injected
        if Path(destination).name == failed_target and not failure_injected:
            failure_injected = True
            raise OSError(f"failed {failed_target}")
        original_replace(source, destination)

    monkeypatch.setattr(io_module.os, "replace", fail_one_publication)

    with pytest.raises(OSError, match=f"failed {failed_target}"):
        io_module._publish_download_summary(
            _aggregate_records(),
            output_dir,
            overwrite=True,
        )

    assert failure_injected is True
    assert {path: path.read_bytes() for path in originals} == originals
    assert list(output_dir.glob(".dmsroute-download-summary-*")) == []


def test_new_download_summary_failure_removes_earlier_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "output"
    original_replace = io_module.os.replace
    failure_injected = False

    def fail_json_publication(source: str | Path, destination: str | Path) -> None:
        nonlocal failure_injected
        if Path(destination).name == "download-summary.json" and not failure_injected:
            failure_injected = True
            raise OSError("aggregate publication failed")
        original_replace(source, destination)

    monkeypatch.setattr(io_module.os, "replace", fail_json_publication)

    with pytest.raises(OSError, match="aggregate publication failed"):
        io_module._publish_download_summary(
            _aggregate_records(),
            output_dir,
            overwrite=False,
        )

    assert failure_injected is True
    assert not (output_dir / "download-summary.csv").exists()
    assert not (output_dir / "download-summary.json").exists()
    assert list(output_dir.glob(".dmsroute-download-summary-*")) == []


def test_download_summary_rejects_nan_before_publication(tmp_path: Path) -> None:
    records = _aggregate_records()
    records[0]["raw_rows"] = float("nan")
    output_dir = tmp_path / "output"

    with pytest.raises(ValueError, match="Out of range float values"):
        io_module._publish_download_summary(
            records,
            output_dir,
            overwrite=False,
        )

    assert not (output_dir / "download-summary.csv").exists()
    assert not (output_dir / "download-summary.json").exists()
