from __future__ import annotations

import importlib
import importlib.util
import logging
from pathlib import Path

import pandas as pd
import pytest
import requests

import dms_parser
import dms_parser.fetch as fetch_module
import dms_parser.io as io_module
from dms_parser import (
    FilesystemCache,
    MaveDBCatalog,
    ProteinGymCatalog,
    build_proteingym_dataset,
    fetch_to_cache,
)
from dms_parser.exceptions import DownloadError

RUNNER_PATH = (
    Path(__file__).parents[1] / "examples" / "yml_parser" / "run_dms_parser.py"
)


class JsonResponse:
    """Offline successful JSON response."""

    def __init__(self, payload: object) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        """Represent a successful HTTP status."""

    def json(self) -> object:
        """Return the configured JSON payload."""
        return self.payload


class MetadataSession:
    """Offline session for one MaveDB metadata lookup."""

    def __init__(self, response: JsonResponse) -> None:
        self.response = response

    def get(self, url: str, *, timeout: int) -> JsonResponse:
        """Return the configured response."""
        del url, timeout
        return self.response


class StreamingResponse:
    """Offline streaming download response."""

    def __enter__(self) -> StreamingResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def raise_for_status(self) -> None:
        """Represent a successful HTTP status."""

    def iter_content(self, *, chunk_size: int):
        """Yield deterministic response bytes."""
        del chunk_size
        yield b"offline content"


def test_import_does_not_configure_root_logger() -> None:
    root_logger = logging.getLogger()
    original_handlers = tuple(root_logger.handlers)
    original_level = root_logger.level

    importlib.reload(dms_parser)

    assert tuple(root_logger.handlers) == original_handlers
    assert root_logger.level == original_level
    assert any(
        isinstance(handler, logging.NullHandler)
        for handler in logging.getLogger("dms_parser").handlers
    )


def test_info_reports_cache_miss_and_hit_without_changing_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def offline_download(
        url: str,
        output_path: str | Path,
        *,
        overwrite: bool = False,
        chunk_size: int = 8192,
        timeout: int = 60,
    ) -> Path:
        del url, overwrite, chunk_size, timeout
        path = Path(output_path)
        path.write_bytes(b"cached content")
        return path

    monkeypatch.setattr(fetch_module, "download_file", offline_download)
    cache = FilesystemCache(tmp_path / "cache")
    caplog.set_level(logging.INFO, logger="dms_parser")

    first = fetch_to_cache(
        "https://example.test/data.csv",
        source="example",
        dataset_id="dataset-1",
        cache=cache,
    )
    second = fetch_to_cache(
        "https://example.test/data.csv",
        source="example",
        dataset_id="dataset-1",
        cache=cache,
    )

    assert first == second
    assert first.read_bytes() == b"cached content"
    assert "Cache miss source=example dataset_id=dataset-1" in caplog.text
    assert "Cache hit source=example dataset_id=dataset-1" in caplog.text


def test_debug_reports_cache_diagnostics(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cache = FilesystemCache(tmp_path / "cache")
    cached_path = cache.store_bytes(
        "example",
        "dataset-1",
        "https://example.test/data.csv",
        b"cached content",
    )
    caplog.clear()
    caplog.set_level(logging.DEBUG, logger="dms_parser.cache")

    result = cache.resolve("example", "dataset-1")

    assert result == cached_path
    assert "Resolving cache entry source=example dataset_id=dataset-1" in caplog.text
    assert "Validated cache manifest source=example dataset_id=dataset-1" in caplog.text
    assert "checksum=" in caplog.text


def test_catalog_listing_reports_source_and_result_count(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    substitutions_path = tmp_path / "DMS_substitutions.csv"
    pd.DataFrame(
        [{"DMS_id": "ASSAY_1", "UniProt_ID": "P12345"}]
    ).to_csv(substitutions_path, index=False)
    catalog = ProteinGymCatalog(substitutions_path=substitutions_path)
    caplog.set_level(
        logging.INFO,
        logger="dms_parser.sources.proteingym_catalog",
    )

    records = catalog.list_datasets(variant_type="substitutions")

    assert [record.dataset_id for record in records] == ["ASSAY_1"]
    assert "Starting catalog listing source=proteingym" in caplog.text
    assert (
        "Completed catalog listing source=proteingym result_count=1"
        in caplog.text
    )


def test_metadata_lookup_reports_source_and_dataset_id(
    caplog: pytest.LogCaptureFixture,
) -> None:
    dataset_id = "urn:mavedb:00000001-a-1"
    catalog = MaveDBCatalog(
        session=MetadataSession(JsonResponse({"urn": dataset_id}))
    )
    caplog.set_level(
        logging.INFO,
        logger="dms_parser.sources.mavedb_catalog",
    )

    record = catalog.get_metadata(dataset_id)

    assert record.dataset_id == dataset_id
    assert (
        f"Retrieving dataset metadata source=mavedb dataset_id={dataset_id}"
        in caplog.text
    )
    assert (
        f"Completed metadata retrieval source=mavedb dataset_id={dataset_id}"
        in caplog.text
    )


def test_builder_logs_aggregate_counts_without_wt_or_row_flood(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    wt_sequence = "MKT"
    row_count = 60
    input_path = tmp_path / "scores.csv"
    pd.DataFrame(
        {
            "mutant": ["M1A", "K2R", "T3S"] * (row_count // 3),
            "DMS_score": list(range(row_count)),
        }
    ).to_csv(input_path, index=False)
    caplog.set_level(logging.DEBUG, logger="dms_parser")

    result = build_proteingym_dataset(
        input_path,
        score_col="DMS_score",
        variant_col="mutant",
        dataset_id="assay-1",
        wt_sequence=wt_sequence,
    )

    builder_records = [
        record
        for record in caplog.records
        if record.name == "dms_parser.builders"
    ]
    assert len(result) == row_count
    assert (
        "Completed dataset build source=proteingym dataset_id=assay-1 "
        "input_rows=60 output_rows=60 validated_rows=60 "
        "unsupported_rows=0 error_rows=0"
    ) in caplog.text
    assert wt_sequence not in caplog.text
    assert len(builder_records) <= 6


def test_download_logging_redacts_url_secrets_and_preserves_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    url = (
        "https://api-user:api-secret@example.test/data.csv"
        "?token=top-secret&authorization=Bearer-value#fragment"
    )
    output_path = tmp_path / "data.csv"

    monkeypatch.setattr(
        io_module.requests,
        "get",
        lambda *args, **kwargs: StreamingResponse(),
    )
    caplog.set_level(logging.DEBUG, logger="dms_parser.io")

    result = io_module.download_file(url, output_path)

    assert result == output_path
    assert output_path.read_bytes() == b"offline content"
    assert "endpoint=https://example.test/data.csv" in caplog.text
    for secret in (
        "api-user",
        "api-secret",
        "top-secret",
        "authorization",
        "Bearer-value",
        "fragment",
    ):
        assert secret not in caplog.text


def test_download_exception_type_and_message_are_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    url = "https://example.test/data.csv"
    cause = requests.ConnectionError("offline")

    def failed_request(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise cause

    monkeypatch.setattr(io_module.requests, "get", failed_request)
    caplog.set_level(logging.DEBUG, logger="dms_parser.io")

    with pytest.raises(DownloadError) as exc_info:
        io_module.download_file(url, tmp_path / "data.csv")

    assert str(exc_info.value) == (
        "Failed to download file from 'https://example.test/data.csv': offline"
    )
    assert exc_info.value.__cause__ is cause


def test_example_runner_honors_log_level(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec = importlib.util.spec_from_file_location("logging_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the YAML example runner.")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    configured: dict[str, object] = {}

    def record_configuration(**kwargs: object) -> None:
        configured.update(kwargs)

    monkeypatch.setattr(runner.logging, "basicConfig", record_configuration)
    monkeypatch.setattr(
        runner,
        "load_config",
        lambda path: {"output": {}},
    )
    monkeypatch.setattr(runner, "print_report", lambda summary: None)

    result = runner.main(
        [
            "--config",
            str(tmp_path / "config.yml"),
            "--log-level",
            "DEBUG",
        ]
    )

    assert result == 0
    assert configured["level"] == logging.DEBUG
    assert "%(name)s" in str(configured["format"])
    assert runner.logger.name == "dms_parser.example_runner"
