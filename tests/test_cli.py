from __future__ import annotations

import ast
import json
import logging
import re
from pathlib import Path
from typing import Any

import pytest
import requests

import dms_parser.cli as cli_module
from dms_parser import (
    DatasetBatchDownloadEntry,
    DatasetBatchDownloadResult,
    DatasetDownloadResult,
    DatasetRecord,
    FilesystemCache,
    MaveDBSnapshot,
    MaveDBSnapshotRecord,
    PipelineResult,
)
from dms_parser.exceptions import (
    CatalogError,
    DatasetNotFoundError,
    InvalidCacheEntryError,
    SourceConfigurationError,
)


def _catalog_record(**overrides: Any) -> DatasetRecord:
    """Return a representative normalized catalog record."""
    values: dict[str, Any] = {
        "source": "proteingym",
        "dataset_id": "ASSAY_1",
        "title": "Example assay",
        "target_id": "P12345",
        "variant_type": "substitutions",
        "n_variants": 42,
        "raw_metadata": {
            "nested": {"label": "café", "missing": None},
            "values": [1, None],
        },
    }
    values.update(overrides)
    return DatasetRecord(**values)


@pytest.mark.parametrize(
    ("arguments", "expected_text"),
    [
        (
            ["--help"],
            ("run", "list", "metadata", "download", "download-many", "snapshot"),
        ),
        (["run", "--help"], ("--config", "--dry-run")),
        (["list", "--help"], ("--source", "--query", "--limit", "--format")),
        (["metadata", "--help"], ("--source", "--dataset-id", "--format")),
        (
            ["download", "--help"],
            ("--source", "--dataset-id", "--output-dir", "--overwrite"),
        ),
        (
            ["download-many", "--help"],
            ("--source", "--dataset-id", "--output-dir", "--overwrite"),
        ),
        (["snapshot", "--help"], ("fetch",)),
        (
            ["snapshot", "fetch", "--help"],
            ("--latest", "--record", "--cache-dir", "--format"),
        ),
    ],
    ids=(
        "root",
        "run",
        "list",
        "metadata",
        "download",
        "download-many",
        "snapshot",
        "snapshot-fetch",
    ),
)
def test_help_exits_successfully(
    arguments: list[str],
    expected_text: tuple[str, ...],
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli_module.main(arguments)

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert all(text in output for text in expected_text)


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["unknown"],
        ["run"],
        ["run", "--config", "config.yml", "--only", "invalid"],
        ["run", "--config", "config.yml", "--log-level", "TRACE"],
        ["list"],
        ["metadata"],
        ["metadata", "--source", "mavedb"],
        ["metadata", "--source", "mavedb", "--dataset-id", "   "],
        ["list", "--source", "mavedb", "--limit", "0"],
        ["list", "--source", "mavedb", "--limit", "not-an-integer"],
        ["list", "--source", "mavedb", "--offset", "-1"],
        ["download"],
        ["download", "--source", "mavedb"],
        [
            "download",
            "--source",
            "mavedb",
            "--dataset-id",
            "urn:mavedb:00000001-a-1",
        ],
        [
            "download",
            "--source",
            "mavedb",
            "--dataset-id",
            "   ",
            "--output-dir",
            "output",
        ],
        [
            "download",
            "--source",
            "mavedb",
            "--dataset-id",
            "urn:mavedb:00000001-a-1",
            "--output-dir",
            "   ",
        ],
        ["download-many"],
        ["download-many", "--source", "proteingym"],
        [
            "download-many",
            "--source",
            "proteingym",
            "--dataset-id",
            "ASSAY_1",
        ],
        [
            "download-many",
            "--source",
            "proteingym",
            "--dataset-id",
            " ASSAY_1",
            "--output-dir",
            "output",
        ],
        [
            "download-many",
            "--source",
            "proteingym",
            "--dataset-id",
            "ASSAY\n1",
            "--output-dir",
            "output",
        ],
        ["snapshot"],
        ["snapshot", "fetch"],
        ["snapshot", "fetch", "--record", "0"],
        ["snapshot", "fetch", "--record", "not-a-record"],
        [
            "snapshot",
            "fetch",
            "--latest",
            "--record",
            "20840937",
        ],
    ],
)
def test_invalid_usage_preserves_argparse_exit_status(
    arguments: list[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli_module.main(arguments)

    assert exc_info.value.code == 2


def test_run_forwards_explicit_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yml"
    loaded_config = {"output": {}}
    calls: dict[str, Any] = {}

    def load_config(path: Path) -> dict[str, Any]:
        calls["path"] = path
        return loaded_config

    def run(config: object, *, only: str, dry_run: bool) -> PipelineResult:
        calls["config"] = config
        calls["only"] = only
        calls["dry_run"] = dry_run
        return PipelineResult(summary=[])

    monkeypatch.setattr(cli_module, "load_pipeline_config", load_config)
    monkeypatch.setattr(cli_module, "run_pipeline", run)
    monkeypatch.setattr(
        cli_module,
        "list_datasets",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("run invoked list_datasets")
        ),
    )
    monkeypatch.setattr(
        cli_module,
        "get_dataset_metadata",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("run invoked get_dataset_metadata")
        ),
    )
    monkeypatch.setattr(
        cli_module,
        "download_and_standardize_dataset",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("run invoked download_and_standardize_dataset")
        ),
    )
    monkeypatch.setattr(
        cli_module.logging,
        "basicConfig",
        lambda **kwargs: calls.update(logging_kwargs=kwargs),
    )

    exit_code = cli_module.main(
        [
            "run",
            "--config",
            str(config_path),
            "--only",
            "mavedb",
            "--dry-run",
            "--log-level",
            "DEBUG",
        ]
    )

    assert exit_code == 0
    assert calls["path"] == config_path
    assert calls["config"] is loaded_config
    assert calls["only"] == "mavedb"
    assert calls["dry_run"] is True
    assert calls["logging_kwargs"]["level"] == logging.DEBUG


def test_run_preserves_default_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}

    def load_config(path: Path) -> dict[str, Any]:
        calls["path"] = path
        return {"output": {}}

    monkeypatch.setattr(cli_module, "load_pipeline_config", load_config)

    def run(config: object, *, only: str, dry_run: bool) -> PipelineResult:
        calls.update(config=config, only=only, dry_run=dry_run)
        return PipelineResult(summary=[])

    monkeypatch.setattr(cli_module, "run_pipeline", run)
    monkeypatch.setattr(
        cli_module.logging,
        "basicConfig",
        lambda **kwargs: calls.update(logging_kwargs=kwargs),
    )
    config_path = tmp_path / "config.yml"

    exit_code = cli_module.main(["run", "--config", str(config_path)])

    assert exit_code == 0
    assert calls["path"] == config_path
    assert calls["only"] == "all"
    assert calls["dry_run"] is False
    assert calls["logging_kwargs"]["level"] == logging.INFO


@pytest.mark.parametrize(
    ("summary", "expected_exit_code"),
    [
        ([{"status": "OK"}], 0),
        ([{"status": "ERROR"}], 1),
    ],
)
def test_run_returns_pipeline_result_exit_code(
    summary: list[dict[str, str]],
    expected_exit_code: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli_module,
        "load_pipeline_config",
        lambda path: {"output": {}},
    )
    monkeypatch.setattr(
        cli_module,
        "run_pipeline",
        lambda config, **kwargs: PipelineResult(summary=summary),
    )
    monkeypatch.setattr(cli_module.logging, "basicConfig", lambda **kwargs: None)

    assert (
        cli_module.main(["run", "--config", "config.yml"])
        == expected_exit_code
    )


def test_configuration_error_returns_one_without_running_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def invalid_config(path: Path) -> None:
        raise SourceConfigurationError(f"Invalid config: {path.name}")

    def unexpected_pipeline(*args: object, **kwargs: object) -> None:
        raise AssertionError("Invalid configuration reached the pipeline.")

    monkeypatch.setattr(cli_module, "load_pipeline_config", invalid_config)
    monkeypatch.setattr(cli_module, "run_pipeline", unexpected_pipeline)
    monkeypatch.setattr(cli_module.logging, "basicConfig", lambda **kwargs: None)
    caplog.set_level(logging.ERROR, logger="dms_parser.cli")

    exit_code = cli_module.main(["run", "--config", "invalid.yml"])

    assert exit_code == 1
    assert "Configuration error: Invalid config: invalid.yml" in caplog.text


def test_unexpected_configuration_exception_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cause = RuntimeError("unexpected")

    def fail_unexpectedly(path: Path) -> None:
        raise cause

    monkeypatch.setattr(cli_module, "load_pipeline_config", fail_unexpectedly)
    monkeypatch.setattr(cli_module.logging, "basicConfig", lambda **kwargs: None)

    with pytest.raises(RuntimeError) as exc_info:
        cli_module.main(["run", "--config", "config.yml"])

    assert exc_info.value is cause


def test_list_forwards_defaults_and_never_runs_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: dict[str, Any] = {}

    def list_records(source: str, **kwargs: Any) -> list[DatasetRecord]:
        calls.update(source=source, **kwargs)
        return []

    monkeypatch.setattr(cli_module, "list_datasets", list_records)
    monkeypatch.setattr(
        cli_module,
        "run_pipeline",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("list invoked run_pipeline")
        ),
    )
    monkeypatch.setattr(cli_module.logging, "basicConfig", lambda **kwargs: None)

    exit_code = cli_module.main(["list", "--source", "mavedb", "--format", "json"])

    assert exit_code == 0
    assert calls == {
        "source": "mavedb",
        "query": None,
        "limit": None,
        "offset": 0,
        "variant_type": None,
        "cache": None,
        "refresh": False,
    }
    assert json.loads(capsys.readouterr().out) == []


def test_default_proteingym_cache_root_is_lazy_and_forwarded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root = tmp_path / "home" / ".cache" / "dms-parser"
    calls: dict[str, Any] = {}
    monkeypatch.setattr(cli_module, "_default_cache_root", lambda: cache_root)

    def list_records(source: str, **kwargs: Any) -> list[DatasetRecord]:
        calls.update(source=source, **kwargs)
        return []

    monkeypatch.setattr(cli_module, "list_datasets", list_records)
    monkeypatch.setattr(cli_module.logging, "basicConfig", lambda **kwargs: None)

    assert cli_module.main(["list", "--source", "proteingym"]) == 0

    assert isinstance(calls["cache"], FilesystemCache)
    assert calls["cache"].root == cache_root
    assert not cache_root.exists()


def test_list_forwards_every_explicit_filter_and_expands_cache_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}
    supplied_cache = Path("~") / "catalog-cli-test-cache"

    def list_records(source: str, **kwargs: Any) -> list[DatasetRecord]:
        calls.update(source=source, **kwargs)
        return []

    monkeypatch.setattr(cli_module, "list_datasets", list_records)
    monkeypatch.setattr(cli_module.logging, "basicConfig", lambda **kwargs: None)

    exit_code = cli_module.main(
        [
            "list",
            "--source",
            "proteingym",
            "--query",
            "BRCA1",
            "--limit",
            "10",
            "--offset",
            "5",
            "--variant-type",
            "substitutions",
            "--cache-dir",
            str(supplied_cache),
            "--refresh",
        ]
    )

    assert exit_code == 0
    assert calls["source"] == "proteingym"
    assert calls["query"] == "BRCA1"
    assert calls["limit"] == 10
    assert calls["offset"] == 5
    assert calls["variant_type"] == "substitutions"
    assert calls["refresh"] is True
    assert calls["cache"].root == supplied_cache.expanduser()


def test_metadata_forwards_identifier_variant_type_cache_and_refresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}
    cache_root = tmp_path / "cache"

    def get_record(source: str, dataset_id: str, **kwargs: Any) -> DatasetRecord:
        calls.update(source=source, dataset_id=dataset_id, **kwargs)
        return _catalog_record()

    monkeypatch.setattr(cli_module, "get_dataset_metadata", get_record)
    monkeypatch.setattr(cli_module.logging, "basicConfig", lambda **kwargs: None)

    exit_code = cli_module.main(
        [
            "metadata",
            "--source",
            "proteingym",
            "--dataset-id",
            "ASSAY_1",
            "--variant-type",
            "indels",
            "--cache-dir",
            str(cache_root),
            "--refresh",
            "--format",
            "json",
        ]
    )

    assert exit_code == 0
    assert calls["source"] == "proteingym"
    assert calls["dataset_id"] == "ASSAY_1"
    assert calls["variant_type"] == "indels"
    assert calls["refresh"] is True
    assert calls["cache"].root == cache_root


@pytest.mark.parametrize(
    ("command", "option"),
    [
        ("list", ["--variant-type", "substitutions"]),
        ("list", ["--cache-dir", "unused-cache"]),
        ("list", ["--refresh"]),
        ("metadata", ["--variant-type", "substitutions"]),
        ("metadata", ["--cache-dir", "unused-cache"]),
        ("metadata", ["--refresh"]),
    ],
)
def test_mavedb_rejects_proteingym_options_before_side_effects(
    command: str,
    option: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "output" / "result.txt"
    arguments = [command, "--source", "mavedb"]
    if command == "metadata":
        arguments.extend(["--dataset-id", "urn:mavedb:00000001-a-1"])
    arguments.extend(option + ["--output", str(output_path)])

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Invalid MaveDB options performed a side effect")

    monkeypatch.setattr(cli_module, "FilesystemCache", forbidden)
    monkeypatch.setattr(cli_module, "list_datasets", forbidden)
    monkeypatch.setattr(cli_module, "get_dataset_metadata", forbidden)

    with pytest.raises(SystemExit) as exc_info:
        cli_module.main(arguments)

    assert exc_info.value.code == 2
    assert not output_path.parent.exists()


def test_list_text_is_aligned_without_mutating_displayed_values(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    record = _catalog_record(title="Line one\nLine\ttwo", target_id=None, n_variants=None)
    monkeypatch.setattr(cli_module, "list_datasets", lambda *args, **kwargs: [record])
    monkeypatch.setattr(cli_module.logging, "basicConfig", lambda **kwargs: None)

    assert cli_module.main(["list", "--source", "mavedb"]) == 0

    output = capsys.readouterr().out
    assert output.splitlines()[0].split() == [
        "SOURCE",
        "DATASET_ID",
        "TITLE",
        "TARGET_ID",
        "VARIANT_TYPE",
        "N_VARIANTS",
    ]
    assert "Line one Line two" in output
    assert "raw_metadata" not in output
    assert "-" in output
    assert output.endswith("\n")
    assert record.title == "Line one\nLine\ttwo"


def test_metadata_text_contains_summary_and_pretty_raw_metadata(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    record = _catalog_record(title=None, target_id=None, variant_type=None, n_variants=None)
    monkeypatch.setattr(cli_module, "get_dataset_metadata", lambda *args, **kwargs: record)
    monkeypatch.setattr(cli_module.logging, "basicConfig", lambda **kwargs: None)

    assert cli_module.main(
        ["metadata", "--source", "mavedb", "--dataset-id", "urn:mavedb:1-a-1"]
    ) == 0

    output = capsys.readouterr().out
    for field in (
        "source:",
        "dataset_id:",
        "title: -",
        "target_id: -",
        "variant_type: -",
        "n_variants: -",
        "raw_metadata:",
    ):
        assert field in output
    assert '    "label": "café"' in output
    assert output.endswith("\n")


@pytest.mark.parametrize("command", ["list", "metadata"])
def test_catalog_json_contains_all_fields_and_preserves_whitespace(
    command: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    record = _catalog_record(title="Line one\nLine\ttwo")
    monkeypatch.setattr(cli_module, "list_datasets", lambda *args, **kwargs: [record])
    monkeypatch.setattr(cli_module, "get_dataset_metadata", lambda *args, **kwargs: record)
    monkeypatch.setattr(cli_module.logging, "basicConfig", lambda **kwargs: None)
    arguments = [command, "--source", "mavedb", "--format", "json"]
    if command == "metadata":
        arguments.extend(["--dataset-id", "urn:mavedb:1-a-1"])

    assert cli_module.main(arguments) == 0

    output = capsys.readouterr().out
    parsed = json.loads(output)
    values = parsed[0] if command == "list" else parsed
    assert list(values) == [
        "source",
        "dataset_id",
        "title",
        "target_id",
        "variant_type",
        "n_variants",
        "raw_metadata",
    ]
    assert values["title"] == "Line one\nLine\ttwo"
    assert values["raw_metadata"] == record.raw_metadata
    assert "NaN" not in output
    assert output.endswith("\n")


def test_json_renderer_disables_nonstandard_nan(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}
    original_dumps = json.dumps

    def dumps(value: object, **kwargs: Any) -> str:
        calls.update(kwargs)
        return original_dumps(value, **kwargs)

    monkeypatch.setattr(cli_module.json, "dumps", dumps)

    assert json.loads(cli_module._render_json({"missing": None})) == {"missing": None}
    assert calls["allow_nan"] is False
    assert calls["ensure_ascii"] is False
    assert calls["indent"] == 2


@pytest.mark.parametrize(
    ("output_format", "expected"),
    [
        ("text", "No datasets found."),
        ("json", "[]"),
    ],
)
def test_empty_list_output_is_successful(
    output_format: str,
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli_module, "list_datasets", lambda *args, **kwargs: [])
    monkeypatch.setattr(cli_module.logging, "basicConfig", lambda **kwargs: None)

    assert cli_module.main(
        ["list", "--source", "mavedb", "--format", output_format]
    ) == 0
    assert expected in capsys.readouterr().out


def test_output_creates_parents_overwrites_utf8_and_suppresses_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    record = _catalog_record()
    output_path = tmp_path / "nested" / "catalog.json"
    monkeypatch.setattr(cli_module, "list_datasets", lambda *args, **kwargs: [record])
    monkeypatch.setattr(cli_module.logging, "basicConfig", lambda **kwargs: None)
    caplog.set_level(logging.INFO, logger="dms_parser.cli")
    arguments = [
        "list",
        "--source",
        "mavedb",
        "--format",
        "json",
        "--output",
        str(output_path),
    ]

    assert cli_module.main(arguments) == 0
    assert capsys.readouterr().out == ""
    first_bytes = output_path.read_bytes()
    assert "café".encode() in first_bytes
    assert first_bytes.endswith(b"\n") and not first_bytes.endswith(b"\n\n")

    output_path.write_text("obsolete", encoding="utf-8")
    assert cli_module.main(arguments) == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))[0]["dataset_id"] == "ASSAY_1"
    assert "Catalog output written path=" in caplog.text


def test_output_format_is_not_inferred_from_filename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "catalog.json"
    monkeypatch.setattr(
        cli_module,
        "list_datasets",
        lambda *args, **kwargs: [_catalog_record()],
    )
    monkeypatch.setattr(cli_module.logging, "basicConfig", lambda **kwargs: None)

    assert cli_module.main(
        ["list", "--source", "mavedb", "--output", str(output_path)]
    ) == 0

    output = output_path.read_text(encoding="utf-8")
    assert output.startswith("SOURCE")
    assert output.endswith("\n") and not output.endswith("\n\n")


def test_failed_atomic_publication_preserves_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "catalog.json"
    output_path.write_text("existing", encoding="utf-8")
    monkeypatch.setattr(
        cli_module,
        "list_datasets",
        lambda *args, **kwargs: [_catalog_record()],
    )
    monkeypatch.setattr(cli_module.logging, "basicConfig", lambda **kwargs: None)
    monkeypatch.setattr(
        cli_module.os,
        "replace",
        lambda source, destination: (_ for _ in ()).throw(OSError("publication failed")),
    )

    exit_code = cli_module.main(
        [
            "list",
            "--source",
            "mavedb",
            "--format",
            "json",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 1
    assert output_path.read_text(encoding="utf-8") == "existing"
    assert list(tmp_path.glob(".catalog.json.*")) == []


@pytest.mark.parametrize(
    "cause",
    [
        CatalogError("catalog failed"),
        InvalidCacheEntryError("cache failed"),
        requests.ConnectionError("network failed"),
        OSError("filesystem failed"),
    ],
)
def test_expected_list_failures_return_one(
    cause: Exception,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise cause

    monkeypatch.setattr(cli_module, "list_datasets", fail)
    monkeypatch.setattr(cli_module.logging, "basicConfig", lambda **kwargs: None)

    assert cli_module.main(["list", "--source", "mavedb"]) == 1


def test_cache_construction_failure_returns_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("cache unavailable")

    monkeypatch.setattr(cli_module, "FilesystemCache", fail)
    monkeypatch.setattr(cli_module.logging, "basicConfig", lambda **kwargs: None)

    assert cli_module.main(["list", "--source", "proteingym"]) == 1


def test_dataset_not_found_returns_one_without_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "metadata.json"

    def not_found(*args: object, **kwargs: object) -> None:
        raise DatasetNotFoundError("missing dataset")

    monkeypatch.setattr(cli_module, "get_dataset_metadata", not_found)
    monkeypatch.setattr(cli_module.logging, "basicConfig", lambda **kwargs: None)

    assert cli_module.main(
        [
            "metadata",
            "--source",
            "mavedb",
            "--dataset-id",
            "urn:mavedb:missing",
            "--output",
            str(output_path),
        ]
    ) == 1
    assert not output_path.exists()


def test_unexpected_catalog_exception_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    cause = RuntimeError("unexpected")

    def fail(*args: object, **kwargs: object) -> None:
        raise cause

    monkeypatch.setattr(cli_module, "list_datasets", fail)
    monkeypatch.setattr(cli_module.logging, "basicConfig", lambda **kwargs: None)

    with pytest.raises(RuntimeError) as exc_info:
        cli_module.main(["list", "--source", "mavedb"])

    assert exc_info.value is cause


def _snapshot_result(
    cache_root: Path,
    *,
    cache_hit: bool = False,
) -> MaveDBSnapshot:
    """Return a representative managed snapshot result for CLI tests."""
    root = cache_root / "mavedb" / "snapshots" / "20840937"
    return MaveDBSnapshot(
        record=MaveDBSnapshotRecord(
            record_id="20840937",
            doi="10.5281/zenodo.20840937",
            concept_doi="10.5281/zenodo.11201736",
            publication_date="2026-06-24",
            filename="mavedb-dump.test.tar.gz",
            size=123,
            checksum="md5:900150983cd24fb0d6963f7d28e17f72",
            download_url="https://zenodo.org/content",
        ),
        archive_path=root / "mavedb-dump.test.tar.gz",
        main_json_path=root / "main.json",
        cache_hit=cache_hit,
    )


def test_snapshot_latest_forwards_defaults_and_renders_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cache_root = tmp_path / "default-cache"
    calls: dict[str, Any] = {}
    monkeypatch.setattr(cli_module, "_default_cache_root", lambda: cache_root)

    def fetch(selector: str, **kwargs: Any) -> MaveDBSnapshot:
        calls.update(selector=selector, **kwargs)
        return _snapshot_result(kwargs["cache"].root)

    monkeypatch.setattr(cli_module, "fetch_mavedb_snapshot", fetch)
    monkeypatch.setattr(cli_module.logging, "basicConfig", lambda **kwargs: None)

    assert cli_module.main(["snapshot", "fetch", "--latest"]) == 0

    assert calls["selector"] == "latest"
    assert calls["cache"].root == cache_root
    assert calls["refresh"] is False
    output = capsys.readouterr().out
    for text in (
        "record_id: 20840937",
        "doi: 10.5281/zenodo.20840937",
        "concept_doi: 10.5281/zenodo.11201736",
        "publication_date: 2026-06-24",
        "archive_filename: mavedb-dump.test.tar.gz",
        "size: 123",
        "checksum: md5:900150983cd24fb0d6963f7d28e17f72",
        "archive_path:",
        "main_json_path:",
        "cache_hit: False",
    ):
        assert text in output
    assert output.endswith("\n")


def test_snapshot_record_forwards_options_and_renders_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cache_root = tmp_path / "explicit-cache"
    calls: dict[str, Any] = {}
    logging_calls: list[dict[str, Any]] = []

    def fetch(selector: str, **kwargs: Any) -> MaveDBSnapshot:
        calls.update(selector=selector, **kwargs)
        return _snapshot_result(kwargs["cache"].root, cache_hit=True)

    monkeypatch.setattr(cli_module, "fetch_mavedb_snapshot", fetch)
    monkeypatch.setattr(
        cli_module.logging,
        "basicConfig",
        lambda **kwargs: logging_calls.append(kwargs),
    )

    assert cli_module.main(
        [
            "snapshot",
            "fetch",
            "--record",
            "20840937",
            "--cache-dir",
            str(cache_root),
            "--refresh",
            "--format",
            "json",
            "--log-level",
            "DEBUG",
        ]
    ) == 0

    assert calls["selector"] == "20840937"
    assert calls["cache"].root == cache_root
    assert calls["refresh"] is True
    assert logging_calls[0]["level"] == logging.DEBUG
    values = json.loads(capsys.readouterr().out)
    assert list(values) == [
        "record_id",
        "doi",
        "concept_doi",
        "publication_date",
        "archive_filename",
        "size",
        "checksum",
        "archive_path",
        "main_json_path",
        "cache_hit",
    ]
    assert values["record_id"] == "20840937"
    assert values["cache_hit"] is True


@pytest.mark.parametrize(
    "cause",
    [
        InvalidCacheEntryError("cache failed"),
        requests.ConnectionError("network failed"),
        OSError("filesystem failed"),
    ],
)
def test_expected_snapshot_failures_return_one(
    cause: Exception,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        cli_module,
        "fetch_mavedb_snapshot",
        lambda *args, **kwargs: (_ for _ in ()).throw(cause),
    )
    monkeypatch.setattr(cli_module.logging, "basicConfig", lambda **kwargs: None)

    assert cli_module.main(["snapshot", "fetch", "--latest"]) == 1
    assert capsys.readouterr().out == ""


def test_unexpected_snapshot_exception_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cause = RuntimeError("unexpected")
    monkeypatch.setattr(
        cli_module,
        "fetch_mavedb_snapshot",
        lambda *args, **kwargs: (_ for _ in ()).throw(cause),
    )
    monkeypatch.setattr(cli_module.logging, "basicConfig", lambda **kwargs: None)

    with pytest.raises(RuntimeError) as exc_info:
        cli_module.main(["snapshot", "fetch", "--latest"])

    assert exc_info.value is cause


@pytest.mark.parametrize(
    ("arguments", "api_name"),
    [
        (["list", "--source", "mavedb"], "list_datasets"),
        (
            ["metadata", "--source", "mavedb", "--dataset-id", "urn:mavedb:1-a-1"],
            "get_dataset_metadata",
        ),
    ],
)
def test_catalog_log_level_and_json_stdout_are_separate(
    arguments: list[str],
    api_name: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    logging_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        cli_module.logging,
        "basicConfig",
        lambda **kwargs: logging_calls.append(kwargs),
    )
    monkeypatch.setattr(cli_module, "list_datasets", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        cli_module,
        "get_dataset_metadata",
        lambda *args, **kwargs: _catalog_record(),
    )
    monkeypatch.setattr(
        cli_module,
        "run_pipeline",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError(f"{api_name} invoked run_pipeline")
        ),
    )

    assert cli_module.main(arguments + ["--format", "json", "--log-level", "DEBUG"]) == 0

    json.loads(capsys.readouterr().out)
    assert logging_calls[0]["level"] == logging.DEBUG
    assert "stream" not in logging_calls[0]


def _download_result(output_dir: Path) -> DatasetDownloadResult:
    """Return a successful download result rooted at ``output_dir``."""
    return DatasetDownloadResult(
        dataset_path=output_dir / "standardized.csv",
        summary_csv_path=output_dir / "summary.csv",
        summary_json_path=output_dir / "summary.json",
        summary={"status": "OK"},
    )


def test_download_help_excludes_deferred_options(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli_module.main(["download", "--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "--variant-type" not in output
    assert "--keep-failed" not in output


@pytest.mark.parametrize(
    ("source", "dataset_id"),
    [
        ("mavedb", "not-a-permanent-urn"),
        ("mavedb", "urn:mavedb:00000001-a-0"),
        ("proteingym", "ASSAY.csv"),
        ("proteingym", "ASSAY.CSV"),
    ],
)
def test_download_rejects_source_ids_before_side_effects(
    source: str,
    dataset_id: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "output"

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Invalid dataset ID performed a side effect")

    monkeypatch.setattr(cli_module, "FilesystemCache", forbidden)
    monkeypatch.setattr(
        cli_module,
        "download_and_standardize_dataset",
        forbidden,
    )

    with pytest.raises(SystemExit) as exc_info:
        cli_module.main(
            [
                "download",
                "--source",
                source,
                "--dataset-id",
                dataset_id,
                "--output-dir",
                str(output_dir),
            ]
        )

    assert exc_info.value.code == 2
    assert not output_dir.exists()


def test_download_forwards_defaults_and_uses_default_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    cache_root = tmp_path / "home" / ".cache" / "dms-parser"
    output_dir = tmp_path / "output"
    calls: dict[str, Any] = {}
    monkeypatch.setattr(cli_module, "_default_cache_root", lambda: cache_root)

    def download(source: str, dataset_id: str, **kwargs: Any) -> DatasetDownloadResult:
        calls.update(source=source, dataset_id=dataset_id, **kwargs)
        return _download_result(kwargs["output_dir"])

    monkeypatch.setattr(cli_module, "download_and_standardize_dataset", download)
    monkeypatch.setattr(
        cli_module,
        "run_pipeline",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("download invoked run_pipeline")
        ),
    )
    monkeypatch.setattr(cli_module.logging, "basicConfig", lambda **kwargs: None)
    caplog.set_level(logging.INFO, logger="dms_parser.cli")

    assert cli_module.main(
        [
            "download",
            "--source",
            "proteingym",
            "--dataset-id",
            "ASSAY_1",
            "--output-dir",
            str(output_dir),
        ]
    ) == 0

    assert calls == {
        "source": "proteingym",
        "dataset_id": "ASSAY_1",
        "output_dir": output_dir,
        "cache": calls["cache"],
        "refresh": False,
        "drop_failed": False,
        "add_wildtype_row": False,
        "overwrite": False,
    }
    assert isinstance(calls["cache"], FilesystemCache)
    assert calls["cache"].root == cache_root
    assert not cache_root.exists()
    assert capsys.readouterr().out == ""
    assert "Dataset download completed" in caplog.text
    assert str(output_dir / "standardized.csv") in caplog.text
    assert str(output_dir / "summary.csv") in caplog.text
    assert str(output_dir / "summary.json") in caplog.text


def test_download_forwards_explicit_options_and_expands_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}
    logging_calls: list[dict[str, Any]] = []
    cache_dir = Path("~") / "download-cache"
    output_dir = Path("~") / "download-output"

    def download(source: str, dataset_id: str, **kwargs: Any) -> DatasetDownloadResult:
        calls.update(source=source, dataset_id=dataset_id, **kwargs)
        return _download_result(kwargs["output_dir"])

    monkeypatch.setattr(cli_module, "download_and_standardize_dataset", download)
    monkeypatch.setattr(
        cli_module.logging,
        "basicConfig",
        lambda **kwargs: logging_calls.append(kwargs),
    )

    assert cli_module.main(
        [
            "download",
            "--source",
            "mavedb",
            "--dataset-id",
            "urn:mavedb:00000001-a-1",
            "--output-dir",
            str(output_dir),
            "--cache-dir",
            str(cache_dir),
            "--refresh",
            "--drop-failed",
            "--add-wildtype-row",
            "--overwrite",
            "--log-level",
            "DEBUG",
        ]
    ) == 0

    assert calls["source"] == "mavedb"
    assert calls["dataset_id"] == "urn:mavedb:00000001-a-1"
    assert calls["output_dir"] == output_dir.expanduser()
    assert calls["cache"].root == cache_dir.expanduser()
    assert calls["refresh"] is True
    assert calls["drop_failed"] is True
    assert calls["add_wildtype_row"] is True
    assert calls["overwrite"] is True
    assert len(logging_calls) == 1
    assert logging_calls[0]["level"] == logging.DEBUG
    assert "stream" not in logging_calls[0]


def test_expected_download_failure_returns_one_without_stdout(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        cli_module,
        "download_and_standardize_dataset",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            DatasetNotFoundError("missing")
        ),
    )
    monkeypatch.setattr(cli_module.logging, "basicConfig", lambda **kwargs: None)

    assert cli_module.main(
        [
            "download",
            "--source",
            "proteingym",
            "--dataset-id",
            "MISSING",
            "--output-dir",
            "output",
        ]
    ) == 1
    assert capsys.readouterr().out == ""


def test_unexpected_download_exception_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cause = RuntimeError("unexpected")
    monkeypatch.setattr(
        cli_module,
        "download_and_standardize_dataset",
        lambda *args, **kwargs: (_ for _ in ()).throw(cause),
    )
    monkeypatch.setattr(cli_module.logging, "basicConfig", lambda **kwargs: None)

    with pytest.raises(RuntimeError) as exc_info:
        cli_module.main(
            [
                "download",
                "--source",
                "proteingym",
                "--dataset-id",
                "ASSAY_1",
                "--output-dir",
                "output",
            ]
        )

    assert exc_info.value is cause


def _batch_download_result(
    output_dir: Path,
    *,
    failed: bool = False,
) -> DatasetBatchDownloadResult:
    """Return one representative result for download-many CLI tests."""
    dataset_output_dir = output_dir / "proteingym" / "id-ASSAY_1--digest"
    result = None if failed else _download_result(dataset_output_dir)
    entry = DatasetBatchDownloadEntry(
        source="proteingym",
        dataset_id="ASSAY_1",
        output_dir=dataset_output_dir,
        result=result,
        error_type="DatasetNotFoundError" if failed else None,
        error="missing" if failed else None,
    )
    return DatasetBatchDownloadResult(
        entries=(entry,),
        summary_csv_path=output_dir / "download-summary.csv",
        summary_json_path=output_dir / "download-summary.json",
    )


def test_download_many_help_excludes_deferred_options(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli_module.main(["download-many", "--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    for option in (
        "--continue-on-error",
        "--fail-fast",
        "--jobs",
        "--variant-type",
        "--keep-failed",
        "--all",
        "--query",
    ):
        assert option not in output


@pytest.mark.parametrize(
    ("source", "dataset_ids"),
    [
        ("mavedb", ["urn:mavedb:00000001-a-1", "invalid"]),
        ("proteingym", ["ASSAY_1", "ASSAY_1"]),
        ("proteingym", ["ASSAY_1", "ASSAY.csv"]),
    ],
)
def test_download_many_validates_every_id_before_cache_construction(
    source: str,
    dataset_ids: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("invalid batch constructed or used a cache")

    monkeypatch.setattr(cli_module, "FilesystemCache", forbidden)
    monkeypatch.setattr(
        cli_module,
        "download_and_standardize_datasets",
        forbidden,
    )
    arguments = ["download-many", "--source", source]
    for dataset_id in dataset_ids:
        arguments.extend(["--dataset-id", dataset_id])
    arguments.extend(["--output-dir", str(tmp_path / "output")])

    with pytest.raises(SystemExit) as exc_info:
        cli_module.main(arguments)

    assert exc_info.value.code == 2
    assert not (tmp_path / "output").exists()


def test_download_many_collision_preflight_precedes_cache_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    existing = output_dir / "download-summary.csv"
    existing.write_text("existing", encoding="utf-8")

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("collision constructed or used a cache")

    monkeypatch.setattr(cli_module, "FilesystemCache", forbidden)
    monkeypatch.setattr(
        cli_module,
        "download_and_standardize_datasets",
        forbidden,
    )
    monkeypatch.setattr(cli_module.logging, "basicConfig", lambda **kwargs: None)

    assert cli_module.main(
        [
            "download-many",
            "--source",
            "proteingym",
            "--dataset-id",
            "ASSAY_1",
            "--output-dir",
            str(output_dir),
        ]
    ) == 1
    assert existing.read_text(encoding="utf-8") == "existing"


def test_download_many_forwards_defaults_order_and_default_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "output"
    cache_root = tmp_path / "home" / ".cache" / "dms-parser"
    calls: dict[str, Any] = {}
    monkeypatch.setattr(cli_module, "_default_cache_root", lambda: cache_root)

    def batch(source: str, dataset_ids: list[str], **kwargs: Any):
        calls.update(source=source, dataset_ids=dataset_ids, **kwargs)
        return _batch_download_result(kwargs["output_dir"])

    monkeypatch.setattr(cli_module, "download_and_standardize_datasets", batch)
    monkeypatch.setattr(
        cli_module,
        "run_pipeline",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("download-many invoked run_pipeline")
        ),
    )
    monkeypatch.setattr(cli_module.logging, "basicConfig", lambda **kwargs: None)

    assert cli_module.main(
        [
            "download-many",
            "--source",
            "proteingym",
            "--dataset-id",
            "ASSAY_2",
            "--dataset-id",
            "ASSAY_1",
            "--output-dir",
            str(output_dir),
        ]
    ) == 0

    assert calls["source"] == "proteingym"
    assert calls["dataset_ids"] == ["ASSAY_2", "ASSAY_1"]
    assert calls["output_dir"] == output_dir
    assert calls["cache"].root == cache_root
    assert calls["refresh"] is False
    assert calls["drop_failed"] is False
    assert calls["add_wildtype_row"] is False
    assert calls["overwrite"] is False
    assert not cache_root.exists()
    assert capsys.readouterr().out == ""


def test_download_many_forwards_explicit_options_and_expands_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = Path("~") / "batch-output"
    cache_dir = Path("~") / "batch-cache"
    calls: dict[str, Any] = {}
    logging_calls: list[dict[str, Any]] = []

    def batch(source: str, dataset_ids: list[str], **kwargs: Any):
        calls.update(source=source, dataset_ids=dataset_ids, **kwargs)
        return _batch_download_result(kwargs["output_dir"])

    monkeypatch.setattr(cli_module, "download_and_standardize_datasets", batch)
    monkeypatch.setattr(
        cli_module.logging,
        "basicConfig",
        lambda **kwargs: logging_calls.append(kwargs),
    )

    assert cli_module.main(
        [
            "download-many",
            "--source",
            "mavedb",
            "--dataset-id",
            "urn:mavedb:00000001-a-1",
            "--output-dir",
            str(output_dir),
            "--cache-dir",
            str(cache_dir),
            "--refresh",
            "--drop-failed",
            "--add-wildtype-row",
            "--overwrite",
            "--log-level",
            "DEBUG",
        ]
    ) == 0

    assert calls["output_dir"] == output_dir.expanduser()
    assert calls["cache"].root == cache_dir.expanduser()
    assert calls["refresh"] is True
    assert calls["drop_failed"] is True
    assert calls["add_wildtype_row"] is True
    assert calls["overwrite"] is True
    assert logging_calls[0]["level"] == logging.DEBUG


def test_download_many_partial_failure_returns_one_without_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        cli_module,
        "download_and_standardize_datasets",
        lambda *args, **kwargs: _batch_download_result(
            kwargs["output_dir"],
            failed=True,
        ),
    )
    monkeypatch.setattr(cli_module.logging, "basicConfig", lambda **kwargs: None)

    assert cli_module.main(
        [
            "download-many",
            "--source",
            "proteingym",
            "--dataset-id",
            "ASSAY_1",
            "--output-dir",
            str(tmp_path / "output"),
        ]
    ) == 1
    assert capsys.readouterr().out == ""


def test_unexpected_download_many_exception_propagates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cause = RuntimeError("unexpected")
    monkeypatch.setattr(
        cli_module,
        "download_and_standardize_datasets",
        lambda *args, **kwargs: (_ for _ in ()).throw(cause),
    )
    monkeypatch.setattr(cli_module.logging, "basicConfig", lambda **kwargs: None)

    with pytest.raises(RuntimeError) as exc_info:
        cli_module.main(
            [
                "download-many",
                "--source",
                "proteingym",
                "--dataset-id",
                "ASSAY_1",
                "--output-dir",
                str(tmp_path / "output"),
            ]
        )

    assert exc_info.value is cause


def test_pyproject_registers_cli_entry_point() -> None:
    pyproject_path = Path(__file__).parents[1] / "pyproject.toml"
    pyproject = pyproject_path.read_text(encoding="utf-8")
    scripts_section = pyproject.split("[project.scripts]\n", maxsplit=1)[1]
    scripts_section = scripts_section.split("\n[", maxsplit=1)[0]
    entries = [
        line.strip()
        for line in scripts_section.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert entries == ['dms-parser = "dms_parser.cli:main"']


def test_pyproject_declares_cli_runtime_dependencies() -> None:
    pyproject_path = Path(__file__).parents[1] / "pyproject.toml"
    pyproject = pyproject_path.read_text(encoding="utf-8")

    def section_array(section: str, key: str) -> list[str]:
        section_body = pyproject.split(f"[{section}]\n", maxsplit=1)[1]
        section_body = section_body.split("\n[", maxsplit=1)[0]
        match = re.search(
            rf"(?ms)^{re.escape(key)}\s*=\s*(\[.*?^\])",
            section_body,
        )
        assert match is not None
        return ast.literal_eval(match.group(1))

    def package_names(requirements: list[str]) -> set[str]:
        names = set()
        for requirement in requirements:
            match = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", requirement)
            assert match is not None
            names.add(re.sub(r"[-_.]+", "-", match.group()).lower())
        return names

    runtime_names = package_names(section_array("project", "dependencies"))
    dev_names = package_names(
        section_array("project.optional-dependencies", "dev")
    )

    assert {"pyyaml", "pyarrow"} <= runtime_names
    assert {"pyyaml", "pyarrow"}.isdisjoint(dev_names)
