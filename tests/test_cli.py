from __future__ import annotations

import ast
import json
import logging
import re
from pathlib import Path
from typing import Any

import pytest
import requests
from typer.testing import CliRunner

import dms_parser.cli as cli_module
from dms_parser import (
    DatasetBatchDownloadEntry,
    DatasetBatchDownloadResult,
    DatasetDownloadResult,
    DatasetRecord,
    FilesystemCache,
    MaveDBDiscoveredExperiment,
    MaveDBDiscoveredScoreSet,
    MaveDBDiscoveryResult,
    MaveDBSnapshot,
    MaveDBSnapshotRecord,
    MaveDBSnapshotTable,
    MaveDBSnapshotTableExtractionResult,
    PipelineResult,
)
from dms_parser.exceptions import (
    CatalogError,
    DatasetNotFoundError,
    InvalidCacheEntryError,
    MaveDBSnapshotError,
    MaveDBSnapshotTableError,
    SourceConfigurationError,
)

SNAPSHOT_TABLE_ID = "urn:mavedb:00000003-a-1"
SNAPSHOT_EXTRACT_LATEST = (
    "snapshot",
    "extract",
    "--latest",
    "--dataset-id",
    SNAPSHOT_TABLE_ID,
)
RUNNER = CliRunner()


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


HELP_CASES = (
        (
            ["--help"],
            (
                "run",
                "list",
                "metadata",
                "download",
                "download-many",
                "snapshot",
                "discover",
            ),
        ),
        (["run", "--help"], ("--config", "WT fallbacks", "--dry-run")),
        (["list", "--help"], ("--source", "--query", "--limit", "--format")),
        (["metadata", "--help"], ("--source", "--dataset-id", "--format")),
        (
            ["download", "--help"],
            (
                "--source",
                "--dataset-id",
                "--output-dir",
                "--wt-sequence",
                "--wt-score",
                "--add-relative-score",
                "--overwrite",
            ),
        ),
        (
            ["download-many", "--help"],
            (
                "--source",
                "--dataset-id",
                "--output-dir",
                "DATASET_ID=SEQUENCE",
                "DATASET_ID=SCORE",
                "--add-relative-score",
                "--overwrite",
            ),
        ),
        (["snapshot", "--help"], ("fetch", "extract")),
        (
            ["snapshot", "fetch", "--help"],
            ("--latest", "--record", "--cache-dir", "--format"),
        ),
        (
            ["snapshot", "extract", "--help"],
            (
                "--latest",
                "--record",
                "--dataset-id",
                "--include-superseded",
                "--format",
                "--output",
                "1.9 GB",
            ),
        ),
        (
            ["discover", "--help"],
            (
                "--main-json",
                "--snapshot",
                "--query",
                "--include-superseded",
                "--format",
                "--output",
            ),
        ),
)


def test_help_exits_successfully() -> None:
    for arguments, expected_text in HELP_CASES:
        result = RUNNER.invoke(cli_module.app, arguments)

        assert result.exit_code == 0, arguments
        assert result.stderr == "", arguments
        output = result.stdout
        assert all(text in output for text in expected_text), arguments
        assert "--no-" not in output, arguments


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["unknown"],
        ["run"],
        ["run", "--config", "config.yml", "--only", "invalid"],
        ["list", "--source", "mavedb", "--limit", "0"],
        ["snapshot"],
        [
            "snapshot",
            "fetch",
            "--latest",
            "--record",
            "20840937",
        ],
        [
            "snapshot",
            "extract",
            "--latest",
            "--dataset-id",
            "urn:mavedb:00000003-a-0",
        ],
        [
            "discover",
            "--main-json",
            "main.json",
            "--snapshot",
            "latest",
            "--query",
            "BRCA1",
        ],
    ],
)
def test_invalid_usage_preserves_exit_status(
    arguments: list[str],
) -> None:
    result = RUNNER.invoke(cli_module.app, arguments)

    assert result.exit_code == 2
    assert result.stdout == ""
    assert result.stderr


@pytest.mark.parametrize(
    "command",
    [
        [],
        ["run"],
        ["list"],
        ["metadata"],
        ["download"],
        ["download-many"],
        ["snapshot"],
        ["snapshot", "fetch"],
        ["snapshot", "extract"],
        ["discover"],
    ],
)
def test_short_help_alias_is_available_for_every_help_page(
    command: list[str],
) -> None:
    result = RUNNER.invoke(cli_module.app, [*command, "-h"])

    assert result.exit_code == 0
    assert "Usage:" in result.stdout
    assert result.stderr == ""


@pytest.mark.parametrize(
    ("arguments", "expected_code"),
    [
        (["list", "--source", "mavedb", "--format", "json"], 0),
        (["--help"], 0),
        (["run", "--only", "invalid"], 2),
    ],
)
def test_main_argv_compatibility_wrapper(
    arguments: list[str],
    expected_code: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli_module, "list_datasets", lambda *args, **kwargs: [])
    monkeypatch.setattr(cli_module.logging, "basicConfig", lambda **kwargs: None)

    if expected_code == 0 and arguments != ["--help"]:
        assert cli_module.main(arguments) == 0
    else:
        with pytest.raises(SystemExit) as exc_info:
            cli_module.main(arguments)
        assert exc_info.value.code == expected_code


def test_public_typer_app_disables_completion_options() -> None:
    assert cli_module.app.info.name == "dms-parser"
    assert cli_module.snapshot_app.info.name == "snapshot"

    result = RUNNER.invoke(cli_module.app, ["--help"])

    assert result.exit_code == 0
    assert "--install-completion" not in result.stdout
    assert "--show-completion" not in result.stdout


@pytest.mark.parametrize(
    "arguments",
    [
        ["list", "--source", "unknown"],
        ["list", "--source", "mavedb", "--format", "yaml"],
        [
            "list",
            "--source",
            "proteingym",
            "--variant-type",
            "deletions",
        ],
        ["run", "--config", "config.yml", "--log-level", "TRACE"],
        [
            "download",
            "--source",
            "mavedb",
            "--dataset-id",
            "urn:mavedb:00000001-a-1",
            "--output-dir",
            "output",
            "--acquisition",
            "bulk",
        ],
    ],
)
def test_invalid_enum_values_are_usage_errors(arguments: list[str]) -> None:
    result = RUNNER.invoke(cli_module.app, arguments)

    assert result.exit_code == 2
    assert result.stdout == ""
    assert result.stderr


@pytest.mark.parametrize(
    "arguments",
    [
        ["list"],
        ["metadata", "--source", "mavedb"],
        [
            "download",
            "--source",
            "mavedb",
            "--dataset-id",
            "urn:mavedb:00000001-a-1",
        ],
        ["snapshot", "extract", "--latest"],
        ["discover", "--main-json", "main.json"],
    ],
)
def test_missing_required_options_are_usage_errors(arguments: list[str]) -> None:
    result = RUNNER.invoke(cli_module.app, arguments)

    assert result.exit_code == 2
    assert result.stdout == ""
    assert result.stderr


def test_run_short_config_alias_forwards_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yml"
    calls: dict[str, Any] = {}
    monkeypatch.setattr(
        cli_module,
        "load_pipeline_config",
        lambda path: calls.setdefault("path", path) or {"output": {}},
    )
    monkeypatch.setattr(
        cli_module,
        "run_pipeline",
        lambda config, **kwargs: PipelineResult(summary=[]),
    )
    monkeypatch.setattr(cli_module.logging, "basicConfig", lambda **kwargs: None)

    result = RUNNER.invoke(cli_module.app, ["run", "-c", str(config_path)])

    assert result.exit_code == 0
    assert calls["path"] == config_path


@pytest.mark.parametrize("option", ["--output-dir", "--cache-dir"])
def test_download_rejects_empty_path_values_before_side_effects(
    option: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = [
        "download",
        "--source",
        "proteingym",
        "--dataset-id",
        "ASSAY_1",
        "--output-dir",
        "output",
    ]
    if option == "--output-dir":
        arguments[-1] = ""
    else:
        arguments.extend([option, ""])
    monkeypatch.setattr(
        cli_module,
        "download_and_standardize_dataset",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("empty path reached the download API")
        ),
    )

    result = RUNNER.invoke(cli_module.app, arguments)

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "non-empty path" in result.stderr


def test_cli_runner_preserves_stdout_and_stderr_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_logger = logging.getLogger()
    monkeypatch.setattr(root_logger, "handlers", [])
    monkeypatch.setattr(
        cli_module,
        "load_pipeline_config",
        lambda path: (_ for _ in ()).throw(
            SourceConfigurationError("invalid configuration")
        ),
    )

    result = RUNNER.invoke(
        cli_module.app,
        ["run", "--config", "invalid.yml"],
    )

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "Configuration error: invalid configuration" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_runner_propagates_unexpected_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cause = RuntimeError("unexpected")
    monkeypatch.setattr(
        cli_module,
        "list_datasets",
        lambda *args, **kwargs: (_ for _ in ()).throw(cause),
    )
    monkeypatch.setattr(cli_module.logging, "basicConfig", lambda **kwargs: None)

    with pytest.raises(RuntimeError) as exc_info:
        RUNNER.invoke(
            cli_module.app,
            ["list", "--source", "mavedb"],
            catch_exceptions=False,
        )

    assert exc_info.value is cause


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
    "option",
    [
        ["--variant-type", "substitutions"],
        ["--cache-dir", "unused-cache"],
        ["--refresh"],
    ],
)
def test_mavedb_rejects_proteingym_options_before_side_effects(
    option: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "output" / "result.txt"
    arguments = ["list", "--source", "mavedb"]
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


def test_expected_list_failures_return_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cause = CatalogError("catalog failed")
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


def test_expected_snapshot_failures_return_one(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cause = InvalidCacheEntryError("cache failed")
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


def _snapshot_table_result(
    cache_root: Path,
    *,
    counts: bool = True,
    cache_hit: bool = False,
    superseded: bool = False,
) -> MaveDBSnapshotTableExtractionResult:
    """Return one representative raw snapshot-table extraction result."""
    snapshot = _snapshot_result(cache_root)
    table_root = (
        cache_root
        / "mavedb"
        / "snapshot_tables"
        / "20840937"
        / SNAPSHOT_TABLE_ID.replace(":", "-")
    )
    return MaveDBSnapshotTableExtractionResult(
        record=snapshot.record,
        tables=(
            MaveDBSnapshotTable(
                dataset_id=SNAPSHOT_TABLE_ID,
                scores_path=table_root / "scores.csv",
                counts_path=table_root / "counts.csv" if counts else None,
                is_superseded=superseded,
                cache_hit=cache_hit,
            ),
        ),
    )


def _forbidden_snapshot_extract_side_effect(
    *args: object,
    **kwargs: object,
) -> None:
    """Fail when snapshot extraction invokes an unrelated API."""
    raise AssertionError("snapshot extract invoked a forbidden operation")


def _patch_snapshot_extract_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    extract: Any,
) -> None:
    """Install the common local fetch and extraction CLI test doubles."""
    monkeypatch.setattr(
        cli_module,
        "fetch_mavedb_snapshot",
        lambda selector, **kwargs: _snapshot_result(kwargs["cache"].root),
    )
    monkeypatch.setattr(cli_module, "extract_mavedb_snapshot_tables", extract)
    monkeypatch.setattr(cli_module.logging, "basicConfig", lambda **kwargs: None)


def test_snapshot_extract_latest_calls_each_layer_once_and_renders_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cache_root = tmp_path / "default-cache"
    calls: dict[str, Any] = {"fetch_count": 0, "extract_count": 0}
    monkeypatch.setattr(cli_module, "_default_cache_root", lambda: cache_root)

    def fetch(selector: str, **kwargs: Any) -> MaveDBSnapshot:
        calls["fetch_count"] += 1
        calls["selector"] = selector
        calls["fetch_cache"] = kwargs["cache"]
        calls["fetch_refresh"] = kwargs["refresh"]
        return _snapshot_result(kwargs["cache"].root)

    def extract(
        snapshot: MaveDBSnapshot,
        dataset_ids: object,
        **kwargs: Any,
    ) -> MaveDBSnapshotTableExtractionResult:
        calls["extract_count"] += 1
        calls["snapshot"] = snapshot
        calls["dataset_ids"] = dataset_ids
        calls["extract_cache"] = kwargs["cache"]
        calls["include_superseded"] = kwargs["include_superseded"]
        calls["extract_refresh"] = kwargs["refresh"]
        return _snapshot_table_result(kwargs["cache"].root)

    monkeypatch.setattr(cli_module, "fetch_mavedb_snapshot", fetch)
    monkeypatch.setattr(cli_module, "extract_mavedb_snapshot_tables", extract)
    for name in (
        "get_dataset_metadata",
        "download_and_standardize_dataset",
        "download_and_standardize_datasets",
    ):
        monkeypatch.setattr(
            cli_module,
            name,
            _forbidden_snapshot_extract_side_effect,
        )
    monkeypatch.setattr(cli_module.logging, "basicConfig", lambda **kwargs: None)

    assert cli_module.main(SNAPSHOT_EXTRACT_LATEST) == 0

    assert calls["fetch_count"] == calls["extract_count"] == 1
    assert calls["selector"] == "latest"
    assert calls["dataset_ids"] == (SNAPSHOT_TABLE_ID,)
    assert calls["fetch_cache"] is calls["extract_cache"]
    assert calls["fetch_cache"].root == cache_root
    assert calls["fetch_refresh"] is calls["extract_refresh"] is False
    assert calls["include_superseded"] is False
    output = capsys.readouterr().out
    for text in (
        "record_id: 20840937",
        "doi: 10.5281/zenodo.20840937",
        "archive_filename: mavedb-dump.test.tar.gz",
        "dataset_count: 1",
        f"dataset_id: {SNAPSHOT_TABLE_ID}",
        "status: current",
        "scores_path:",
        "counts_path:",
        "cache: extracted",
    ):
        assert text in output


def test_snapshot_extract_record_forwards_options_and_renders_json_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cache_root = tmp_path / "cache"
    output_path = tmp_path / "nested" / "tables.json"
    calls: dict[str, Any] = {}
    logging_calls: list[dict[str, Any]] = []

    def fetch(selector: str, **kwargs: Any) -> MaveDBSnapshot:
        calls.update(selector=selector, fetch_kwargs=kwargs)
        return _snapshot_result(kwargs["cache"].root)

    def extract(
        snapshot: MaveDBSnapshot,
        dataset_ids: object,
        **kwargs: Any,
    ) -> MaveDBSnapshotTableExtractionResult:
        calls.update(
            snapshot=snapshot,
            dataset_ids=dataset_ids,
            extract_kwargs=kwargs,
        )
        return _snapshot_table_result(
            kwargs["cache"].root,
            counts=False,
            cache_hit=True,
            superseded=True,
        )

    monkeypatch.setattr(cli_module, "fetch_mavedb_snapshot", fetch)
    monkeypatch.setattr(cli_module, "extract_mavedb_snapshot_tables", extract)
    monkeypatch.setattr(
        cli_module.logging,
        "basicConfig",
        lambda **kwargs: logging_calls.append(kwargs),
    )

    assert cli_module.main(
        [
            "snapshot",
            "extract",
            "--record",
            "20840937",
            "--dataset-id",
            "urn:mavedb:00000003-a-2",
            "--dataset-id",
            SNAPSHOT_TABLE_ID,
            "--include-superseded",
            "--cache-dir",
            str(cache_root),
            "--refresh",
            "--format",
            "json",
            "--output",
            str(output_path),
            "--log-level",
            "DEBUG",
        ]
    ) == 0

    assert capsys.readouterr().out == ""
    assert calls["selector"] == "20840937"
    assert calls["dataset_ids"] == (
        "urn:mavedb:00000003-a-2",
        SNAPSHOT_TABLE_ID,
    )
    assert calls["fetch_kwargs"]["cache"] is calls["extract_kwargs"]["cache"]
    assert calls["fetch_kwargs"]["cache"].root == cache_root
    assert calls["fetch_kwargs"]["refresh"] is True
    assert calls["extract_kwargs"]["refresh"] is True
    assert calls["extract_kwargs"]["include_superseded"] is True
    assert logging_calls[0]["level"] == logging.DEBUG
    values = json.loads(output_path.read_text(encoding="utf-8"))
    assert list(values) == [
        "record_id",
        "doi",
        "concept_doi",
        "publication_date",
        "archive",
        "dataset_count",
        "tables",
    ]
    assert values["archive"] == {
        "filename": "mavedb-dump.test.tar.gz",
        "size": 123,
        "checksum": "md5:900150983cd24fb0d6963f7d28e17f72",
    }
    assert values["tables"][0]["counts_path"] is None
    assert values["tables"][0]["is_superseded"] is True
    assert values["tables"][0]["cache_hit"] is True
    assert ".extract-" not in output_path.read_text(encoding="utf-8")


def test_snapshot_extract_counts_absent_and_cache_hit_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_snapshot_extract_dependencies(
        monkeypatch,
        lambda snapshot, dataset_ids, **kwargs: _snapshot_table_result(
            kwargs["cache"].root,
            counts=False,
            cache_hit=True,
        ),
    )

    assert cli_module.main(SNAPSHOT_EXTRACT_LATEST) == 0

    output = capsys.readouterr().out
    assert "counts_path: not available" in output
    assert "cache: hit" in output


def test_snapshot_extract_duplicate_ids_fail_before_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "FilesystemCache",
        "fetch_mavedb_snapshot",
        "extract_mavedb_snapshot_tables",
    ):
        monkeypatch.setattr(
            cli_module,
            name,
            _forbidden_snapshot_extract_side_effect,
        )

    with pytest.raises(SystemExit) as exc_info:
        cli_module.main(
            [*SNAPSHOT_EXTRACT_LATEST, "--dataset-id", SNAPSHOT_TABLE_ID]
        )

    assert exc_info.value.code == 2


def test_snapshot_extract_help_performs_no_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("fetch_mavedb_snapshot", "extract_mavedb_snapshot_tables"):
        monkeypatch.setattr(
            cli_module,
            name,
            _forbidden_snapshot_extract_side_effect,
        )

    with pytest.raises(SystemExit) as exc_info:
        cli_module.main(["snapshot", "extract", "--help"])

    assert exc_info.value.code == 0


def test_expected_snapshot_extract_failures_return_one_without_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cause = MaveDBSnapshotTableError("table failed")
    output_path = tmp_path / "tables.json"
    _patch_snapshot_extract_dependencies(
        monkeypatch,
        lambda *args, **kwargs: (_ for _ in ()).throw(cause),
    )

    assert cli_module.main(
        [*SNAPSHOT_EXTRACT_LATEST, "--output", str(output_path)]
    ) == 1
    assert capsys.readouterr().out == ""
    assert not output_path.exists()


def test_unexpected_snapshot_extract_error_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cause = RuntimeError("unexpected")
    _patch_snapshot_extract_dependencies(
        monkeypatch,
        lambda *args, **kwargs: (_ for _ in ()).throw(cause),
    )

    with pytest.raises(RuntimeError) as exc_info:
        cli_module.main(SNAPSHOT_EXTRACT_LATEST)

    assert exc_info.value is cause


def _discovery_result(*, empty: bool = False) -> MaveDBDiscoveryResult:
    """Return deterministic grouped discovery data for CLI tests."""
    experiments: tuple[MaveDBDiscoveredExperiment, ...] = ()
    if not empty:
        experiments = (
            MaveDBDiscoveredExperiment(
                experiment_set_id="urn:mavedb:00000001-a",
                experiment_id="urn:mavedb:00000001-a",
                title="BRCA1 saturation editing",
                score_sets=(
                    MaveDBDiscoveredScoreSet(
                        dataset_id="urn:mavedb:00000001-a-1",
                        title="Current BRCA1 scores",
                        n_variants=123,
                        targets=("BRCA1", "P38398"),
                        is_superseded=False,
                    ),
                    MaveDBDiscoveredScoreSet(
                        dataset_id="urn:mavedb:00000001-a-2",
                        title="Earlier BRCA1 scores",
                        n_variants=None,
                        targets=("BRCA1",),
                        is_superseded=True,
                    ),
                ),
            ),
        )
    return MaveDBDiscoveryResult(
        query="BRCA1",
        snapshot_title="MaveDB public data dump v5",
        as_of="2026-06-24T18:13:01Z",
        experiments=experiments,
    )


def _install_fake_discovery_catalog(
    monkeypatch: pytest.MonkeyPatch,
    calls: dict[str, Any],
    *,
    empty: bool = False,
) -> None:
    """Install a catalog fake that records loading and search arguments."""

    class FakeCatalog:
        @classmethod
        def from_file(cls, path: Path) -> FakeCatalog:
            calls["catalog_path"] = path
            return cls()

        def search_by_gene(
            self,
            query: str,
            *,
            include_superseded: bool = False,
        ) -> MaveDBDiscoveryResult:
            calls["query"] = query
            calls["include_superseded"] = include_superseded
            return _discovery_result(empty=empty)

    monkeypatch.setattr(cli_module, "MaveDBBulkCatalog", FakeCatalog)


def _forbidden_discovery_side_effect(
    *args: object,
    **kwargs: object,
) -> None:
    """Fail when discovery invokes an unrelated side effect."""
    raise AssertionError("discovery invoked a forbidden operation")


def test_discover_local_forwards_search_and_renders_grouped_text(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: dict[str, Any] = {}
    logging_calls: list[dict[str, Any]] = []
    main_json = Path("~") / "snapshots" / "main.json"
    _install_fake_discovery_catalog(monkeypatch, calls)
    monkeypatch.setattr(
        cli_module,
        "fetch_mavedb_snapshot",
        _forbidden_discovery_side_effect,
    )
    monkeypatch.setattr(
        cli_module,
        "download_and_standardize_dataset",
        _forbidden_discovery_side_effect,
    )
    monkeypatch.setattr(
        cli_module,
        "download_and_standardize_datasets",
        _forbidden_discovery_side_effect,
    )
    monkeypatch.setattr(
        cli_module.logging,
        "basicConfig",
        lambda **kwargs: logging_calls.append(kwargs),
    )

    assert cli_module.main(
        [
            "discover",
            "--main-json",
            str(main_json),
            "--query",
            "  BrCa1  ",
            "--include-superseded",
            "--log-level",
            "DEBUG",
        ]
    ) == 0

    assert calls == {
        "catalog_path": main_json.expanduser(),
        "query": "  BrCa1  ",
        "include_superseded": True,
    }
    assert logging_calls[0]["level"] == logging.DEBUG
    output = capsys.readouterr().out
    for text in (
        "query: BRCA1",
        "snapshot_title: MaveDB public data dump v5",
        "as_of: 2026-06-24T18:13:01Z",
        "source: local",
        "experiment_count: 1",
        "score_set_count: 2",
        "experiment: urn:mavedb:00000001-a",
        "experiment_set: urn:mavedb:00000001-a",
        "score_set: urn:mavedb:00000001-a-1",
        "targets: BRCA1, P38398",
        "n_variants: 123",
        "status: current",
        "score_set: urn:mavedb:00000001-a-2",
        "n_variants: -",
        "status: superseded",
    ):
        assert text in output


def test_discover_local_json_output_file_is_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: dict[str, Any] = {}
    output_path = tmp_path / "nested" / "discovery.json"
    main_json = tmp_path / "main.json"
    _install_fake_discovery_catalog(monkeypatch, calls)
    monkeypatch.setattr(
        cli_module,
        "fetch_mavedb_snapshot",
        _forbidden_discovery_side_effect,
    )
    monkeypatch.setattr(cli_module.logging, "basicConfig", lambda **kwargs: None)

    assert cli_module.main(
        [
            "discover",
            "--main-json",
            str(main_json),
            "-q",
            "BRCA1",
            "--format",
            "json",
            "--output",
            str(output_path),
        ]
    ) == 0

    assert capsys.readouterr().out == ""
    values = json.loads(output_path.read_text(encoding="utf-8"))
    assert list(values) == [
        "query",
        "snapshot_title",
        "as_of",
        "experiment_count",
        "score_set_count",
        "source",
        "experiments",
    ]
    assert values["source"] == {"kind": "local"}
    assert str(main_json.resolve()) not in output_path.read_text(encoding="utf-8")
    assert values["experiments"][0]["score_sets"][0]["targets"] == [
        "BRCA1",
        "P38398",
    ]


@pytest.mark.parametrize(
    ("selector", "extra_arguments", "expected_refresh"),
    [
        ("latest", [], False),
        ("20840937", ["--refresh"], True),
    ],
)
def test_discover_snapshot_fetches_once_and_preserves_provenance(
    selector: str,
    extra_arguments: list[str],
    expected_refresh: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: dict[str, Any] = {"fetch_count": 0}
    cache_root = tmp_path / "cache"
    _install_fake_discovery_catalog(monkeypatch, calls)

    def fetch(value: str, **kwargs: Any) -> MaveDBSnapshot:
        calls["fetch_count"] += 1
        calls["selector"] = value
        calls["cache"] = kwargs["cache"]
        calls["refresh"] = kwargs["refresh"]
        return _snapshot_result(kwargs["cache"].root, cache_hit=True)

    monkeypatch.setattr(cli_module, "fetch_mavedb_snapshot", fetch)
    monkeypatch.setattr(
        cli_module,
        "download_and_standardize_dataset",
        _forbidden_discovery_side_effect,
    )
    monkeypatch.setattr(
        cli_module,
        "download_and_standardize_datasets",
        _forbidden_discovery_side_effect,
    )
    monkeypatch.setattr(cli_module.logging, "basicConfig", lambda **kwargs: None)
    arguments = [
        "discover",
        "--snapshot",
        selector,
        "--query",
        "BRCA1",
        "--cache-dir",
        str(cache_root),
        "--format",
        "json",
    ]

    assert cli_module.main(arguments + extra_arguments) == 0

    assert calls["fetch_count"] == 1
    assert calls["selector"] == selector
    assert calls["cache"].root == cache_root
    assert calls["refresh"] is expected_refresh
    assert calls["catalog_path"] == _snapshot_result(cache_root).main_json_path
    values = json.loads(capsys.readouterr().out)
    assert values["source"] == {
        "kind": "snapshot",
        "record_id": "20840937",
        "doi": "10.5281/zenodo.20840937",
        "concept_doi": "10.5281/zenodo.11201736",
        "publication_date": "2026-06-24",
        "filename": "mavedb-dump.test.tar.gz",
        "size": 123,
        "checksum": "md5:900150983cd24fb0d6963f7d28e17f72",
    }
    assert "cache_hit" not in values["source"]
    assert "main_json_path" not in values["source"]


@pytest.mark.parametrize("option", ["--refresh", "--cache-dir"])
def test_discover_local_rejects_snapshot_options_before_side_effects(
    option: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli_module,
        "FilesystemCache",
        _forbidden_discovery_side_effect,
    )
    monkeypatch.setattr(
        cli_module,
        "fetch_mavedb_snapshot",
        _forbidden_discovery_side_effect,
    )
    monkeypatch.setattr(
        cli_module,
        "MaveDBBulkCatalog",
        _forbidden_discovery_side_effect,
    )
    arguments = [
        "discover",
        "--main-json",
        str(tmp_path / "main.json"),
        "--query",
        "BRCA1",
        option,
    ]
    if option == "--cache-dir":
        arguments.append(str(tmp_path / "cache"))

    with pytest.raises(SystemExit) as exc_info:
        cli_module.main(arguments)

    assert exc_info.value.code == 2


def test_discover_empty_result_is_successful(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: dict[str, Any] = {}
    _install_fake_discovery_catalog(monkeypatch, calls, empty=True)
    monkeypatch.setattr(
        cli_module,
        "fetch_mavedb_snapshot",
        _forbidden_discovery_side_effect,
    )
    monkeypatch.setattr(cli_module.logging, "basicConfig", lambda **kwargs: None)

    assert cli_module.main(
        [
            "discover",
            "--main-json",
            str(tmp_path / "main.json"),
            "--query",
            "missing",
        ]
    ) == 0

    output = capsys.readouterr().out
    assert "experiment_count: 0" in output
    assert "score_set_count: 0" in output
    assert "No matching score sets found." in output


def test_expected_discovery_failures_return_one_without_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cause = CatalogError("invalid catalog")
    class FailingCatalog:
        @classmethod
        def from_file(cls, path: Path) -> None:
            raise cause

    monkeypatch.setattr(cli_module, "MaveDBBulkCatalog", FailingCatalog)
    monkeypatch.setattr(cli_module.logging, "basicConfig", lambda **kwargs: None)

    assert cli_module.main(
        [
            "discover",
            "--main-json",
            str(tmp_path / "main.json"),
            "--query",
            "BRCA1",
        ]
    ) == 1
    assert capsys.readouterr().out == ""


def test_unexpected_discovery_exception_propagates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cause = RuntimeError("unexpected")

    class FailingCatalog:
        @classmethod
        def from_file(cls, path: Path) -> None:
            raise cause

    monkeypatch.setattr(cli_module, "MaveDBBulkCatalog", FailingCatalog)
    monkeypatch.setattr(cli_module.logging, "basicConfig", lambda **kwargs: None)

    with pytest.raises(RuntimeError) as exc_info:
        cli_module.main(
            [
                "discover",
                "--main-json",
                str(tmp_path / "main.json"),
                "--query",
                "BRCA1",
            ]
        )

    assert exc_info.value is cause


def test_discover_help_does_not_resolve_a_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli_module,
        "fetch_mavedb_snapshot",
        _forbidden_discovery_side_effect,
    )

    with pytest.raises(SystemExit) as exc_info:
        cli_module.main(["discover", "--help"])

    assert exc_info.value.code == 0


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
    assert "--acquisition" in output
    assert "api" in output and "snapshot" in output
    assert "--snapshot-record" in output
    assert "--include-superseded" in output
    assert "MaveDB-only" in output
    assert "prefetched" in output
    assert "1.9 GB" in output


@pytest.mark.parametrize(
    ("source", "dataset_id"),
    [
        ("mavedb", "not-a-permanent-urn"),
        ("mavedb", "urn:mavedb:00000001-a-0"),
        ("proteingym", "ASSAY.csv"),
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


@pytest.mark.parametrize("command", ["download", "download-many"])
@pytest.mark.parametrize(
    ("source", "extra"),
    [
        ("mavedb", ["--acquisition", "snapshot"]),
        (
            "mavedb",
            ["--acquisition", "api", "--snapshot-record", "20840937"],
        ),
        ("mavedb", ["--include-superseded"]),
        ("proteingym", ["--acquisition", "api"]),
        (
            "proteingym",
            [
                "--acquisition",
                "snapshot",
                "--snapshot-record",
                "20840937",
            ],
        ),
        (
            "mavedb",
            ["--acquisition", "snapshot", "--snapshot-record", "latest"],
        ),
    ],
)
def test_download_acquisition_options_fail_before_side_effects(
    command: str,
    source: str,
    extra: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forbidden = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("invalid acquisition constructed a cache or downloaded")
    )
    monkeypatch.setattr(cli_module, "FilesystemCache", forbidden)
    monkeypatch.setattr(
        cli_module,
        "download_and_standardize_dataset",
        forbidden,
    )
    monkeypatch.setattr(
        cli_module,
        "download_and_standardize_datasets",
        forbidden,
    )
    dataset_id = (
        "urn:mavedb:00000001-a-1"
        if source == "mavedb"
        else "ASSAY_1"
    )
    arguments = [
        command,
        "--source",
        source,
        "--dataset-id",
        dataset_id,
        "--output-dir",
        str(tmp_path / "output"),
        *extra,
    ]

    with pytest.raises(SystemExit) as exc_info:
        cli_module.main(arguments)

    assert exc_info.value.code == 2
    assert not (tmp_path / "output").exists()


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
        "wt_sequence": None,
        "wt_score": None,
        "add_relative_score": False,
        "relative_method": "log_ratio",
        "relative_output_col": "score_log_ratio",
        "add_binary_label": False,
        "delta": 0.1,
        "higher_is_better": True,
        "binary_output_col": "score_binary_like",
        "overwrite": False,
        "acquisition": None,
        "snapshot_record_id": None,
        "include_superseded": False,
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
            "--wt-sequence",
            "MKT",
            "--wt-score",
            "1.25",
            "--add-relative-score",
            "--relative-method",
            "difference",
            "--relative-output-col",
            "score_difference",
            "--add-binary-label",
            "--delta",
            "0.2",
            "--higher-is-better",
            "false",
            "--binary-output-col",
            "activity_label",
            "--overwrite",
            "--acquisition",
            "snapshot",
            "--snapshot-record",
            "20840937",
            "--include-superseded",
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
    assert calls["wt_sequence"] == "MKT"
    assert calls["wt_score"] == 1.25
    assert calls["add_relative_score"] is True
    assert calls["relative_method"] == "difference"
    assert calls["relative_output_col"] == "score_difference"
    assert calls["add_binary_label"] is True
    assert calls["delta"] == 0.2
    assert calls["higher_is_better"] is False
    assert calls["binary_output_col"] == "activity_label"
    assert calls["overwrite"] is True
    assert calls["acquisition"] == "snapshot"
    assert calls["snapshot_record_id"] == "20840937"
    assert calls["include_superseded"] is True
    assert len(logging_calls) == 1
    assert logging_calls[0]["level"] == logging.DEBUG
    assert "stream" not in logging_calls[0]


def test_download_snapshot_missing_cache_does_not_fetch_implicitly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "output"
    root_logger = logging.getLogger()
    monkeypatch.setattr(root_logger, "handlers", [])
    monkeypatch.setattr(
        cli_module,
        "fetch_mavedb_snapshot",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("download invoked implicit snapshot fetching")
        ),
    )

    def missing_snapshot(
        source: str,
        dataset_id: str,
        **kwargs: Any,
    ) -> None:
        assert source == "mavedb"
        assert kwargs["acquisition"] == "snapshot"
        assert kwargs["snapshot_record_id"] == "20840937"
        raise MaveDBSnapshotError(
            "MaveDB snapshot 20840937 is not cached; run "
            "'dms-parser snapshot fetch --record 20840937' first."
        )

    monkeypatch.setattr(
        cli_module,
        "download_and_standardize_dataset",
        missing_snapshot,
    )

    result = RUNNER.invoke(
        cli_module.app,
        [
            "download",
            "--source",
            "mavedb",
            "--dataset-id",
            "urn:mavedb:00000001-a-1",
            "--output-dir",
            str(output_dir),
            "--acquisition",
            "snapshot",
            "--snapshot-record",
            "20840937",
        ],
    )

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "snapshot fetch --record 20840937" in result.stderr
    assert "Traceback" not in result.stderr
    assert not output_dir.exists()


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
    assert "--acquisition" in output
    assert "api" in output and "snapshot" in output
    assert "--snapshot-record" in output
    assert "--include-superseded" in output
    assert "1.9 GB" in output


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
    assert calls["acquisition"] is None
    assert calls["snapshot_record_id"] is None
    assert calls["include_superseded"] is False
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
            "--wt-sequence",
            "urn:mavedb:00000001-a-1=MKT",
            "--wt-score",
            "urn:mavedb:00000001-a-1=-0.5",
            "--add-relative-score",
            "--relative-method",
            "difference",
            "--relative-output-col",
            "score_difference",
            "--overwrite",
            "--acquisition",
            "snapshot",
            "--snapshot-record",
            "20840937",
            "--include-superseded",
            "--log-level",
            "DEBUG",
        ]
    ) == 0

    assert calls["output_dir"] == output_dir.expanduser()
    assert calls["cache"].root == cache_dir.expanduser()
    assert calls["refresh"] is True
    assert calls["drop_failed"] is True
    assert calls["add_wildtype_row"] is True
    assert calls["wt_sequence"] == {"urn:mavedb:00000001-a-1": "MKT"}
    assert calls["wt_score"] == {"urn:mavedb:00000001-a-1": -0.5}
    assert calls["add_relative_score"] is True
    assert calls["relative_method"] == "difference"
    assert calls["relative_output_col"] == "score_difference"
    assert calls["overwrite"] is True
    assert calls["acquisition"] == "snapshot"
    assert calls["snapshot_record_id"] == "20840937"
    assert calls["include_superseded"] is True
    assert logging_calls[0]["level"] == logging.DEBUG


@pytest.mark.parametrize(
    "fallback_arguments",
    [
        ["--wt-score", "missing-equals"],
        ["--wt-score", "ASSAY_1=1.0", "--wt-score", "ASSAY_1=1.0"],
        ["--wt-score", "ASSAY_2=1.0"],
        ["--wt-sequence", "ASSAY_1="],
    ],
)
def test_download_many_rejects_invalid_wt_mappings_before_side_effects(
    fallback_arguments: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli_module,
        "download_and_standardize_datasets",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("invalid fallback reached acquisition")
        ),
    )
    output_dir = tmp_path / "output"

    with pytest.raises(SystemExit) as exc_info:
        cli_module.main(
            [
                "download-many",
                "--source",
                "proteingym",
                "--dataset-id",
                "ASSAY_1",
                "--output-dir",
                str(output_dir),
                *fallback_arguments,
            ]
        )

    assert exc_info.value.code == 2
    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("command", "extra"),
    [
        (
            "download",
            ["--add-relative-score", "--relative-output-col", "score_raw"],
        ),
        (
            "download-many",
            [
                "--add-relative-score",
                "--add-binary-label",
                "--relative-output-col",
                "generated",
                "--binary-output-col",
                "generated",
            ],
        ),
    ],
)
def test_cli_rejects_output_collisions_before_domain_side_effects(
    command: str,
    extra: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forbidden = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("invalid output column reached download API")
    )
    monkeypatch.setattr(
        cli_module,
        "download_and_standardize_dataset",
        forbidden,
    )
    monkeypatch.setattr(
        cli_module,
        "download_and_standardize_datasets",
        forbidden,
    )
    output_dir = tmp_path / "output"

    with pytest.raises(SystemExit) as exc_info:
        cli_module.main(
            [
                command,
                "--source",
                "proteingym",
                "--dataset-id",
                "ASSAY_1",
                "--output-dir",
                str(output_dir),
                *extra,
            ]
        )

    assert exc_info.value.code == 2
    assert not output_dir.exists()


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

    assert {"pyyaml", "pyarrow", "typer"} <= runtime_names
    assert {"pyyaml", "pyarrow", "typer"}.isdisjoint(dev_names)
    assert "typer>=0.27.1,<0.28" in section_array("project", "dependencies")
