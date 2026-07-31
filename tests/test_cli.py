from __future__ import annotations

import ast
import logging
import re
from pathlib import Path
from typing import Any

import pytest

import dms_parser.cli as cli_module
from dms_parser import PipelineResult
from dms_parser.exceptions import SourceConfigurationError


def test_root_help_exits_successfully(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli_module.main(["--help"])

    assert exc_info.value.code == 0
    assert "run" in capsys.readouterr().out


def test_run_help_exits_successfully(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli_module.main(["run", "--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "--config" in output
    assert "--dry-run" in output


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["unknown"],
        ["run"],
        ["run", "--config", "config.yml", "--only", "invalid"],
        ["run", "--config", "config.yml", "--log-level", "TRACE"],
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
