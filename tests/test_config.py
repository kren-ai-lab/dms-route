from __future__ import annotations

import ast
from pathlib import Path

import pytest

import dms_parser.pipeline as pipeline_module
from dms_parser import (
    SourceConfigurationError,
    UnknownSourceResourceError,
    UnsupportedSourceResourceError,
    load_pipeline_config,
    validate_pipeline_config,
)


def _valid_config(root: Path) -> dict:
    """Return a minimal valid pipeline configuration."""
    return {
        "proteingym": {
            "resource": "dms_substitutions",
            "dir_base": root / "proteingym",
            "datasets": [{"dataset_id": "ASSAY_1"}],
        },
        "mavedb": {
            "dir_base": root / "mavedb",
            "datasets": [
                {"dataset_id": "urn:mavedb:00000001-a-1"}
            ],
        },
        "output": {"summary_dir": root / "summaries"},
    }


@pytest.mark.parametrize("path_kind", ["str", "path"])
def test_load_pipeline_config_accepts_str_and_path(
    path_kind: str,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        "\n".join(
            [
                "proteingym:",
                "  resource: dms_substitutions",
                "  dir_base: data/proteingym",
                "  datasets:",
                "    - dataset_id: ASSAY_1",
            ]
        ),
        encoding="utf-8",
    )
    supplied_path: str | Path = (
        str(config_path) if path_kind == "str" else config_path
    )

    config = load_pipeline_config(supplied_path)

    assert config["proteingym"]["resource"] == "dms_substitutions"
    assert config["proteingym"]["datasets"] == [
        {"dataset_id": "ASSAY_1"}
    ]


def test_load_pipeline_config_rejects_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.yml"

    with pytest.raises(
        SourceConfigurationError,
        match="does not exist",
    ) as exc_info:
        load_pipeline_config(missing_path)

    assert isinstance(exc_info.value.__cause__, FileNotFoundError)


def test_load_pipeline_config_rejects_unreadable_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text("output: {}\n", encoding="utf-8")
    original_open = Path.open

    def denied_open(path: Path, *args: object, **kwargs: object):
        if path == config_path:
            raise PermissionError("permission denied")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", denied_open)

    with pytest.raises(
        SourceConfigurationError,
        match="Could not read",
    ) as exc_info:
        load_pipeline_config(config_path)

    assert isinstance(exc_info.value.__cause__, PermissionError)


def test_load_pipeline_config_rejects_malformed_yaml(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "malformed.yml"
    config_path.write_text("proteingym: [\n", encoding="utf-8")

    with pytest.raises(
        SourceConfigurationError,
        match="malformed YAML",
    ):
        load_pipeline_config(config_path)


@pytest.mark.parametrize("content", ["", "   \n", "{}\n"])
def test_load_pipeline_config_rejects_empty_document(
    content: str,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "empty.yml"
    config_path.write_text(content, encoding="utf-8")

    with pytest.raises(SourceConfigurationError, match="empty"):
        load_pipeline_config(config_path)


def test_load_pipeline_config_rejects_non_mapping_root(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "list.yml"
    config_path.write_text("- one\n- two\n", encoding="utf-8")

    with pytest.raises(SourceConfigurationError, match="root.*mapping"):
        load_pipeline_config(config_path)


@pytest.mark.parametrize(
    "content",
    [
        "proteingym:\n  enabled: true\n",
        "mavedb:\n  dir_base: data\n  datasets: []\n",
    ],
)
def test_loading_applies_source_validation(
    content: str,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "invalid.yml"
    config_path.write_text(content, encoding="utf-8")

    with pytest.raises(SourceConfigurationError):
        load_pipeline_config(config_path)


@pytest.mark.parametrize(
    ("resource_id", "error_type"),
    [
        ("not_registered", UnknownSourceResourceError),
        ("dms_indels", UnsupportedSourceResourceError),
    ],
)
def test_validation_preserves_distinct_resource_errors(
    resource_id: str,
    error_type: type[SourceConfigurationError],
    tmp_path: Path,
) -> None:
    config = _valid_config(tmp_path)
    config["proteingym"]["resource"] = resource_id

    with pytest.raises(error_type):
        validate_pipeline_config(config)


def test_validation_performs_no_io_or_directory_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "output"

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Configuration validation attempted I/O.")

    monkeypatch.setattr(Path, "mkdir", forbidden)
    monkeypatch.setattr(pipeline_module.requests, "get", forbidden)
    monkeypatch.setattr(pipeline_module, "download_file", forbidden)

    validated = validate_pipeline_config(_valid_config(output_root))

    assert "proteingym" in validated
    assert "mavedb" in validated
    assert not output_root.exists()


def test_installed_package_sources_do_not_import_examples() -> None:
    source_root = Path(__file__).parents[1] / "src" / "dms_parser"
    violations: list[str] = []

    for source_path in source_root.rglob("*.py"):
        tree = ast.parse(
            source_path.read_text(encoding="utf-8"),
            filename=str(source_path),
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name == "examples" or name.startswith("examples.") for name in names):
                violations.append(str(source_path.relative_to(source_root)))

    assert violations == []
