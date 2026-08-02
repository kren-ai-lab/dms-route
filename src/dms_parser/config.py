"""Loading and validation for configuration-driven DMS pipelines."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

from dms_parser.exceptions import SourceConfigurationError
from dms_parser.sources.proteingym_resources import (
    ProteinGymResource,
    get_proteingym_resource,
)

_MAVEDB_SCORE_SET_URN_PATTERN = re.compile(
    r"urn:mavedb:[0-9]{8}-(?:[a-z]+|0)-[1-9][0-9]*"
)


def validate_source_dataset_id(source: str, dataset_id: object) -> None:
    """Validate one source-specific dataset identifier without performing I/O."""
    if source not in {"mavedb", "proteingym"}:
        raise SourceConfigurationError(
            "source must be 'mavedb' or 'proteingym'."
        )
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise SourceConfigurationError(
            f"The {source!r} dataset_id must be a non-empty string."
        )
    if source == "proteingym" and dataset_id.lower().endswith(".csv"):
        raise SourceConfigurationError(
            "ProteinGym dataset_id must be the canonical DMS_id, not a "
            "source filename."
        )
    if (
        source == "mavedb"
        and _MAVEDB_SCORE_SET_URN_PATTERN.fullmatch(dataset_id) is None
    ):
        raise SourceConfigurationError(
            "MaveDB dataset_id must be a complete permanent score-set URN "
            "matching 'urn:mavedb:<8 digits>-<lowercase experiment letters "
            "or 0>-<positive index>'."
        )


def load_pipeline_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a pipeline configuration from a YAML file.

    The path is interpreted exactly as supplied. Relative paths therefore
    remain relative to the caller's current working directory.
    """
    try:
        config_path = Path(path)
    except (TypeError, ValueError) as exc:
        raise SourceConfigurationError(
            "Pipeline configuration path must be a string or pathlib.Path."
        ) from exc

    try:
        import yaml
    except ImportError as exc:
        raise SourceConfigurationError(
            "PyYAML is required to load pipeline configuration files."
        ) from exc

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise SourceConfigurationError(
            f"Pipeline configuration file does not exist: {config_path}"
        ) from exc
    except yaml.YAMLError as exc:
        raise SourceConfigurationError(
            f"Pipeline configuration file contains malformed YAML: {config_path}"
        ) from exc
    except (OSError, UnicodeError) as exc:
        raise SourceConfigurationError(
            f"Could not read pipeline configuration file {config_path}: {exc}"
        ) from exc

    if config is None or config == {}:
        raise SourceConfigurationError(
            f"Pipeline configuration file is empty: {config_path}"
        )
    return validate_pipeline_config(config)


def validate_pipeline_config(config: object) -> dict[str, Any]:
    """Validate the logical source configuration without performing I/O."""
    if not isinstance(config, dict):
        raise SourceConfigurationError(
            "The pipeline configuration root must be a YAML mapping."
        )
    if "proteingym" in config:
        _validate_proteingym_config(config["proteingym"])
    if "mavedb" in config:
        _validate_mavedb_config(config["mavedb"])
    output = config.get("output")
    if output is not None and not isinstance(output, dict):
        raise SourceConfigurationError("'output' must be a mapping when provided.")
    return cast(dict[str, Any], config)


def _validate_proteingym_config(config: object) -> ProteinGymResource:
    """Validate ProteinGym configuration and return its selected resource."""
    source_config = _validate_source_mapping(config, "proteingym")
    _reject_legacy_keys(
        source_config,
        "proteingym",
        {"enabled", "metadata_url", "benchmark_url"},
    )
    _validate_dir_base(source_config, "proteingym")
    if "resource" not in source_config:
        raise SourceConfigurationError(
            "ProteinGym configuration requires 'resource'."
        )
    resource_id = source_config["resource"]
    if not isinstance(resource_id, str) or not resource_id.strip():
        raise SourceConfigurationError(
            "ProteinGym 'resource' must be a non-empty string."
        )
    resource = get_proteingym_resource(
        resource_id,
        require_processing=True,
    )
    _validate_dataset_entries(source_config, "proteingym")
    return resource


def _validate_mavedb_config(config: object) -> None:
    """Validate MaveDB configuration and permanent score-set identifiers."""
    source_config = _validate_source_mapping(config, "mavedb")
    _reject_legacy_keys(source_config, "mavedb", {"enabled", "base_url"})
    _validate_dir_base(source_config, "mavedb")
    entries = _validate_dataset_entries(source_config, "mavedb")
    for entry in entries:
        validate_source_dataset_id("mavedb", entry["dataset_id"])


def _validate_source_mapping(
    config: object,
    source: str,
) -> dict[str, Any]:
    """Return a validated source mapping."""
    if not isinstance(config, dict):
        raise SourceConfigurationError(
            f"The {source!r} source section must be a mapping."
        )
    return cast(dict[str, Any], config)


def _reject_legacy_keys(
    config: dict[str, Any],
    source: str,
    keys: set[str],
) -> None:
    """Reject legacy source keys that are not part of the YAML contract."""
    present = sorted(keys.intersection(config))
    if present:
        joined = ", ".join(present)
        raise SourceConfigurationError(
            f"The {source!r} source section contains unsupported legacy "
            f"configuration keys: {joined}."
        )


def _validate_dir_base(config: dict[str, Any], source: str) -> None:
    """Validate a source output base directory."""
    if "dir_base" not in config:
        raise SourceConfigurationError(
            f"The {source!r} source section requires 'dir_base'."
        )
    value = config["dir_base"]
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise SourceConfigurationError(
            f"The {source!r} 'dir_base' must be a non-empty path."
        )


def _validate_dataset_entries(
    config: dict[str, Any],
    source: str,
) -> list[dict[str, Any]]:
    """Validate canonical dataset entries and reject duplicates."""
    if "datasets" not in config:
        raise SourceConfigurationError(
            f"The {source!r} source section requires 'datasets'."
        )
    entries = config["datasets"]
    if not isinstance(entries, list) or not entries:
        raise SourceConfigurationError(
            f"The {source!r} 'datasets' value must be a non-empty list."
        )
    default_build_kwargs = config.get("default_build_kwargs", {})
    if not isinstance(default_build_kwargs, dict):
        raise SourceConfigurationError(
            f"The {source!r} 'default_build_kwargs' value must be a mapping."
        )

    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise SourceConfigurationError(
                f"The {source!r} dataset entry at index {index} must be a mapping."
            )
        legacy_key = "filename" if source == "proteingym" else "urn"
        if legacy_key in entry:
            raise SourceConfigurationError(
                f"The {source!r} dataset key {legacy_key!r} is unsupported; "
                "use 'dataset_id'."
            )
        if "dataset_id" not in entry:
            raise SourceConfigurationError(
                f"The {source!r} dataset entry at index {index} requires "
                "'dataset_id'."
            )
        dataset_id = entry["dataset_id"]
        if not isinstance(dataset_id, str) or not dataset_id.strip():
            raise SourceConfigurationError(
                f"The {source!r} dataset_id at index {index} must be a "
                "non-empty string."
            )
        if source == "proteingym":
            validate_source_dataset_id(source, dataset_id)
        if dataset_id in seen:
            raise SourceConfigurationError(
                f"Duplicate {source} dataset_id {dataset_id!r}."
            )
        build_kwargs = entry.get("build_kwargs", {})
        if not isinstance(build_kwargs, dict):
            raise SourceConfigurationError(
                f"The {source!r} build_kwargs at index {index} must be a mapping."
            )
        seen.add(dataset_id)
        validated.append(cast(dict[str, Any], entry))
    return validated
