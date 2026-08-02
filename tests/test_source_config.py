from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

import dms_parser.pipeline as pipeline_module
from dms_parser.config import validate_source_dataset_id
from dms_parser import (
    PROTEINGYM_RESOURCES,
    SourceConfigurationError,
    UnknownSourceResourceError,
    UnsupportedSourceResourceError,
    get_proteingym_resource,
    list_proteingym_resources,
    run_pipeline,
    validate_pipeline_config,
)
from dms_parser.sources.mavedb_catalog import MAVEDB_API_URL, MaveDBCatalog
from dms_parser.sources.proteingym_catalog import (
    PROTEINGYM_INDELS_URL,
    PROTEINGYM_SUBSTITUTIONS_URL,
    ProteinGymCatalog,
)

CONFIG_PATH = Path(__file__).parents[1] / "examples" / "pipeline.yml"

VALID_MAVEDB_SCORE_SET_URNS = (
    "urn:mavedb:00000001-a-1",
    "urn:mavedb:12345678-aa-42",
    "urn:mavedb:00000055-0-1",
)

INVALID_MAVEDB_DATASET_IDS = (
    "urn:mavedb:00000055",
    "urn:mavedb:00000055-a",
    "tmp:446191af-c1f8-4891-9f67-de152e9d328b",
    "urn:mavedb:calibration-11111111-2222-3333-aaaa-bbbbccccdddd",
    "urn:mavedb:collection-11111111-2222-3333-aaaa-bbbbccccdddd",
    "urn:mavedb:00000055-a-1trailing",
    " urn:mavedb:00000055-a-1",
    "urn:mavedb:00000055-a-1 ",
    "urn:mavedb:00000055-a-1/scores",
    r"urn:mavedb:00000055-a-1\scores",
    "../urn:mavedb:00000055-a-1",
    "urn:mavedb:00000055-a-1/../scores",
    "urn:mavedb:00000055-A-1",
    "urn:mavedb:00000055-a-0",
)


@pytest.mark.parametrize(
    ("source", "dataset_id"),
    [
        ("proteingym", "ASSAY_1"),
        ("mavedb", "urn:mavedb:00000001-a-1"),
    ],
)
def test_shared_source_dataset_id_validation_accepts_supported_identifiers(
    source: str,
    dataset_id: str,
) -> None:
    validate_source_dataset_id(source, dataset_id)


@pytest.mark.parametrize(
    ("source", "dataset_id", "message"),
    [
        ("proteingym", "ASSAY.csv", "canonical DMS_id"),
        ("mavedb", "not-a-urn", "complete permanent score-set URN"),
        ("unknown", "dataset", "source must be"),
        ("proteingym", "   ", "non-empty string"),
    ],
)
def test_shared_source_dataset_id_validation_rejects_invalid_identifiers(
    source: str,
    dataset_id: str,
    message: str,
) -> None:
    with pytest.raises(SourceConfigurationError, match=message):
        validate_source_dataset_id(source, dataset_id)

EXPECTED_RESOURCES = {
    "dms_substitutions": {
        "collection": "dms",
        "variant_type": "substitutions",
        "metadata_url": (
            "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/"
            "reference_files/DMS_substitutions.csv"
        ),
        "data_url": (
            "https://proteingym.s3.us-east-2.amazonaws.com/"
            "DMS_substitutions.parquet"
        ),
        "metadata_filename": "DMS_substitutions.csv",
        "data_filename": "DMS_substitutions.parquet",
        "processing_supported": True,
    },
    "dms_indels": {
        "collection": "dms",
        "variant_type": "indels",
        "metadata_url": (
            "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/"
            "reference_files/DMS_indels.csv"
        ),
        "data_url": (
            "https://proteingym.s3.us-east-2.amazonaws.com/DMS_indels.parquet"
        ),
        "metadata_filename": "DMS_indels.csv",
        "data_filename": "DMS_indels.parquet",
        "processing_supported": False,
    },
    "clinical_substitutions": {
        "collection": "clinical",
        "variant_type": "substitutions",
        "metadata_url": (
            "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/"
            "reference_files/clinical_substitutions.csv"
        ),
        "data_url": (
            "https://proteingym.s3.us-east-2.amazonaws.com/"
            "clinical_substitutions.parquet"
        ),
        "metadata_filename": "clinical_substitutions.csv",
        "data_filename": "clinical_substitutions.parquet",
        "processing_supported": False,
    },
    "clinical_indels": {
        "collection": "clinical",
        "variant_type": "indels",
        "metadata_url": (
            "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/"
            "reference_files/clinical_indels.csv"
        ),
        "data_url": (
            "https://proteingym.s3.us-east-2.amazonaws.com/"
            "clinical_indels.parquet"
        ),
        "metadata_filename": "clinical_indels.csv",
        "data_filename": "clinical_indels.parquet",
        "processing_supported": False,
    },
}


def _mock_built_table() -> pd.DataFrame:
    """Return the standardized columns needed by pipeline accounting."""
    return pd.DataFrame(
        {"status": ["OK"], "is_wildtype": [False], "is_synthetic": [False]}
    )

def test_registry_contains_exact_official_resources() -> None:
    resources = list_proteingym_resources()

    assert tuple(resource.resource_id for resource in resources) == tuple(
        EXPECTED_RESOURCES
    )
    assert set(PROTEINGYM_RESOURCES) == set(EXPECTED_RESOURCES)
    for resource in resources:
        expected = EXPECTED_RESOURCES[resource.resource_id]
        for field, value in expected.items():
            assert getattr(resource, field) == value

    assert [
        resource.resource_id
        for resource in resources
        if resource.processing_supported
    ] == ["dms_substitutions"]


def test_source_and_catalog_defaults_resolve_registry() -> None:
    substitutions = get_proteingym_resource("dms_substitutions")
    indels = get_proteingym_resource("dms_indels")
    catalog_signature = inspect.signature(ProteinGymCatalog)

    assert PROTEINGYM_SUBSTITUTIONS_URL == substitutions.metadata_url
    assert PROTEINGYM_INDELS_URL == indels.metadata_url
    assert (
        catalog_signature.parameters["substitutions_url"].default
        == substitutions.metadata_url
    )
    assert (
        catalog_signature.parameters["indels_url"].default
        == indels.metadata_url
    )
    assert (
        inspect.signature(MaveDBCatalog).parameters["base_url"].default
        == MAVEDB_API_URL
    )


def test_python_endpoint_overrides_remain_available() -> None:
    catalog = ProteinGymCatalog(
        substitutions_url="https://mirror.example/substitutions.csv",
        indels_url="https://mirror.example/indels.csv",
    )
    mavedb = MaveDBCatalog(base_url="https://mirror.example/api/v1/")

    assert catalog.urls == {
        "substitutions": "https://mirror.example/substitutions.csv",
        "indels": "https://mirror.example/indels.csv",
    }
    assert mavedb.base_url == "https://mirror.example/api/v1"


def test_unknown_and_unsupported_resources_are_distinct() -> None:
    with pytest.raises(UnknownSourceResourceError, match="Unknown"):
        get_proteingym_resource("not_registered")
    with pytest.raises(UnsupportedSourceResourceError, match="not currently supported"):
        get_proteingym_resource("dms_indels", require_processing=True)


def test_unsupported_resource_fails_before_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "proteingym"

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Unsupported resource attempted I/O.")

    monkeypatch.setattr(pipeline_module, "download_file", forbidden)
    with pytest.raises(UnsupportedSourceResourceError):
        pipeline_module.process_proteingym(
            {
                "resource": "dms_indels",
                "dir_base": source_root,
                "datasets": [{"dataset_id": "ASSAY_1"}],
            }
        )

    assert not source_root.exists()


def test_yaml_uses_logical_source_contract() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    forbidden = {
        "metadata_url",
        "benchmark_url",
        "base_url",
        "enabled",
        "filename",
        "urn",
    }

    assert not forbidden.intersection(_nested_keys(config))
    assert config["proteingym"]["resource"] == "dms_substitutions"
    assert all(
        "dataset_id" in entry for entry in config["proteingym"]["datasets"]
    )
    assert all("dataset_id" in entry for entry in config["mavedb"]["datasets"])
    assert (
        config["proteingym"]["default_build_kwargs"]["add_relative_score"]
        is False
    )
    assert (
        config["proteingym"]["default_build_kwargs"]["add_binary_label"]
        is False
    )
    assert config["proteingym"]["default_build_kwargs"]["add_wildtype_row"] is False
    assert config["proteingym"]["default_build_kwargs"]["drop_failed"] is False
    assert config["mavedb"]["default_build_kwargs"]["add_relative_score"] is False
    assert config["mavedb"]["default_build_kwargs"]["add_binary_label"] is False
    assert config["mavedb"]["default_build_kwargs"]["add_wildtype_row"] is False
    assert config["mavedb"]["default_build_kwargs"]["drop_failed"] is False


@pytest.mark.parametrize(
    ("only", "config", "expected"),
    [
        (
            "all",
            {
                "proteingym": {
                    "resource": "dms_substitutions",
                    "dir_base": "unused",
                    "datasets": [{"dataset_id": "ASSAY_1"}],
                }
            },
            ["proteingym"],
        ),
        (
            "mavedb",
            {
                "proteingym": {
                    "resource": "dms_substitutions",
                    "dir_base": "unused",
                    "datasets": [{"dataset_id": "ASSAY_1"}],
                },
                "mavedb": {
                    "dir_base": "unused",
                    "datasets": [
                        {"dataset_id": "urn:mavedb:00000001-a-1"}
                    ],
                },
            },
            ["mavedb"],
        ),
        (
            "proteingym",
            {
                "proteingym": {
                    "resource": "dms_substitutions",
                    "dir_base": "unused",
                    "datasets": [{"dataset_id": "ASSAY_1"}],
                },
                "mavedb": {
                    "dir_base": "unused",
                    "datasets": [
                        {"dataset_id": "urn:mavedb:00000001-a-1"}
                    ],
                }
            },
            ["proteingym"],
        ),
    ],
)
def test_source_presence_and_only_control_execution(
    only: str,
    config: dict[str, Any],
    expected: list[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    monkeypatch.setattr(
        pipeline_module,
        "process_proteingym",
        lambda cfg, dry_run=False: calls.append("proteingym")
        or [{"source": "proteingym", "input": "p", "status": "DRY_RUN"}],
    )
    monkeypatch.setattr(
        pipeline_module,
        "process_mavedb",
        lambda cfg, dry_run=False: calls.append("mavedb")
        or [{"source": "mavedb", "input": "m", "status": "DRY_RUN"}],
    )
    monkeypatch.setattr(
        pipeline_module,
        "write_summary",
        lambda summary, output: tmp_path / "summary.csv",
    )
    monkeypatch.setattr(pipeline_module, "log_summary", lambda summary: None)

    result = run_pipeline(config, only=only)

    assert result.exit_code == 0
    assert calls == expected


@pytest.mark.parametrize(
    ("source", "config", "message"),
    [
        (
            "proteingym",
            {"dir_base": "data", "datasets": [{"dataset_id": "ASSAY_1"}]},
            "resource",
        ),
        (
            "proteingym",
            {
                "resource": "",
                "dir_base": "data",
                "datasets": [{"dataset_id": "ASSAY_1"}],
            },
            "resource",
        ),
        (
            "proteingym",
            {
                "resource": "dms_substitutions",
                "dir_base": "data",
                "datasets": [{}],
            },
            "requires 'dataset_id'",
        ),
        (
            "proteingym",
            {
                "resource": "dms_substitutions",
                "dir_base": "data",
                "datasets": [{"dataset_id": ""}],
            },
            "non-empty string",
        ),
        (
            "proteingym",
            {
                "resource": "dms_substitutions",
                "dir_base": "data",
                "datasets": [{"dataset_id": 123}],
            },
            "non-empty string",
        ),
        (
            "proteingym",
            {
                "resource": "dms_substitutions",
                "dir_base": "data",
                "datasets": [
                    {"dataset_id": "ASSAY_1"},
                    {"dataset_id": "ASSAY_1"},
                ],
            },
            "Duplicate",
        ),
        (
            "proteingym",
            {
                "resource": "dms_substitutions",
                "dir_base": "data",
                "datasets": [{"filename": "ASSAY_1.csv"}],
            },
            "unsupported",
        ),
        (
            "mavedb",
            {
                "dir_base": "data",
                "datasets": [{"dataset_id": "GENE1"}],
            },
            "complete permanent score-set URN",
        ),
        (
            "mavedb",
            {
                "dir_base": "data",
                "datasets": [{"urn": "urn:mavedb:00000001-a-1"}],
            },
            "unsupported",
        ),
        (
            "mavedb",
            {
                "dir_base": [],
                "datasets": [
                    {"dataset_id": "urn:mavedb:00000001-a-1"}
                ],
            },
            "non-empty path",
        ),
    ],
)
def test_invalid_configuration_fails_clearly(
    source: str,
    config: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(SourceConfigurationError, match=message):
        validate_pipeline_config({source: config})


@pytest.mark.parametrize("dataset_id", VALID_MAVEDB_SCORE_SET_URNS)
def test_mavedb_accepts_complete_permanent_score_set_urns(
    dataset_id: str,
) -> None:
    validate_pipeline_config(
        {
            "mavedb": {
                "dir_base": "unused",
                "datasets": [{"dataset_id": dataset_id}],
            }
        }
    )


@pytest.mark.parametrize("dataset_id", INVALID_MAVEDB_DATASET_IDS)
def test_invalid_mavedb_identifiers_fail_before_io(
    dataset_id: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "mavedb"

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Invalid MaveDB identifier attempted I/O.")

    monkeypatch.setattr(pipeline_module, "_ensure_dirs", forbidden)
    monkeypatch.setattr(pipeline_module.requests, "get", forbidden)

    with pytest.raises(
        SourceConfigurationError,
        match="complete permanent score-set URN",
    ):
        pipeline_module.process_mavedb(
            {
                "dir_base": source_root,
                "datasets": [{"dataset_id": dataset_id}],
            }
        )

    assert not source_root.exists()


@pytest.mark.parametrize(
    ("metadata", "expected_gene", "expected_target", "expected_uniprot"),
    [
        (
            {
                "targetGenes": [
                    {
                        "name": "   ",
                        "uniprotIdFromMappedMetadata": " Q9MAPPED ",
                    },
                    {"name": "SECOND_TARGET"},
                ],
                "title": "Fallback protein title",
                "targetSequence": {"sequence": "MKT"},
            },
            None,
            "Fallback",
            "Q9MAPPED",
        ),
        (
            {
                "targetGenes": [{}],
                "targetSequence": {"sequence": "MKT"},
            },
            None,
            "Unknown",
            None,
        ),
    ],
)
def test_mavedb_gene_and_display_fallback_are_separate(
    metadata: dict[str, Any],
    expected_gene: str | None,
    expected_target: str,
    expected_uniprot: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_id = "urn:mavedb:00000055-aa-2"
    builder_calls: list[dict[str, Any]] = []

    class MetadataResponse:
        status_code = 200

        def json(self) -> dict[str, Any]:
            return metadata

    class ScoresResponse:
        text = "hgvs_pro,score\np.Met1Ala,0.5\n"

        def raise_for_status(self) -> None:
            """Represent a successful score-table response."""

    def source_request(url: str, *, timeout: int):
        assert timeout == 60
        if url.endswith("/scores"):
            return ScoresResponse()
        return MetadataResponse()

    def build_dataset(**kwargs: Any) -> pd.DataFrame:
        builder_calls.append(kwargs)
        return _mock_built_table()

    monkeypatch.setattr(pipeline_module.requests, "get", source_request)
    monkeypatch.setattr(
        pipeline_module,
        "build_mavedb_dataset",
        build_dataset,
    )

    result = pipeline_module.process_mavedb(
        {
            "dir_base": tmp_path / "mavedb",
            "datasets": [{"dataset_id": dataset_id}],
        },
        base_url="https://api.example.test",
    )

    assert result[0]["status"] == "OK"
    assert result[0]["target_protein"] == expected_target
    assert builder_calls[0]["gene"] == expected_gene
    assert builder_calls[0]["uniprot_id"] == expected_uniprot


@pytest.mark.parametrize("drop_failed", [False, True])
def test_drop_failed_controls_saved_rows_and_transform_defaults(
    drop_failed: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / str(drop_failed).lower()
    metadata = pd.DataFrame(
        {
            "DMS_filename": ["assay.csv"],
            "DMS_id": ["ASSAY_1"],
            "target_seq": ["MKT"],
            "UniProt_ID": ["P12345"],
        }
    )
    benchmark = pd.DataFrame(
        {
            "DMS_id": ["ASSAY_1", "ASSAY_1"],
            "mutant": ["M1A", "not-a-variant"],
            "DMS_score": [-1.5, 2.0],
        }
    )

    monkeypatch.setattr(
        pipeline_module,
        "download_file",
        lambda url, output_path, overwrite=False: Path(output_path),
    )
    monkeypatch.setattr(
        pipeline_module,
        "read_table",
        lambda path: (
            metadata
            if Path(path).name == "DMS_substitutions.csv"
            else benchmark
        ),
    )

    summary = pipeline_module.process_proteingym(
        {
            "resource": "dms_substitutions",
            "dir_base": source_root,
            "default_build_kwargs": {"drop_failed": drop_failed},
            "datasets": [{"dataset_id": "ASSAY_1"}],
        }
    )

    output = pd.read_csv(source_root / "processed" / "ASSAY_1_processed.csv")
    assert summary[0]["status"] == "OK"
    assert list(output["score_raw"]) == (
        [-1.5] if drop_failed else [-1.5, 2.0]
    )
    assert list(output["status"]) == (
        ["OK"] if drop_failed else ["OK", "Error"]
    )
    assert "score_log_ratio" not in output.columns
    assert "score_binary_like" not in output.columns


def _nested_keys(value: object) -> set[str]:
    """Return every mapping key contained in a nested YAML value."""
    if isinstance(value, dict):
        keys = set(value)
        for item in value.values():
            keys.update(_nested_keys(item))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for item in value:
            keys.update(_nested_keys(item))
        return keys
    return set()
