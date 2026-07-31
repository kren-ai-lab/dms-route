from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

import dms_parser.fetch as fetch_module
import dms_parser.sources.mavedb as mavedb_module
import dms_parser.sources.proteingym as proteingym_module

RUNNER_PATH = (
    Path(__file__).parents[1] / "examples" / "yml_parser" / "run_dms_parser.py"
)


@pytest.fixture
def runner(monkeypatch):
    """Load the example runner without requiring PyYAML in package tests."""
    yaml_stub = types.ModuleType("yaml")
    yaml_stub.safe_load = lambda handle: {}
    monkeypatch.setitem(sys.modules, "yaml", yaml_stub)

    spec = importlib.util.spec_from_file_location("test_dms_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the example DMS runner.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _forbid_skipped_operations(monkeypatch, runner) -> None:
    """Make every dataset acquisition and row-processing operation fail."""

    def forbidden(*args, **kwargs):
        raise AssertionError("Metadata-only execution invoked a skipped operation.")

    monkeypatch.setattr(runner, "build_mavedb_dataset", forbidden)
    monkeypatch.setattr(runner, "build_proteingym_dataset", forbidden)
    monkeypatch.setattr(runner, "write_table", forbidden)
    monkeypatch.setattr(fetch_module, "fetch_to_cache", forbidden)
    monkeypatch.setattr(mavedb_module, "download_mavedb_dataset", forbidden)
    monkeypatch.setattr(mavedb_module, "load_mavedb_from_url", forbidden)
    monkeypatch.setattr(proteingym_module, "download_proteingym_dataset", forbidden)
    monkeypatch.setattr(proteingym_module, "load_proteingym_from_url", forbidden)


def _assert_no_data_artifacts(root: Path) -> None:
    """Assert that metadata-only execution created no dataset artifacts."""
    assert not (root / "raw").exists()
    assert not (root / "processed").exists()
    assert not (root / "cache").exists()
    assert list(root.rglob(".download-*")) == []
    assert list(root.rglob("artifact-*")) == []
    assert list(root.rglob("*_processed.*")) == []


class MetadataResponse:
    """Offline MaveDB metadata response."""

    status_code = 200

    def __init__(self, metadata: dict) -> None:
        self._metadata = metadata

    def json(self) -> dict:
        """Return the configured metadata object."""
        return self._metadata


def test_mavedb_metadata_only_skips_dataset_acquisition(
    tmp_path,
    monkeypatch,
    runner,
):
    _forbid_skipped_operations(monkeypatch, runner)
    base_url = "https://api.example.test"
    urn = "urn:mavedb:00000001-a-1"
    source_root = tmp_path / "mavedb"
    requested_urls: list[str] = []
    metadata = {
        "targetGenes": [{"name": "GENE1"}],
        "targetSequence": {"sequence": "MKT"},
    }

    def metadata_only_request(url: str, *, timeout: int):
        assert timeout == 60
        requested_urls.append(url)
        if url.endswith("/scores"):
            raise AssertionError("Metadata-only mode requested MaveDB scores.")
        return MetadataResponse(metadata)

    monkeypatch.setattr(runner.requests, "get", metadata_only_request)
    monkeypatch.setattr(
        runner.pd,
        "read_csv",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Metadata-only mode read a score table.")
        ),
    )

    result = runner.process_mavedb(
        {
            "dir_base": source_root,
            "datasets": [{"dataset_id": urn}],
        },
        dry_run=True,
        base_url=base_url,
    )

    assert requested_urls == [f"{base_url}/score-sets/{urn}"]
    assert result == [
        {
            "source": "mavedb",
            "input": urn,
            "status": "DRY_RUN",
            "dataset_id": urn,
            "target_protein": "GENE1",
            "wt_length": 3,
            "raw_rows": None,
        }
    ]
    assert not source_root.exists()


def test_proteingym_metadata_only_skips_benchmark_and_processing(
    tmp_path,
    monkeypatch,
    runner,
):
    _forbid_skipped_operations(monkeypatch, runner)
    source_root = tmp_path / "proteingym"
    source_root.mkdir()
    metadata_path = source_root / "DMS_substitutions.csv"
    pd.DataFrame(
        {
            "DMS_filename": ["experiment.csv"],
            "DMS_id": ["experiment-1"],
            "target_seq": ["MKT"],
            "UniProt_ID": ["P12345"],
        }
    ).to_csv(metadata_path, index=False)
    resource = runner.get_proteingym_resource("dms_substitutions")
    requested_urls: list[str] = []

    def metadata_download(url, output_path, *, overwrite=False):
        assert overwrite is False
        requested_urls.append(url)
        if url == resource.data_url:
            raise AssertionError("Metadata-only mode requested the benchmark archive.")
        assert Path(output_path) == metadata_path
        return metadata_path

    monkeypatch.setattr(runner, "download_file", metadata_download)

    result = runner.process_proteingym(
        {
            "resource": "dms_substitutions",
            "dir_base": source_root,
            "datasets": [{"dataset_id": "experiment-1"}],
        },
        dry_run=True,
    )

    assert requested_urls == [resource.metadata_url]
    assert result == [
        {
            "source": "proteingym",
            "input": "experiment-1",
            "status": "DRY_RUN",
            "dataset_id": "experiment-1",
            "target_protein": "P12345",
            "wt_length": 3,
            "raw_rows": None,
        }
    ]
    assert metadata_path.exists()
    assert not (source_root / "DMS_substitutions.parquet").exists()
    _assert_no_data_artifacts(source_root)


def test_normal_mavedb_execution_keeps_existing_acquisition_path(
    tmp_path,
    monkeypatch,
    runner,
):
    source_root = tmp_path / "mavedb"
    base_url = "https://api.example.test"
    urn = "urn:mavedb:00000001-a-1"
    requested_urls: list[str] = []
    builder_calls: list[dict] = []

    class ScoresResponse:
        text = "hgvs_pro,score\np.Met1Ala,0.5\n"

        def raise_for_status(self) -> None:
            """Represent a successful scores response."""

    def source_request(url: str, *, timeout: int):
        assert timeout == 60
        requested_urls.append(url)
        if url.endswith("/scores"):
            return ScoresResponse()
        return MetadataResponse(
            {
                "targetGenes": [
                    {
                        "name": "GENE1",
                        "externalIdentifiers": [
                            {
                                "identifier": {
                                    "dbName": "UniProt",
                                    "identifier": "P12345",
                                }
                            }
                        ],
                    }
                ],
                "targetSequence": {"sequence": "MKT"},
            }
        )

    def build_dataset(**kwargs):
        builder_calls.append(kwargs)
        return pd.DataFrame({"status": ["OK"]})

    monkeypatch.setattr(runner.requests, "get", source_request)
    monkeypatch.setattr(runner, "build_mavedb_dataset", build_dataset)

    result = runner.process_mavedb(
        {
            "dir_base": source_root,
            "datasets": [{"dataset_id": urn}],
        },
        dry_run=False,
        base_url=base_url,
    )

    assert requested_urls == [
        f"{base_url}/score-sets/{urn}",
        f"{base_url}/score-sets/{urn}/scores",
    ]
    assert len(builder_calls) == 1
    assert builder_calls[0]["dataset_id"] == urn
    assert builder_calls[0]["gene"] == "GENE1"
    assert builder_calls[0]["protein_id"] is None
    assert builder_calls[0]["uniprot_id"] == "P12345"
    assert builder_calls[0]["add_relative_score"] is False
    assert builder_calls[0]["add_binary_label"] is False
    assert builder_calls[0]["drop_failed"] is False
    assert result[0]["status"] == "OK"
    assert (source_root / "raw").exists()
    assert (source_root / "processed").exists()


def test_normal_mavedb_rejects_failed_scores_response(
    tmp_path,
    monkeypatch,
    runner,
):
    source_root = tmp_path / "mavedb"
    base_url = "https://api.example.test"
    urn = "urn:mavedb:00000001-a-1"
    requested_urls: list[str] = []
    builder_called = False

    class FailedScoresResponse:
        text = "this content must not be written"

        def raise_for_status(self) -> None:
            """Raise the HTTP failure returned by the score endpoint."""
            raise runner.requests.HTTPError("503 Server Error")

    def source_request(url: str, *, timeout: int):
        assert timeout == 60
        requested_urls.append(url)
        if url.endswith("/scores"):
            return FailedScoresResponse()
        return MetadataResponse(
            {
                "targetGenes": [{"name": "GENE1"}],
                "targetSequence": {"sequence": "MKT"},
            }
        )

    def unexpected_builder(**kwargs):
        nonlocal builder_called
        builder_called = True
        raise AssertionError("A failed score response must not reach the builder.")

    monkeypatch.setattr(runner.requests, "get", source_request)
    monkeypatch.setattr(runner, "build_mavedb_dataset", unexpected_builder)

    result = runner.process_mavedb(
        {
            "dir_base": source_root,
            "datasets": [{"dataset_id": urn}],
        },
        dry_run=False,
        base_url=base_url,
    )

    assert requested_urls == [
        f"{base_url}/score-sets/{urn}",
        f"{base_url}/score-sets/{urn}/scores",
    ]
    assert result[0]["status"] == "ERROR"
    assert "503 Server Error" in result[0]["error"]
    assert builder_called is False
    assert not (
        source_root / "raw" / "urn_mavedb_00000001-a-1_scores.csv"
    ).exists()


def test_normal_proteingym_execution_keeps_existing_acquisition_path(
    tmp_path,
    monkeypatch,
    runner,
):
    source_root = tmp_path / "proteingym"
    resource = runner.get_proteingym_resource("dms_substitutions")
    downloaded_urls: list[str] = []
    builder_calls: list[dict] = []
    metadata = pd.DataFrame(
        {
            "DMS_filename": ["experiment.csv"],
            "DMS_id": ["experiment-1"],
            "target_seq": ["MKT"],
            "UniProt_ID": ["P12345"],
            "molecule_name": ["Example protein"],
            "gene": ["GENE1"],
        }
    )
    benchmark = pd.DataFrame(
        {
            "DMS_id": ["experiment-1"],
            "mutant": ["M1A"],
            "DMS_score": [0.5],
        }
    )

    def offline_download(url, output_path, *, overwrite=False):
        del overwrite
        downloaded_urls.append(url)
        return Path(output_path)

    def offline_read(path):
        if Path(path).name == "DMS_substitutions.csv":
            return metadata
        if Path(path).name == "DMS_substitutions.parquet":
            return benchmark
        raise AssertionError(f"Unexpected read: {path}")

    def build_dataset(**kwargs):
        builder_calls.append(kwargs)
        return pd.DataFrame({"status": ["OK"]})

    monkeypatch.setattr(runner, "download_file", offline_download)
    monkeypatch.setattr(runner, "read_table", offline_read)
    monkeypatch.setattr(runner, "build_proteingym_dataset", build_dataset)

    result = runner.process_proteingym(
        {
            "resource": "dms_substitutions",
            "dir_base": source_root,
            "datasets": [{"dataset_id": "experiment-1"}],
        },
        dry_run=False,
    )

    assert downloaded_urls == [resource.metadata_url, resource.data_url]
    assert len(builder_calls) == 1
    assert builder_calls[0]["dataset_id"] == "experiment-1"
    assert builder_calls[0]["protein_id"] == "Example protein"
    assert builder_calls[0]["gene"] == "GENE1"
    assert builder_calls[0]["uniprot_id"] == "P12345"
    assert builder_calls[0]["add_relative_score"] is False
    assert builder_calls[0]["add_binary_label"] is False
    assert builder_calls[0]["drop_failed"] is False
    assert result[0]["status"] == "OK"
    assert (source_root / "raw").exists()
    assert (source_root / "processed").exists()
