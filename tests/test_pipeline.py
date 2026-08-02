from __future__ import annotations

import copy
import hashlib
import json
import logging
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import requests

import dms_parser.fetch as fetch_module
import dms_parser.pipeline as pipeline_module
from dms_parser import (
    DatasetBatchDownloadEntry,
    DatasetBatchDownloadResult,
    DatasetDownloadResult,
    DatasetRecord,
    FilesystemCache,
    InvalidPipelineOptionError,
    PipelineResult,
    download_and_standardize_dataset,
    download_and_standardize_datasets,
    get_proteingym_resource,
    run_pipeline,
)
from dms_parser.exceptions import (
    DatasetNotFoundError,
    DownloadError,
    InvalidPipelineOptionError,
    SourceConfigurationError,
)
from dms_parser.sources.proteingym_catalog import (
    PROTEINGYM_SUBSTITUTIONS_CACHE_ID,
)


def _proteingym_config(root: Path, *dataset_ids: str) -> dict[str, Any]:
    """Return a minimal ProteinGym pipeline configuration."""
    return {
        "proteingym": {
            "resource": "dms_substitutions",
            "dir_base": root / "proteingym",
            "datasets": [
                {"dataset_id": dataset_id}
                for dataset_id in dataset_ids
            ],
        },
        "output": {"summary_dir": root / "summaries"},
    }


def _mock_built_table() -> pd.DataFrame:
    """Return the standardized columns needed by pipeline accounting."""
    return pd.DataFrame(
        {"status": ["OK"], "is_wildtype": [False], "is_synthetic": [False]}
    )


@pytest.mark.parametrize("only", ["invalid", "", None, 1])
def test_run_pipeline_rejects_invalid_only_values(
    only: object,
) -> None:
    with pytest.raises(InvalidPipelineOptionError, match="only must be"):
        run_pipeline({}, only=only)  # type: ignore[arg-type]


def test_run_pipeline_rejects_non_boolean_dry_run() -> None:
    with pytest.raises(
        InvalidPipelineOptionError,
        match="dry_run must be a boolean",
    ):
        run_pipeline({}, dry_run="yes")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("statuses", "has_errors", "exit_code"),
    [
        ([], False, 0),
        (["OK"], False, 0),
        (["DRY_RUN"], False, 0),
        (["OK", "ERROR"], True, 1),
    ],
)
def test_pipeline_result_exposes_error_and_exit_status(
    statuses: list[str],
    has_errors: bool,
    exit_code: int,
) -> None:
    result = PipelineResult(
        summary=[
            {"source": "test", "input": str(index), "status": status}
            for index, status in enumerate(statuses)
        ]
    )

    assert result.has_errors is has_errors
    assert result.exit_code == exit_code


@pytest.mark.parametrize(
    ("status", "expected_exit"),
    [("OK", 0), ("DRY_RUN", 0), ("ERROR", 1)],
)
def test_run_pipeline_returns_source_status_exit_code(
    status: str,
    expected_exit: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = {"source": "proteingym", "input": "ASSAY_1", "status": status}
    monkeypatch.setattr(
        pipeline_module,
        "process_proteingym",
        lambda config, dry_run=False: [row],
    )
    monkeypatch.setattr(
        pipeline_module,
        "write_summary",
        lambda summary, output: tmp_path / "summary.csv",
    )
    monkeypatch.setattr(pipeline_module, "log_summary", lambda summary: None)

    result = run_pipeline(_proteingym_config(tmp_path, "ASSAY_1"))

    assert result.summary == [row]
    assert result.exit_code == expected_exit


def test_run_pipeline_writes_existing_summary_formats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = {
        "source": "proteingym",
        "input": "ASSAY_1",
        "status": "OK",
        "dataset_id": "ASSAY_1",
        "raw_rows": 2,
        "validated_rows": 1,
        "discarded_rows": 0,
        "output_file": "processed/ASSAY_1_processed.csv",
    }
    monkeypatch.setattr(
        pipeline_module,
        "process_proteingym",
        lambda config, dry_run=False: [row],
    )
    monkeypatch.setattr(pipeline_module, "log_summary", lambda summary: None)

    result = run_pipeline(_proteingym_config(tmp_path, "ASSAY_1"))

    assert result.summary == [row]
    assert result.summary_path is not None
    assert result.summary_path.name.startswith("summary_")
    assert result.summary_path.suffix == ".csv"
    assert pd.read_csv(result.summary_path).to_dict("records") == [row]
    json_path = result.summary_path.with_suffix(".json")
    assert json.loads(json_path.read_text(encoding="utf-8")) == [row]


def test_build_kwargs_are_deep_merged_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = pd.DataFrame(
        {
            "DMS_filename": ["assay.csv"],
            "DMS_id": ["ASSAY_1"],
            "target_seq": ["MKT"],
            "UniProt_ID": ["P12345"],
            "molecule_name": ["Protein"],
            "gene": ["GENE1"],
        }
    )
    benchmark = pd.DataFrame(
        {
            "DMS_id": ["ASSAY_1"],
            "mutant": ["M1A"],
            "DMS_score": [0.5],
        }
    )
    merge_calls: list[tuple[dict[str, Any], dict[str, Any]]] = []
    builder_calls: list[dict[str, Any]] = []

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

    def merge_once(
        defaults: dict[str, Any],
        overrides: dict[str, Any],
    ) -> dict[str, Any]:
        merge_calls.append((copy.deepcopy(defaults), copy.deepcopy(overrides)))
        return {
            "score_col": "DMS_score",
            "variant_col": "mutant",
            "nested": {"default": 1, "dataset": 2},
        }

    def build_dataset(**kwargs: Any) -> pd.DataFrame:
        builder_calls.append(kwargs)
        return _mock_built_table()

    monkeypatch.setattr(pipeline_module, "_deep_merge", merge_once)
    monkeypatch.setattr(
        pipeline_module,
        "build_proteingym_dataset",
        build_dataset,
    )

    pipeline_module.process_proteingym(
        {
            "resource": "dms_substitutions",
            "dir_base": tmp_path / "proteingym",
            "default_build_kwargs": {"nested": {"default": 1}},
            "datasets": [
                {
                    "dataset_id": "ASSAY_1",
                    "build_kwargs": {"nested": {"dataset": 2}},
                }
            ],
        }
    )

    assert merge_calls == [
        (
            {"nested": {"default": 1}},
            {"nested": {"dataset": 2}},
        )
    ]
    assert builder_calls[0]["nested"] == {"default": 1, "dataset": 2}
    assert builder_calls[0]["dataset_id"] == "ASSAY_1"
    assert builder_calls[0]["protein_id"] == "Protein"
    assert builder_calls[0]["gene"] == "GENE1"
    assert builder_calls[0]["uniprot_id"] == "P12345"
    assert builder_calls[0]["add_relative_score"] is False
    assert builder_calls[0]["add_binary_label"] is False
    assert builder_calls[0]["add_wildtype_row"] is False
    assert builder_calls[0]["drop_failed"] is False


def test_mavedb_dataset_columns_and_metadata_reach_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_id = "urn:mavedb:00000001-aa-2"
    builder_calls: list[dict[str, Any]] = []

    class MetadataResponse:
        status_code = 200

        def json(self) -> dict[str, Any]:
            return {
                "targetGenes": [
                    {
                        "name": "GENE1",
                        "uniprotIdFromMappedMetadata": "P12345",
                    }
                ],
                "targetSequence": {"sequence": "MKT"},
            }

    class ScoresResponse:
        text = "custom_hgvs,custom_score\np.Met1Ala,0.5\n"

        def raise_for_status(self) -> None:
            """Represent a successful score response."""

    def source_request(url: str, *, timeout: int):
        assert timeout == 60
        return ScoresResponse() if url.endswith("/scores") else MetadataResponse()

    def build_dataset(**kwargs: Any) -> pd.DataFrame:
        builder_calls.append(kwargs)
        return _mock_built_table()

    monkeypatch.setattr(pipeline_module.requests, "get", source_request)
    monkeypatch.setattr(
        pipeline_module,
        "build_mavedb_dataset",
        build_dataset,
    )

    pipeline_module.process_mavedb(
        {
            "dir_base": tmp_path / "mavedb",
            "datasets": [
                {
                    "dataset_id": dataset_id,
                    "hgvs_col": "custom_hgvs",
                    "score_col": "custom_score",
                }
            ],
        },
        base_url="https://api.example.test",
    )

    assert builder_calls[0]["dataset_id"] == dataset_id
    assert builder_calls[0]["hgvs_col"] == "custom_hgvs"
    assert builder_calls[0]["score_col"] == "custom_score"
    assert builder_calls[0]["protein_id"] is None
    assert builder_calls[0]["gene"] == "GENE1"
    assert builder_calls[0]["uniprot_id"] == "P12345"
    assert builder_calls[0]["add_wildtype_row"] is False


def test_dataset_failure_does_not_abort_later_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = pd.DataFrame(
        {
            "DMS_filename": ["second.csv"],
            "DMS_id": ["SECOND"],
            "target_seq": ["MKT"],
        }
    )
    benchmark = pd.DataFrame(
        {
            "DMS_id": ["SECOND"],
            "mutant": ["M1A"],
            "DMS_score": [0.5],
        }
    )
    builder_calls: list[str] = []

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

    def build_dataset(**kwargs: Any) -> pd.DataFrame:
        builder_calls.append(kwargs["dataset_id"])
        return _mock_built_table()

    monkeypatch.setattr(
        pipeline_module,
        "build_proteingym_dataset",
        build_dataset,
    )

    summary = pipeline_module.process_proteingym(
        {
            "resource": "dms_substitutions",
            "dir_base": tmp_path / "proteingym",
            "datasets": [
                {"dataset_id": "MISSING"},
                {"dataset_id": "SECOND"},
            ],
        }
    )

    assert [row["status"] for row in summary] == ["ERROR", "OK"]
    assert "No information found" in summary[0]["error"]
    assert builder_calls == ["SECOND"]


def test_completed_dataset_counts_exclude_synthetic_wt_from_discarded() -> None:
    built_table = pd.DataFrame(
        {
            "status": ["OK", "OK"],
            "is_wildtype": [True, False],
            "is_synthetic": [True, False],
        }
    )

    counts = pipeline_module._completed_dataset_counts(built_table, raw_rows=1)

    assert counts == {
        "validated_rows": 2,
        "discarded_rows": 0,
        "output_rows": 2,
        "wildtype_rows": 1,
        "synthetic_wildtype_rows": 1,
    }


def test_proteingym_pipeline_accounts_for_generated_wt_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = pd.DataFrame(
        {
            "DMS_filename": ["assay.csv"],
            "DMS_id": ["ASSAY_1"],
            "target_seq": ["MKT"],
        }
    )
    benchmark = pd.DataFrame(
        {"DMS_id": ["ASSAY_1"], "mutant": ["M1A"], "DMS_score": [0.5]}
    )
    source_root = tmp_path / "proteingym"

    monkeypatch.setattr(
        pipeline_module,
        "download_file",
        lambda url, output_path, overwrite=False: Path(output_path),
    )
    monkeypatch.setattr(
        pipeline_module,
        "read_table",
        lambda path: metadata if Path(path).name == "DMS_substitutions.csv" else benchmark,
    )

    summary = pipeline_module.process_proteingym(
        {
            "resource": "dms_substitutions",
            "dir_base": source_root,
            "default_build_kwargs": {"add_wildtype_row": True},
            "datasets": [{"dataset_id": "ASSAY_1"}],
        }
    )

    output = pd.read_csv(source_root / "processed" / "ASSAY_1_processed.csv")
    assert len(output) == 2
    assert summary[0]["output_rows"] == 2
    assert summary[0]["raw_rows"] == 1
    assert summary[0]["discarded_rows"] == 0
    assert summary[0]["wildtype_rows"] == 1
    assert summary[0]["synthetic_wildtype_rows"] == 1


def test_proteingym_wt_row_source_default_and_dataset_override_are_forwarded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = pd.DataFrame(
        {
            "DMS_filename": ["first.csv", "second.csv"],
            "DMS_id": ["FIRST", "SECOND"],
            "target_seq": ["MKT", "MKT"],
        }
    )
    benchmark = pd.DataFrame(
        {
            "DMS_id": ["FIRST", "SECOND"],
            "mutant": ["M1A", "M1A"],
            "DMS_score": [0.5, 0.6],
        }
    )
    builder_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(
        pipeline_module,
        "download_file",
        lambda url, output_path, overwrite=False: Path(output_path),
    )
    monkeypatch.setattr(
        pipeline_module,
        "read_table",
        lambda path: metadata if Path(path).name == "DMS_substitutions.csv" else benchmark,
    )

    def build_dataset(**kwargs: Any) -> pd.DataFrame:
        builder_calls.append(kwargs)
        return _mock_built_table()

    monkeypatch.setattr(pipeline_module, "build_proteingym_dataset", build_dataset)

    pipeline_module.process_proteingym(
        {
            "resource": "dms_substitutions",
            "dir_base": tmp_path / "proteingym",
            "default_build_kwargs": {"add_wildtype_row": True},
            "datasets": [
                {"dataset_id": "FIRST"},
                {"dataset_id": "SECOND", "build_kwargs": {"add_wildtype_row": False}},
            ],
        }
    )

    assert [call["add_wildtype_row"] for call in builder_calls] == [True, False]


def test_mavedb_wt_row_source_default_and_dataset_override_are_forwarded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_ids = ["urn:mavedb:00000001-a-1", "urn:mavedb:00000002-a-1"]
    builder_calls: list[dict[str, Any]] = []

    class MetadataResponse:
        status_code = 200

        def json(self) -> dict[str, Any]:
            return {"targetSequence": {"sequence": "MKT"}}

    class ScoresResponse:
        text = "hgvs_pro,score\np.Met1Ala,0.5\n"

        def raise_for_status(self) -> None:
            """Represent a successful score response."""

    monkeypatch.setattr(
        pipeline_module.requests,
        "get",
        lambda url, timeout: ScoresResponse() if url.endswith("/scores") else MetadataResponse(),
    )

    def build_dataset(**kwargs: Any) -> pd.DataFrame:
        builder_calls.append(kwargs)
        return _mock_built_table()

    monkeypatch.setattr(pipeline_module, "build_mavedb_dataset", build_dataset)

    pipeline_module.process_mavedb(
        {
            "dir_base": tmp_path / "mavedb",
            "default_build_kwargs": {"add_wildtype_row": True},
            "datasets": [
                {"dataset_id": dataset_ids[0]},
                {"dataset_id": dataset_ids[1], "build_kwargs": {"add_wildtype_row": False}},
            ],
        },
        base_url="https://api.example.test",
    )

    assert [call["add_wildtype_row"] for call in builder_calls] == [True, False]


def test_pipeline_logs_no_full_wt_sequence_or_url_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    wt_sequence = "MKTAYIAKQRQISFVKSHFSRQDILDLWQ"
    metadata = pd.DataFrame(
        {
            "DMS_filename": ["assay.csv"],
            "DMS_id": ["ASSAY_1"],
            "target_seq": [wt_sequence],
        }
    )

    monkeypatch.setattr(
        pipeline_module,
        "download_file",
        lambda url, output_path, overwrite=False: Path(output_path),
    )
    monkeypatch.setattr(pipeline_module, "read_table", lambda path: metadata)
    caplog.set_level(logging.DEBUG, logger="dms_parser.pipeline")

    summary = pipeline_module.process_proteingym(
        {
            "resource": "dms_substitutions",
            "dir_base": tmp_path / "proteingym",
            "datasets": [{"dataset_id": "ASSAY_1"}],
        },
        dry_run=True,
    )

    assert summary[0]["status"] == "DRY_RUN"
    assert wt_sequence not in caplog.text
    assert "authorization" not in caplog.text.casefold()
    assert "?" not in caplog.text


def test_download_api_is_public() -> None:
    assert download_and_standardize_dataset is (
        pipeline_module.download_and_standardize_dataset
    )
    assert DatasetDownloadResult is pipeline_module.DatasetDownloadResult


def test_download_collision_preflight_precedes_acquisition_and_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    existing = output_dir / "summary.csv"
    existing.write_text("existing", encoding="utf-8")
    cache = FilesystemCache(tmp_path / "cache")

    monkeypatch.setattr(
        pipeline_module,
        "get_dataset_metadata",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Collision preflight attempted acquisition")
        ),
    )

    with pytest.raises(FileExistsError, match="overwrite=True"):
        download_and_standardize_dataset(
            "proteingym",
            "ASSAY_1",
            output_dir=output_dir,
            cache=cache,
        )

    assert existing.read_text(encoding="utf-8") == "existing"
    assert not cache.root.exists()
    assert list(output_dir.iterdir()) == [existing]


def test_unknown_download_publishes_no_error_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "output"
    monkeypatch.setattr(
        pipeline_module,
        "get_dataset_metadata",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            DatasetNotFoundError("missing")
        ),
    )

    with pytest.raises(DatasetNotFoundError):
        download_and_standardize_dataset(
            "proteingym",
            "MISSING",
            output_dir=output_dir,
            cache=FilesystemCache(tmp_path / "cache"),
        )

    assert not output_dir.exists()


def test_mavedb_download_translates_metadata_404_to_dataset_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = requests.Response()
    response.status_code = 404
    error = requests.HTTPError("missing", response=response)
    monkeypatch.setattr(
        pipeline_module,
        "get_dataset_metadata",
        lambda *args, **kwargs: (_ for _ in ()).throw(error),
    )

    with pytest.raises(DatasetNotFoundError, match="was not found"):
        pipeline_module._get_mavedb_download_metadata(
            "urn:mavedb:00000001-a-1"
        )


def test_proteingym_download_uses_shared_cached_benchmark_and_builds_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resource = get_proteingym_resource("dms_substitutions")
    metadata_source = tmp_path / "DMS_substitutions.csv"
    benchmark_source = tmp_path / "DMS_substitutions.parquet"
    pd.DataFrame(
        {
            "DMS_id": ["ASSAY_1", "ASSAY_2"],
            "target_seq": ["MKT", "MKT"],
            "UniProt_ID": ["P11111", "P22222"],
            "molecule_name": ["Protein one", "Protein two"],
            "gene": ["GENE1", "GENE2"],
        }
    ).to_csv(metadata_source, index=False)
    pd.DataFrame(
        {
            "DMS_id": ["ASSAY_1", "ASSAY_1", "ASSAY_2"],
            "mutant": ["WT", "M1A", "M1A"],
            "DMS_score": [1.5, 0.5, 0.75],
            "source_note": ["observed", "observed", "café"],
        }
    ).to_parquet(benchmark_source, index=False)
    sources = {
        resource.metadata_url: metadata_source,
        resource.data_url: benchmark_source,
    }
    download_calls: list[str] = []

    def offline_download(
        url: str,
        output_path: str | Path,
        **kwargs: Any,
    ) -> Path:
        del kwargs
        download_calls.append(url)
        path = Path(output_path)
        path.write_bytes(sources[url].read_bytes())
        return path

    monkeypatch.setattr(fetch_module, "download_file", offline_download)
    cache = FilesystemCache(tmp_path / "cache")

    first = download_and_standardize_dataset(
        "proteingym",
        "ASSAY_1",
        output_dir=tmp_path / "first",
        cache=cache,
        add_wildtype_row=True,
    )
    second = download_and_standardize_dataset(
        "proteingym",
        "ASSAY_2",
        output_dir=tmp_path / "second",
        cache=cache,
        add_wildtype_row=True,
    )

    assert download_calls == [resource.metadata_url, resource.data_url]
    assert cache.exists("proteingym", PROTEINGYM_SUBSTITUTIONS_CACHE_ID)
    benchmark_manifest = cache.load_manifest(
        "proteingym",
        pipeline_module._PROTEINGYM_SUBSTITUTIONS_BENCHMARK_CACHE_ID,
    )
    assert benchmark_manifest.original_url == resource.data_url
    first_table = pd.read_csv(first.dataset_path)
    second_table = pd.read_csv(second.dataset_path)
    assert first_table["score_raw"].tolist() == [1.5, 0.5]
    assert first_table["is_synthetic"].tolist() == [False, False]
    assert int(first_table["is_wildtype"].sum()) == 1
    assert second_table["score_raw"].iloc[1] == 0.75
    assert pd.isna(second_table["score_raw"].iloc[0])
    assert second_table["is_synthetic"].tolist() == [True, False]
    assert "parsed_variant" in second_table.columns
    assert "source_note" in second_table.columns
    assert "score_log_ratio" not in second_table.columns
    assert "score_binary_like" not in second_table.columns
    assert second.summary["raw_rows"] == 1
    assert second.summary["output_rows"] == 2
    assert second.summary["validated_rows"] == 2
    assert second.summary["discarded_rows"] == 0
    assert second.summary["wildtype_rows"] == 1
    assert second.summary["synthetic_wildtype_rows"] == 1
    assert second.summary["output_file"] == str(second.dataset_path)
    assert list(second.summary) == [
        "source",
        "input",
        "status",
        "dataset_id",
        "target_protein",
        "wt_length",
        "raw_rows",
        "validated_rows",
        "discarded_rows",
        "output_rows",
        "wildtype_rows",
        "synthetic_wildtype_rows",
        "output_file",
    ]
    assert sorted(path.name for path in second.dataset_path.parent.iterdir()) == [
        "standardized.csv",
        "summary.csv",
        "summary.json",
    ]
    assert b"caf\xc3\xa9" in second.dataset_path.read_bytes()
    for path in (
        second.dataset_path,
        second.summary_csv_path,
        second.summary_json_path,
    ):
        content = path.read_bytes()
        assert content.endswith(b"\n")
        assert not content.endswith(b"\n\n")
        assert b"\r\n" not in content

    refreshed = download_and_standardize_dataset(
        "proteingym",
        "ASSAY_1",
        output_dir=tmp_path / "refreshed",
        cache=cache,
        refresh=True,
    )
    assert refreshed.summary["status"] == "OK"
    assert download_calls == [
        resource.metadata_url,
        resource.data_url,
        resource.metadata_url,
        resource.data_url,
    ]


def test_mavedb_download_caches_scores_and_preserves_builder_behavior(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_id = "urn:mavedb:00000001-a-1"
    metadata = DatasetRecord(
        source="mavedb",
        dataset_id=dataset_id,
        title="Non-ASCII assay",
        target_id="GÉNE",
        variant_type=None,
        n_variants=2,
        raw_metadata={
            "urn": dataset_id,
            "targetGenes": [{"name": "GÉNE"}],
            "targetSequence": {"sequence": "MKT"},
        },
    )
    monkeypatch.setattr(
        pipeline_module,
        "get_dataset_metadata",
        lambda source, requested_id: metadata,
    )
    download_calls: list[str] = []

    def offline_download(
        url: str,
        output_path: str | Path,
        **kwargs: Any,
    ) -> Path:
        del kwargs
        download_calls.append(url)
        path = Path(output_path)
        path.write_text(
            "hgvs_pro,score,source_note\n"
            "p.Met1Ala,0.75,observed\n"
            "p.invalid,-2.5,failed\n",
            encoding="utf-8",
        )
        return path

    monkeypatch.setattr(fetch_module, "download_file", offline_download)
    cache = FilesystemCache(tmp_path / "cache")
    first = download_and_standardize_dataset(
        "mavedb",
        dataset_id,
        output_dir=tmp_path / "first",
        cache=cache,
    )
    second = download_and_standardize_dataset(
        "mavedb",
        dataset_id,
        output_dir=tmp_path / "second",
        cache=cache,
        drop_failed=True,
        add_wildtype_row=True,
    )

    assert len(download_calls) == 1
    assert cache.exists("mavedb", dataset_id)
    first_table = pd.read_csv(first.dataset_path)
    assert first_table["score_raw"].tolist() == [0.75, -2.5]
    assert first_table["is_synthetic"].tolist() == [False, False]
    assert first_table["status"].tolist() == ["OK", "Error"]
    second_table = pd.read_csv(second.dataset_path)
    assert second_table["is_synthetic"].tolist() == [True, False]
    assert second_table["status"].tolist() == ["OK", "OK"]
    assert second.summary["raw_rows"] == 2
    assert second.summary["output_rows"] == 2
    assert second.summary["discarded_rows"] == 1
    assert second.summary["synthetic_wildtype_rows"] == 1
    assert json.loads(second.summary_json_path.read_text(encoding="utf-8")) == [
        second.summary
    ]
    assert "GÉNE" in second.summary_json_path.read_text(encoding="utf-8")

    download_and_standardize_dataset(
        "mavedb",
        dataset_id,
        output_dir=tmp_path / "refreshed",
        cache=cache,
        refresh=True,
    )
    assert len(download_calls) == 2


def _batch_metadata_record(dataset_id: str) -> DatasetRecord:
    """Return minimal ProteinGym substitutions metadata for a batch test."""
    return DatasetRecord(
        source="proteingym",
        dataset_id=dataset_id,
        title=None,
        target_id=f"P-{dataset_id}",
        variant_type="substitutions",
        n_variants=1,
        raw_metadata={
            "DMS_id": dataset_id,
            "target_seq": "MKT",
            "UniProt_ID": f"P-{dataset_id}",
        },
    )


def test_batch_download_api_is_public_and_results_are_frozen(tmp_path: Path) -> None:
    entry = DatasetBatchDownloadEntry(
        source="proteingym",
        dataset_id="ASSAY_1",
        output_dir=tmp_path / "output",
        result=None,
        error_type="DatasetNotFoundError",
        error="missing",
    )
    result = DatasetBatchDownloadResult(
        entries=(entry,),
        summary_csv_path=tmp_path / "download-summary.csv",
        summary_json_path=tmp_path / "download-summary.json",
    )

    assert download_and_standardize_datasets is (
        pipeline_module.download_and_standardize_datasets
    )
    assert result.successful_entries == ()
    assert result.failed_entries == (entry,)
    assert result.success_count == 0
    assert result.failure_count == 1
    assert result.has_errors is True
    assert result.exit_code == 1
    with pytest.raises(FrozenInstanceError):
        entry.error = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.entries = ()  # type: ignore[misc]


@pytest.mark.parametrize(
    ("dataset_ids", "exception", "message"),
    [
        ("ASSAY_1", InvalidPipelineOptionError, "non-string sequence"),
        ([], InvalidPipelineOptionError, "at least one"),
        ([" ASSAY_1"], SourceConfigurationError, "whitespace"),
        (["ASSAY_1 "], SourceConfigurationError, "whitespace"),
        (["ASSAY\n1"], SourceConfigurationError, "control"),
        (["ASSAY_1", "ASSAY_1"], SourceConfigurationError, "Duplicate"),
    ],
)
def test_batch_request_validation_rejects_invalid_ids_without_side_effects(
    dataset_ids: object,
    exception: type[Exception],
    message: str,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    cache = FilesystemCache(tmp_path / "cache")

    with pytest.raises(exception, match=message):
        download_and_standardize_datasets(
            "proteingym",
            dataset_ids,  # type: ignore[arg-type]
            output_dir=output_dir,
            cache=cache,
        )

    assert not output_dir.exists()
    assert not cache.root.exists()


@pytest.mark.parametrize(
    "dataset_id",
    ["../outside", r"C:\absolute\path", "CON", "CON.txt", "NUL", "PRN", "..."],
)
def test_portable_dataset_component_is_bounded_and_path_safe(
    dataset_id: str,
) -> None:
    component = pipeline_module._portable_dataset_component(dataset_id)
    digest = hashlib.sha256(dataset_id.encode("utf-8")).hexdigest()[:16]

    assert component.startswith("id-")
    assert component.endswith(f"--{digest}")
    assert "/" not in component
    assert "\\" not in component
    assert not component.endswith((".", " "))
    assert len(component) <= 69


def test_portable_dataset_component_follows_exact_mapping() -> None:
    dataset_id = "urn:mavedb:00000001-a-1"
    digest = hashlib.sha256(dataset_id.encode("utf-8")).hexdigest()[:16]

    assert pipeline_module._portable_dataset_component(dataset_id) == (
        f"id-urn-mavedb-00000001-a-1--{digest}"
    )


def test_batch_rejects_case_insensitive_derived_collision_before_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "output"
    monkeypatch.setattr(
        pipeline_module,
        "_portable_dataset_component",
        lambda dataset_id: "id-SAME--0000" if dataset_id == "A" else "ID-same--0000",
    )

    with pytest.raises(SourceConfigurationError, match="same portable"):
        download_and_standardize_datasets(
            "proteingym",
            ["A", "B"],
            output_dir=output_dir,
            cache=FilesystemCache(tmp_path / "cache"),
        )

    assert not output_dir.exists()


def test_batch_collision_preflight_precedes_cache_and_acquisition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "download-summary.json").write_text(
        "existing",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        pipeline_module,
        "list_datasets",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("collision attempted acquisition")
        ),
    )
    cache = FilesystemCache(tmp_path / "cache")

    with pytest.raises(FileExistsError, match="overwrite=True"):
        download_and_standardize_datasets(
            "proteingym",
            ["ASSAY_1"],
            output_dir=output_dir,
            cache=cache,
        )

    assert not cache.root.exists()


def test_proteingym_batch_acquires_and_loads_shared_artifacts_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resource = get_proteingym_resource("dms_substitutions")
    metadata_source = tmp_path / "metadata.csv"
    benchmark_source = tmp_path / "benchmark.parquet"
    pd.DataFrame(
        {
            "DMS_id": ["ASSAY_1", "ASSAY_2"],
            "target_seq": ["MKT", "MKT"],
            "UniProt_ID": ["P11111", "P22222"],
        }
    ).to_csv(metadata_source, index=False)
    pd.DataFrame(
        {
            "DMS_id": ["ASSAY_1", "ASSAY_1", "ASSAY_2"],
            "mutant": ["WT", "M1A", "M1A"],
            "DMS_score": [1.0, 0.5, 0.75],
            "source_note": ["first", "first", "cafÃ©"],
        }
    ).to_parquet(benchmark_source, index=False)
    source_paths = {
        resource.metadata_url: metadata_source,
        resource.data_url: benchmark_source,
    }
    download_calls: list[str] = []

    def offline_download(url: str, output_path: str | Path, **kwargs: Any) -> Path:
        del kwargs
        download_calls.append(url)
        path = Path(output_path)
        path.write_bytes(source_paths[url].read_bytes())
        return path

    parquet_reads = 0
    original_read_table = pipeline_module.read_table

    def count_reads(path: str | Path, **kwargs: Any) -> pd.DataFrame:
        nonlocal parquet_reads
        if Path(path).suffix == ".parquet":
            parquet_reads += 1
        return original_read_table(path, **kwargs)

    monkeypatch.setattr(fetch_module, "download_file", offline_download)
    monkeypatch.setattr(pipeline_module, "read_table", count_reads)
    output_dir = tmp_path / "output"
    cache = FilesystemCache(tmp_path / "cache")
    result = download_and_standardize_datasets(
        "proteingym",
        ["MISSING", "ASSAY_2", "ASSAY_1"],
        output_dir=output_dir,
        cache=cache,
        refresh=True,
        add_wildtype_row=True,
    )

    assert download_calls == [resource.metadata_url, resource.data_url]
    assert parquet_reads == 1
    assert cache.exists("proteingym", PROTEINGYM_SUBSTITUTIONS_CACHE_ID)
    assert cache.exists(
        "proteingym",
        pipeline_module._PROTEINGYM_SUBSTITUTIONS_BENCHMARK_CACHE_ID,
    )
    assert [entry.dataset_id for entry in result.entries] == [
        "MISSING",
        "ASSAY_2",
        "ASSAY_1",
    ]
    assert [entry.result is not None for entry in result.entries] == [
        False,
        True,
        True,
    ]
    assert result.exit_code == 1
    assay_two = pd.read_csv(result.entries[1].result.dataset_path)  # type: ignore[union-attr]
    assert assay_two["score_raw"].iloc[1] == 0.75
    assert assay_two["is_synthetic"].tolist() == [True, False]
    records = json.loads(result.summary_json_path.read_text(encoding="utf-8"))
    assert [record["status"] for record in records] == [
        "ERROR",
        "SUCCESS",
        "SUCCESS",
    ]
    assert records[1]["dataset_path"].startswith("proteingym/id-")
    assert "\\" not in records[1]["dataset_path"]


def test_proteingym_shared_reference_failure_marks_every_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pipeline_module,
        "list_datasets",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            DownloadError("reference failed")
        ),
    )

    result = download_and_standardize_datasets(
        "proteingym",
        ["ASSAY_1", "ASSAY_2"],
        output_dir=tmp_path / "output",
        cache=FilesystemCache(tmp_path / "cache"),
    )

    assert [entry.error_type for entry in result.entries] == [
        "DownloadError",
        "DownloadError",
    ]
    assert result.failure_count == 2
    assert result.summary_csv_path.exists()
    assert result.summary_json_path.exists()


def test_proteingym_shared_benchmark_failure_preserves_unknown_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pipeline_module,
        "list_datasets",
        lambda *args, **kwargs: [_batch_metadata_record("KNOWN")],
    )
    monkeypatch.setattr(
        pipeline_module,
        "_acquire_proteingym_benchmark",
        lambda **kwargs: (_ for _ in ()).throw(DownloadError("benchmark failed")),
    )

    result = download_and_standardize_datasets(
        "proteingym",
        ["UNKNOWN", "KNOWN"],
        output_dir=tmp_path / "output",
        cache=FilesystemCache(tmp_path / "cache"),
    )

    assert [entry.error_type for entry in result.entries] == [
        "DatasetNotFoundError",
        "DownloadError",
    ]


def test_mavedb_batch_reuses_cache_and_refreshes_each_unique_urn_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_ids = (
        "urn:mavedb:00000001-a-1",
        "urn:mavedb:00000002-a-1",
    )

    def metadata(source: str, dataset_id: str) -> DatasetRecord:
        assert source == "mavedb"
        return DatasetRecord(
            source="mavedb",
            dataset_id=dataset_id,
            title="Batch assay",
            target_id="GENE",
            variant_type=None,
            n_variants=1,
            raw_metadata={
                "urn": dataset_id,
                "targetGenes": [{"name": "GENE"}],
                "targetSequence": {"sequence": "MKT"},
            },
        )

    download_calls: list[str] = []

    def offline_download(url: str, output_path: str | Path, **kwargs: Any) -> Path:
        del kwargs
        download_calls.append(url)
        path = Path(output_path)
        path.write_text("hgvs_pro,score\np.Met1Ala,0.5\n", encoding="utf-8")
        return path

    monkeypatch.setattr(pipeline_module, "get_dataset_metadata", metadata)
    monkeypatch.setattr(fetch_module, "download_file", offline_download)
    cache = FilesystemCache(tmp_path / "cache")

    first = download_and_standardize_datasets(
        "mavedb",
        dataset_ids,
        output_dir=tmp_path / "first",
        cache=cache,
        add_wildtype_row=True,
    )
    second = download_and_standardize_datasets(
        "mavedb",
        dataset_ids,
        output_dir=tmp_path / "second",
        cache=cache,
    )
    refreshed = download_and_standardize_datasets(
        "mavedb",
        dataset_ids,
        output_dir=tmp_path / "refreshed",
        cache=cache,
        refresh=True,
    )

    assert first.success_count == second.success_count == refreshed.success_count == 2
    assert len(download_calls) == 4
    assert all(cache.exists("mavedb", dataset_id) for dataset_id in dataset_ids)
    first_table = pd.read_csv(first.entries[0].result.dataset_path)  # type: ignore[union-attr]
    assert first_table["is_synthetic"].tolist() == [True, False]


def _write_fake_download_result(output_dir: Path) -> DatasetDownloadResult:
    """Write a representative bundle for batch orchestration failure tests."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "standardized.csv"
    summary_csv_path = output_dir / "summary.csv"
    summary_json_path = output_dir / "summary.json"
    dataset_path.write_text("score_raw\n0.5\n", encoding="utf-8")
    summary_csv_path.write_text("status\nOK\n", encoding="utf-8")
    summary_json_path.write_text('[{"status": "OK"}]\n', encoding="utf-8")
    return DatasetDownloadResult(
        dataset_path=dataset_path,
        summary_csv_path=summary_csv_path,
        summary_json_path=summary_json_path,
        summary={
            "target_protein": "GENE",
            "wt_length": 3,
            "raw_rows": 1,
            "validated_rows": 1,
            "discarded_rows": 0,
            "output_rows": 1,
            "wildtype_rows": 0,
            "synthetic_wildtype_rows": 0,
        },
    )


def test_expected_failed_rebuild_preserves_old_bundle_and_publishes_partial_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_ids = (
        "urn:mavedb:00000001-a-1",
        "urn:mavedb:00000002-a-1",
    )
    output_root = tmp_path / "output"
    failed_dir = (
        output_root
        / "mavedb"
        / pipeline_module._portable_dataset_component(dataset_ids[1])
    )
    failed_dir.mkdir(parents=True)
    old_bundle = {}
    for name in ("standardized.csv", "summary.csv", "summary.json"):
        path = failed_dir / name
        path.write_text(f"old {name}", encoding="utf-8")
        old_bundle[path] = path.read_bytes()

    def single(source: str, dataset_id: str, **kwargs: Any):
        assert source == "mavedb"
        if dataset_id == dataset_ids[1]:
            raise DownloadError("expected rebuild failure")
        return _write_fake_download_result(kwargs["output_dir"])

    monkeypatch.setattr(
        pipeline_module,
        "download_and_standardize_dataset",
        single,
    )
    result = download_and_standardize_datasets(
        "mavedb",
        dataset_ids,
        output_dir=output_root,
        cache=FilesystemCache(tmp_path / "cache"),
        overwrite=True,
    )

    assert result.success_count == 1
    assert result.failure_count == 1
    assert {path: path.read_bytes() for path in old_bundle} == old_bundle
    records = json.loads(result.summary_json_path.read_text(encoding="utf-8"))
    assert [record["status"] for record in records] == ["SUCCESS", "ERROR"]


def test_unexpected_batch_error_preserves_completed_bundle_and_old_aggregate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_ids = (
        "urn:mavedb:00000001-a-1",
        "urn:mavedb:00000002-a-1",
    )
    output_root = tmp_path / "output"
    output_root.mkdir()
    aggregate_paths = (
        output_root / "download-summary.csv",
        output_root / "download-summary.json",
    )
    for path in aggregate_paths:
        path.write_text(f"old {path.name}", encoding="utf-8")
    original_aggregate = {path: path.read_bytes() for path in aggregate_paths}
    completed_dir: Path | None = None
    cause = RuntimeError("unexpected")

    def single(source: str, dataset_id: str, **kwargs: Any):
        nonlocal completed_dir
        assert source == "mavedb"
        if dataset_id == dataset_ids[1]:
            raise cause
        completed_dir = kwargs["output_dir"]
        return _write_fake_download_result(completed_dir)

    monkeypatch.setattr(
        pipeline_module,
        "download_and_standardize_dataset",
        single,
    )

    with pytest.raises(RuntimeError) as exc_info:
        download_and_standardize_datasets(
            "mavedb",
            dataset_ids,
            output_dir=output_root,
            cache=FilesystemCache(tmp_path / "cache"),
            overwrite=True,
        )

    assert exc_info.value is cause
    assert completed_dir is not None
    assert (completed_dir / "standardized.csv").exists()
    assert {path: path.read_bytes() for path in aggregate_paths} == original_aggregate
