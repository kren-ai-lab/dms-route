from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import dms_parser.pipeline as pipeline_module
from dms_parser import (
    InvalidPipelineOptionError,
    PipelineResult,
    run_pipeline,
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
