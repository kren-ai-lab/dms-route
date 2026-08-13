from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import dmsroute.pipeline as pipeline_module
from dmsroute import get_proteingym_resource
from tests._factories import standardized_table


def _install_offline_proteingym_tables(
    monkeypatch: pytest.MonkeyPatch,
    metadata: pd.DataFrame,
    shared_data: pd.DataFrame,
    download_calls: list[tuple[str, str]],
    read_calls: list[str],
) -> None:
    """Route configured ProteinGym acquisition to in-memory source tables."""
    def download(url, output_path, overwrite=False):
        assert overwrite is False
        download_calls.append((url, Path(output_path).name))
        return Path(output_path)

    def read(path):
        name = Path(path).name
        read_calls.append(name)
        if name.endswith(".csv") and name.startswith("DMS_"):
            return metadata.copy()
        return shared_data.copy()

    monkeypatch.setattr(pipeline_module, "download_file", download)
    monkeypatch.setattr(pipeline_module, "read_table", read)


def _indel_metadata(*dataset_ids: str) -> pd.DataFrame:
    """Return compact registered-resource metadata for indel assays."""
    return pd.DataFrame(
        {
            "DMS_filename": [f"{dataset_id}.csv" for dataset_id in dataset_ids],
            "DMS_id": list(dataset_ids),
            "target_seq": ["MKT"] * len(dataset_ids),
            "UniProt_ID": ["P12345"] * len(dataset_ids),
            "molecule_name": ["Protein"] * len(dataset_ids),
            "gene": ["GENE1"] * len(dataset_ids),
        }
    )


def test_substitutions_still_dispatch_to_substitution_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = _indel_metadata("ASSAY")
    shared_data = pd.DataFrame(
        {"DMS_id": ["ASSAY"], "mutant": ["M1A"], "DMS_score": [0.5]}
    )
    download_calls: list[tuple[str, str]] = []
    read_calls: list[str] = []
    _install_offline_proteingym_tables(
        monkeypatch,
        metadata,
        shared_data,
        download_calls,
        read_calls,
    )
    builder_calls: list[dict[str, Any]] = []

    def build_substitutions(**kwargs: Any) -> pd.DataFrame:
        builder_calls.append(kwargs)
        return standardized_table()

    monkeypatch.setattr(
        pipeline_module,
        "build_proteingym_dataset",
        build_substitutions,
    )
    monkeypatch.setattr(
        pipeline_module,
        "build_proteingym_indel_dataset",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("substitution resource used indel builder")
        ),
    )

    summary = pipeline_module.process_proteingym(
        {
            "resource": "dms_substitutions",
            "dir_base": tmp_path / "proteingym",
            "datasets": [{"dataset_id": "ASSAY"}],
        }
    )

    assert summary[0]["status"] == "OK"
    assert len(builder_calls) == 1
    assert builder_calls[0]["variant_col"] == "mutant"
    assert "mutated_sequence_col" not in builder_calls[0]


def test_indels_select_registered_resources_and_indel_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resource = get_proteingym_resource("dms_indels", require_processing=True)
    metadata = _indel_metadata("ASSAY")
    shared_data = pd.DataFrame(
        {
            "DMS_id": ["ASSAY"],
            "target_seq": ["MKT"],
            "mutated_sequence": ["MT"],
            "DMS_score": [0.5],
        }
    )
    download_calls: list[tuple[str, str]] = []
    read_calls: list[str] = []
    _install_offline_proteingym_tables(
        monkeypatch,
        metadata,
        shared_data,
        download_calls,
        read_calls,
    )
    builder_calls: list[dict[str, Any]] = []

    def build_indels(**kwargs: Any) -> pd.DataFrame:
        builder_calls.append(kwargs)
        return standardized_table()

    monkeypatch.setattr(
        pipeline_module,
        "build_proteingym_dataset",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("indel resource used substitution builder")
        ),
    )
    monkeypatch.setattr(
        pipeline_module,
        "build_proteingym_indel_dataset",
        build_indels,
    )

    summary = pipeline_module.process_proteingym(
        {
            "resource": "dms_indels",
            "dir_base": tmp_path / "proteingym",
            "default_build_kwargs": {
                "variant_col": "must_not_be_forwarded",
                "strict_variant_parsing": True,
            },
            "datasets": [{"dataset_id": "ASSAY"}],
        }
    )

    assert summary[0]["status"] == "OK"
    assert download_calls == [
        (resource.metadata_url, resource.metadata_filename),
        (resource.data_url, resource.data_filename),
    ]
    assert read_calls == [resource.metadata_filename, resource.data_filename]
    assert len(builder_calls) == 1
    call = builder_calls[0]
    assert call["score_col"] == "DMS_score"
    assert call["mutated_sequence_col"] == "mutated_sequence"
    assert call["target_sequence_col"] == "target_seq"
    assert call["add_relative_score"] is False
    assert call["add_binary_label"] is False
    assert call["add_wildtype_row"] is False
    assert call["drop_failed"] is False
    assert "variant_col" not in call
    assert "strict_variant_parsing" not in call


def test_configured_indel_assay_writes_authoritative_sequences(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = _indel_metadata("ASSAY")
    shared_data = pd.DataFrame(
        {
            "DMS_id": ["ASSAY", "ASSAY"],
            "target_seq": ["MKT", "MKT"],
            "mutated_sequence": ["MKT", "MT"],
            "DMS_score": [1.0, 0.5],
            "DMS_score_bin": [1, 0],
            "mutant": [None, None],
        }
    )
    download_calls: list[tuple[str, str]] = []
    read_calls: list[str] = []
    _install_offline_proteingym_tables(
        monkeypatch,
        metadata,
        shared_data,
        download_calls,
        read_calls,
    )
    source_root = tmp_path / "proteingym"

    summary = pipeline_module.process_proteingym(
        {
            "resource": "dms_indels",
            "dir_base": source_root,
            "datasets": [{"dataset_id": "ASSAY"}],
        }
    )
    output = pd.read_csv(source_root / "processed" / "ASSAY_processed.csv")

    assert summary[0]["status"] == "OK"
    assert output["mutated_sequence"].tolist() == ["MKT", "MT"]
    assert output["variant"].isna().all()
    assert output.loc[0, "n_mutations"] == 0
    assert pd.isna(output.loc[1, "n_mutations"])
    assert output["DMS_score_bin"].tolist() == [1, 0]
    assert "score_log_ratio" not in output.columns
    assert "score_binary_like" not in output.columns


def test_indel_transform_and_drop_failed_remain_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = _indel_metadata("ASSAY")
    shared_data = pd.DataFrame(
        {
            "DMS_id": ["ASSAY", "ASSAY", "ASSAY"],
            "target_seq": ["MKT"] * 3,
            "mutated_sequence": ["MKT", "MT", "MJ"],
            "DMS_score": [1.0, 0.25, -1.0],
        }
    )
    download_calls: list[tuple[str, str]] = []
    read_calls: list[str] = []
    _install_offline_proteingym_tables(
        monkeypatch,
        metadata,
        shared_data,
        download_calls,
        read_calls,
    )
    source_root = tmp_path / "proteingym"

    summary = pipeline_module.process_proteingym(
        {
            "resource": "dms_indels",
            "dir_base": source_root,
            "default_build_kwargs": {
                "drop_failed": True,
                "add_relative_score": True,
                "relative_method": "difference",
                "relative_output_col": "score_delta",
            },
            "datasets": [{"dataset_id": "ASSAY"}],
        }
    )
    output = pd.read_csv(source_root / "processed" / "ASSAY_processed.csv")

    assert summary[0]["status"] == "OK"
    assert output["mutated_sequence"].tolist() == ["MKT", "MT"]
    assert output["score_raw"].tolist() == [1.0, 0.25]
    assert output["score_delta"].tolist() == [0.0, -0.75]
    assert output["status"].tolist() == ["OK", "OK"]


def test_multiple_indel_assays_share_one_parquet_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resource = get_proteingym_resource("dms_indels")
    metadata = _indel_metadata("FIRST", "SECOND")
    shared_data = pd.DataFrame(
        {
            "DMS_id": ["FIRST", "SECOND"],
            "target_seq": ["MKT", "MKT"],
            "mutated_sequence": ["MT", "MKTT"],
            "DMS_score": [0.5, -0.5],
        }
    )
    download_calls: list[tuple[str, str]] = []
    read_calls: list[str] = []
    _install_offline_proteingym_tables(
        monkeypatch,
        metadata,
        shared_data,
        download_calls,
        read_calls,
    )

    summary = pipeline_module.process_proteingym(
        {
            "resource": "dms_indels",
            "dir_base": tmp_path / "proteingym",
            "datasets": [
                {"dataset_id": "FIRST"},
                {"dataset_id": "SECOND"},
            ],
        }
    )

    assert [row["status"] for row in summary] == ["OK", "OK"]
    assert read_calls.count(resource.data_filename) == 1
    assert download_calls.count((resource.data_url, resource.data_filename)) == 1


def test_missing_indel_assay_keeps_expected_failure_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = _indel_metadata("AVAILABLE")
    shared_data = pd.DataFrame(
        {
            "DMS_id": ["AVAILABLE"],
            "target_seq": ["MKT"],
            "mutated_sequence": ["MT"],
            "DMS_score": [0.5],
        }
    )
    download_calls: list[tuple[str, str]] = []
    read_calls: list[str] = []
    _install_offline_proteingym_tables(
        monkeypatch,
        metadata,
        shared_data,
        download_calls,
        read_calls,
    )

    summary = pipeline_module.process_proteingym(
        {
            "resource": "dms_indels",
            "dir_base": tmp_path / "proteingym",
            "datasets": [{"dataset_id": "MISSING"}],
        }
    )

    assert summary[0]["status"] == "ERROR"
    assert "No information found" in summary[0]["error"]


def test_unexpected_indel_builder_exception_propagates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = _indel_metadata("ASSAY")
    shared_data = pd.DataFrame(
        {
            "DMS_id": ["ASSAY"],
            "target_seq": ["MKT"],
            "mutated_sequence": ["MT"],
            "DMS_score": [0.5],
        }
    )
    download_calls: list[tuple[str, str]] = []
    read_calls: list[str] = []
    _install_offline_proteingym_tables(
        monkeypatch,
        metadata,
        shared_data,
        download_calls,
        read_calls,
    )
    cause = RuntimeError("unexpected indel builder failure")
    monkeypatch.setattr(
        pipeline_module,
        "build_proteingym_indel_dataset",
        lambda **kwargs: (_ for _ in ()).throw(cause),
    )

    with pytest.raises(RuntimeError) as error:
        pipeline_module.process_proteingym(
            {
                "resource": "dms_indels",
                "dir_base": tmp_path / "proteingym",
                "datasets": [{"dataset_id": "ASSAY"}],
            }
        )

    assert error.value is cause
