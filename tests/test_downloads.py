from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import requests

import dms_parser
import dms_parser.downloads as downloads_module
import dms_parser.fetch as fetch_module
import dms_parser.pipeline as pipeline_module
from dms_parser import (
    DatasetBatchDownloadEntry,
    DatasetBatchDownloadResult,
    DatasetDownloadResult,
    DatasetRecord,
    FilesystemCache,
    download_and_standardize_dataset,
    download_and_standardize_datasets,
    get_proteingym_resource,
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


def test_download_api_is_public() -> None:
    public_names = (
        "DatasetDownloadResult",
        "DatasetBatchDownloadEntry",
        "DatasetBatchDownloadResult",
        "download_and_standardize_dataset",
        "download_and_standardize_datasets",
    )

    for name in public_names:
        public_value = getattr(dms_parser, name)
        assert public_value is getattr(downloads_module, name)
        assert public_value is getattr(pipeline_module, name)


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
        downloads_module,
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
        downloads_module,
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
        downloads_module,
        "get_dataset_metadata",
        lambda *args, **kwargs: (_ for _ in ()).throw(error),
    )

    with pytest.raises(DatasetNotFoundError, match="was not found"):
        downloads_module._get_mavedb_download_metadata(
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
        downloads_module._PROTEINGYM_SUBSTITUTIONS_BENCHMARK_CACHE_ID,
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
        target_id="G\u00c9NE",
        variant_type=None,
        n_variants=2,
        raw_metadata={
            "urn": dataset_id,
            "targetGenes": [{"name": "G\u00c9NE"}],
            "targetSequence": {"sequence": "MKT"},
        },
    )
    monkeypatch.setattr(
        downloads_module,
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
    assert "G\u00c9NE" in second.summary_json_path.read_text(encoding="utf-8")

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
        downloads_module.download_and_standardize_datasets
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
    component = downloads_module._portable_dataset_component(dataset_id)
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

    assert downloads_module._portable_dataset_component(dataset_id) == (
        f"id-urn-mavedb-00000001-a-1--{digest}"
    )


def test_batch_rejects_case_insensitive_derived_collision_before_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "output"
    monkeypatch.setattr(
        downloads_module,
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
        downloads_module,
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
            "source_note": ["first", "first", "caf\u00c3\u00a9"],
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
    original_read_table = downloads_module.read_table

    def count_reads(path: str | Path, **kwargs: Any) -> pd.DataFrame:
        nonlocal parquet_reads
        if Path(path).suffix == ".parquet":
            parquet_reads += 1
        return original_read_table(path, **kwargs)

    monkeypatch.setattr(fetch_module, "download_file", offline_download)
    monkeypatch.setattr(downloads_module, "read_table", count_reads)
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
        downloads_module._PROTEINGYM_SUBSTITUTIONS_BENCHMARK_CACHE_ID,
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
    assert assay_two["source_note"].iloc[1] == "caf\u00c3\u00a9"
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
        downloads_module,
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
        downloads_module,
        "list_datasets",
        lambda *args, **kwargs: [_batch_metadata_record("KNOWN")],
    )
    monkeypatch.setattr(
        downloads_module,
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

    monkeypatch.setattr(downloads_module, "get_dataset_metadata", metadata)
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
        / downloads_module._portable_dataset_component(dataset_ids[1])
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
        downloads_module,
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
        downloads_module,
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
