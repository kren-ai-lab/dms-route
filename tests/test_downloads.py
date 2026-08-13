from __future__ import annotations

import hashlib
import json
import inspect
import zipfile
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import requests

import dms_parser
import dms_parser.downloads as downloads_module
import dms_parser.fetch as fetch_module
import dms_parser.pipeline as pipeline_module
import dms_parser.sources._mavedb_snapshot_dataset as snapshot_dataset_module
import dms_parser.sources.mavedb_snapshot_tables as snapshot_tables_module
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
    InvalidDatasetError,
    InvalidPipelineOptionError,
    MissingWildTypeError,
    SourceConfigurationError,
    WildTypeConflictError,
)
from dms_parser.sources.mavedb_snapshots import MAVEDB_ZENODO_CONCEPT_DOI
from dms_parser.sources.proteingym_catalog import (
    PROTEINGYM_SUBSTITUTIONS_CACHE_ID,
)


def _write_cached_snapshot(
    tmp_path: Path,
    score_sets: list[dict[str, Any]],
    *,
    current_ids: list[str],
) -> FilesystemCache:
    """Write one tiny fully valid cached ZIP snapshot without network I/O."""
    cache = FilesystemCache(tmp_path / "cache")
    record_id = "20840937"
    entry = cache.root / "mavedb" / "snapshots" / record_id
    entry.mkdir(parents=True)
    main = {
        "title": "Offline test snapshot",
        "asOf": "2026-06-24T18:13:01Z",
        "experimentSets": [
            {
                "urn": "urn:mavedb:00000001",
                "experiments": [
                    {
                        "urn": "urn:mavedb:00000001-a",
                        "title": "Offline experiment",
                        "scoreSetUrns": current_ids,
                        "scoreSets": score_sets,
                    }
                ],
            }
        ],
    }
    main_bytes = (json.dumps(main, sort_keys=True) + "\n").encode("utf-8")
    main_path = entry / "main.json"
    main_path.write_bytes(main_bytes)
    archive_path = entry / "mavedb-dump.test.zip"
    archive_root = "mavedb-dump.2026062418131"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(f"{archive_root}/main.json", main_bytes)
        for score_set in score_sets:
            stem = score_set["urn"].replace(":", "-")
            archive.writestr(
                f"{archive_root}/csv/{stem}.scores.csv",
                (
                    "accession,hgvs_nt,hgvs_splice,hgvs_pro,scores.score,"
                    "scores.sd,scores.se,scores.df\n"
                    f"{score_set['urn']}#1,c.=,,p.=,1.0,0.1,0.01,10\n"
                    f"{score_set['urn']}#2,c.1A>G,,p.Met1Ala,0.5,0.2,0.02,20\n"
                    f"{score_set['urn']}#3,c.4_6del,,p.[Gly2del],-1.0,0.3,0.03,30\n"
                ),
            )
            archive.writestr(
                f"{archive_root}/csv/._{stem}.scores.csv",
                "offline AppleDouble sidecar",
            )
            archive.writestr(
                f"{archive_root}/csv/{stem}.counts.csv",
                "hgvs_pro,count\np.=,10\n",
            )
    archive_bytes = archive_path.read_bytes()
    metadata = {
        "record_id": record_id,
        "doi": f"10.5281/zenodo.{record_id}",
        "concept_doi": MAVEDB_ZENODO_CONCEPT_DOI,
        "publication_date": "2026-06-24",
        "archive_filename": archive_path.name,
        "archive_size": len(archive_bytes),
        "checksum": f"md5:{hashlib.md5(archive_bytes).hexdigest()}",
        "download_url": "https://zenodo.org/offline-test",
        "archive_format": "zip",
        "main_json_size": len(main_bytes),
        "main_json_sha256": hashlib.sha256(main_bytes).hexdigest(),
    }
    (entry / "snapshot.json").write_text(
        json.dumps(metadata, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return cache


def _snapshot_score_set(dataset_id: str, *, gene: str = "GENE") -> dict[str, Any]:
    """Return score-set metadata accepted by discovery and builders."""
    return {
        "urn": dataset_id,
        "title": f"{gene} score set",
        "numVariants": 3,
        "targetGenes": [{"name": gene}],
        "targetSequence": {"sequence": "MKT"},
    }


def _build_offline_mavedb_scores(
    tmp_path: Path,
    table: str,
) -> pd.DataFrame:
    """Build one local MaveDB score table through standard download logic."""
    dataset_id = "urn:mavedb:00000001-a-4"
    scores_path = tmp_path / "scores.csv"
    scores_path.write_text(table, encoding="utf-8")
    built, _ = downloads_module._build_mavedb_download(
        dataset_id,
        downloads_module._MaveDBDownloadInput(
            scores_path=scores_path,
            metadata=_snapshot_score_set(dataset_id),
            provenance={},
        ),
        drop_failed=False,
        add_wildtype_row=False,
        standardization=downloads_module._StandardizationOptions(),
    )
    return built


def _ambiguous_mavedb_download_input(
    tmp_path: Path,
    dataset_id: str,
) -> downloads_module._MaveDBDownloadInput:
    """Return local MaveDB inputs with conflicting protein-WT scores."""
    scores_path = tmp_path / "ambiguous-scores.csv"
    scores_path.write_text(
        "hgvs_pro,score\n"
        "p.=,1.0\n"
        "p.[=;=],2.0\n"
        "p.Met1Ala,0.5\n",
        encoding="utf-8",
    )
    return downloads_module._MaveDBDownloadInput(
        scores_path=scores_path,
        metadata=_snapshot_score_set(dataset_id),
        provenance={},
    )


def test_legacy_mavedb_score_column_remains_supported(tmp_path: Path) -> None:
    table = _build_offline_mavedb_scores(
        tmp_path,
        "hgvs_pro,score\np.=,2.5\np.Met1Ala,-0.125\n",
    )

    assert table["score_raw"].tolist() == [2.5, -0.125]
    assert table["mutated_sequence"].tolist() == ["MKT", "AKT"]


def test_single_mavedb_download_standardizes_bounded_indels_offline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_id = "urn:mavedb:00000001-a-4"
    scores_path = tmp_path / "indel-scores.csv"
    scores_path.write_text(
        "hgvs_pro,score\n"
        "p.Cys2del,0.25\n"
        "p.Asp1_Cys2insLys,1.5\n",
        encoding="utf-8",
    )
    metadata = _snapshot_score_set(dataset_id)
    metadata["targetSequence"] = {"sequence": "DCA"}
    acquired = downloads_module._MaveDBDownloadInput(
        scores_path=scores_path,
        metadata=metadata,
        provenance={},
    )
    monkeypatch.setattr(
        downloads_module,
        "_acquire_mavedb_api_dataset",
        lambda *args, **kwargs: acquired,
    )

    result = download_and_standardize_dataset(
        "mavedb",
        dataset_id,
        output_dir=tmp_path / "indel-download",
        cache=FilesystemCache(tmp_path / "cache"),
    )
    table = pd.read_csv(result.dataset_path)

    assert table["variant"].tolist() == ["C2del", "D1_C2insK"]
    assert table["mutated_sequence"].tolist() == ["DA", "DKCA"]
    assert table["score_raw"].tolist() == [0.25, 1.5]
    assert table["status"].tolist() == ["OK", "OK"]
    assert table["is_wildtype"].tolist() == [False, False]
    assert table["is_synthetic"].tolist() == [False, False]
    assert table["n_mutations"].tolist() == [1, 1]


def test_single_mavedb_download_publishes_ambiguous_raw_wt_scores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_id = "urn:mavedb:00000001-a-4"
    acquired = _ambiguous_mavedb_download_input(tmp_path, dataset_id)
    monkeypatch.setattr(
        downloads_module,
        "_acquire_mavedb_api_dataset",
        lambda *args, **kwargs: acquired,
    )

    result = download_and_standardize_dataset(
        "mavedb",
        dataset_id,
        output_dir=tmp_path / "single",
        cache=FilesystemCache(tmp_path / "cache"),
    )

    assert result.dataset_path.is_file()
    assert result.summary_csv_path.is_file()
    assert result.summary_json_path.is_file()
    assert pd.read_csv(result.dataset_path)["score_raw"].tolist() == [
        1.0,
        2.0,
        0.5,
    ]
    assert result.summary["status"] == "OK"
    assert result.summary["wt_score"] is None
    assert result.summary["wt_score_provenance"] is None
    assert result.summary["observed_wildtype_row"] is True
    assert (
        result.summary["wt_score_unavailable_reason"]
        == "conflicting_observed_wildtype_scores"
    )


def test_mavedb_batch_distinguishes_raw_and_transformed_ambiguous_wt_scores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_id = "urn:mavedb:00000001-a-4"
    acquired = _ambiguous_mavedb_download_input(tmp_path, dataset_id)
    monkeypatch.setattr(
        downloads_module,
        "_acquire_mavedb_api_dataset",
        lambda *args, **kwargs: acquired,
    )
    cache = FilesystemCache(tmp_path / "cache")

    raw = download_and_standardize_datasets(
        "mavedb",
        [dataset_id],
        output_dir=tmp_path / "raw-batch",
        cache=cache,
    )
    transformed = download_and_standardize_datasets(
        "mavedb",
        [dataset_id],
        output_dir=tmp_path / "transformed-batch",
        cache=cache,
        add_relative_score=True,
        relative_method="difference",
    )

    assert raw.exit_code == 0
    assert raw.success_count == 1
    assert raw.entries[0].result is not None
    raw_records = json.loads(raw.summary_json_path.read_text(encoding="utf-8"))
    assert raw_records[0]["status"] == "SUCCESS"
    assert (
        raw_records[0]["wt_score_unavailable_reason"]
        == "conflicting_observed_wildtype_scores"
    )
    assert transformed.exit_code == 1
    assert transformed.failure_count == 1
    assert transformed.entries[0].result is None
    assert transformed.entries[0].error_type == "WildTypeConflictError"
    transformed_records = json.loads(
        transformed.summary_json_path.read_text(encoding="utf-8")
    )
    assert transformed_records[0]["status"] == "ERROR"


def test_mavedb_dna_target_sequence_is_translated_before_standardization(
    tmp_path: Path,
) -> None:
    dataset_id = "urn:mavedb:00000001-a-4"
    scores_path = tmp_path / "scores.csv"
    scores_path.write_text(
        "hgvs_pro,score\np.=,1.0\np.Met1Ala,0.5\n",
        encoding="utf-8",
    )
    metadata = _snapshot_score_set(dataset_id)
    metadata["targetSequence"] = {"sequence": "ATGAAAACC"}

    built, _ = downloads_module._build_mavedb_download(
        dataset_id,
        downloads_module._MaveDBDownloadInput(
            scores_path=scores_path,
            metadata=metadata,
            provenance={},
        ),
        drop_failed=False,
        add_wildtype_row=False,
        standardization=downloads_module._StandardizationOptions(),
    )

    assert built["wt_sequence"].tolist() == ["MKT", "MKT"]
    assert built["mutated_sequence"].tolist() == ["MKT", "AKT"]


@pytest.mark.parametrize(
    "table",
    [
        (
            "accession,hgvs_pro,scores.sd,scores.se,scores.df\n"
            "urn:mavedb:00000001-a-4#1,p.=,0.1,0.01,10\n"
        ),
        (
            "accession,scores.score,scores.sd\n"
            "urn:mavedb:00000001-a-4#1,1.0,0.1\n"
        ),
    ],
)
def test_mavedb_missing_primary_score_or_hgvs_is_rejected(
    table: str,
    tmp_path: Path,
) -> None:
    with pytest.raises(InvalidDatasetError, match="score nor HGVS"):
        _build_offline_mavedb_scores(tmp_path, table)


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

    for function_name in (
        "download_and_standardize_dataset",
        "download_and_standardize_datasets",
    ):
        parameters = inspect.signature(
            getattr(downloads_module, function_name)
        ).parameters
        assert parameters["acquisition"].default is None
        assert parameters["snapshot_record_id"].default is None
        assert parameters["include_superseded"].default is False
        assert parameters["add_relative_score"].default is False
        assert parameters["relative_method"].default == "log_ratio"
        assert parameters["add_binary_label"].default is False
        assert parameters["wt_sequence"].default is None
        assert parameters["wt_score"].default is None


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


def test_batch_wt_mapping_validation_precedes_cache_and_output(tmp_path: Path) -> None:
    cache = FilesystemCache(tmp_path / "cache")
    output_dir = tmp_path / "output"

    with pytest.raises(InvalidPipelineOptionError, match="unrequested"):
        download_and_standardize_datasets(
            "proteingym",
            ["ASSAY_1"],
            output_dir=output_dir,
            cache=cache,
            wt_score={"ASSAY_2": 1.0},
        )

    assert not cache.root.exists()
    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("batch", "options"),
    [
        (False, {"add_relative_score": True, "relative_output_col": "score_raw"}),
        (
            False,
            {
                "add_relative_score": True,
                "relative_output_col": "parsed_variant",
            },
        ),
        (
            False,
            {
                "add_relative_score": True,
                "relative_output_col": "parsed_position",
            },
        ),
        (
            False,
            {
                "add_relative_score": True,
                "relative_output_col": "parsed_wt_aa",
            },
        ),
        (
            False,
            {
                "add_relative_score": True,
                "relative_output_col": "parsed_mut_aa",
            },
        ),
        (
            False,
            {
                "add_relative_score": True,
                "relative_output_col": "parsed_is_wildtype",
            },
        ),
        (
            False,
            {
                "add_relative_score": True,
                "relative_output_col": "parsed_n_mutations",
            },
        ),
        (
            False,
            {
                "add_relative_score": True,
                "relative_output_col": "parsed_mutations",
            },
        ),
        (
            False,
            {
                "add_relative_score": True,
                "add_binary_label": True,
                "binary_output_col": "status",
            },
        ),
        (
            False,
            {
                "add_relative_score": True,
                "add_binary_label": True,
                "binary_output_col": "parsed_variant",
            },
        ),
        (
            True,
            {
                "add_relative_score": True,
                "add_binary_label": True,
                "relative_output_col": "generated",
                "binary_output_col": "generated",
            },
        ),
    ],
)
def test_output_collision_validation_precedes_acquisition_cache_and_output(
    batch: bool,
    options: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        downloads_module,
        "get_dataset_metadata",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("invalid output name reached acquisition")
        ),
    )
    cache = FilesystemCache(tmp_path / "cache")
    output_dir = tmp_path / "output"

    with pytest.raises(InvalidPipelineOptionError):
        if batch:
            download_and_standardize_datasets(
                "proteingym",
                ["ASSAY_1"],
                output_dir=output_dir,
                cache=cache,
                **options,
            )
        else:
            download_and_standardize_dataset(
                "proteingym",
                "ASSAY_1",
                output_dir=output_dir,
                cache=cache,
                **options,
            )

    assert not cache.root.exists()
    assert not output_dir.exists()


@pytest.mark.parametrize(
    "options",
    [
        {"add_relative_score": True, "relative_output_col": "parsed_variant"},
        {"add_relative_score": True, "relative_output_col": "parsed_position"},
        {"add_relative_score": True, "relative_output_col": "parsed_wt_aa"},
        {"add_relative_score": True, "relative_output_col": "parsed_mut_aa"},
        {
            "add_relative_score": True,
            "relative_output_col": "parsed_is_wildtype",
        },
        {
            "add_relative_score": True,
            "relative_output_col": "parsed_n_mutations",
        },
        {"add_relative_score": True, "relative_output_col": "parsed_mutations"},
        {
            "add_relative_score": True,
            "add_binary_label": True,
            "binary_output_col": "parsed_variant",
        },
    ],
)
def test_parsed_variant_collision_precedes_snapshot_extraction(
    options: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acquisition_calls = 0

    def forbidden_acquisition(*args: Any, **kwargs: Any) -> None:
        nonlocal acquisition_calls
        acquisition_calls += 1
        raise AssertionError("invalid output name reached snapshot extraction")

    monkeypatch.setattr(
        downloads_module,
        "acquire_cached_mavedb_snapshot_datasets",
        forbidden_acquisition,
    )
    cache = FilesystemCache(tmp_path / "cache")
    output_dir = tmp_path / "output"

    protected_column = options.get("relative_output_col", "parsed_variant")
    with pytest.raises(InvalidPipelineOptionError, match=str(protected_column)):
        download_and_standardize_dataset(
            "mavedb",
            "urn:mavedb:00000001-a-1",
            output_dir=output_dir,
            cache=cache,
            acquisition="snapshot",
            snapshot_record_id="20840937",
            **options,
        )

    assert acquisition_calls == 0
    assert not cache.root.exists()
    assert not output_dir.exists()


def test_proteingym_manual_sequence_cannot_replace_reference_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        downloads_module,
        "get_dataset_metadata",
        lambda *args, **kwargs: DatasetRecord(
            source="proteingym",
            dataset_id="ASSAY_1",
            title=None,
            target_id=None,
            variant_type="substitutions",
            n_variants=1,
            raw_metadata={"DMS_id": "ASSAY_1", "target_seq": "MKT"},
        ),
    )

    with pytest.raises(WildTypeConflictError, match="fallback conflicts"):
        download_and_standardize_dataset(
            "proteingym",
            "ASSAY_1",
            output_dir=tmp_path / "output",
            cache=FilesystemCache(tmp_path / "cache"),
            wt_sequence="AAA",
        )

    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "cache").exists()


def test_invalid_proteingym_metadata_sequence_is_invalid_dataset_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        downloads_module,
        "get_dataset_metadata",
        lambda *args, **kwargs: DatasetRecord(
            source="proteingym",
            dataset_id="ASSAY_1",
            title=None,
            target_id=None,
            variant_type="substitutions",
            n_variants=1,
            raw_metadata={
                "DMS_id": "ASSAY_1",
                "target_seq": "NOT-A-PROTEIN",
            },
        ),
    )

    with pytest.raises(InvalidDatasetError, match="proteingym_reference_metadata"):
        download_and_standardize_dataset(
            "proteingym",
            "ASSAY_1",
            output_dir=tmp_path / "output",
            cache=FilesystemCache(tmp_path / "cache"),
        )

    assert not (tmp_path / "cache").exists()
    assert not (tmp_path / "output").exists()


def test_invalid_user_sequence_fallback_is_invalid_option_before_acquisition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        downloads_module,
        "get_dataset_metadata",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("invalid fallback reached acquisition")
        ),
    )

    with pytest.raises(InvalidPipelineOptionError, match="user_fallback"):
        download_and_standardize_dataset(
            "proteingym",
            "ASSAY_1",
            output_dir=tmp_path / "output",
            cache=FilesystemCache(tmp_path / "cache"),
            wt_sequence="NOT-A-PROTEIN",
        )

    assert not (tmp_path / "cache").exists()
    assert not (tmp_path / "output").exists()


def test_conflicting_mavedb_metadata_sequences_publish_no_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acquired = downloads_module._MaveDBDownloadInput(
        scores_path=tmp_path / "unused.csv",
        metadata={
            "targetSequence": {"sequence": "MKT"},
            "nested": {"targetSequence": {"sequence": "AAA"}},
        },
        provenance={},
    )
    monkeypatch.setattr(
        downloads_module,
        "_acquire_mavedb_api_dataset",
        lambda *args, **kwargs: acquired,
    )
    output_dir = tmp_path / "output"

    with pytest.raises(WildTypeConflictError, match="Conflicting WT sequences"):
        download_and_standardize_dataset(
            "mavedb",
            "urn:mavedb:00000001-a-1",
            output_dir=output_dir,
            cache=FilesystemCache(tmp_path / "cache"),
        )

    assert not output_dir.exists()


def test_missing_wt_score_for_requested_transform_publishes_no_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        downloads_module,
        "get_dataset_metadata",
        lambda *args, **kwargs: DatasetRecord(
            source="proteingym",
            dataset_id="ASSAY_1",
            title=None,
            target_id=None,
            variant_type="substitutions",
            n_variants=1,
            raw_metadata={"DMS_id": "ASSAY_1", "target_seq": "MKT"},
        ),
    )
    monkeypatch.setattr(
        downloads_module,
        "_acquire_proteingym_benchmark",
        lambda **kwargs: pd.DataFrame(
            {"DMS_id": ["ASSAY_1"], "mutant": ["M1A"], "DMS_score": [0.5]}
        ),
    )
    output_dir = tmp_path / "output"

    with pytest.raises(MissingWildTypeError, match="provide wt_score"):
        download_and_standardize_dataset(
            "proteingym",
            "ASSAY_1",
            output_dir=output_dir,
            cache=FilesystemCache(tmp_path / "cache"),
            add_relative_score=True,
            relative_method="difference",
        )

    assert not output_dir.exists()


def test_download_numerical_transform_error_includes_dataset_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_id = "OVERFLOW_ASSAY"
    benchmark_table = pd.DataFrame(
        {
            "DMS_id": [dataset_id],
            "mutant": ["M1A"],
            "DMS_score": [np.finfo(float).max],
        }
    )
    monkeypatch.setattr(
        downloads_module,
        "get_dataset_metadata",
        lambda *args, **kwargs: DatasetRecord(
            source="proteingym",
            dataset_id=dataset_id,
            title=None,
            target_id=None,
            variant_type="substitutions",
            n_variants=1,
            raw_metadata={"DMS_id": dataset_id, "target_seq": "MKT"},
        ),
    )
    monkeypatch.setattr(
        downloads_module,
        "_acquire_proteingym_benchmark",
        lambda **kwargs: benchmark_table,
    )
    output_dir = tmp_path / "output"
    cache = FilesystemCache(tmp_path / "cache")

    with pytest.raises(InvalidDatasetError) as exc_info:
        download_and_standardize_dataset(
            "proteingym",
            dataset_id,
            output_dir=output_dir,
            cache=cache,
            wt_score=-np.finfo(float).max,
            add_relative_score=True,
            relative_method="difference",
            relative_output_col="score_difference",
        )

    message = str(exc_info.value)
    assert dataset_id in message
    assert "difference" in message
    assert "produced a non-finite result" in message
    assert isinstance(exc_info.value.__cause__, InvalidDatasetError)
    assert "score_difference" not in benchmark_table.columns
    assert benchmark_table["DMS_score"].iloc[0] == np.finfo(float).max
    assert not output_dir.exists()
    assert not cache.root.exists()


@pytest.mark.parametrize(
    ("source", "options"),
    [
        ("mavedb", {"acquisition": "snapshot"}),
        ("mavedb", {"acquisition": "snapshot", "snapshot_record_id": "latest"}),
        ("mavedb", {"acquisition": "api", "snapshot_record_id": "20840937"}),
        ("mavedb", {"include_superseded": True}),
        ("proteingym", {"acquisition": "api"}),
        (
            "proteingym",
            {"acquisition": "snapshot", "snapshot_record_id": "20840937"},
        ),
    ],
)
def test_acquisition_validation_precedes_cache_output_and_source_work(
    source: str,
    options: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        downloads_module,
        "_preflight_dataset_bundle",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("invalid acquisition reached output preflight")
        ),
    )
    monkeypatch.setattr(
        downloads_module,
        "get_dataset_metadata",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("invalid acquisition reached source metadata")
        ),
    )
    dataset_id = (
        "urn:mavedb:00000001-a-1"
        if source == "mavedb"
        else "ASSAY_1"
    )

    with pytest.raises(InvalidPipelineOptionError):
        download_and_standardize_dataset(
            source,
            dataset_id,
            output_dir=tmp_path / "output",
            cache=FilesystemCache(tmp_path / "cache"),
            **options,
        )

    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "cache").exists()


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
    assert second.summary["wt_sequence_provenance"] == (
        "proteingym_reference_metadata"
    )
    assert second.summary["wt_score"] is None
    assert second.summary["wt_score_provenance"] is None
    assert second.summary["observed_wildtype_row"] is False
    assert second.summary["synthetic_wildtype_inserted"] is True
    assert second.summary["output_file"] == str(second.dataset_path)
    assert list(second.summary) == [
        "source",
        "input",
        "status",
        "dataset_id",
        "target_protein",
        "wt_length",
        "wt_sequence_sha256",
        "wt_sequence_provenance",
        "wt_score",
        "wt_score_provenance",
        "observed_wildtype_row",
        "synthetic_wildtype_inserted",
        "requested_transformation",
        "transformed_output_column",
        "wt_score_unavailable_reason",
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

    rebuilt = download_and_standardize_dataset(
        "proteingym",
        "ASSAY_2",
        output_dir=tmp_path / "second",
        cache=cache,
        wt_score=1.0,
        add_relative_score=True,
        relative_method="difference",
        relative_output_col="score_difference",
        overwrite=True,
    )
    rebuilt_table = pd.read_csv(rebuilt.dataset_path)
    assert rebuilt_table["score_raw"].tolist() == [0.75]
    assert rebuilt_table["score_difference"].tolist() == [-0.25]
    assert rebuilt.summary["wt_score"] == 1.0
    assert rebuilt.summary["wt_score_provenance"] == "user_fallback"
    assert rebuilt.summary["requested_transformation"] == "difference"
    assert download_calls == [resource.metadata_url, resource.data_url]

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
    metadata_calls: list[str] = []

    def offline_metadata(source: str, requested_id: str) -> DatasetRecord:
        assert source == "mavedb"
        metadata_calls.append(requested_id)
        return metadata

    monkeypatch.setattr(downloads_module, "get_dataset_metadata", offline_metadata)
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
    assert metadata_calls == [dataset_id]
    assert cache.exists("mavedb", dataset_id)
    assert cache.exists(downloads_module._MAVEDB_METADATA_CACHE_SOURCE, dataset_id)
    metadata_manifest = cache.load_manifest(
        downloads_module._MAVEDB_METADATA_CACHE_SOURCE,
        dataset_id,
    )
    assert metadata_manifest.original_url == (
        f"{downloads_module.MAVEDB_API_URL.rstrip('/')}/score-sets/{dataset_id}"
    )
    cached_metadata_path = cache.resolve(
        downloads_module._MAVEDB_METADATA_CACHE_SOURCE,
        dataset_id,
    )
    assert cached_metadata_path is not None
    assert cached_metadata_path.read_text(encoding="utf-8") == (
        json.dumps(
            metadata.raw_metadata,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )
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

    rebuilt = download_and_standardize_dataset(
        "mavedb",
        dataset_id,
        output_dir=tmp_path / "second",
        cache=cache,
        wt_score=1.0,
        add_relative_score=True,
        relative_method="difference",
        relative_output_col="score_difference",
        overwrite=True,
    )
    rebuilt_table = pd.read_csv(rebuilt.dataset_path)
    assert rebuilt_table["score_raw"].tolist() == [0.75, -2.5]
    assert rebuilt_table["score_difference"].tolist() == [-0.25, -3.5]

    download_and_standardize_dataset(
        "mavedb",
        dataset_id,
        output_dir=tmp_path / "refreshed",
        cache=cache,
        refresh=True,
    )
    assert len(download_calls) == 2
    assert metadata_calls == [dataset_id, dataset_id]


def test_failed_mavedb_metadata_refresh_preserves_cached_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_id = "urn:mavedb:00000001-a-1"
    metadata = DatasetRecord(
        source="mavedb",
        dataset_id=dataset_id,
        title="Cached assay",
        target_id="GENE",
        variant_type=None,
        n_variants=1,
        raw_metadata={
            "urn": dataset_id,
            "targetGenes": [{"name": "GENE"}],
            "targetSequence": {"sequence": "MKT"},
        },
    )
    monkeypatch.setattr(
        downloads_module,
        "get_dataset_metadata",
        lambda *args, **kwargs: metadata,
    )

    def offline_download(url: str, output_path: str | Path, **kwargs: Any) -> Path:
        path = Path(output_path)
        path.write_text("hgvs_pro,score\np.Met1Ala,0.5\n", encoding="utf-8")
        return path

    monkeypatch.setattr(fetch_module, "download_file", offline_download)
    cache = FilesystemCache(tmp_path / "cache")
    download_and_standardize_dataset(
        "mavedb",
        dataset_id,
        output_dir=tmp_path / "first",
        cache=cache,
    )
    scores_path = cache.resolve("mavedb", dataset_id)
    metadata_path = cache.resolve(
        downloads_module._MAVEDB_METADATA_CACHE_SOURCE,
        dataset_id,
    )
    assert scores_path is not None and metadata_path is not None
    original_scores = scores_path.read_bytes()
    original_metadata = metadata_path.read_bytes()
    monkeypatch.setattr(
        downloads_module,
        "get_dataset_metadata",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            DownloadError("metadata refresh failed")
        ),
    )

    with pytest.raises(DownloadError, match="metadata refresh failed"):
        download_and_standardize_dataset(
            "mavedb",
            dataset_id,
            output_dir=tmp_path / "failed-refresh",
            cache=cache,
            refresh=True,
        )

    preserved_scores_path = cache.resolve("mavedb", dataset_id)
    preserved_metadata_path = cache.resolve(
        downloads_module._MAVEDB_METADATA_CACHE_SOURCE,
        dataset_id,
    )
    assert preserved_scores_path is not None
    assert preserved_metadata_path is not None
    assert preserved_scores_path.read_bytes() == original_scores
    assert preserved_metadata_path.read_bytes() == original_metadata
    assert not (tmp_path / "failed-refresh").exists()


@pytest.mark.parametrize(
    ("invalid_kind", "expected_error", "message"),
    [
        ("malformed", InvalidDatasetError, "cannot be serialized"),
        ("non-object", InvalidDatasetError, "must be a JSON object"),
        ("mismatched-urn", InvalidDatasetError, "does not match"),
        ("invalid-wt", InvalidDatasetError, "mavedb_score_set_metadata"),
        ("conflicting-wt", WildTypeConflictError, "Conflicting WT sequences"),
    ],
)
def test_invalid_mavedb_metadata_refresh_preserves_valid_cached_bytes(
    invalid_kind: str,
    expected_error: type[Exception],
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_id = "urn:mavedb:00000001-a-1"
    valid_metadata = {
        "urn": dataset_id,
        "targetGenes": [{"name": "GENE"}],
        "targetSequence": {"sequence": "MKT"},
    }

    def record(raw_metadata: Any) -> DatasetRecord:
        return DatasetRecord(
            source="mavedb",
            dataset_id=dataset_id,
            title="Cached assay",
            target_id="GENE",
            variant_type=None,
            n_variants=1,
            raw_metadata=raw_metadata,
        )

    current_metadata = record(valid_metadata)
    metadata_calls = 0

    def offline_metadata(*args: Any, **kwargs: Any) -> DatasetRecord:
        nonlocal metadata_calls
        metadata_calls += 1
        return current_metadata

    download_calls = 0

    def offline_download(url: str, output_path: str | Path, **kwargs: Any) -> Path:
        nonlocal download_calls
        download_calls += 1
        path = Path(output_path)
        path.write_text("hgvs_pro,score\np.Met1Ala,0.5\n", encoding="utf-8")
        return path

    monkeypatch.setattr(downloads_module, "get_dataset_metadata", offline_metadata)
    monkeypatch.setattr(fetch_module, "download_file", offline_download)
    cache = FilesystemCache(tmp_path / "cache")
    download_and_standardize_dataset(
        "mavedb",
        dataset_id,
        output_dir=tmp_path / "first",
        cache=cache,
    )
    metadata_path = cache.resolve(
        downloads_module._MAVEDB_METADATA_CACHE_SOURCE,
        dataset_id,
    )
    assert metadata_path is not None
    original_metadata = metadata_path.read_bytes()
    real_normalize = downloads_module.normalize_raw_metadata

    if invalid_kind == "malformed":
        invalid_metadata: Any = {
            **valid_metadata,
            "unserializable": object(),
        }
    elif invalid_kind == "non-object":
        invalid_metadata = valid_metadata
        monkeypatch.setattr(
            downloads_module,
            "normalize_raw_metadata",
            lambda values: [values],
        )
    elif invalid_kind == "mismatched-urn":
        invalid_metadata = {**valid_metadata, "urn": f"{dataset_id}-other"}
    elif invalid_kind == "invalid-wt":
        invalid_metadata = {
            **valid_metadata,
            "targetSequence": {"sequence": "NOT-A-PROTEIN"},
        }
    else:
        invalid_metadata = {
            **valid_metadata,
            "nested": {"targetSequence": {"sequence": "AAA"}},
        }
    current_metadata = record(invalid_metadata)

    with pytest.raises(expected_error, match=message):
        download_and_standardize_dataset(
            "mavedb",
            dataset_id,
            output_dir=tmp_path / "failed-refresh",
            cache=cache,
            refresh=True,
        )

    preserved_path = cache.resolve(
        downloads_module._MAVEDB_METADATA_CACHE_SOURCE,
        dataset_id,
    )
    assert preserved_path is not None
    assert preserved_path.read_bytes() == original_metadata
    assert not (tmp_path / "failed-refresh").exists()
    assert metadata_calls == 2
    assert download_calls == 1

    monkeypatch.setattr(
        downloads_module,
        "normalize_raw_metadata",
        real_normalize,
    )
    monkeypatch.setattr(
        downloads_module,
        "get_dataset_metadata",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("cache reconstruction contacted MaveDB metadata")
        ),
    )
    monkeypatch.setattr(
        fetch_module,
        "download_file",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("cache reconstruction downloaded MaveDB scores")
        ),
    )
    rebuilt = download_and_standardize_dataset(
        "mavedb",
        dataset_id,
        output_dir=tmp_path / "cached",
        cache=cache,
    )
    assert rebuilt.summary["status"] == "OK"
    assert preserved_path.read_bytes() == original_metadata


def test_invalid_mavedb_api_metadata_sequence_is_invalid_dataset_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_id = "urn:mavedb:00000001-a-1"
    metadata = DatasetRecord(
        source="mavedb",
        dataset_id=dataset_id,
        title=None,
        target_id=None,
        variant_type=None,
        n_variants=1,
        raw_metadata={
            "urn": dataset_id,
            "targetSequence": {"sequence": "NOT-A-PROTEIN"},
        },
    )
    monkeypatch.setattr(
        downloads_module,
        "get_dataset_metadata",
        lambda *args, **kwargs: metadata,
    )

    def offline_download(url: str, output_path: str | Path, **kwargs: Any) -> Path:
        path = Path(output_path)
        path.write_text("hgvs_pro,score\np.Met1Ala,0.5\n", encoding="utf-8")
        return path

    monkeypatch.setattr(fetch_module, "download_file", offline_download)

    with pytest.raises(InvalidDatasetError, match="mavedb_score_set_metadata"):
        download_and_standardize_dataset(
            "mavedb",
            dataset_id,
            output_dir=tmp_path / "output",
            cache=FilesystemCache(tmp_path / "cache"),
        )

    assert not (tmp_path / "output").exists()


def test_invalid_mavedb_snapshot_sequence_is_invalid_without_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_id = "urn:mavedb:00000001-a-1"
    metadata = _snapshot_score_set(dataset_id)
    metadata["targetSequence"] = {"sequence": "NOT-A-PROTEIN"}
    cache = _write_cached_snapshot(
        tmp_path,
        [metadata],
        current_ids=[dataset_id],
    )
    forbidden = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("snapshot validation contacted MaveDB API")
    )
    monkeypatch.setattr(downloads_module, "get_dataset_metadata", forbidden)
    monkeypatch.setattr(downloads_module, "download_mavedb_dataset", forbidden)

    with pytest.raises(InvalidDatasetError, match="mavedb_score_set_metadata"):
        download_and_standardize_dataset(
            "mavedb",
            dataset_id,
            output_dir=tmp_path / "output",
            cache=cache,
            acquisition="snapshot",
            snapshot_record_id="20840937",
        )

    assert not (tmp_path / "output").exists()


def test_snapshot_single_uses_offline_tables_common_builder_and_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_id = "urn:mavedb:00000001-a-1"
    metadata = _snapshot_score_set(dataset_id, gene="SNAPGENE")
    cache = _write_cached_snapshot(
        tmp_path,
        [metadata],
        current_ids=[dataset_id],
    )
    forbidden = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("snapshot standardization contacted live MaveDB")
    )
    monkeypatch.setattr(downloads_module, "get_dataset_metadata", forbidden)
    monkeypatch.setattr(downloads_module, "download_mavedb_dataset", forbidden)
    monkeypatch.setattr(
        snapshot_dataset_module,
        "fetch_mavedb_snapshot",
        forbidden,
        raising=False,
    )
    monkeypatch.setattr(
        snapshot_dataset_module,
        "resolve_mavedb_snapshot",
        forbidden,
        raising=False,
    )

    first = download_and_standardize_dataset(
        "mavedb",
        dataset_id,
        output_dir=tmp_path / "snapshot-first",
        cache=cache,
        acquisition="snapshot",
        snapshot_record_id="20840937",
    )
    second = download_and_standardize_dataset(
        "mavedb",
        dataset_id,
        output_dir=tmp_path / "snapshot-second",
        cache=cache,
        acquisition="snapshot",
        snapshot_record_id="20840937",
    )

    first_table = pd.read_csv(first.dataset_path)
    assert first_table["score_raw"].tolist() == [1.0, 0.5, -1.0]
    assert first_table["score"].tolist() == [1.0, 0.5, -1.0]
    assert first_table["sd"].tolist() == [0.1, 0.2, 0.3]
    assert first_table["se"].tolist() == [0.01, 0.02, 0.03]
    assert first_table["df"].tolist() == [10, 20, 30]
    assert not {
        "scores.score",
        "scores.sd",
        "scores.se",
        "scores.df",
    }.intersection(first_table.columns)
    assert first_table["accession"].tolist() == [
        f"{dataset_id}#1",
        f"{dataset_id}#2",
        f"{dataset_id}#3",
    ]
    assert first_table["hgvs_pro"].tolist() == [
        "p.=",
        "p.Met1Ala",
        "p.[Gly2del]",
    ]
    assert first_table["status"].tolist() == ["OK", "OK", "Unsupported"]
    assert first_table["is_synthetic"].tolist() == [False, False, False]
    assert first_table["mutated_sequence"].iloc[0] == "MKT"
    assert first_table["mutated_sequence"].iloc[1] == "AKT"
    assert "score_log_ratio" not in first_table.columns
    assert "score_binary_like" not in first_table.columns
    assert str(first_table["score_raw"].dtype) == "float64"
    assert first.dataset_path.read_bytes() == second.dataset_path.read_bytes()
    assert sorted(path.name for path in first.dataset_path.parent.iterdir()) == [
        "standardized.csv",
        "summary.csv",
        "summary.json",
    ]
    assert not any(
        path.name in {"scores.csv", "counts.csv"}
        for path in first.dataset_path.parent.iterdir()
    )

    provenance_fields = [
        "acquisition_method",
        "snapshot_record_id",
        "snapshot_doi",
        "snapshot_concept_doi",
        "snapshot_publication_date",
        "snapshot_archive_filename",
        "snapshot_archive_size",
        "snapshot_archive_checksum",
        "snapshot_catalog_as_of",
        "snapshot_score_set_superseded",
    ]
    assert list(first.summary)[-11:-1] == provenance_fields
    assert first.summary["acquisition_method"] == "snapshot"
    assert first.summary["snapshot_record_id"] == "20840937"
    assert first.summary["snapshot_catalog_as_of"] == "2026-06-24T18:13:01Z"
    assert first.summary["snapshot_score_set_superseded"] is False
    assert {
        key: first.summary[key]
        for key in provenance_fields
    } == {
        key: second.summary[key]
        for key in provenance_fields
    }

    extracted_scores = next(
        cache.root.glob("mavedb/snapshot_tables/20840937/*/scores.csv")
    )
    assert extracted_scores.read_text(encoding="utf-8").splitlines()[0] == (
        "accession,hgvs_nt,hgvs_splice,hgvs_pro,score,sd,se,df"
    )
    api_scores = tmp_path / "api-like-scores.csv"
    api_scores.write_text(
        "accession,hgvs_nt,hgvs_splice,hgvs_pro,score,sd,se,df\n"
        f"{dataset_id}#1,c.=,,p.=,1.0,0.1,0.01,10\n"
        f"{dataset_id}#2,c.1A>G,,p.Met1Ala,0.5,0.2,0.02,20\n"
        f"{dataset_id}#3,c.4_6del,,p.[Gly2del],-1.0,0.3,0.03,30\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        downloads_module,
        "_acquire_mavedb_api_dataset",
        lambda *args, **kwargs: downloads_module._MaveDBDownloadInput(
            scores_path=api_scores,
            metadata=metadata,
            provenance={},
        ),
    )
    api = download_and_standardize_dataset(
        "mavedb",
        dataset_id,
        output_dir=tmp_path / "api-equivalent",
        cache=cache,
        acquisition="api",
    )
    api_table = pd.read_csv(api.dataset_path)
    assert first_table.columns.tolist() == api_table.columns.tolist()
    pd.testing.assert_frame_equal(
        first_table,
        api_table,
        check_dtype=True,
        check_exact=True,
        check_like=False,
    )
    assert first.dataset_path.read_bytes() == api.dataset_path.read_bytes()
    for result in (first, api):
        assert result.summary["wt_score"] == 1.0
        assert result.summary["wt_score_provenance"] == "observed_wildtype_row"
        assert result.summary["wt_sequence_provenance"] == (
            "mavedb_score_set_metadata"
        )
        assert result.summary["observed_wildtype_row"] is True


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
        wt_score={"ASSAY_2": 0.25},
        add_relative_score=True,
        relative_method="difference",
        relative_output_col="score_difference",
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
    assert assay_two["score_difference"].iloc[1] == 0.5
    assert assay_two["is_synthetic"].tolist() == [True, False]
    assert assay_two["source_note"].iloc[1] == "caf\u00c3\u00a9"
    records = json.loads(result.summary_json_path.read_text(encoding="utf-8"))
    assert [record["status"] for record in records] == [
        "ERROR",
        "SUCCESS",
        "SUCCESS",
    ]
    assert records[1]["dataset_path"].startswith("proteingym/id-")
    assert records[1]["wt_score"] == 0.25
    assert records[1]["wt_score_provenance"] == "user_fallback"
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

    metadata_calls: list[str] = []

    def metadata(source: str, dataset_id: str) -> DatasetRecord:
        assert source == "mavedb"
        metadata_calls.append(dataset_id)
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
    assert metadata_calls == [*dataset_ids, *dataset_ids]
    assert all(cache.exists("mavedb", dataset_id) for dataset_id in dataset_ids)
    assert all(
        cache.exists(downloads_module._MAVEDB_METADATA_CACHE_SOURCE, dataset_id)
        for dataset_id in dataset_ids
    )
    assert len(
        {
            cache.entry_path(
                downloads_module._MAVEDB_METADATA_CACHE_SOURCE,
                dataset_id,
            )
            for dataset_id in dataset_ids
        }
    ) == 2
    first_table = pd.read_csv(first.entries[0].result.dataset_path)  # type: ignore[union-attr]
    assert first_table["is_synthetic"].tolist() == [True, False]


def test_snapshot_batch_acquires_once_isolates_catalog_outcomes_and_refreshes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_id = "urn:mavedb:00000001-a-1"
    superseded_id = "urn:mavedb:00000001-a-2"
    missing_id = "urn:mavedb:00000001-a-3"
    dataset_ids = (current_id, superseded_id, missing_id)
    cache = _write_cached_snapshot(
        tmp_path,
        [
            _snapshot_score_set(current_id, gene="CURRENT"),
            _snapshot_score_set(superseded_id, gene="LEGACY"),
        ],
        current_ids=[current_id],
    )
    forbidden = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("snapshot batch contacted live MaveDB")
    )
    monkeypatch.setattr(downloads_module, "get_dataset_metadata", forbidden)
    monkeypatch.setattr(downloads_module, "download_mavedb_dataset", forbidden)

    load_calls = 0
    extraction_calls: list[tuple[tuple[str, ...], bool, bool]] = []
    archive_open_calls = 0
    real_load = snapshot_dataset_module.load_cached_mavedb_snapshot
    real_extract = snapshot_dataset_module.extract_mavedb_snapshot_tables
    real_zip_file = snapshot_tables_module.zipfile.ZipFile

    def load(*args: Any, **kwargs: Any):
        nonlocal load_calls
        load_calls += 1
        return real_load(*args, **kwargs)

    def extract(snapshot: Any, ids: Any, **kwargs: Any):
        extraction_calls.append(
            (tuple(ids), kwargs["include_superseded"], kwargs["refresh"])
        )
        return real_extract(snapshot, ids, **kwargs)

    def open_zip(*args: Any, **kwargs: Any):
        nonlocal archive_open_calls
        archive_open_calls += 1
        return real_zip_file(*args, **kwargs)

    monkeypatch.setattr(
        snapshot_dataset_module,
        "load_cached_mavedb_snapshot",
        load,
    )
    monkeypatch.setattr(
        snapshot_dataset_module,
        "extract_mavedb_snapshot_tables",
        extract,
    )
    monkeypatch.setattr(snapshot_tables_module.zipfile, "ZipFile", open_zip)

    first = download_and_standardize_datasets(
        "mavedb",
        dataset_ids,
        output_dir=tmp_path / "first",
        cache=cache,
        acquisition="snapshot",
        snapshot_record_id="20840937",
    )
    assert [entry.dataset_id for entry in first.entries] == list(dataset_ids)
    assert [entry.error_type for entry in first.entries] == [
        None,
        "SupersededSnapshotDatasetError",
        "SnapshotDatasetNotFoundError",
    ]
    assert first.success_count == 1
    assert load_calls == 1
    assert extraction_calls == [((current_id,), False, False)]
    assert archive_open_calls == 1

    second = download_and_standardize_datasets(
        "mavedb",
        dataset_ids,
        output_dir=tmp_path / "second",
        cache=cache,
        acquisition="snapshot",
        snapshot_record_id="20840937",
        include_superseded=True,
    )
    assert [entry.dataset_id for entry in second.entries] == list(dataset_ids)
    assert [entry.error_type for entry in second.entries] == [
        None,
        None,
        "SnapshotDatasetNotFoundError",
    ]
    assert load_calls == 2
    assert extraction_calls[-1] == (
        (current_id, superseded_id),
        True,
        False,
    )
    assert archive_open_calls == 2

    cache_hits = download_and_standardize_datasets(
        "mavedb",
        (current_id, superseded_id),
        output_dir=tmp_path / "cache-hits",
        cache=cache,
        acquisition="snapshot",
        snapshot_record_id="20840937",
        include_superseded=True,
    )
    assert cache_hits.success_count == 2
    assert archive_open_calls == 2

    refreshed = download_and_standardize_datasets(
        "mavedb",
        (current_id, superseded_id),
        output_dir=tmp_path / "refreshed",
        cache=cache,
        acquisition="snapshot",
        snapshot_record_id="20840937",
        include_superseded=True,
        refresh=True,
    )
    assert refreshed.success_count == 2
    assert extraction_calls[-1] == (
        (current_id, superseded_id),
        True,
        True,
    )
    assert archive_open_calls == 3

    records = json.loads(
        second.summary_json_path.read_text(encoding="utf-8")
    )
    assert [record["dataset_id"] for record in records] == list(dataset_ids)
    assert all(record["acquisition_method"] == "snapshot" for record in records)
    assert records[0]["snapshot_score_set_superseded"] is False
    assert records[1]["snapshot_score_set_superseded"] is True
    assert records[2]["snapshot_score_set_superseded"] is None


@pytest.mark.parametrize("failure_stage", ["builder", "publication"])
def test_snapshot_batch_dataset_failure_preserves_successful_sibling(
    failure_stage: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_ids = (
        "urn:mavedb:00000001-a-1",
        "urn:mavedb:00000001-a-2",
    )
    cache = _write_cached_snapshot(
        tmp_path,
        [_snapshot_score_set(dataset_id) for dataset_id in dataset_ids],
        current_ids=list(dataset_ids),
    )
    real_build = downloads_module._build_mavedb_download
    real_publish = downloads_module._publish_download_result
    publish_calls = 0

    def build(dataset_id: str, *args: Any, **kwargs: Any):
        if failure_stage == "builder" and dataset_id == dataset_ids[1]:
            raise InvalidDatasetError("offline builder failure")
        return real_build(dataset_id, *args, **kwargs)

    def publish(*args: Any, **kwargs: Any):
        nonlocal publish_calls
        publish_calls += 1
        if failure_stage == "publication" and publish_calls == 2:
            raise DownloadError("offline publication failure")
        return real_publish(*args, **kwargs)

    monkeypatch.setattr(downloads_module, "_build_mavedb_download", build)
    monkeypatch.setattr(downloads_module, "_publish_download_result", publish)
    result = download_and_standardize_datasets(
        "mavedb",
        dataset_ids,
        output_dir=tmp_path / "output",
        cache=cache,
        acquisition="snapshot",
        snapshot_record_id="20840937",
    )

    assert result.success_count == 1
    assert result.failure_count == 1
    assert result.entries[0].result is not None
    assert result.entries[0].result.dataset_path.is_file()
    expected_error = (
        "InvalidDatasetError"
        if failure_stage == "builder"
        else "DownloadError"
    )
    assert result.entries[1].error_type == expected_error
    assert not result.entries[1].output_dir.exists()


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
