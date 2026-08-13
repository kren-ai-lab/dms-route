from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import dms_parser.downloads as downloads_module
import dms_parser.acquisition.fetch as fetch_module
import dms_parser.acquisition.io as io_module
from dms_parser import (
    FilesystemCache,
    download_and_standardize_dataset,
    download_and_standardize_datasets,
    get_proteingym_resource,
)
from dms_parser.core.exceptions import InvalidPipelineOptionError


def _install_offline_resource(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variant_type: str,
    metadata: pd.DataFrame,
    benchmark: pd.DataFrame,
) -> tuple[FilesystemCache, list[str], list[Path]]:
    resource = get_proteingym_resource(f"dms_{variant_type}")
    tmp_path.mkdir(parents=True, exist_ok=True)
    metadata_path = tmp_path / resource.metadata_filename
    benchmark_path = tmp_path / resource.data_filename
    metadata.to_csv(metadata_path, index=False)
    benchmark.to_parquet(benchmark_path, index=False)
    sources = {
        resource.metadata_url: metadata_path,
        resource.data_url: benchmark_path,
    }
    downloads: list[str] = []

    def offline_download(url: str, output_path: str | Path, **kwargs: Any) -> Path:
        downloads.append(url)
        destination = Path(output_path)
        destination.write_bytes(sources[url].read_bytes())
        return destination

    reads: list[Path] = []
    original_read = downloads_module.read_table

    def count_reads(path: str | Path, **kwargs: Any) -> pd.DataFrame:
        reads.append(Path(path))
        return original_read(path, **kwargs)

    monkeypatch.setattr(fetch_module, "download_file", offline_download)
    monkeypatch.setattr(downloads_module, "read_table", count_reads)
    return FilesystemCache(tmp_path / "cache"), downloads, reads

def _metadata(*dataset_ids: str) -> pd.DataFrame:
    count = len(dataset_ids)
    return pd.DataFrame({
        "DMS_id": dataset_ids, "target_seq": ["MKT"] * count,
        "UniProt_ID": [f"P-{value}" for value in dataset_ids],
        "molecule_name": ["Protein"] * count, "gene": ["GENE"] * count,
    })


def test_public_variant_type_signature_and_validation_precede_side_effects(
    tmp_path: Path,
) -> None:
    for function in (download_and_standardize_dataset, download_and_standardize_datasets):
        parameter = inspect.signature(function).parameters["variant_type"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default == "substitutions"

    cache = FilesystemCache(tmp_path / "cache")
    for variant_type in (None, "deletions"):
        with pytest.raises(InvalidPipelineOptionError, match="variant_type"):
            download_and_standardize_dataset(
                "proteingym", "ASSAY", output_dir=tmp_path / "output",
                cache=cache, variant_type=variant_type,  # type: ignore[arg-type]
            )
    with pytest.raises(InvalidPipelineOptionError, match="only for ProteinGym"):
        download_and_standardize_datasets(
            "mavedb", ["urn:mavedb:00000001-a-1"],
            output_dir=tmp_path / "output", cache=cache, variant_type="indels",
        )
    assert not (tmp_path / "output").exists()
    assert not cache.root.exists()

def test_single_indel_uses_exact_cached_resources_and_output_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resource = get_proteingym_resource("dms_indels")
    benchmark = pd.DataFrame(
        {
            "DMS_id": ["ASSAY"] * 3,
            "target_seq": ["MKT"] * 3,
            "mutated_sequence": ["MKT", "MT", "MJ"],
            "DMS_score": [1.0, 0.25, -1.0],
            "DMS_score_bin": [1, 0, 0],
            "source_note": ["wt", "deletion", "invalid"],
        }
    )
    cache, calls, _ = _install_offline_resource(
        tmp_path, monkeypatch, "indels", _metadata("ASSAY"), benchmark
    )

    result = download_and_standardize_dataset(
        "proteingym", "ASSAY", output_dir=tmp_path / "output",
        cache=cache, variant_type="indels",
    )
    table = pd.read_csv(result.dataset_path)
    refreshed = download_and_standardize_dataset(
        "proteingym", "ASSAY", output_dir=tmp_path / "refreshed",
        cache=cache, refresh=True, variant_type="indels",
    )

    assert refreshed.summary["status"] == "OK"
    assert calls == [
        resource.metadata_url, resource.data_url,
        resource.metadata_url, resource.data_url,
    ]
    assert cache.exists("proteingym", "reference-files-dms-indels")
    assert cache.exists("proteingym", "resource-data-dms-indels")
    assert table["mutated_sequence"].iloc[:2].tolist() == ["MKT", "MT"]
    assert table["variant"].isna().all()
    assert table.loc[0, "n_mutations"] == 0
    assert pd.isna(table.loc[1, "n_mutations"])
    assert table["status"].tolist() == ["OK", "OK", "Error"]
    assert table["DMS_score_bin"].tolist() == [1, 0, 0]
    assert sorted(path.name for path in result.dataset_path.parent.iterdir()) == [
        "standardized.csv", "summary.csv", "summary.json"
    ]
    assert result.summary["raw_rows"] == 3


def test_indel_transforms_and_drop_failed_remain_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = pd.DataFrame(
        {
            "DMS_id": ["ASSAY"] * 2,
            "target_seq": ["MKT"] * 2,
            "mutated_sequence": ["MT", "MJ"],
            "DMS_score": [0.25, -1.0],
        }
    )
    cache, _, _ = _install_offline_resource(
        tmp_path, monkeypatch, "indels", _metadata("ASSAY"), benchmark
    )
    result = download_and_standardize_dataset(
        "proteingym",
        "ASSAY",
        output_dir=tmp_path / "output",
        cache=cache,
        variant_type="indels",
        drop_failed=True,
        add_wildtype_row=True,
        wt_score=1.0,
        add_relative_score=True,
        relative_method="difference",
        relative_output_col="score_delta",
    )
    table = pd.read_csv(result.dataset_path)

    assert table["mutated_sequence"].tolist() == ["MKT", "MT"]
    assert pd.isna(table.loc[0, "score_raw"])
    assert table.loc[1, ["score_raw", "score_delta"]].tolist() == [0.25, -0.75]
    assert table["is_synthetic"].tolist() == [True, False]
    assert result.summary["discarded_rows"] == 1

def test_indel_batch_reads_shared_parquet_once_and_isolates_missing_assay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resource = get_proteingym_resource("dms_indels")
    benchmark = pd.DataFrame(
        {
            "DMS_id": ["FIRST", "SECOND"],
            "target_seq": ["MKT", "MKT"],
            "mutated_sequence": ["MT", "MKTT"],
            "DMS_score": [0.5, -0.5],
        }
    )
    cache, calls, reads = _install_offline_resource(
        tmp_path, monkeypatch, "indels", _metadata("FIRST", "SECOND"), benchmark
    )
    result = download_and_standardize_datasets(
        "proteingym", ["MISSING", "SECOND", "FIRST"],
        output_dir=tmp_path / "output", cache=cache, variant_type="indels",
    )

    assert calls == [resource.metadata_url, resource.data_url]
    assert sum(path.suffix == ".parquet" for path in reads) == 1
    assert [entry.dataset_id for entry in result.entries] == [
        "MISSING", "SECOND", "FIRST"
    ]
    assert [entry.result is not None for entry in result.entries] == [
        False, True, True
    ]
    assert result.summary_csv_path.exists()
    assert result.summary_json_path.exists()

def test_single_indel_preflight_and_publication_rollback_preserve_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = pd.DataFrame({
        "DMS_id": ["ASSAY"], "target_seq": ["MKT"],
        "mutated_sequence": ["MT"], "DMS_score": [0.5],
    })
    cache, calls, _ = _install_offline_resource(
        tmp_path, monkeypatch, "indels", _metadata("ASSAY"), benchmark
    )
    output = tmp_path / "output"
    output.mkdir()
    originals: dict[Path, bytes] = {}
    for name in ("standardized.csv", "summary.csv", "summary.json"):
        path = output / name
        originals[path] = f"old {name}".encode()
        path.write_bytes(originals[path])

    with pytest.raises(FileExistsError, match="overwrite=True"):
        download_and_standardize_dataset(
            "proteingym", "ASSAY", output_dir=output, cache=cache,
            variant_type="indels",
        )
    assert calls == []

    original_replace = io_module.os.replace
    injected = False

    def fail_summary(source: str | Path, destination: str | Path) -> None:
        nonlocal injected
        if Path(destination) == output / "summary.csv" and not injected:
            injected = True
            raise OSError("summary publication failed")
        original_replace(source, destination)

    monkeypatch.setattr(io_module.os, "replace", fail_summary)
    with pytest.raises(OSError, match="summary publication failed"):
        download_and_standardize_dataset(
            "proteingym", "ASSAY", output_dir=output, cache=cache,
            variant_type="indels", overwrite=True,
        )
    assert {path: path.read_bytes() for path in originals} == originals

@pytest.mark.parametrize("batch", [False, True])
def test_substitution_defaults_and_unexpected_indel_errors_are_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    batch: bool,
) -> None:
    substitution_resource = get_proteingym_resource("dms_substitutions")
    substitutions = pd.DataFrame(
        {"DMS_id": ["SUB"], "mutant": ["M1A"], "DMS_score": [0.5]}
    )
    cache, calls, reads = _install_offline_resource(
        tmp_path, monkeypatch, "substitutions", _metadata("SUB"), substitutions
    )
    original_builder = downloads_module.build_proteingym_dataset
    builder_calls: list[dict[str, Any]] = []

    def substitution_builder(**kwargs: Any) -> pd.DataFrame:
        builder_calls.append(kwargs)
        return original_builder(**kwargs)

    monkeypatch.setattr(downloads_module, "build_proteingym_dataset", substitution_builder)
    download_and_standardize_dataset(
        "proteingym", "SUB", output_dir=tmp_path / "sub-output", cache=cache
    )
    assert calls == [substitution_resource.metadata_url, substitution_resource.data_url]
    assert sum(path.suffix == ".parquet" for path in reads) == 1
    assert builder_calls[0]["variant_col"] == "mutant"
    assert cache.exists("proteingym", "reference-files-dms-substitutions")
    assert cache.exists("proteingym", "resource-data-dms-substitutions")

    cause = RuntimeError("unexpected indel builder failure")
    monkeypatch.setattr(
        downloads_module,
        "build_proteingym_indel_dataset",
        lambda **kwargs: (_ for _ in ()).throw(cause),
    )
    indel_cache, _, _ = _install_offline_resource(
        tmp_path / "indel", monkeypatch, "indels", _metadata("INDEL"),
        pd.DataFrame({
            "DMS_id": ["INDEL"], "target_seq": ["MKT"],
            "mutated_sequence": ["MT"], "DMS_score": [0.5],
        }),
    )
    download = download_and_standardize_datasets if batch else download_and_standardize_dataset
    requested = ["INDEL"] if batch else "INDEL"
    with pytest.raises(RuntimeError) as error:
        download(
            "proteingym", requested, output_dir=tmp_path / "indel-output",
            cache=indel_cache, variant_type="indels",
        )
    assert error.value is cause
