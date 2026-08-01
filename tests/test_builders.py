from __future__ import annotations

import warnings
from inspect import signature

import pandas as pd
import pytest

from dms_parser.builders import build_mavedb_dataset, build_proteingym_dataset


def _build_source_dataset(
    tmp_path,
    source: str,
    variants: list[str],
    scores: list[float],
    **kwargs,
) -> pd.DataFrame:
    """Build a small source-specific dataset with shared metadata."""
    if source == "proteingym":
        path = tmp_path / "proteingym.csv"
        pd.DataFrame({"mutant": variants, "DMS_score": scores}).to_csv(path, index=False)
        return build_proteingym_dataset(
            input_path=path,
            score_col="DMS_score",
            variant_col="mutant",
            **kwargs,
        )

    path = tmp_path / "mavedb.csv"
    pd.DataFrame({"hgvs_pro": variants, "score": scores}).to_csv(path, index=False)
    return build_mavedb_dataset(
        input_path=path,
        score_col="score",
        hgvs_col="hgvs_pro",
        **kwargs,
    )


def _source_variants(source: str, *, include_wt: bool = False) -> list[str]:
    """Return equivalent source-specific variant strings."""
    if source == "proteingym":
        return ["WT", "M1A"] if include_wt else ["M1A"]
    return ["p.=", "p.Met1Ala"] if include_wt else ["p.Met1Ala"]


@pytest.mark.parametrize(
    "builder",
    [build_mavedb_dataset, build_proteingym_dataset],
)
def test_builder_score_transform_defaults_are_disabled(builder):
    parameters = signature(builder).parameters

    assert parameters["add_relative_score"].default is False
    assert parameters["add_binary_label"].default is False
    assert parameters["add_wildtype_row"].default is False


@pytest.mark.parametrize("source", ["proteingym", "mavedb"])
def test_dataset_without_wt_keeps_source_rows_by_default(
    source: str,
    tmp_path,
    wt_sequence: str,
) -> None:
    result = _build_source_dataset(
        tmp_path,
        source,
        _source_variants(source),
        [0.5],
        wt_sequence=wt_sequence,
    )

    assert len(result) == 1
    assert result["is_synthetic"].tolist() == [False]
    assert result["score_raw"].tolist() == [0.5]


@pytest.mark.parametrize("source", ["proteingym", "mavedb"])
def test_add_wildtype_row_prepends_exact_scoreless_row(
    source: str,
    tmp_path,
    wt_sequence: str,
) -> None:
    result = _build_source_dataset(
        tmp_path,
        source,
        _source_variants(source),
        [0.5],
        wt_sequence=wt_sequence,
        dataset_id="dataset-1",
        protein_id="protein-1",
        gene="GENE1",
        uniprot_id="P12345",
        add_wildtype_row=True,
    )

    synthetic = result.iloc[0]
    assert isinstance(result.index, pd.RangeIndex)
    assert len(result) == 2
    assert synthetic[
        ["dataset_id", "source", "protein_id", "gene", "uniprot_id"]
    ].to_dict() == {
        "dataset_id": "dataset-1",
        "source": source,
        "protein_id": "protein-1",
        "gene": "GENE1",
        "uniprot_id": "P12345",
    }
    assert synthetic["variant"] == ""
    assert synthetic["mutated_sequence"] == wt_sequence
    assert synthetic["wt_sequence"] == wt_sequence
    assert synthetic["is_wildtype"] == True
    assert synthetic["n_mutations"] == 0
    assert pd.isna(synthetic["score_raw"])
    assert synthetic["status"] == "OK"
    assert synthetic["error"] == ""
    assert synthetic["is_synthetic"] == True
    assert pd.isna(synthetic["mutant" if source == "proteingym" else "hgvs_pro"])
    assert result.iloc[1]["is_synthetic"] == False
    assert result.iloc[1]["score_raw"] == 0.5


@pytest.mark.parametrize("source", ["proteingym", "mavedb"])
def test_add_wildtype_row_emits_no_future_warning(
    source: str,
    tmp_path,
    wt_sequence: str,
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        result = _build_source_dataset(
            tmp_path,
            source,
            _source_variants(source),
            [0.5],
            wt_sequence=wt_sequence,
            add_wildtype_row=True,
        )

    assert isinstance(result.index, pd.RangeIndex)
    assert result["is_synthetic"].tolist() == [True, False]


@pytest.mark.parametrize("source", ["proteingym", "mavedb"])
def test_existing_observed_wt_is_not_duplicated_or_rescored(
    source: str,
    tmp_path,
    wt_sequence: str,
) -> None:
    result = _build_source_dataset(
        tmp_path,
        source,
        _source_variants(source, include_wt=True),
        [2.5, 0.5],
        wt_sequence=wt_sequence,
        add_wildtype_row=True,
    )

    assert len(result) == 2
    assert int(result["is_wildtype"].sum()) == 1
    assert result["is_synthetic"].tolist() == [False, False]
    assert result["score_raw"].tolist() == [2.5, 0.5]


def test_mavedb_complete_identity_builds_observed_wt(
    tmp_path,
    wt_sequence: str,
) -> None:
    result = _build_source_dataset(
        tmp_path,
        "mavedb",
        ["p.="],
        [2.5],
        wt_sequence=wt_sequence,
    )

    observed = result.iloc[0]
    assert observed["variant"] == ""
    assert observed["mutated_sequence"] == wt_sequence
    assert observed["is_wildtype"] == True
    assert observed["n_mutations"] == 0
    assert observed["is_synthetic"] == False
    assert observed["score_raw"] == 2.5


@pytest.mark.parametrize(
    ("source", "variants"),
    [
        ("proteingym", ["not-a-variant", "M1A"]),
        ("mavedb", ["p.invalid", "p.Met1Ala"]),
    ],
)
def test_drop_failed_precedes_synthetic_wt_insertion(
    source: str,
    variants: list[str],
    tmp_path,
    wt_sequence: str,
) -> None:
    result = _build_source_dataset(
        tmp_path,
        source,
        variants,
        [9.0, 0.5],
        wt_sequence=wt_sequence,
        drop_failed=True,
        add_wildtype_row=True,
    )

    assert isinstance(result.index, pd.RangeIndex)
    assert result["is_synthetic"].tolist() == [True, False]
    assert result["status"].tolist() == ["OK", "OK"]
    assert result["score_raw"].iloc[1:].tolist() == [0.5]


@pytest.mark.parametrize("source", ["proteingym", "mavedb"])
def test_scoreless_synthetic_wt_skips_relative_transform(
    source: str,
    tmp_path,
    wt_sequence: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("WARNING", logger="dms_parser.builders")

    result = _build_source_dataset(
        tmp_path,
        source,
        _source_variants(source),
        [0.5],
        wt_sequence=wt_sequence,
        add_wildtype_row=True,
        add_relative_score=True,
    )

    assert "score_log_ratio" not in result.columns
    assert "reason=no_valid_numeric_wild_type_score" in caplog.text


@pytest.mark.parametrize("source", ["proteingym", "mavedb"])
def test_scoreless_synthetic_wt_can_be_required_for_transform(
    source: str,
    tmp_path,
    wt_sequence: str,
) -> None:
    with pytest.raises(ValueError, match="no valid numeric WT score"):
        _build_source_dataset(
            tmp_path,
            source,
            _source_variants(source),
            [0.5],
            wt_sequence=wt_sequence,
            add_wildtype_row=True,
            add_relative_score=True,
            require_wt_for_transforms=True,
        )


@pytest.mark.parametrize("transform_kwargs", [{}, {"add_relative_score": False}])
def test_proteingym_scores_are_unchanged_without_transformations(
    tmp_path,
    wt_sequence: str,
    transform_kwargs,
):
    scores = pd.Series([-1.0, 0.0, 2.0, None], name="DMS_score")
    path = tmp_path / "proteingym_scores.csv"
    pd.DataFrame(
        {
            "variant": ["WT", "M1A", "K2R", "T3Y"],
            "DMS_score": scores,
        }
    ).to_csv(path, index=False)

    result = build_proteingym_dataset(
        input_path=path,
        score_col="DMS_score",
        variant_col="variant",
        wt_sequence=wt_sequence,
        **transform_kwargs,
    )

    pd.testing.assert_series_equal(
        result["score_raw"].reset_index(drop=True),
        scores,
        check_names=False,
    )
    assert "score_log_ratio" not in result.columns
    assert "score_binary_like" not in result.columns
    assert (result["status"] == "OK").all()


@pytest.mark.parametrize("transform_kwargs", [{}, {"add_relative_score": False}])
def test_mavedb_scores_are_unchanged_without_transformations(
    tmp_path,
    wt_sequence: str,
    transform_kwargs,
):
    scores = pd.Series([-1.0, 0.0, 2.0, None], name="score")
    path = tmp_path / "mavedb_scores.csv"
    pd.DataFrame(
        {
            "hgvs_pro": [
                "p.Met1Ala",
                "p.Lys2Arg",
                "p.Thr3Tyr",
                "p.Ala4Val",
            ],
            "score": scores,
        }
    ).to_csv(path, index=False)

    result = build_mavedb_dataset(
        input_path=path,
        score_col="score",
        hgvs_col="hgvs_pro",
        wt_sequence=wt_sequence,
        **transform_kwargs,
    )

    pd.testing.assert_series_equal(
        result["score_raw"].reset_index(drop=True),
        scores,
        check_names=False,
    )
    assert "score_log_ratio" not in result.columns
    assert "score_binary_like" not in result.columns
    assert (result["status"] == "OK").all()


def test_build_proteingym_dataset_basic(tmp_path, proteingym_like_df: pd.DataFrame, wt_sequence: str):
    path = tmp_path / "proteingym.csv"
    proteingym_like_df.to_csv(path, index=False)

    result = build_proteingym_dataset(
        input_path=path,
        score_col="DMS_score",
        variant_col="variant",
        wt_sequence=wt_sequence,
        add_relative_score=True,
        add_binary_label=True,
    )

    assert "score_raw" in result.columns
    assert "mutated_sequence" in result.columns
    assert "status" in result.columns
    assert "score_log_ratio" in result.columns
    assert "score_binary_like" in result.columns
    assert (result["status"] == "OK").all()


def test_build_proteingym_dataset_with_invalid_variant(tmp_path, wt_sequence: str):
    df = pd.DataFrame(
        {
            "variant": ["WT", "BADVARIANT"],
            "DMS_score": [1.0, 0.5],
        }
    )
    path = tmp_path / "proteingym.csv"
    df.to_csv(path, index=False)

    result = build_proteingym_dataset(
        input_path=path,
        score_col="DMS_score",
        variant_col="variant",
        wt_sequence=wt_sequence,
        strict_variant_parsing=False,
        add_relative_score=True,
    )

    assert list(result["status"]) == ["OK", "Error"]


def test_build_proteingym_dataset_drop_failed(tmp_path, wt_sequence: str):
    df = pd.DataFrame(
        {
            "variant": ["WT", "BADVARIANT"],
            "DMS_score": [1.0, 0.5],
        }
    )
    path = tmp_path / "proteingym.csv"
    df.to_csv(path, index=False)

    result = build_proteingym_dataset(
        input_path=path,
        score_col="DMS_score",
        variant_col="variant",
        wt_sequence=wt_sequence,
        strict_variant_parsing=False,
        drop_failed=True,
    )

    assert len(result) == 1
    assert (result["status"] == "OK").all()


def test_build_proteingym_dataset_requires_columns(tmp_path, wt_sequence: str):
    df = pd.DataFrame({"wrong_col": ["WT"], "DMS_score": [1.0]})
    path = tmp_path / "proteingym.csv"
    df.to_csv(path, index=False)

    with pytest.raises(Exception):
        build_proteingym_dataset(
            input_path=path,
            score_col="DMS_score",
            variant_col="variant",
            wt_sequence=wt_sequence,
        )


def test_build_mavedb_dataset_basic(tmp_path, mavedb_like_df: pd.DataFrame, wt_sequence: str):
    path = tmp_path / "mavedb.csv"
    mavedb_like_df.to_csv(path, index=False)

    result = build_mavedb_dataset(
        input_path=path,
        score_col="score",
        hgvs_col="hgvs_pro",
        wt_sequence=wt_sequence,
        add_relative_score=False,
    )

    assert "score_raw" in result.columns
    assert "mutated_sequence" in result.columns
    assert "status" in result.columns
    assert (result["status"] == "OK").all()


def test_build_mavedb_dataset_with_unsupported_variant(tmp_path, wt_sequence: str):
    df = pd.DataFrame(
        {
            "hgvs_pro": ["p.Met1Ala", "p.Gly10del"],
            "score": [0.8, 0.2],
        }
    )
    path = tmp_path / "mavedb.csv"
    df.to_csv(path, index=False)

    result = build_mavedb_dataset(
        input_path=path,
        score_col="score",
        hgvs_col="hgvs_pro",
        wt_sequence=wt_sequence,
        add_relative_score=False,
    )

    assert list(result["status"]) == ["OK", "Unsupported"]


def test_build_mavedb_dataset_drop_failed(tmp_path, wt_sequence: str):
    df = pd.DataFrame(
        {
            "hgvs_pro": ["p.Met1Ala", "p.invalid"],
            "score": [0.8, 0.2],
        }
    )
    path = tmp_path / "mavedb.csv"
    df.to_csv(path, index=False)

    result = build_mavedb_dataset(
        input_path=path,
        score_col="score",
        hgvs_col="hgvs_pro",
        wt_sequence=wt_sequence,
        drop_failed=True,
        add_relative_score=False,
    )

    assert len(result) == 1
    assert result.iloc[0]["status"] == "OK"


def test_build_proteingym_dataset_with_fasta(tmp_path, proteingym_like_df: pd.DataFrame, wt_sequence: str):
    csv_path = tmp_path / "proteingym.csv"
    proteingym_like_df.to_csv(csv_path, index=False)

    fasta_path = tmp_path / "wt.fasta"
    fasta_path.write_text(f">wt\n{wt_sequence}\n", encoding="utf-8")

    result = build_proteingym_dataset(
        input_path=csv_path,
        score_col="DMS_score",
        variant_col="variant",
        wt_fasta_path=fasta_path,
    )

    assert (result["wt_sequence"] == wt_sequence).all()


def test_build_proteingym_dataset_with_dna_wt(tmp_path):
    df = pd.DataFrame(
        {
            "variant": ["WT", "M1A"],
            "DMS_score": [1.0, 0.5],
        }
    )
    csv_path = tmp_path / "proteingym.csv"
    df.to_csv(csv_path, index=False)

    dna_wt = "ATGAAAACC"  # MKT

    result = build_proteingym_dataset(
        input_path=csv_path,
        score_col="DMS_score",
        variant_col="variant",
        wt_sequence=dna_wt,
        wt_sequence_is_dna=True,
        add_relative_score=True,
    )

    assert result.iloc[0]["wt_sequence"] == "MKT"
    assert result.iloc[1]["mutated_sequence"] == "AKT"
