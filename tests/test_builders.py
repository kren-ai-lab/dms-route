from __future__ import annotations

import pandas as pd
import pytest

from dms_parser.builders import build_mavedb_dataset, build_proteingym_dataset


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