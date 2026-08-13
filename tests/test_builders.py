from __future__ import annotations

import warnings
from inspect import signature

import numpy as np
import pandas as pd
import pytest

import dms_parser.builders as builders_module
from dms_parser._wildtype import resolve_wt_score
from dms_parser.builders import build_mavedb_dataset, build_proteingym_dataset
from dms_parser.constants import NEUTRAL_LABEL
from dms_parser.exceptions import (
    InvalidDatasetError,
    InvalidPipelineOptionError,
    MissingWildTypeError,
    WildTypeConflictError,
)


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


@pytest.mark.parametrize(
    "builder",
    [build_mavedb_dataset, build_proteingym_dataset],
)
def test_builder_dna_options_are_absent(builder) -> None:
    parameters = signature(builder).parameters

    assert not {
        "wt_sequence_is_dna",
        "dna_frame",
        "stop_at_stop",
    }.intersection(parameters)


@pytest.mark.parametrize(
    "options",
    [
        {"add_relative_score": True, "relative_output_col": "score_raw"},
        {
            "add_relative_score": True,
            "relative_output_col": "parsed_position",
        },
        {
            "add_relative_score": True,
            "relative_output_col": "parsed_wt_aa",
        },
        {
            "add_relative_score": True,
            "relative_output_col": "parsed_mut_aa",
        },
        {
            "add_relative_score": True,
            "relative_output_col": "parsed_is_wildtype",
        },
        {
            "add_relative_score": True,
            "relative_output_col": "parsed_n_mutations",
        },
        {
            "add_relative_score": True,
            "relative_output_col": "parsed_mutations",
        },
        {
            "add_relative_score": True,
            "add_binary_label": True,
            "binary_output_col": "score_raw",
        },
        {
            "add_relative_score": True,
            "add_binary_label": True,
            "relative_output_col": "generated",
            "binary_output_col": "generated",
        },
        {"add_relative_score": True, "relative_output_col": "status"},
    ],
)
@pytest.mark.parametrize(
    "builder",
    [build_mavedb_dataset, build_proteingym_dataset],
)
def test_builder_rejects_static_output_collisions_before_input_io(
    builder,
    options: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        builders_module,
        "read_table",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("invalid output column reached input I/O")
        ),
    )

    with pytest.raises(InvalidPipelineOptionError):
        builder(
            input_path="unused.csv",
            score_col="DMS_score",
            wt_sequence="MKT",
            **options,
        )


def test_builder_rejects_collision_with_acquired_source_column(
    tmp_path,
    wt_sequence: str,
) -> None:
    path = tmp_path / "proteingym.csv"
    pd.DataFrame(
        {
            "mutant": ["WT", "M1A"],
            "DMS_score": [1.0, 0.5],
            "custom_relative": [9.0, 8.0],
        }
    ).to_csv(path, index=False)

    with pytest.raises(InvalidDatasetError, match="cannot overwrite"):
        build_proteingym_dataset(
            input_path=path,
            score_col="DMS_score",
            variant_col="mutant",
            wt_sequence=wt_sequence,
            add_relative_score=True,
            relative_method="difference",
            relative_output_col="custom_relative",
        )


def test_valid_custom_output_columns_preserve_score_raw(
    tmp_path,
    wt_sequence: str,
) -> None:
    result = _build_source_dataset(
        tmp_path,
        "proteingym",
        ["WT", "M1A"],
        [1.0, 0.5],
        wt_sequence=wt_sequence,
        add_relative_score=True,
        relative_method="difference",
        relative_output_col="custom_relative",
        add_binary_label=True,
        binary_output_col="custom_label",
    )

    assert result["score_raw"].tolist() == [1.0, 0.5]
    assert result["custom_relative"].tolist() == [0.0, -0.5]
    assert result["custom_label"].tolist() == [NEUTRAL_LABEL, 0]


def test_non_finite_transform_result_reports_method_and_dataset(
    tmp_path,
    wt_sequence: str,
) -> None:
    with pytest.raises(
        InvalidDatasetError,
        match="Transformation 'difference' for 'overflow-assay' failed",
    ):
        _build_source_dataset(
            tmp_path,
            "proteingym",
            ["M1A"],
            [np.finfo(float).max],
            wt_sequence=wt_sequence,
            wt_score=-np.finfo(float).max,
            dataset_id="overflow-assay",
            add_relative_score=True,
            relative_method="difference",
            relative_output_col="relative",
            require_wt_for_transforms=True,
        )


def test_dataset_without_wt_keeps_source_rows_by_default(
    tmp_path,
    wt_sequence: str,
) -> None:
    result = _build_source_dataset(
        tmp_path,
        "proteingym",
        _source_variants("proteingym"),
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
    assert observed[
        ["parsed_position", "parsed_wt_aa", "parsed_mut_aa"]
    ].isna().all()


@pytest.mark.parametrize("hgvs_pro", ["p.[=]", "p.[=;=]"])
def test_mavedb_bracketed_equality_builds_observed_wt(
    hgvs_pro: str,
    tmp_path,
    wt_sequence: str,
) -> None:
    result = _build_source_dataset(
        tmp_path,
        "mavedb",
        [hgvs_pro],
        [2.5],
        wt_sequence=wt_sequence,
    )

    observed = result.iloc[0]
    assert observed["status"] == "OK"
    assert observed["variant"] == ""
    assert observed["mutated_sequence"] == wt_sequence
    assert observed["is_wildtype"] == True
    assert observed["n_mutations"] == 0
    assert observed["is_synthetic"] == False
    assert observed["score_raw"] == 2.5


def test_mavedb_mixed_equality_builds_only_substitutions(
    tmp_path,
    wt_sequence: str,
) -> None:
    result = _build_source_dataset(
        tmp_path,
        "mavedb",
        ["p.[=;Met1Ala]", "p.[Met1Ala;=;Lys2Arg]"],
        [0.8, 1.2],
        wt_sequence=wt_sequence,
    )

    assert result["status"].tolist() == ["OK", "OK"]
    assert result["variant"].tolist() == ["M1A", "M1A;K2R"]
    assert result["n_mutations"].tolist() == [1, 2]
    assert result["is_wildtype"].tolist() == [False, False]
    assert result["is_synthetic"].tolist() == [False, False]
    assert result["score_raw"].tolist() == [0.8, 1.2]
    assert result["mutated_sequence"].tolist() == [
        "AKTAYIAKQRQISFVKSHFSRQDILDLWQ",
        "ARTAYIAKQRQISFVKSHFSRQDILDLWQ",
    ]
    assert result.loc[
        0,
        ["parsed_position", "parsed_wt_aa", "parsed_mut_aa"],
    ].tolist() == [1, "M", "A"]
    assert result.loc[
        1,
        ["parsed_position", "parsed_wt_aa", "parsed_mut_aa"],
    ].isna().all()


def test_raw_mavedb_build_preserves_ambiguous_observed_wt_scores(
    tmp_path,
    wt_sequence: str,
) -> None:
    result = _build_source_dataset(
        tmp_path,
        "mavedb",
        ["p.=", "p.[=;=]", "p.Met1Ala"],
        [1.0, 2.0, 0.5],
        wt_sequence=wt_sequence,
    )

    assert result["score_raw"].tolist() == [1.0, 2.0, 0.5]
    assert result["is_wildtype"].tolist() == [True, True, False]
    assert result["is_synthetic"].tolist() == [False, False, False]
    resolution = result.attrs["dms_parser_wt_resolution"]
    assert resolution["score"] is None
    assert resolution["score_provenance"] is None
    assert resolution["observed_wildtype_row"] is True
    assert (
        resolution["score_unavailable_reason"]
        == "conflicting_observed_wildtype_scores"
    )


@pytest.mark.parametrize("require_wt_for_transforms", [False, True])
def test_mavedb_transform_rejects_ambiguous_observed_wt_scores(
    require_wt_for_transforms: bool,
    tmp_path,
    wt_sequence: str,
) -> None:
    with pytest.raises(WildTypeConflictError, match="Conflicting WT scores"):
        _build_source_dataset(
            tmp_path,
            "mavedb",
            ["p.=", "p.[=;=]", "p.Met1Ala"],
            [1.0, 2.0, 0.5],
            wt_sequence=wt_sequence,
            add_relative_score=True,
            relative_method="difference",
            require_wt_for_transforms=require_wt_for_transforms,
        )


def test_wt_score_fallback_rejects_ambiguous_observed_evidence(
    tmp_path,
    wt_sequence: str,
) -> None:
    with pytest.raises(WildTypeConflictError, match="Conflicting WT scores"):
        _build_source_dataset(
            tmp_path,
            "mavedb",
            ["p.=", "p.[=;=]"],
            [1.0, 2.0],
            wt_sequence=wt_sequence,
            wt_score=1.0,
        )


@pytest.mark.parametrize("scores", [[1.0, 1.0], [1.0, 1.0 + 1e-10]])
def test_equal_or_equivalent_observed_wt_scores_still_resolve(
    scores: list[float],
    tmp_path,
    wt_sequence: str,
) -> None:
    result = _build_source_dataset(
        tmp_path,
        "mavedb",
        ["p.=", "p.[=;=]"],
        scores,
        wt_sequence=wt_sequence,
    )

    resolution = result.attrs["dms_parser_wt_resolution"]
    assert resolution["score"] == 1.0
    assert resolution["score_provenance"] == "observed_wildtype_row"
    assert resolution["score_unavailable_reason"] is None


def test_automatic_wt_score_evidence_keeps_ambiguous_rows_strict() -> None:
    table = pd.DataFrame(
        {
            "score_raw": [1.0, 2.0],
            "is_wildtype": [True, True],
            "status": ["OK", "OK"],
        }
    )

    with pytest.raises(WildTypeConflictError, match="Conflicting WT scores"):
        resolve_wt_score(
            table,
            None,
            dataset_id="dataset",
            automatic=(("metadata", 1.0),),
            allow_ambiguous_observed_scores=True,
        )


def test_drop_failed_precedes_synthetic_wt_insertion(
    tmp_path,
    wt_sequence: str,
) -> None:
    result = _build_source_dataset(
        tmp_path,
        "proteingym",
        ["not-a-variant", "M1A"],
        [9.0, 0.5],
        wt_sequence=wt_sequence,
        drop_failed=True,
        add_wildtype_row=True,
    )

    assert isinstance(result.index, pd.RangeIndex)
    assert result["is_synthetic"].tolist() == [True, False]
    assert result["status"].tolist() == ["OK", "OK"]
    assert result["score_raw"].iloc[1:].tolist() == [0.5]


def test_scoreless_synthetic_wt_skips_relative_transform(
    tmp_path,
    wt_sequence: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("WARNING", logger="dms_parser.builders")

    result = _build_source_dataset(
        tmp_path,
        "proteingym",
        _source_variants("proteingym"),
        [0.5],
        wt_sequence=wt_sequence,
        add_wildtype_row=True,
        add_relative_score=True,
    )

    assert "score_log_ratio" not in result.columns
    assert "reason=no_valid_numeric_wild_type_score" in caplog.text


def test_scoreless_synthetic_wt_can_be_required_for_transform(
    tmp_path,
    wt_sequence: str,
) -> None:
    with pytest.raises(MissingWildTypeError, match="provide wt_score"):
        _build_source_dataset(
            tmp_path,
            "proteingym",
            _source_variants("proteingym"),
            [0.5],
            wt_sequence=wt_sequence,
            add_wildtype_row=True,
            add_relative_score=True,
            require_wt_for_transforms=True,
        )


def test_manual_wt_score_enables_difference_without_observed_wt(
    tmp_path,
    wt_sequence: str,
) -> None:
    result = _build_source_dataset(
        tmp_path,
        "proteingym",
        ["M1A"],
        [0.5],
        wt_sequence=wt_sequence,
        wt_score=-0.25,
        add_relative_score=True,
        relative_method="difference",
        relative_output_col="score_difference",
        add_wildtype_row=True,
        require_wt_for_transforms=True,
    )

    assert pd.isna(result.iloc[0]["score_raw"])
    assert pd.isna(result.iloc[0]["score_difference"])
    assert result.iloc[1]["score_raw"] == 0.5
    assert result.iloc[1]["score_difference"] == 0.75
    assert result.attrs["dms_parser_wt_resolution"]["score"] == -0.25
    assert (
        result.attrs["dms_parser_wt_resolution"]["score_provenance"]
        == "user_fallback"
    )


def test_manual_wt_score_cannot_replace_observed_wt(
    tmp_path,
    wt_sequence: str,
) -> None:
    with pytest.raises(WildTypeConflictError, match="fallback conflicts"):
        _build_source_dataset(
            tmp_path,
            "proteingym",
            ["WT", "M1A"],
            [1.0, 0.5],
            wt_sequence=wt_sequence,
            wt_score=0.0,
        )


def test_matching_manual_wt_score_preserves_observed_provenance(
    tmp_path,
    wt_sequence: str,
) -> None:
    result = _build_source_dataset(
        tmp_path,
        "proteingym",
        ["WT", "M1A"],
        [1.0, 0.5],
        wt_sequence=wt_sequence,
        wt_score=1.0,
        add_relative_score=True,
        relative_method="difference",
    )

    resolution = result.attrs["dms_parser_wt_resolution"]
    assert resolution["score"] == 1.0
    assert resolution["score_provenance"] == "observed_wildtype_row"
    assert result["score_raw"].tolist() == [1.0, 0.5]


def test_proteingym_scores_are_unchanged_without_transformations(
    tmp_path,
    wt_sequence: str,
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
    assert result["variant"].tolist() == ["M1A", "K2K", "T3Y"]
    assert result["parsed_position"].tolist() == [1, 2, 3]
    assert result["parsed_wt_aa"].tolist() == ["M", "K", "T"]
    assert result["parsed_mut_aa"].tolist() == ["A", "K", "Y"]
    pd.testing.assert_series_equal(
        result["score_raw"].reset_index(drop=True),
        mavedb_like_df["score"],
        check_names=False,
    )


def test_build_mavedb_dataset_with_unsupported_variant(tmp_path, wt_sequence: str):
    df = pd.DataFrame(
        {
            "hgvs_pro": [
                "p.Met1Ala",
                "p.[Gly10del]",
                "p.[=;Gly10del]",
                "p.Ala1del",
            ],
            "score": [0.8, 0.2, 0.1, -0.3],
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

    assert list(result["status"]) == [
        "OK",
        "Unsupported",
        "Unsupported",
        "Error",
    ]
    assert result.loc[
        0,
        ["parsed_position", "parsed_wt_aa", "parsed_mut_aa"],
    ].tolist() == [1, "M", "A"]
    assert result.loc[
        1:,
        ["parsed_position", "parsed_wt_aa", "parsed_mut_aa"],
    ].isna().all().all()


def test_build_mavedb_dataset_supports_bounded_indels_and_transforms(tmp_path):
    path = tmp_path / "mavedb.csv"
    pd.DataFrame(
        {
            "hgvs_pro": ["p.=", "p.Cys2del", "p.Asp1_Cys2insLys"],
            "score": [1.0, 0.25, 1.5],
        }
    ).to_csv(path, index=False)

    result = build_mavedb_dataset(
        input_path=path,
        score_col="score",
        wt_sequence="DCA",
        add_relative_score=True,
        relative_method="difference",
        relative_output_col="score_delta",
    )

    assert result["variant"].tolist() == ["", "C2del", "D1_C2insK"]
    assert result["mutated_sequence"].tolist() == ["DCA", "DA", "DKCA"]
    assert result["score_raw"].tolist() == [1.0, 0.25, 1.5]
    assert result["score_delta"].tolist() == [0.0, -0.75, 0.5]
    assert result["status"].tolist() == ["OK", "OK", "OK"]
    assert result["is_wildtype"].tolist() == [True, False, False]
    assert result["is_synthetic"].tolist() == [False, False, False]
    assert result["n_mutations"].tolist() == [0, 1, 1]
    assert result[
        ["parsed_position", "parsed_wt_aa", "parsed_mut_aa"]
    ].isna().all().all()


def test_build_mavedb_dataset_indels_preserve_drop_failed_and_synthetic_wt(tmp_path):
    path = tmp_path / "mavedb.csv"
    pd.DataFrame(
        {
            "hgvs_pro": ["p.Cys2del", "p.[Cys2del]"],
            "score": [0.25, -1.0],
        }
    ).to_csv(path, index=False)

    result = build_mavedb_dataset(
        input_path=path,
        score_col="score",
        wt_sequence="DCA",
        drop_failed=True,
        add_wildtype_row=True,
        add_relative_score=False,
    )

    assert result["variant"].tolist() == ["", "C2del"]
    assert result["mutated_sequence"].tolist() == ["DCA", "DA"]
    assert result["status"].tolist() == ["OK", "OK"]
    assert result["is_synthetic"].tolist() == [True, False]
    assert result["n_mutations"].tolist() == [0, 1]
    assert result[
        ["parsed_position", "parsed_wt_aa", "parsed_mut_aa"]
    ].isna().all().all()
    assert pd.isna(result.iloc[0]["score_raw"])
    assert result.iloc[1]["score_raw"] == 0.25


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


@pytest.mark.parametrize("source", ["mavedb", "proteingym"])
def test_builder_accepts_protein_fasta(
    tmp_path,
    source: str,
    wt_sequence: str,
) -> None:
    fasta_path = tmp_path / "wt.fasta"
    fasta_path.write_text(f">wt\n{wt_sequence}\n", encoding="utf-8")

    result = _build_source_dataset(
        tmp_path,
        source,
        _source_variants(source, include_wt=True),
        [1.0, 0.5],
        wt_fasta_path=fasta_path,
    )

    assert (result["wt_sequence"] == wt_sequence).all()


@pytest.mark.parametrize("source", ["mavedb", "proteingym"])
def test_builder_preserves_dna_looking_manual_protein_sequence(
    tmp_path,
    source: str,
) -> None:
    result = _build_source_dataset(
        tmp_path,
        source,
        _source_variants(source, include_wt=True)[:1],
        [1.0],
        wt_sequence="ACGT",
    )

    assert result.iloc[0]["wt_sequence"] == "ACGT"
    assert result.iloc[0]["mutated_sequence"] == "ACGT"
